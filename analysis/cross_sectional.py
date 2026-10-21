"""Causal cross-sectional intraday portfolio research on aligned OHLCV bars.

Signals are calculated at a completed bar close, filled at the next bar open,
and exited at a later bar open in the same session.  Each portfolio holds the
same gross long and short exposure, so results measure stock-selection edge
rather than a hidden directional bet on the market.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Any, Iterable

import numpy as np
import pandas as pd

from analysis.metrics import cagr, max_drawdown, sharpe
from analysis.scalping import prepare_intraday_bars
from analysis.vectorized_screen import resolve_device


PARAM_FIELDS = [
    "signal_family", "decision_bar", "lookback_bars", "hold_bars", "top_n",
    "min_dispersion_bps", "min_relative_volume", "max_market_move_bps",
]


@dataclass(frozen=True)
class CrossSectionalGrammar:
    signal_families: tuple[str, ...] = ("momentum", "reversal")
    decision_bars: tuple[int, ...] = (6, 24, 48)
    lookback_bars: tuple[int, ...] = (3, 6, 12)
    hold_bars: tuple[int, ...] = (3, 6, 12)
    top_n: tuple[int, ...] = (2, 3, 5)
    min_dispersion_bps: tuple[float, ...] = (10.0, 25.0, 50.0)
    min_relative_volume: tuple[float, ...] = (0.75, 1.0)
    max_market_move_bps: tuple[float, ...] = (50.0,)

    def __post_init__(self) -> None:
        if not self.signal_families or set(self.signal_families) - {"momentum", "reversal"}:
            raise ValueError("signal family must be momentum or reversal")
        groups = (
            self.decision_bars, self.lookback_bars, self.hold_bars, self.top_n,
            self.min_dispersion_bps, self.min_relative_volume,
            self.max_market_move_bps,
        )
        if any(not values or any(value <= 0 for value in values) for values in groups):
            raise ValueError("all cross-sectional grammar values must be positive")


@dataclass(frozen=True)
class CrossSectionalPanel:
    tickers: tuple[str, ...]
    timestamps: pd.DatetimeIndex
    session_dates: np.ndarray
    sessions: np.ndarray
    bar_in_session: np.ndarray
    open: np.ndarray
    close: np.ndarray
    relative_volume: np.ndarray
    beta: np.ndarray
    benchmark_open: np.ndarray
    benchmark_close: np.ndarray

    def __post_init__(self) -> None:
        rows, assets = self.open.shape
        if assets != len(self.tickers) or self.close.shape != (rows, assets):
            raise ValueError("panel prices do not match ticker dimensions")
        if rows < 2 or len(self.sessions) != rows:
            raise ValueError("panel requires at least two aligned bars")
        if len(self.benchmark_close) != rows:
            raise ValueError("benchmark does not align with panel")


def generate_cross_sectional_candidates(
    grammar: CrossSectionalGrammar = CrossSectionalGrammar(),
) -> list[dict[str, Any]]:
    return [
        {
            "signal_family": family,
            "decision_bar": int(decision_bar),
            "lookback_bars": int(lookback),
            "hold_bars": int(hold),
            "top_n": int(top_n),
            "min_dispersion_bps": float(dispersion),
            "min_relative_volume": float(relative_volume),
            "max_market_move_bps": float(market_move),
        }
        for family, decision_bar, lookback, hold, top_n, dispersion, relative_volume, market_move
        in product(
            grammar.signal_families, grammar.decision_bars,
            grammar.lookback_bars, grammar.hold_bars,
            grammar.top_n, grammar.min_dispersion_bps,
            grammar.min_relative_volume, grammar.max_market_move_bps,
        )
    ]


def _causal_relative_volume(volume: pd.DataFrame, bar_index: pd.Series) -> pd.DataFrame:
    expected = pd.DataFrame(index=volume.index, columns=volume.columns, dtype=float)
    for column in volume:
        expected[column] = volume[column].groupby(bar_index, sort=False).transform(
            lambda values: values.shift(1).rolling(20, min_periods=5).median()
        )
    return volume / expected.where(expected > 0)


def _causal_beta(close: pd.DataFrame, benchmark: pd.Series, sessions: pd.Series) -> pd.DataFrame:
    same_session = sessions.eq(sessions.shift(1))
    market_return = benchmark.pct_change(fill_method=None).where(same_session)
    result = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    market_variance = market_return.rolling(390, min_periods=60).var(ddof=0)
    for column in close:
        stock_return = close[column].pct_change(fill_method=None).where(same_session)
        covariance = stock_return.rolling(390, min_periods=60).cov(market_return, ddof=0)
        result[column] = (covariance / market_variance.where(market_variance > 1e-14)).clip(0.2, 3.0)
    return result.fillna(1.0)


def prepare_cross_sectional_panel(
    bars_by_ticker: dict[str, pd.DataFrame],
    benchmark_bars: pd.DataFrame,
) -> CrossSectionalPanel:
    """Inner-align stocks and benchmark and create strictly causal features."""
    if len(bars_by_ticker) < 4:
        raise ValueError("cross-sectional panel requires at least four stocks")
    tickers = tuple(bars_by_ticker)
    prepared = {ticker: prepare_intraday_bars(frame) for ticker, frame in bars_by_ticker.items()}
    benchmark = prepare_intraday_bars(benchmark_bars)
    indexed = {
        ticker: frame.set_index("ts")[["open", "close", "volume"]]
        for ticker, frame in prepared.items()
    }
    benchmark_indexed = benchmark.set_index("ts")[["open", "close"]]
    common = benchmark_indexed.index
    for frame in indexed.values():
        common = common.intersection(frame.index)
    common = common.sort_values()
    if len(common) < 2:
        raise ValueError("stocks and benchmark have no aligned intraday bars")

    opens = pd.DataFrame({ticker: indexed[ticker].loc[common, "open"] for ticker in tickers})
    closes = pd.DataFrame({ticker: indexed[ticker].loc[common, "close"] for ticker in tickers})
    volume = pd.DataFrame({ticker: indexed[ticker].loc[common, "volume"] for ticker in tickers})
    eastern = common.tz_convert("America/New_York")
    session_dates = np.asarray(eastern.date, dtype=object)
    session_codes, _ = pd.factorize(session_dates, sort=False)
    bar_index = pd.Series((eastern.hour * 60 + eastern.minute - 570) // 5, index=common)
    session_series = pd.Series(session_codes, index=common)
    benchmark_close = benchmark_indexed.loc[common, "close"]
    relative_volume = _causal_relative_volume(volume, bar_index)
    beta = _causal_beta(closes, benchmark_close, session_series)
    return CrossSectionalPanel(
        tickers=tickers,
        timestamps=common,
        session_dates=session_dates,
        sessions=session_codes.astype(np.int64),
        bar_in_session=bar_index.to_numpy(dtype=np.int64),
        open=opens.to_numpy(dtype=np.float64),
        close=closes.to_numpy(dtype=np.float64),
        relative_volume=relative_volume.to_numpy(dtype=np.float64),
        beta=beta.to_numpy(dtype=np.float64),
        benchmark_open=benchmark_indexed.loc[common, "open"].to_numpy(dtype=np.float64),
        benchmark_close=benchmark_close.to_numpy(dtype=np.float64),
    )


def slice_panel_sessions(panel: CrossSectionalPanel, session_ids: Iterable[int]) -> CrossSectionalPanel:
    wanted = np.asarray(list(session_ids), dtype=np.int64)
    mask = np.isin(panel.sessions, wanted)
    if not mask.any():
        raise ValueError("session slice is empty")
    new_sessions, _ = pd.factorize(panel.session_dates[mask], sort=False)
    return CrossSectionalPanel(
        tickers=panel.tickers,
        timestamps=panel.timestamps[mask],
        session_dates=panel.session_dates[mask],
        sessions=new_sessions.astype(np.int64),
        bar_in_session=panel.bar_in_session[mask],
        open=panel.open[mask], close=panel.close[mask],
        relative_volume=panel.relative_volume[mask], beta=panel.beta[mask],
        benchmark_open=panel.benchmark_open[mask],
        benchmark_close=panel.benchmark_close[mask],
    )


def split_panel_discovery_holdout(
    panel: CrossSectionalPanel, *, discovery_fraction: float = 0.8, n_windows: int = 4,
) -> tuple[list[CrossSectionalPanel], CrossSectionalPanel]:
    if not 0.5 <= discovery_fraction < 1.0:
        raise ValueError("discovery_fraction must be in [0.5, 1)")
    sessions = np.unique(panel.sessions)
    cut = int(len(sessions) * discovery_fraction)
    if cut < n_windows or len(sessions) - cut < 1:
        raise ValueError("not enough sessions for discovery and holdout")
    chunks = np.array_split(sessions[:cut], n_windows)
    return [slice_panel_sessions(panel, chunk) for chunk in chunks], slice_panel_sessions(panel, sessions[cut:])


def combinatorial_probability_overfit(
    evaluations: pd.DataFrame,
    param_fields: list[str] = PARAM_FIELDS,
) -> dict[str, Any]:
    """Estimate selection overfit by symmetric train/test window splits.

    For every half-window training combination, select the candidate with the
    highest mean training Sharpe and measure its percentile rank on the
    complementary windows. PBO is the fraction landing in the test-set bottom
    half. This is intentionally reported as a diagnostic, not a p-value.
    """
    windows = sorted(evaluations["window_idx"].unique())
    if len(windows) < 4 or len(windows) % 2:
        raise ValueError("PBO requires an even number of at least four windows")
    grouped = evaluations.groupby(param_fields, dropna=False, sort=False)
    keys = list(grouped.groups)
    matrix = np.asarray([
        grouped.get_group(key).set_index("window_idx")["sharpe"].reindex(windows).to_numpy(dtype=float)
        for key in keys
    ])
    matrix = matrix[np.isfinite(matrix).all(axis=1)]
    if len(matrix) < 2:
        raise ValueError("PBO requires at least two finite candidates in every window")
    percentiles = []
    half = len(windows) // 2
    for train_columns in combinations(range(len(windows)), half):
        test_columns = tuple(index for index in range(len(windows)) if index not in train_columns)
        train_score = matrix[:, train_columns].mean(axis=1)
        winner = int(np.argmax(train_score))
        test_score = matrix[:, test_columns].mean(axis=1)
        percentile = float((np.count_nonzero(test_score < test_score[winner]) + 0.5) / len(matrix))
        percentiles.append(percentile)
    return {
        "probability_overfit": float(np.mean(np.asarray(percentiles) <= 0.5)),
        "split_count": len(percentiles),
        "median_test_percentile": float(np.median(percentiles)),
        "test_percentiles": percentiles,
    }


def _candidate_result(
    panel: CrossSectionalPanel,
    candidate: dict[str, Any],
    *,
    cost_per_side_bps: float,
    device: str,
    contributions: bool,
) -> tuple[dict[str, Any], np.ndarray | None]:
    import torch

    used_device = resolve_device(device)
    dtype = torch.float64
    open_px = torch.as_tensor(panel.open, dtype=dtype, device=used_device)
    close_px = torch.as_tensor(panel.close, dtype=dtype, device=used_device)
    rvol = torch.as_tensor(panel.relative_volume, dtype=dtype, device=used_device)
    beta = torch.as_tensor(panel.beta, dtype=dtype, device=used_device)
    market_open = torch.as_tensor(panel.benchmark_open, dtype=dtype, device=used_device)
    market_close = torch.as_tensor(panel.benchmark_close, dtype=dtype, device=used_device)
    sessions = torch.as_tensor(panel.sessions, dtype=torch.int64, device=used_device)
    bars = torch.as_tensor(panel.bar_in_session, dtype=torch.int64, device=used_device)
    rows, assets = open_px.shape
    top_n = int(candidate["top_n"])
    lookback = int(candidate["lookback_bars"])
    hold = int(candidate["hold_bars"])
    if top_n * 2 > assets:
        raise ValueError("top_n needs at least twice as many instruments")
    if rows <= lookback + hold + 1:
        raise ValueError("panel is too short for candidate")

    # There is exactly one configured decision time per session. Restrict the
    # tensor before ranking, rather than sorting every intraday row and masking
    # almost all of them afterward.
    all_rows = torch.arange(rows, device=used_device)
    decision = all_rows[
        (all_rows >= lookback)
        & (all_rows < rows - hold - 1)
        & (bars == int(candidate["decision_bar"]))
    ]
    if not len(decision):
        raise ValueError("candidate has no scheduled decision bars")
    enter = decision + 1
    exit_index = enter + hold
    same_history = sessions[decision] == sessions[decision - lookback]
    same_trade = sessions[decision] == sessions[exit_index]
    stock_move = close_px[decision] / close_px[decision - lookback] - 1.0
    market_move = market_close[decision] / market_close[decision - lookback] - 1.0
    residual = stock_move - beta[decision] * market_move[:, None]
    if candidate["signal_family"] == "reversal":
        residual = -residual
    elif candidate["signal_family"] != "momentum":
        raise ValueError("unsupported signal family")

    liquid = torch.isfinite(rvol[decision]) & (rvol[decision] >= float(candidate["min_relative_volume"]))
    finite = liquid & torch.isfinite(residual)
    high_values, high_indices = torch.topk(
        torch.where(finite, residual, -torch.inf), top_n, dim=1,
    )
    low_values, low_indices = torch.topk(
        torch.where(finite, -residual, -torch.inf), top_n, dim=1,
    )
    dispersion = high_values[:, 0] + low_values[:, 0]
    valid = (
        same_history & same_trade
        & torch.isfinite(high_values).all(dim=1)
        & torch.isfinite(low_values).all(dim=1)
        & (dispersion >= float(candidate["min_dispersion_bps"]) / 10_000.0)
        & (market_move.abs() <= float(candidate["max_market_move_bps"]) / 10_000.0)
    )
    weights = torch.zeros((len(decision), assets), dtype=dtype, device=used_device)
    side_weight = 0.5 / top_n
    weights.scatter_add_(1, high_indices, torch.full_like(high_values, side_weight))
    weights.scatter_add_(1, low_indices, torch.full_like(low_values, -side_weight))
    weights = torch.where(valid[:, None], weights, 0.0)
    # Remove both cash and rolling-SPY-beta exposure from selected weights,
    # then restore unit gross exposure. A degenerate beta cross-section falls
    # back to the already dollar-neutral equal-weight book.
    selected = weights.ne(0).to(dtype)
    selected_count = selected.sum(dim=1)
    selected_beta = selected * beta[decision]
    beta_sum = selected_beta.sum(dim=1)
    beta_sq_sum = (selected_beta * beta[decision]).sum(dim=1)
    raw_beta = (weights * beta[decision]).sum(dim=1)
    determinant = selected_count * beta_sq_sum - beta_sum.square()
    safe = determinant.abs() > 1e-12
    intercept = torch.where(safe, -raw_beta * beta_sum / determinant, 0.0)
    slope = torch.where(safe, selected_count * raw_beta / determinant, 0.0)
    projected = weights - selected * (
        intercept[:, None] + slope[:, None] * beta[decision]
    )
    projected_gross = projected.abs().sum(dim=1)
    weights = torch.where(
        (safe & (projected_gross > 1e-12))[:, None],
        projected / projected_gross.clamp(min=1e-12)[:, None],
        weights,
    )
    asset_returns = open_px[exit_index] / open_px[enter] - 1.0
    gross_contribution = weights * asset_returns
    allocated_cost = weights.abs() * (2.0 * cost_per_side_bps / 10_000.0)
    net_by_asset = gross_contribution - allocated_cost
    event_returns = net_by_asset.sum(dim=1)
    event_returns = torch.where(valid, event_returns, 0.0)
    daily_log = torch.zeros(int(sessions.max().item()) + 1, dtype=dtype, device=used_device)
    daily_log.scatter_add_(0, sessions[decision], torch.log1p(event_returns.clamp(min=-0.999999)))
    daily_returns = torch.expm1(daily_log).cpu().numpy()
    equity = np.cumprod(1.0 + daily_returns)
    returns_series = pd.Series(daily_returns)
    curve = pd.Series(np.concatenate([[1.0], equity]))
    event_count = int(valid.sum().item())
    beta_exposure = (weights * beta[decision]).sum(dim=1)
    active_beta = beta_exposure[valid]
    row = {
        **candidate,
        "sharpe": float(sharpe(returns_series)),
        "cagr": float(cagr(returns_series)),
        "max_drawdown": float(max_drawdown(curve)),
        "total_return": float(equity[-1] - 1.0),
        "n_trades": event_count * top_n * 2,
        "portfolio_events": event_count,
        "positive_sessions": int((daily_returns > 0).sum()),
        "mean_abs_beta_exposure": float(active_beta.abs().mean().item()) if event_count else float("nan"),
        "simulation_device": used_device,
    }
    contribution_array = net_by_asset.sum(dim=0).cpu().numpy() if contributions else None
    return row, contribution_array


def simulate_cross_sectional_candidates(
    panel: CrossSectionalPanel,
    candidates: list[dict[str, Any]],
    *,
    cost_per_side_bps: float,
    device: str = "cuda",
) -> pd.DataFrame:
    if not candidates:
        raise ValueError("candidates must be non-empty")
    if cost_per_side_bps < 0:
        raise ValueError("cost must be non-negative")
    return pd.DataFrame([
        _candidate_result(
            panel, candidate, cost_per_side_bps=cost_per_side_bps,
            device=device, contributions=False,
        )[0]
        for candidate in candidates
    ])


def simulate_cross_sectional_winner(
    panel: CrossSectionalPanel,
    candidate: dict[str, Any],
    *,
    cost_per_side_bps: float,
    device: str = "cuda",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row, contribution = _candidate_result(
        panel, candidate, cost_per_side_bps=cost_per_side_bps,
        device=device, contributions=True,
    )
    by_instrument = pd.DataFrame({
        "ticker": panel.tickers,
        "net_additive_contribution": contribution,
    }).sort_values("net_additive_contribution", ascending=False, kind="stable")
    return pd.DataFrame([row]), by_instrument.reset_index(drop=True)


def describe_cross_sectional(candidate: dict[str, Any] | pd.Series) -> str:
    minute = 570 + int(candidate["decision_bar"]) * 5
    return (
        f"{candidate['signal_family']} residual ranking over "
        f"{int(candidate['lookback_bars']) * 5} minutes at "
        f"{minute // 60:02d}:{minute % 60:02d} ET; long/short top "
        f"{int(candidate['top_n'])}, hold {int(candidate['hold_bars']) * 5} minutes, "
        f"dispersion >= {float(candidate['min_dispersion_bps']):g} bps, "
        f"relative volume >= {float(candidate['min_relative_volume']):g}x and "
        f"market move <= {float(candidate['max_market_move_bps']):g} bps."
    )
