"""Mechanical support/resistance confluence for cross-sectional portfolios."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from analysis.cross_sectional import CrossSectionalPanel, prepare_cross_sectional_panel
from analysis.metrics import cagr, max_drawdown, sharpe
from analysis.scalping import prepare_intraday_bars
from analysis.vectorized_screen import resolve_device


LEVEL_TYPES = (
    "previous_day", "previous_week", "premarket_proxy",
    "opening_range", "swing", "vwap_band",
)
SEARCH_LEVEL_TYPES = (*LEVEL_TYPES, "any_level")
BEHAVIORS = ("breakout_retest", "rejection", "failed_breakout")
PARAM_FIELDS = [
    "level_type", "behavior", "signal_family", "decision_bar",
    "lookback_bars", "hold_bars", "top_n", "zone_atr",
    "stop_atr", "reward_risk", "min_relative_volume", "max_market_move_bps",
]


@dataclass(frozen=True)
class SupportResistanceGrammar:
    level_types: tuple[str, ...] = SEARCH_LEVEL_TYPES
    behaviors: tuple[str, ...] = BEHAVIORS
    signal_families: tuple[str, ...] = ("momentum", "reversal")
    decision_bars: tuple[int, ...] = (6, 24, 48)
    lookback_bars: tuple[int, ...] = (6, 12)
    hold_bars: tuple[int, ...] = (12, 24)
    top_n: tuple[int, ...] = (1, 2)
    zone_atr: tuple[float, ...] = (0.10, 0.25)
    stop_atr: tuple[float, ...] = (0.50,)
    reward_risk: tuple[float, ...] = (2.0,)
    min_relative_volume: tuple[float, ...] = (1.0,)
    max_market_move_bps: tuple[float, ...] = (50.0,)

    def __post_init__(self) -> None:
        if not self.level_types or set(self.level_types) - set(SEARCH_LEVEL_TYPES):
            raise ValueError("unsupported support/resistance level")
        if not self.behaviors or set(self.behaviors) - set(BEHAVIORS):
            raise ValueError("unsupported support/resistance behavior")
        if not self.signal_families or set(self.signal_families) - {"momentum", "reversal"}:
            raise ValueError("signal family must be momentum or reversal")


@dataclass(frozen=True)
class SupportResistancePanel:
    base: CrossSectionalPanel
    high: np.ndarray
    low: np.ndarray
    atr_fraction: np.ndarray
    levels: dict[str, tuple[np.ndarray, np.ndarray]]

    def __post_init__(self) -> None:
        shape = self.base.close.shape
        if self.high.shape != shape or self.low.shape != shape or self.atr_fraction.shape != shape:
            raise ValueError("support/resistance features do not match base panel")
        if set(self.levels) != set(LEVEL_TYPES):
            raise ValueError("support/resistance panel has incomplete levels")
        if any(support.shape != shape or resistance.shape != shape for support, resistance in self.levels.values()):
            raise ValueError("level matrices do not match base panel")


def generate_support_resistance_candidates(
    grammar: SupportResistanceGrammar = SupportResistanceGrammar(),
) -> list[dict[str, Any]]:
    candidates = []
    for values in product(
        grammar.level_types, grammar.behaviors, grammar.signal_families,
        grammar.decision_bars, grammar.lookback_bars, grammar.hold_bars,
        grammar.top_n, grammar.zone_atr,
        grammar.stop_atr, grammar.reward_risk,
        grammar.min_relative_volume, grammar.max_market_move_bps,
    ):
        row = dict(zip(PARAM_FIELDS, values, strict=True))
        if int(row["lookback_bars"]) > int(row["decision_bar"]):
            continue
        candidates.append(row)
    return candidates


def _premarket_levels(raw: pd.DataFrame, dates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame = raw.copy()
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    eastern = frame["ts"].dt.tz_convert("America/New_York")
    minute = eastern.dt.hour * 60 + eastern.dt.minute
    frame["session_date"] = eastern.dt.date
    premarket = frame[(minute >= 8 * 60) & (minute < 9 * 60 + 30)]
    grouped = premarket.groupby("session_date").agg(
        support=("low", "min"), resistance=("high", "max"), count=("close", "size"),
    )
    grouped.loc[grouped["count"] < 3, ["support", "resistance"]] = np.nan
    return (
        pd.Series(dates).map(grouped["support"]).to_numpy(dtype=float),
        pd.Series(dates).map(grouped["resistance"]).to_numpy(dtype=float),
    )


def _ticker_features(
    raw: pd.DataFrame,
    common: pd.DatetimeIndex,
    sessions: np.ndarray,
    bars: np.ndarray,
    dates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    regular = prepare_intraday_bars(raw).set_index("ts").loc[common].copy()
    regular["session"] = sessions
    regular["bar"] = bars
    grouped = regular.groupby("session", sort=False)

    daily = grouped.agg(support=("low", "min"), resistance=("high", "max"))
    prior_day = daily.shift(1)
    previous_day = (
        regular["session"].map(prior_day["support"]).to_numpy(dtype=float),
        regular["session"].map(prior_day["resistance"]).to_numpy(dtype=float),
    )

    week = pd.Series(pd.to_datetime(dates).to_period("W"), index=regular.index)
    weekly = regular.assign(week=week).groupby("week", sort=False).agg(
        support=("low", "min"), resistance=("high", "max"),
    ).shift(1)
    previous_week = (
        week.map(weekly["support"]).to_numpy(dtype=float),
        week.map(weekly["resistance"]).to_numpy(dtype=float),
    )

    included = regular["bar"] < 6
    opening_support = regular["low"].where(included).groupby(regular["session"]).transform("min")
    opening_resistance = regular["high"].where(included).groupby(regular["session"]).transform("max")
    opening_count = included.groupby(regular["session"]).transform("sum")
    opening_range = (
        opening_support.where(opening_count == 6).to_numpy(dtype=float),
        opening_resistance.where(opening_count == 6).to_numpy(dtype=float),
    )

    swing = (
        grouped["low"].transform(lambda values: values.shift(1).rolling(24, min_periods=6).min()).to_numpy(dtype=float),
        grouped["high"].transform(lambda values: values.shift(1).rolling(24, min_periods=6).max()).to_numpy(dtype=float),
    )

    typical = (regular["high"] + regular["low"] + regular["close"]) / 3.0
    cumulative_value = (typical * regular["volume"]).groupby(regular["session"]).cumsum()
    cumulative_volume = regular["volume"].groupby(regular["session"]).cumsum()
    vwap = cumulative_value / cumulative_volume.where(cumulative_volume > 0)
    deviation = regular["close"] - vwap
    deviation_std = deviation.groupby(regular["session"]).transform(
        lambda values: values.expanding(min_periods=6).std(ddof=0)
    )
    vwap_band = (
        (vwap - deviation_std).to_numpy(dtype=float),
        (vwap + deviation_std).to_numpy(dtype=float),
    )

    previous_close = regular["close"].shift(1)
    same_session = regular["session"].eq(regular["session"].shift(1))
    true_range = pd.concat([
        regular["high"] - regular["low"],
        (regular["high"] - previous_close).abs().where(same_session),
        (regular["low"] - previous_close).abs().where(same_session),
    ], axis=1).max(axis=1)
    atr = true_range.groupby(regular["session"]).transform(
        lambda values: values.rolling(24, min_periods=6).mean()
    ) / regular["close"]
    levels = {
        "previous_day": previous_day,
        "previous_week": previous_week,
        "premarket_proxy": _premarket_levels(raw, dates),
        "opening_range": opening_range,
        "swing": swing,
        "vwap_band": vwap_band,
    }
    return (
        regular["high"].to_numpy(dtype=float),
        regular["low"].to_numpy(dtype=float),
        atr.to_numpy(dtype=float),
        levels,
    )


def prepare_support_resistance_panel(
    bars_by_ticker: dict[str, pd.DataFrame], benchmark_bars: pd.DataFrame,
) -> SupportResistancePanel:
    base = prepare_cross_sectional_panel(bars_by_ticker, benchmark_bars)
    rows, assets = base.close.shape
    high = np.empty((rows, assets), dtype=float)
    low = np.empty((rows, assets), dtype=float)
    atr = np.empty((rows, assets), dtype=float)
    level_columns = {name: [np.empty((rows, assets)), np.empty((rows, assets))] for name in LEVEL_TYPES}
    for column, ticker in enumerate(base.tickers):
        item_high, item_low, item_atr, item_levels = _ticker_features(
            bars_by_ticker[ticker], base.timestamps, base.sessions,
            base.bar_in_session, base.session_dates,
        )
        high[:, column], low[:, column], atr[:, column] = item_high, item_low, item_atr
        for name, (support, resistance) in item_levels.items():
            level_columns[name][0][:, column] = support
            level_columns[name][1][:, column] = resistance
    return SupportResistancePanel(
        base=base, high=high, low=low, atr_fraction=atr,
        levels={name: (pair[0], pair[1]) for name, pair in level_columns.items()},
    )


def _slice(panel: SupportResistancePanel, session_ids: np.ndarray) -> SupportResistancePanel:
    from analysis.cross_sectional import slice_panel_sessions

    mask = np.isin(panel.base.sessions, session_ids)
    return SupportResistancePanel(
        base=slice_panel_sessions(panel.base, session_ids),
        high=panel.high[mask], low=panel.low[mask], atr_fraction=panel.atr_fraction[mask],
        levels={name: (support[mask], resistance[mask]) for name, (support, resistance) in panel.levels.items()},
    )


def split_support_resistance_panel(
    panel: SupportResistancePanel, *, discovery_fraction: float = 0.8, n_windows: int = 4,
) -> tuple[list[SupportResistancePanel], SupportResistancePanel]:
    sessions = np.unique(panel.base.sessions)
    cut = int(len(sessions) * discovery_fraction)
    if cut < n_windows or len(sessions) - cut < 1:
        raise ValueError("not enough sessions for discovery and evaluation")
    chunks = np.array_split(sessions[:cut], n_windows)
    return [_slice(panel, chunk) for chunk in chunks], _slice(panel, sessions[cut:])


def _behavior_masks(close, high, low, support, resistance, zone, previous_close, behavior):
    import torch

    finite = torch.isfinite(support) & torch.isfinite(resistance) & torch.isfinite(zone)
    if behavior == "breakout_retest":
        long = (previous_close > resistance + zone) & (low <= resistance + zone) & (close > resistance)
        short = (previous_close < support - zone) & (high >= support - zone) & (close < support)
    elif behavior == "rejection":
        long = (low <= support + zone) & (close > support + zone)
        short = (high >= resistance - zone) & (close < resistance - zone)
    elif behavior == "failed_breakout":
        long = (low < support - zone) & (close > support)
        short = (high > resistance + zone) & (close < resistance)
    else:
        raise ValueError("unsupported support/resistance behavior")
    return finite & long, finite & short


def simulate_support_resistance_candidates(
    panel: SupportResistancePanel,
    candidates: list[dict[str, Any]],
    *, cost_per_side_bps: float, device: str = "cuda",
    contribution_candidate: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Evaluate fixed-time confluence portfolios; tensors are cached per call."""
    import torch

    if not candidates or cost_per_side_bps < 0:
        raise ValueError("non-empty candidates and non-negative cost are required")
    used_device = resolve_device(device)
    base = panel.base
    dtype = torch.float64
    tensors = {
        "open": torch.as_tensor(base.open, dtype=dtype, device=used_device),
        "close": torch.as_tensor(base.close, dtype=dtype, device=used_device),
        "high": torch.as_tensor(panel.high, dtype=dtype, device=used_device),
        "low": torch.as_tensor(panel.low, dtype=dtype, device=used_device),
        "atr": torch.as_tensor(panel.atr_fraction, dtype=dtype, device=used_device),
        "rvol": torch.as_tensor(base.relative_volume, dtype=dtype, device=used_device),
        "beta": torch.as_tensor(base.beta, dtype=dtype, device=used_device),
        "market": torch.as_tensor(base.benchmark_close, dtype=dtype, device=used_device),
        "sessions": torch.as_tensor(base.sessions, dtype=torch.int64, device=used_device),
        "bars": torch.as_tensor(base.bar_in_session, dtype=torch.int64, device=used_device),
    }
    level_tensors = {
        name: (
            torch.as_tensor(support, dtype=dtype, device=used_device),
            torch.as_tensor(resistance, dtype=dtype, device=used_device),
        ) for name, (support, resistance) in panel.levels.items()
    }
    rows, assets = base.close.shape
    results = []
    saved_contribution = None
    all_rows = torch.arange(rows, device=used_device)
    for candidate in candidates:
        lookback, hold, top_n = (
            int(candidate["lookback_bars"]), int(candidate["hold_bars"]), int(candidate["top_n"]),
        )
        decision = all_rows[
            (all_rows >= lookback) & (all_rows < rows - hold - 1)
            & (tensors["bars"] == int(candidate["decision_bar"]))
        ]
        enter, exit_index = decision + 1, decision + 1 + hold
        same = (
            (tensors["sessions"][decision] == tensors["sessions"][decision - lookback])
            & (tensors["sessions"][decision] == tensors["sessions"][exit_index])
            & (tensors["sessions"][decision] == tensors["sessions"][decision - 1])
        )
        stock_move = tensors["close"][decision] / tensors["close"][decision - lookback] - 1
        market_move = tensors["market"][decision] / tensors["market"][decision - lookback] - 1
        score = stock_move - tensors["beta"][decision] * market_move[:, None]
        if candidate["signal_family"] == "reversal":
            score = -score
        selected_levels = (
            tuple(level_tensors.values())
            if candidate["level_type"] == "any_level"
            else (level_tensors[str(candidate["level_type"])],)
        )
        long_setup = torch.zeros_like(score, dtype=torch.bool)
        short_setup = torch.zeros_like(score, dtype=torch.bool)
        # A setup may complete on the decision bar or either of the two prior
        # completed bars. This remains causal and avoids requiring the retest
        # to coincide with one exact five-minute timestamp.
        for lag in range(3):
            setup_row = decision - lag
            zone = (
                tensors["atr"][setup_row] * tensors["close"][setup_row]
                * float(candidate["zone_atr"])
            )
            for support, resistance in selected_levels:
                item_long, item_short = _behavior_masks(
                    tensors["close"][setup_row], tensors["high"][setup_row],
                    tensors["low"][setup_row], support[setup_row],
                    resistance[setup_row], zone,
                    tensors["close"][setup_row - 1], str(candidate["behavior"]),
                )
                long_setup |= item_long
                short_setup |= item_short
        liquid = torch.isfinite(tensors["rvol"][decision]) & (
            tensors["rvol"][decision] >= float(candidate["min_relative_volume"])
        )
        long_valid = long_setup & liquid & torch.isfinite(score)
        short_valid = short_setup & liquid & torch.isfinite(score)
        high_values, high_indices = torch.topk(torch.where(long_valid, score, -torch.inf), top_n, dim=1)
        low_values, low_indices = torch.topk(torch.where(short_valid, -score, -torch.inf), top_n, dim=1)
        valid = (
            same & torch.isfinite(high_values).all(dim=1) & torch.isfinite(low_values).all(dim=1)
            & (market_move.abs() <= float(candidate["max_market_move_bps"]) / 10_000)
        )
        weights = torch.zeros((len(decision), assets), dtype=dtype, device=used_device)
        weights.scatter_add_(1, high_indices, torch.full_like(high_values, 0.5 / top_n))
        weights.scatter_add_(1, low_indices, torch.full_like(low_values, -0.5 / top_n))
        weights = torch.where(valid[:, None], weights, 0.0)

        selected = weights.ne(0).to(dtype)
        beta = tensors["beta"][decision]
        count = selected.sum(dim=1)
        beta_sum = (selected * beta).sum(dim=1)
        beta_sq = (selected * beta.square()).sum(dim=1)
        raw_beta = (weights * beta).sum(dim=1)
        determinant = count * beta_sq - beta_sum.square()
        safe = determinant.abs() > 1e-12
        intercept = torch.where(safe, -raw_beta * beta_sum / determinant, 0.0)
        slope = torch.where(safe, count * raw_beta / determinant, 0.0)
        projected = weights - selected * (intercept[:, None] + slope[:, None] * beta)
        gross = projected.abs().sum(dim=1)
        weights = torch.where(
            (safe & (gross > 1e-12))[:, None], projected / gross.clamp(min=1e-12)[:, None], weights,
        )
        entry_price = tensors["open"][enter]
        risk_distance = (
            tensors["atr"][decision] * tensors["close"][decision]
            * float(candidate["stop_atr"])
        )
        reward_distance = risk_distance * float(candidate["reward_risk"])
        future_high = torch.stack([
            tensors["high"][enter + offset] for offset in range(hold)
        ], dim=0)
        future_low = torch.stack([
            tensors["low"][enter + offset] for offset in range(hold)
        ], dim=0)
        step = torch.arange(hold, device=used_device)[:, None, None]
        sentinel = torch.full_like(step, hold)
        long_stop_at = torch.where(
            future_low <= entry_price[None] - risk_distance[None], step, sentinel,
        ).amin(dim=0)
        long_target_at = torch.where(
            future_high >= entry_price[None] + reward_distance[None], step, sentinel,
        ).amin(dim=0)
        short_stop_at = torch.where(
            future_high >= entry_price[None] + risk_distance[None], step, sentinel,
        ).amin(dim=0)
        short_target_at = torch.where(
            future_low <= entry_price[None] - reward_distance[None], step, sentinel,
        ).amin(dim=0)
        terminal_price = tensors["open"][exit_index]
        long_exit = torch.where(
            (long_stop_at < hold) & (long_stop_at <= long_target_at),
            entry_price - risk_distance,
            torch.where(long_target_at < hold, entry_price + reward_distance, terminal_price),
        )
        short_exit = torch.where(
            (short_stop_at < hold) & (short_stop_at <= short_target_at),
            entry_price + risk_distance,
            torch.where(short_target_at < hold, entry_price - reward_distance, terminal_price),
        )
        exit_price = torch.where(weights > 0, long_exit, short_exit)
        asset_return = exit_price / entry_price - 1
        by_asset = weights * asset_return - weights.abs() * (2 * cost_per_side_bps / 10_000)
        event_return = torch.where(valid, by_asset.sum(dim=1), 0.0)
        daily_log = torch.zeros(int(tensors["sessions"].max().item()) + 1, dtype=dtype, device=used_device)
        daily_log.scatter_add_(0, tensors["sessions"][decision], torch.log1p(event_return.clamp(min=-0.999999)))
        daily_return = torch.expm1(daily_log).cpu().numpy()
        equity = np.cumprod(1 + daily_return)
        curve = pd.Series(np.concatenate([[1.0], equity]))
        events = int(valid.sum().item())
        active_beta = (weights * beta).sum(dim=1)[valid]
        results.append({
            **candidate,
            "sharpe": float(sharpe(pd.Series(daily_return))),
            "cagr": float(cagr(pd.Series(daily_return))),
            "max_drawdown": float(max_drawdown(curve)),
            "total_return": float(equity[-1] - 1),
            "n_trades": events * top_n * 2,
            "portfolio_events": events,
            "positive_sessions": int((daily_return > 0).sum()),
            "mean_abs_beta_exposure": float(active_beta.abs().mean().item()) if events else float("nan"),
            "simulation_device": used_device,
        })
        if contribution_candidate is not None and all(candidate[field] == contribution_candidate[field] for field in PARAM_FIELDS):
            saved_contribution = by_asset.sum(dim=0).cpu().numpy()
    contribution_frame = None
    if saved_contribution is not None:
        contribution_frame = pd.DataFrame({
            "ticker": base.tickers, "net_additive_contribution": saved_contribution,
        }).sort_values("net_additive_contribution", ascending=False, kind="stable").reset_index(drop=True)
    return pd.DataFrame(results), contribution_frame


def describe_support_resistance(candidate: dict[str, Any] | pd.Series) -> str:
    minute = 570 + int(candidate["decision_bar"]) * 5
    return (
        f"3-bar {candidate['behavior'].replace('_', ' ')} at {candidate['level_type'].replace('_', ' ')} "
        f"using {candidate['signal_family']} ranking at {minute // 60:02d}:{minute % 60:02d} ET; "
        f"{int(candidate['lookback_bars']) * 5}-minute lookback, long/short top "
        f"{int(candidate['top_n'])}, {int(candidate['hold_bars']) * 5}-minute hold, "
        f"{float(candidate['zone_atr']):g} ATR zone and relative volume >= "
        f"{float(candidate['min_relative_volume']):g}x; "
        f"{float(candidate['stop_atr']):g} ATR stop / "
        f"{float(candidate['reward_risk']):g}R target."
    )
