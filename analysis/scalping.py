"""Execution-aware intraday scalping search over OHLCV bars.

Signals use only completed bars. Entries and signal exits fill at the next bar
open; every position is flattened by the final regular-session bar. Explicit
per-side costs stand in for spread, fees, and slippage absent from IEX bars.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Literal

import numpy as np
import pandas as pd

from analysis.metrics import cagr, max_drawdown, sharpe
from analysis.vectorized_screen import resolve_device


FAMILIES = ("mean_reversion", "momentum", "breakout")


@dataclass(frozen=True)
class ScalpingGrammar:
    families: tuple[str, ...] = FAMILIES
    lookback_bars: tuple[int, ...] = (3, 6, 12, 24)
    mean_reversion_thresholds: tuple[float, ...] = (0.5, 1.0, 1.5)
    return_threshold_bps: tuple[float, ...] = (2.0, 5.0, 10.0)
    max_hold_bars: tuple[int, ...] = (1, 3, 6)
    take_profit_bps: tuple[float, ...] = (5.0, 10.0, 20.0)
    stop_loss_bps: tuple[float, ...] = (5.0, 10.0, 20.0)

    def __post_init__(self) -> None:
        if not self.families or set(self.families) - set(FAMILIES):
            raise ValueError("unsupported or empty scalping family set")
        numeric_groups = (
            self.lookback_bars, self.mean_reversion_thresholds,
            self.return_threshold_bps, self.max_hold_bars,
            self.take_profit_bps, self.stop_loss_bps,
        )
        if any(not values or any(value <= 0 for value in values) for values in numeric_groups):
            raise ValueError("all scalping grammar values must be positive")


def prepare_intraday_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and restrict bars to the US regular session."""
    required = {"ts", "open", "high", "low", "close", "volume"}
    if missing := required - set(frame.columns):
        raise ValueError(f"intraday bars missing columns: {sorted(missing)}")
    bars = frame[list(required)].copy()
    bars["ts"] = pd.to_datetime(bars["ts"], utc=True)
    bars = bars.sort_values("ts", kind="stable").reset_index(drop=True)
    if bars["ts"].duplicated().any():
        raise ValueError("intraday bars contain duplicate timestamps")
    if bars[list({"open", "high", "low", "close"})].isna().any().any():
        raise ValueError("intraday OHLC contains missing values")
    if (bars[list({"open", "high", "low", "close"})] <= 0).any().any():
        raise ValueError("intraday OHLC prices must be positive")

    eastern = bars["ts"].dt.tz_convert("America/New_York")
    clock = eastern.dt.time
    regular = (
        (clock >= pd.Timestamp("09:30").time())
        & (clock <= pd.Timestamp("16:00").time())
        & (eastern.dt.dayofweek < 5)
    )
    bars = bars.loc[regular].reset_index(drop=True)
    if bars.empty:
        raise ValueError("no regular-session bars")
    eastern = bars["ts"].dt.tz_convert("America/New_York")
    dates = pd.Series(eastern.dt.date, index=bars.index)
    bars["session_date"] = dates
    bars["session"], _ = pd.factorize(dates, sort=False)
    bars["minute_of_day"] = eastern.dt.hour * 60 + eastern.dt.minute
    bars["last_bar"] = bars["session"].ne(bars["session"].shift(-1))
    return bars


def split_discovery_holdout(
    bars: pd.DataFrame,
    *,
    discovery_fraction: float = 0.8,
    n_windows: int = 4,
) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    """Split whole sessions into discovery windows plus untouched holdout."""
    if not 0.5 <= discovery_fraction < 1.0:
        raise ValueError("discovery_fraction must be in [0.5, 1)")
    sessions = np.sort(bars["session"].unique())
    cut = int(len(sessions) * discovery_fraction)
    if cut < n_windows or len(sessions) - cut < 1:
        raise ValueError("not enough sessions for discovery and holdout")
    discovery_sessions = sessions[:cut]
    chunks = np.array_split(discovery_sessions, n_windows)
    windows = [
        bars[bars["session"].isin(chunk)].reset_index(drop=True)
        for chunk in chunks
    ]
    holdout = bars[bars["session"].isin(sessions[cut:])].reset_index(drop=True)
    return windows, holdout

def align_common_sessions(
    bars_by_ticker: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Restrict every instrument to identical complete trading sessions."""
    if len(bars_by_ticker) < 2:
        raise ValueError("at least two instruments are required")
    session_sets = [set(frame["session_date"]) for frame in bars_by_ticker.values()]
    common = set.intersection(*session_sets)
    if len(common) < 2:
        raise ValueError("instruments have fewer than two common sessions")
    ordered = sorted(common)
    session_ids = {date: idx for idx, date in enumerate(ordered)}
    aligned: dict[str, pd.DataFrame] = {}
    for ticker, frame in bars_by_ticker.items():
        selected = frame[frame["session_date"].isin(common)].copy()
        selected["session"] = selected["session_date"].map(session_ids).astype(int)
        selected["last_bar"] = selected["session"].ne(selected["session"].shift(-1))
        aligned[ticker] = selected.reset_index(drop=True)
    return aligned


def aggregate_entry_screens(screened_by_instrument: pd.DataFrame) -> pd.DataFrame:
    """Rank entry rules by their median edge across distinct instruments."""
    required = {"ticker", "entry_rule_id", "screen_score", "screen_trades"}
    if missing := required - set(screened_by_instrument.columns):
        raise ValueError(f"instrument screens missing columns: {sorted(missing)}")
    keys = [
        "family", "lookback_bars", "entry_threshold",
        "max_hold_bars", "entry_rule_id",
    ]
    return (
        screened_by_instrument.groupby(keys, as_index=False, dropna=False)
        .agg(
            screen_trades=("screen_trades", "sum"),
            screen_edge_bps=("screen_edge_bps", "median"),
            screen_win_rate=("screen_win_rate", "median"),
            screen_t_stat=("screen_t_stat", "median"),
            screen_score=("screen_score", "median"),
            instrument_screen_count=("ticker", "nunique"),
            screen_device=("screen_device", "first"),
        )
        .sort_values("screen_score", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


def aggregate_instrument_holdout(holdout: pd.DataFrame) -> pd.DataFrame:
    """Summarize a generic rule's holdout robustness across instruments."""
    required = {
        "ticker", "cost_scenario_bps_per_side", "total_return", "sharpe",
        "n_trades", "excess_total_return", "benchmark_total_return",
        "benchmark_sharpe",
    }
    if missing := required - set(holdout.columns):
        raise ValueError(f"instrument holdout missing columns: {sorted(missing)}")
    rows = []
    for cost, group in holdout.groupby("cost_scenario_bps_per_side", sort=True):
        rows.append({
            "cost_scenario_bps_per_side": float(cost),
            "instrument_count": int(group["ticker"].nunique()),
            "total_return": float(group["total_return"].mean()),
            "median_total_return": float(group["total_return"].median()),
            "worst_total_return": float(group["total_return"].min()),
            "sharpe": float(group["sharpe"].median()),
            "benchmark_total_return": float(group["benchmark_total_return"].median()),
            "benchmark_sharpe": float(group["benchmark_sharpe"].median()),
            "excess_total_return": float(group["excess_total_return"].median()),
            "n_trades": int(group["n_trades"].sum()),
            "positive_instruments": int((group["total_return"] > 0).sum()),
            "beat_spy_instruments": int((group["excess_total_return"] > 0).sum()),
        })
    return pd.DataFrame(rows)


def aggregate_group_holdout(holdout: pd.DataFrame) -> pd.DataFrame:
    """Summarize holdout robustness independently for each asset sleeve."""
    if "asset_group" not in holdout.columns:
        raise ValueError("instrument holdout missing columns: ['asset_group']")
    rows = []
    for asset_group, group in holdout.groupby("asset_group", sort=True):
        aggregate = aggregate_instrument_holdout(group)
        aggregate.insert(0, "asset_group", asset_group)
        rows.append(aggregate)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()



def generate_entry_rules(grammar: ScalpingGrammar = ScalpingGrammar()) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for family in grammar.families:
        thresholds = (
            grammar.mean_reversion_thresholds
            if family == "mean_reversion"
            else grammar.return_threshold_bps
        )
        for lookback, threshold, hold in product(
            grammar.lookback_bars, thresholds, grammar.max_hold_bars,
        ):
            rules.append({
                "family": family,
                "lookback_bars": int(lookback),
                "entry_threshold": float(threshold),
                "max_hold_bars": int(hold),
                "entry_rule_id": f"{family}__lb{lookback}__th{threshold:g}__hold{hold}",
            })
    return rules


def expand_exit_rules(
    entry_rules: list[dict[str, Any]],
    grammar: ScalpingGrammar,
) -> list[dict[str, Any]]:
    return [
        {
            **rule,
            "take_profit_bps": float(take),
            "stop_loss_bps": float(stop),
        }
        for rule in entry_rules
        for take, stop in product(grammar.take_profit_bps, grammar.stop_loss_bps)
    ]


def _signal_cache(
    bars: pd.DataFrame,
    rules: list[dict[str, Any]],
) -> dict[tuple[str, int, float], np.ndarray]:
    close = bars["close"].to_numpy(dtype=float)
    session = bars["session"].to_numpy()
    cache: dict[tuple[str, int, float], np.ndarray] = {}
    for rule in rules:
        key = (
            str(rule["family"]),
            int(rule["lookback_bars"]),
            float(rule["entry_threshold"]),
        )
        if key in cache:
            continue
        family, lookback, threshold = key
        signal = np.zeros(len(bars), dtype=bool)
        for i in range(lookback, len(bars)):
            if session[i] != session[i - lookback]:
                continue
            history = close[i - lookback : i]
            if family == "mean_reversion":
                std = float(history.std(ddof=0))
                if std > 1e-12:
                    signal[i] = (close[i] - float(history.mean())) / std <= -threshold
            elif family == "momentum":
                signal[i] = close[i] / close[i - lookback] - 1.0 >= threshold / 10_000.0
            else:
                signal[i] = close[i] / float(history.max()) - 1.0 >= threshold / 10_000.0
        cache[key] = signal
    return cache


def screen_entry_rules(
    bars: pd.DataFrame,
    rules: list[dict[str, Any]],
    *,
    cost_per_side_bps: float = 1.0,
    device: Literal["auto", "cpu", "cuda"] = "auto",
) -> pd.DataFrame:
    """Rank entry structures by same-session forward returns on tensors."""
    import torch

    if not rules:
        raise ValueError("entry rules must be non-empty")
    used_device = resolve_device(device)
    open_px = torch.tensor(bars["open"].to_numpy(), dtype=torch.float64, device=used_device)
    close_px = torch.tensor(bars["close"].to_numpy(), dtype=torch.float64, device=used_device)
    sessions = torch.tensor(bars["session"].to_numpy(), device=used_device)
    if used_device == "cuda":
        from analysis.cuda_scalping import _signal_matrix

        signal_matrix = _signal_matrix(close_px, sessions, rules)
        cache = None
    else:
        signal_matrix = None
        cache = _signal_cache(bars, rules)
    entry_ok = torch.tensor(
        bars["minute_of_day"].between(9 * 60 + 45, 15 * 60 + 30).to_numpy(),
        dtype=torch.bool,
        device=used_device,
    )
    cost = cost_per_side_bps / 10_000.0
    rows: list[dict[str, Any]] = []
    for rule_idx, rule in enumerate(rules):
        hold = int(rule["max_hold_bars"])
        signal = (
            signal_matrix[:, rule_idx]
            if signal_matrix is not None
            else torch.tensor(
                cache[(rule["family"], int(rule["lookback_bars"]), float(rule["entry_threshold"]))],
                dtype=torch.bool,
                device=used_device,
            )
        )
        if len(bars) <= hold + 1:
            valid_returns = torch.empty(0, device=used_device)
        else:
            decision = torch.arange(0, len(bars) - hold - 1, device=used_device)
            enter = decision + 1
            exit_idx = enter + hold
            mask = signal[decision] & entry_ok[enter] & (sessions[decision] == sessions[exit_idx])
            net = (
                close_px[exit_idx] * (1.0 - cost)
                / (open_px[enter] * (1.0 + cost))
                - 1.0
            )
            valid_returns = net[mask]
        count = int(valid_returns.numel())
        if count:
            mean = float(valid_returns.mean().item())
            std = float(valid_returns.std(unbiased=False).item())
            win_rate = float((valid_returns > 0).double().mean().item())
            t_stat = mean / std * np.sqrt(count) if std > 1e-14 else float("nan")
        else:
            mean = std = win_rate = t_stat = float("nan")
        rows.append({
            **rule,
            "screen_trades": count,
            "screen_edge_bps": mean * 10_000.0,
            "screen_win_rate": win_rate,
            "screen_t_stat": t_stat,
            "screen_score": t_stat,
            "screen_device": used_device,
        })
    return pd.DataFrame(rows).sort_values(
        "screen_score", ascending=False, na_position="last", kind="stable",
    ).reset_index(drop=True)


def select_entry_finalists(
    screened: pd.DataFrame,
    count: int,
    *,
    min_screen_trades: int = 30,
) -> list[dict[str, Any]]:
    """Preserve each family, then fill remaining places by screen score."""
    valid = screened.replace([np.inf, -np.inf], np.nan).dropna(subset=["screen_score"])
    valid = valid[valid["screen_trades"] >= min_screen_trades].sort_values("screen_score", ascending=False)
    if valid.empty:
        return []
    chosen = list(valid.drop_duplicates("family", keep="first").head(count).index)
    for idx in valid.index:
        if len(chosen) >= count:
            break
        if idx not in chosen:
            chosen.append(idx)
    columns = [
        "family", "lookback_bars", "entry_threshold",
        "max_hold_bars", "entry_rule_id",
    ]
    return valid.loc[chosen, columns].to_dict(orient="records")


def simulate_scalping_candidates(
    bars: pd.DataFrame,
    candidates: list[dict[str, Any]],
    *,
    cost_per_side_bps: float = 1.0,
    benchmark_bars: pd.DataFrame | None = None,
    device: Literal["auto", "cpu", "cuda"] = "auto",
) -> pd.DataFrame:
    """Backtest candidates with next-open fills and mandatory daily flattening."""
    if not candidates:
        raise ValueError("candidates must be non-empty")
    if cost_per_side_bps < 0:
        raise ValueError("cost_per_side_bps must be non-negative")
    used_device = resolve_device(device)
    if used_device == "cuda":
        from analysis.cuda_scalping import simulate_scalping_candidates_cuda

        return simulate_scalping_candidates_cuda(
            bars,
            candidates,
            cost_per_side_bps=cost_per_side_bps,
            benchmark_bars=benchmark_bars,
        )
    cache = _signal_cache(bars, candidates)
    open_px = bars["open"].to_numpy(dtype=float)
    high_px = bars["high"].to_numpy(dtype=float)
    low_px = bars["low"].to_numpy(dtype=float)
    close_px = bars["close"].to_numpy(dtype=float)
    session = bars["session"].to_numpy()
    minutes = bars["minute_of_day"].to_numpy()
    last_bar = bars["last_bar"].to_numpy(dtype=bool)
    count = len(candidates)
    equity = np.ones(count, dtype=float)
    position = np.zeros(count, dtype=bool)
    entry_price = np.ones(count, dtype=float)
    held = np.zeros(count, dtype=int)
    trades = np.zeros(count, dtype=int)
    wins = np.zeros(count, dtype=int)
    gross_win = np.zeros(count, dtype=float)
    gross_loss = np.zeros(count, dtype=float)
    max_hold = np.array([int(c["max_hold_bars"]) for c in candidates])
    take = np.array([float(c["take_profit_bps"]) for c in candidates]) / 10_000.0
    stop = np.array([float(c["stop_loss_bps"]) for c in candidates]) / 10_000.0
    signals = np.stack([
        cache[(c["family"], int(c["lookback_bars"]), float(c["entry_threshold"]))]
        for c in candidates
    ])
    cost = cost_per_side_bps / 10_000.0
    daily_equity: list[np.ndarray] = []

    def close_positions(mask: np.ndarray, price: float | np.ndarray) -> None:
        if not mask.any():
            return
        prices = np.broadcast_to(price, (count,))[mask]
        trade_return = prices * (1.0 - cost) / entry_price[mask] - 1.0
        equity[mask] *= 1.0 + trade_return
        trades[mask] += 1
        wins[mask] += trade_return > 0
        gross_win[mask] += np.maximum(trade_return, 0.0)
        gross_loss[mask] += np.maximum(-trade_return, 0.0)
        position[mask] = False
        held[mask] = 0

    for i in range(1, len(bars)):
        same_session = session[i] == session[i - 1]
        if position.any():
            mark_return = close_px[i - 1] / entry_price - 1.0
            exit_mask = position & (
                (mark_return >= take)
                | (mark_return <= -stop)
                | (held >= max_hold)
                | (not same_session)
            )
            close_positions(exit_mask, open_px[i])

        allowed_time = 9 * 60 + 45 <= minutes[i] <= 15 * 60 + 30
        enter_mask = (
            (~position)
            & signals[:, i - 1]
            & same_session
            & allowed_time
            & (not last_bar[i])
        )
        if enter_mask.any():
            entry_price[enter_mask] = open_px[i] * (1.0 + cost)
            position[enter_mask] = True
            held[enter_mask] = 0

        held[position] += 1
        target_price = entry_price * (1.0 + take) / (1.0 - cost)
        stop_price = entry_price * (1.0 - stop) / (1.0 - cost)
        target_hit = position & (high_px[i] >= target_price)
        stop_hit = position & (low_px[i] <= stop_price)
        intrabar_exit = target_hit | stop_hit
        if intrabar_exit.any():
            close_positions(
                intrabar_exit, np.where(stop_hit, stop_price, target_price),
            )
        if last_bar[i]:
            close_positions(position.copy(), close_px[i])
            daily_equity.append(equity.copy())

    if not daily_equity:
        raise ValueError("simulation produced no complete sessions")
    curves = np.vstack(daily_equity)
    daily_returns = curves[1:] / curves[:-1] - 1.0
    benchmark_source = bars if benchmark_bars is None else benchmark_bars
    traded_dates = set(bars["session_date"])
    benchmark_close = benchmark_source.loc[
        benchmark_source["last_bar"]
        & benchmark_source["session_date"].isin(traded_dates),
        "close",
    ].to_numpy(dtype=float)
    if len(benchmark_close) < 2:
        raise ValueError("benchmark has fewer than two aligned sessions")
    benchmark_curve = benchmark_close / benchmark_close[0]
    benchmark_returns = pd.Series(benchmark_curve).pct_change().dropna()
    benchmark_metrics = {
        "benchmark_total_return": float(benchmark_curve[-1] - 1.0),
        "benchmark_sharpe": float(sharpe(benchmark_returns)),
        "benchmark_max_drawdown": float(max_drawdown(pd.Series(benchmark_curve))),
    }

    rows: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates):
        returns = pd.Series(daily_returns[:, idx])
        curve = pd.Series(curves[:, idx])
        total_return = float(curve.iloc[-1] - 1.0)
        rows.append({
            **candidate,
            "sharpe": float(sharpe(returns)),
            "cagr": float(cagr(returns)),
            "max_drawdown": float(max_drawdown(curve)),
            "total_return": total_return,
            "n_trades": int(trades[idx]),
            "win_rate": float(wins[idx] / trades[idx]) if trades[idx] else float("nan"),
            "profit_factor": (
                float(gross_win[idx] / gross_loss[idx])
                if gross_loss[idx] > 0 else float("nan")
            ),
            "excess_total_return": total_return - benchmark_metrics["benchmark_total_return"],
            "cost_per_side_bps": cost_per_side_bps,
            "simulation_device": "cpu",
            **benchmark_metrics,
        })
    return pd.DataFrame(rows)


def describe_scalper(candidate: dict[str, Any] | pd.Series) -> str:
    threshold = (
        f"{float(candidate['entry_threshold']):g}σ below its rolling mean"
        if candidate["family"] == "mean_reversion"
        else f"{float(candidate['entry_threshold']):g} bps "
             f"{'return' if candidate['family'] == 'momentum' else 'breakout'}"
    )
    return (
        f"{candidate['family']} entry at {threshold} over "
        f"{int(candidate['lookback_bars'])} five-minute bars; "
        f"{float(candidate['take_profit_bps']):g} bps target, "
        f"{float(candidate['stop_loss_bps']):g} bps stop, "
        f"maximum {int(candidate['max_hold_bars'])} bars, flat by session close."
    )
