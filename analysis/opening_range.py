"""Selective long/short opening-range breakout research primitives."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from analysis.metrics import cagr, max_drawdown, sharpe
from analysis.vectorized_screen import resolve_device


@dataclass(frozen=True)
class OpeningRangeGrammar:
    opening_range_bars: tuple[int, ...] = (3, 6, 12)
    breakout_atr: tuple[float, ...] = (0.0, 0.25, 0.5)
    min_relative_volume: tuple[float, ...] = (1.0, 1.5, 2.0)
    market_alignment: tuple[bool, ...] = (False, True)
    fresh_breakout: tuple[bool, ...] = (False, True)
    context_relation: tuple[str, ...] = ("none",)
    min_regime_strength: tuple[float, ...] = (0.0,)
    directions: tuple[str, ...] = ("long", "short", "both")
    max_hold_bars: tuple[int, ...] = (1, 2, 3, 6, 12, 24)
    take_profit_atr: tuple[float, ...] = (0.5, 1.0, 1.5)
    stop_loss_atr: tuple[float, ...] = (0.5, 0.75, 1.0)
    max_trades_per_session: int = 1

    def __post_init__(self) -> None:
        if not self.opening_range_bars or any(v <= 0 for v in self.opening_range_bars):
            raise ValueError("opening-range bars must be positive")
        if not self.breakout_atr or any(v < 0 for v in self.breakout_atr):
            raise ValueError("breakout ATR values must be non-negative")
        if not self.min_relative_volume or any(v <= 0 for v in self.min_relative_volume):
            raise ValueError("relative-volume thresholds must be positive")
        if not self.fresh_breakout or any(type(v) is not bool for v in self.fresh_breakout):
            raise ValueError("fresh-breakout values must be booleans")
        if set(self.context_relation) - {"none", "same", "inverse"}:
            raise ValueError("context relation must be none, same, or inverse")
        if not self.min_regime_strength or any(v < 0 for v in self.min_regime_strength):
            raise ValueError("regime-strength thresholds must be non-negative")
        if set(self.directions) - {"long", "short", "both"}:
            raise ValueError("direction must be long, short, or both")
        positive = (
            self.max_hold_bars, self.take_profit_atr, self.stop_loss_atr,
        )
        if any(not values or any(v <= 0 for v in values) for values in positive):
            raise ValueError("exit parameters must be positive")
        if self.max_trades_per_session <= 0:
            raise ValueError("max trades per session must be positive")


ENTRY_FIELDS = [
    "family", "opening_range_bars", "breakout_atr",
    "min_relative_volume", "market_alignment", "fresh_breakout",
    "context_relation", "min_regime_strength", "direction",
    "entry_rule_id",
]
PARAM_FIELDS = [
    *ENTRY_FIELDS,
    "max_hold_bars", "take_profit_atr", "stop_loss_atr",
    "max_trades_per_session",
]


def generate_opening_range_entries(
    grammar: OpeningRangeGrammar = OpeningRangeGrammar(),
) -> list[dict[str, Any]]:
    entries = []
    for opening_bars, distance, relative_volume, alignment, fresh, context, regime, direction in product(
        grammar.opening_range_bars,
        grammar.breakout_atr,
        grammar.min_relative_volume,
        grammar.market_alignment,
        grammar.fresh_breakout,
        grammar.context_relation,
        grammar.min_regime_strength,
        grammar.directions,
    ):
        entries.append({
            "family": "opening_range_breakout",
            "opening_range_bars": int(opening_bars),
            "breakout_atr": float(distance),
            "min_relative_volume": float(relative_volume),
            "market_alignment": bool(alignment),
            "fresh_breakout": bool(fresh),
            "context_relation": str(context),
            "min_regime_strength": float(regime),
            "direction": direction,
            "entry_rule_id": (
                f"orb__range{opening_bars}__atr{distance:g}"
                f"__rvol{relative_volume:g}__market{int(alignment)}"
                f"__fresh{int(fresh)}__context{context}__regime{regime:g}__{direction}"
            ),
        })
    return entries


def expand_opening_range_exits(
    entries: list[dict[str, Any]],
    grammar: OpeningRangeGrammar,
) -> list[dict[str, Any]]:
    return [
        {
            **entry,
            "max_hold_bars": int(hold),
            "take_profit_atr": float(take),
            "stop_loss_atr": float(stop),
            "max_trades_per_session": grammar.max_trades_per_session,
        }
        for entry in entries
        for hold, take, stop in product(
            grammar.max_hold_bars,
            grammar.take_profit_atr,
            grammar.stop_loss_atr,
        )
    ]


def add_pooled_context_features(
    features: pd.DataFrame,
    context_by_ticker: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Attach causal equal-weight context direction and breadth features."""
    if not context_by_ticker:
        raise ValueError("at least one pooled-context instrument is required")
    result = features.copy().sort_values("ts", kind="stable").reset_index(drop=True)
    directions = []
    for ticker, frame in context_by_ticker.items():
        required = {"ts", "open", "close", "session"}
        if missing := required - set(frame.columns):
            raise ValueError(f"context {ticker} missing columns: {sorted(missing)}")
        context = frame.sort_values("ts", kind="stable").copy()
        opening = context.groupby("session", sort=False)["open"].transform("first")
        context["direction"] = context["close"] / opening - 1.0
        aligned = pd.merge_asof(
            result[["ts"]], context[["ts", "direction"]], on="ts",
            direction="backward", tolerance=pd.Timedelta(minutes=5),
        )
        directions.append(aligned["direction"].to_numpy(dtype=float))
    matrix = np.column_stack(directions)
    finite = np.isfinite(matrix)
    counts = finite.sum(axis=1)
    result["context_direction"] = np.divide(
        np.nansum(matrix, axis=1), counts,
        out=np.full(len(result), np.nan), where=counts > 0,
    )
    result["context_breadth"] = np.divide(
        np.nansum(np.sign(matrix), axis=1), counts,
        out=np.full(len(result), np.nan), where=counts > 0,
    )
    return result


def split_opening_range_nested(
    features: pd.DataFrame,
    *,
    holdout_fraction: float = 0.20,
    orientation_sessions: int = 280,
    purge_sessions: int = 5,
    validation_windows: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.DataFrame], pd.DataFrame]:
    """Create orientation, purge, nested-validation, and sealed holdout slices."""
    if not 0 < holdout_fraction < 0.5:
        raise ValueError("holdout_fraction must be between zero and one half")
    if orientation_sessions <= 0 or purge_sessions < 0 or validation_windows <= 0:
        raise ValueError("nested split counts must be positive; purge may be zero")
    sessions = np.sort(features["session"].unique())
    holdout_start = int(len(sessions) * (1.0 - holdout_fraction))
    discovery_sessions = sessions[:holdout_start]
    if orientation_sessions + purge_sessions + validation_windows > len(discovery_sessions):
        raise ValueError("not enough sessions for nested opening-range validation")
    orientation_ids = discovery_sessions[:orientation_sessions]
    purge_ids = discovery_sessions[
        orientation_sessions:orientation_sessions + purge_sessions
    ]
    validation_ids = discovery_sessions[orientation_sessions + purge_sessions:]
    chunks = np.array_split(validation_ids, validation_windows)

    def select(ids: np.ndarray) -> pd.DataFrame:
        return features[features["session"].isin(ids)].reset_index(drop=True)

    return (
        select(orientation_ids), select(purge_ids),
        [select(chunk) for chunk in chunks], select(sessions[holdout_start:]),
    )


def prepare_opening_range_features(
    bars: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
) -> pd.DataFrame:
    """Add strictly causal ATR, relative-volume, and SPY-direction features."""
    required = {
        "ts", "open", "high", "low", "close", "volume", "session",
        "session_date", "minute_of_day", "last_bar",
    }
    if missing := required - set(bars.columns):
        raise ValueError(f"prepared bars missing columns: {sorted(missing)}")
    result = bars.copy().sort_values("ts", kind="stable").reset_index(drop=True)
    result["bar_in_session"] = (
        (result["minute_of_day"] - (9 * 60 + 30)) // 5
    ).astype(int)

    previous_close = result["close"].shift(1)
    same_session = result["session"].eq(result["session"].shift(1))
    high_gap = (result["high"] - previous_close).abs().where(
        same_session, result["high"] - result["low"],
    )
    low_gap = (result["low"] - previous_close).abs().where(
        same_session, result["high"] - result["low"],
    )
    true_range = pd.concat(
        [result["high"] - result["low"], high_gap, low_gap], axis=1,
    ).max(axis=1)
    result["atr_fraction"] = (
        (true_range / result["close"])
        .shift(1)
        .rolling(24, min_periods=6)
        .mean()
    )

    expected_volume = result.groupby(
        "minute_of_day", sort=False,
    )["volume"].transform(
        lambda values: values.shift(1).rolling(20, min_periods=5).median()
    )
    result["relative_volume"] = result["volume"] / expected_volume.where(
        expected_volume > 0,
    )

    benchmark = benchmark_bars.copy().sort_values("ts", kind="stable")
    opening = benchmark.groupby("session", sort=False)["open"].transform("first")
    benchmark["market_direction"] = benchmark["close"] / opening - 1.0
    aligned = pd.merge_asof(
        result[["ts"]],
        benchmark[["ts", "market_direction"]],
        on="ts",
        direction="backward",
        tolerance=pd.Timedelta(minutes=5),
    )
    result["market_direction"] = aligned["market_direction"].to_numpy()
    daily_close = result.groupby("session", sort=False)["close"].last()
    daily_return = daily_close.pct_change(fill_method=None)
    trailing_volatility = daily_return.shift(1).rolling(20, min_periods=10).std(ddof=0)
    trailing_trend = daily_close.shift(1) / daily_close.shift(21) - 1.0
    regime_strength = trailing_trend.abs() / (trailing_volatility * np.sqrt(20)).where(
        trailing_volatility > 1e-12,
    )
    result["regime_strength"] = result["session"].map(regime_strength)
    result["regime_direction"] = result["session"].map(np.sign(trailing_trend))
    return result


def opening_range_signal_matrix(
    features: pd.DataFrame,
    entries: list[dict[str, Any]],
    *,
    device: str = "cuda",
):
    """Build signed (-1/0/+1) ORB signals on CPU or CUDA tensors."""
    import torch

    used_device = resolve_device(device)
    high = torch.as_tensor(
        features["high"].to_numpy(dtype=np.float64),
        dtype=torch.float64, device=used_device,
    )
    low = torch.as_tensor(
        features["low"].to_numpy(dtype=np.float64),
        dtype=torch.float64, device=used_device,
    )
    close = torch.as_tensor(
        features["close"].to_numpy(dtype=np.float64),
        dtype=torch.float64, device=used_device,
    )
    sessions = torch.as_tensor(
        features["session"].to_numpy(dtype=np.int64), device=used_device,
    )
    bar_index = torch.as_tensor(
        features["bar_in_session"].to_numpy(dtype=np.int64), device=used_device,
    )
    risk = torch.as_tensor(
        features["atr_fraction"].to_numpy(dtype=np.float64),
        dtype=torch.float64, device=used_device,
    )
    relative_volume = torch.as_tensor(
        features["relative_volume"].to_numpy(dtype=np.float64),
        dtype=torch.float64, device=used_device,
    )
    regime_strength = torch.as_tensor(
        features["regime_strength"].to_numpy(dtype=np.float64),
        dtype=torch.float64, device=used_device,
    )
    market = torch.as_tensor(
        features["market_direction"].to_numpy(dtype=np.float64),
        dtype=torch.float64, device=used_device,
    )
    session_count = int(sessions.max().item()) + 1
    range_cache = {}
    columns = []
    for entry in entries:
        opening_bars = int(entry["opening_range_bars"])
        if opening_bars not in range_cache:
            included = bar_index < opening_bars
            opening_high = torch.full(
                (session_count,), -torch.inf, dtype=torch.float64,
                device=used_device,
            )
            opening_low = torch.full(
                (session_count,), torch.inf, dtype=torch.float64,
                device=used_device,
            )
            opening_high.scatter_reduce_(
                0, sessions, torch.where(included, high, -torch.inf),
                reduce="amax", include_self=True,
            )
            opening_low.scatter_reduce_(
                0, sessions, torch.where(included, low, torch.inf),
                reduce="amin", include_self=True,
            )
            opening_count = torch.zeros(
                session_count, dtype=torch.int32, device=used_device,
            )
            opening_count.scatter_add_(0, sessions, included.to(torch.int32))
            range_cache[opening_bars] = (
                opening_high[sessions], opening_low[sessions],
                opening_count[sessions] == opening_bars,
            )
        opening_high, opening_low, complete_range = range_cache[opening_bars]
        distance = float(entry["breakout_atr"]) * risk * close
        minimum_regime = float(entry["min_regime_strength"])
        regime_ok = (
            torch.ones_like(regime_strength, dtype=torch.bool)
            if minimum_regime <= 0 else
            torch.isfinite(regime_strength) & (regime_strength >= minimum_regime)
        )
        eligible = (
            (bar_index >= opening_bars)
            & complete_range
            & torch.isfinite(risk)
            & torch.isfinite(relative_volume)
            & (relative_volume >= float(entry["min_relative_volume"]))
            & regime_ok
        )
        long_signal = eligible & (close >= opening_high + distance)
        short_signal = eligible & (close <= opening_low - distance)
        if bool(entry["fresh_breakout"]):
            previous_close = torch.roll(close, 1)
            same_session = sessions == torch.roll(sessions, 1)
            long_signal &= same_session & (previous_close < opening_high + distance)
            short_signal &= same_session & (previous_close > opening_low - distance)
        context = str(entry["context_relation"])
        if context != "none":
            context_direction = torch.as_tensor(
                features["context_direction"].to_numpy(dtype=np.float64),
                dtype=torch.float64, device=used_device,
            )
            relation = 1.0 if context == "same" else -1.0
            long_signal &= context_direction * relation > 0
            short_signal &= context_direction * relation < 0
        if bool(entry["market_alignment"]):
            long_signal &= market > 0
            short_signal &= market < 0
        direction = str(entry["direction"])
        signal = torch.zeros(len(features), dtype=torch.int8, device=used_device)
        if direction in {"long", "both"}:
            signal = torch.where(long_signal, 1, signal)
        if direction in {"short", "both"}:
            signal = torch.where(short_signal, -1, signal)
        columns.append(signal)
    return torch.stack(columns, dim=1).contiguous()


def screen_opening_range_entries(
    features: pd.DataFrame,
    entries: list[dict[str, Any]],
    *,
    cost_per_side_bps: float,
    device: str = "cuda",
    forward_bars: int = 12,
) -> pd.DataFrame:
    """CUDA screen signed entries by fixed-horizon same-session return."""
    import torch

    if not entries:
        raise ValueError("opening-range entries must be non-empty")
    used_device = resolve_device(device)
    signals = opening_range_signal_matrix(features, entries, device=used_device)
    open_px = torch.as_tensor(
        features["open"].to_numpy(dtype=np.float64),
        dtype=torch.float64, device=used_device,
    )
    close_px = torch.as_tensor(
        features["close"].to_numpy(dtype=np.float64),
        dtype=torch.float64, device=used_device,
    )
    sessions = torch.as_tensor(
        features["session"].to_numpy(dtype=np.int64), device=used_device,
    )
    if len(features) <= forward_bars + 1:
        raise ValueError("not enough bars for opening-range screen")
    decision = torch.arange(
        0, len(features) - forward_bars - 1, device=used_device,
    )
    enter = decision + 1
    exit_index = enter + forward_bars
    side = signals[decision]
    valid = (
        (side != 0)
        & (sessions[decision, None] == sessions[exit_index, None])
    )
    cost = cost_per_side_bps / 10_000.0
    long_return = (
        close_px[exit_index, None] * (1.0 - cost)
        / (open_px[enter, None] * (1.0 + cost))
        - 1.0
    )
    short_return = (
        open_px[enter, None] * (1.0 - cost)
        / (close_px[exit_index, None] * (1.0 + cost))
        - 1.0
    )
    returns = torch.where(side > 0, long_return, short_return)
    masked = torch.where(valid, returns, 0.0)
    counts = valid.sum(dim=0)
    safe_counts = counts.clamp(min=1)
    means = masked.sum(dim=0) / safe_counts
    centered = torch.where(valid, returns - means, 0.0)
    std = torch.sqrt((centered * centered).sum(dim=0) / safe_counts)
    win_rate = ((returns > 0) & valid).sum(dim=0) / safe_counts
    t_stat = torch.where(
        std > 1e-14,
        means / std * torch.sqrt(safe_counts.to(torch.float64)),
        torch.nan,
    )
    rows = []
    for index, entry in enumerate(entries):
        count = int(counts[index].item())
        rows.append({
            **entry,
            "screen_trades": count,
            "screen_edge_bps": (
                float(means[index].item()) * 10_000.0 if count else float("nan")
            ),
            "screen_win_rate": (
                float(win_rate[index].item()) if count else float("nan")
            ),
            "screen_t_stat": (
                float(t_stat[index].item()) if count else float("nan")
            ),
            "screen_score": (
                float(t_stat[index].item()) if count else float("nan")
            ),
            "screen_device": used_device,
        })
    return pd.DataFrame(rows).sort_values(
        "screen_score", ascending=False, na_position="last", kind="stable",
    ).reset_index(drop=True)


def select_opening_range_finalists(
    screened: pd.DataFrame,
    count: int,
    *,
    min_screen_trades: int,
) -> list[dict[str, Any]]:
    """Preserve direction/alignment variants, then fill by screen score."""
    valid = screened.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["screen_score"],
    )
    valid = valid[valid["screen_trades"] >= min_screen_trades].sort_values(
        "screen_score", ascending=False, kind="stable",
    )
    if valid.empty:
        return []
    chosen = list(
        valid.drop_duplicates(
            ["direction", "market_alignment", "fresh_breakout", "context_relation"]
        ).head(count).index
    )
    for index in valid.index:
        if len(chosen) >= count:
            break
        if index not in chosen:
            chosen.append(index)
    return valid.loc[chosen, ENTRY_FIELDS].to_dict(orient="records")


def replay_opening_range_candidate(
    features: pd.DataFrame,
    candidate: dict[str, Any],
    *,
    cost_per_side_bps: float,
) -> pd.DataFrame:
    """Replay one signed candidate and return an execution-level trade ledger."""
    if cost_per_side_bps < 0:
        raise ValueError("cost_per_side_bps must be non-negative")
    signals = opening_range_signal_matrix(
        features, [candidate], device="cpu",
    ).numpy()[:, 0]
    open_px = features["open"].to_numpy(dtype=float)
    high_px = features["high"].to_numpy(dtype=float)
    low_px = features["low"].to_numpy(dtype=float)
    close_px = features["close"].to_numpy(dtype=float)
    sessions = features["session"].to_numpy(dtype=int)
    minutes = features["minute_of_day"].to_numpy(dtype=int)
    last_bar = features["last_bar"].to_numpy(dtype=bool)
    risk = features["atr_fraction"].to_numpy(dtype=float)
    timestamps = pd.to_datetime(features["ts"], utc=True)
    dates = features["session_date"].to_numpy()
    cost = cost_per_side_bps / 10_000.0
    maximum_hold = int(candidate["max_hold_bars"])
    maximum_trades = int(candidate["max_trades_per_session"])
    take_multiple = float(candidate["take_profit_atr"])
    stop_multiple = float(candidate["stop_loss_atr"])
    side = 0
    entry_price = 1.0
    entry_raw_price = 1.0
    entry_index = -1
    decision_index = -1
    entry_take = 0.0
    entry_stop = 0.0
    held = 0
    entries_this_session = 0
    rows: list[dict[str, Any]] = []

    def close_trade(index: int, raw_price: float, reason: str) -> None:
        nonlocal side, held
        exit_price = raw_price * (1.0 - cost) if side > 0 else raw_price * (1.0 + cost)
        net_return = exit_price / entry_price - 1.0 if side > 0 else entry_price / exit_price - 1.0
        gross_return = raw_price / entry_raw_price - 1.0 if side > 0 else entry_raw_price / raw_price - 1.0
        if side > 0:
            maximum_favorable_excursion = high_px[entry_index:index + 1].max() / entry_raw_price - 1.0
            maximum_adverse_excursion = low_px[entry_index:index + 1].min() / entry_raw_price - 1.0
        else:
            maximum_favorable_excursion = entry_raw_price / low_px[entry_index:index + 1].min() - 1.0
            maximum_adverse_excursion = entry_raw_price / high_px[entry_index:index + 1].max() - 1.0
        rows.append({
            "session_date": dates[index],
            "signal_ts": timestamps.iloc[decision_index],
            "entry_ts": timestamps.iloc[entry_index],
            "exit_ts": timestamps.iloc[index],
            "side": "long" if side > 0 else "short",
            "entry_price": entry_raw_price,
            "exit_price": raw_price,
            "hold_bars": index - entry_index,
            "exit_reason": reason,
            "gross_return": gross_return,
            "maximum_favorable_excursion": maximum_favorable_excursion,
            "maximum_adverse_excursion": maximum_adverse_excursion,
            "cost_drag": gross_return - net_return,
            "net_return": net_return,
            "cost_per_side_bps": cost_per_side_bps,
        })
        side = 0
        held = 0

    for index in range(1, len(features)):
        same_session = sessions[index] == sessions[index - 1]
        if not same_session:
            entries_this_session = 0
        if side:
            mark_return = (
                close_px[index - 1] / entry_price - 1.0
                if side > 0 else entry_price / close_px[index - 1] - 1.0
            )
            reason = None
            if mark_return >= entry_take:
                reason = "target"
            elif mark_return <= -entry_stop:
                reason = "stop"
            elif held >= maximum_hold:
                reason = "time"
            elif not same_session:
                reason = "session_boundary"
            if reason:
                close_trade(index, open_px[index], reason)
        allowed_time = 9 * 60 + 45 <= minutes[index] <= 15 * 60 + 30
        signal = int(signals[index - 1])
        if (
            not side and signal and same_session and allowed_time
            and not last_bar[index] and entries_this_session < maximum_trades
        ):
            side = signal
            decision_index = index - 1
            entry_index = index
            entry_raw_price = open_px[index]
            entry_price = open_px[index] * (1.0 + cost if side > 0 else 1.0 - cost)
            entry_take = risk[index - 1] * take_multiple
            entry_stop = risk[index - 1] * stop_multiple
            entries_this_session += 1
            held = 0
        if side:
            held += 1
            if side > 0:
                target_price = entry_price * (1.0 + entry_take) / (1.0 - cost)
                stop_price = entry_price * (1.0 - entry_stop) / (1.0 - cost)
                target_hit = high_px[index] >= target_price
                stop_hit = low_px[index] <= stop_price
            else:
                target_price = entry_price / ((1.0 + entry_take) * (1.0 + cost))
                stop_price = entry_price / ((1.0 - entry_stop) * (1.0 + cost))
                target_hit = low_px[index] <= target_price
                stop_hit = high_px[index] >= stop_price
            if stop_hit or target_hit:
                close_trade(
                    index, stop_price if stop_hit else target_price,
                    "stop" if stop_hit else "target",
                )
        if side and last_bar[index]:
            close_trade(index, close_px[index], "session_close")
    return pd.DataFrame(rows)


def block_bootstrap_return_confidence(
    returns: np.ndarray,
    *,
    block_size: int = 5,
    samples: int = 2_000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Estimate a deterministic moving-block confidence interval for return."""
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values) or block_size <= 0 or samples <= 0:
        raise ValueError("finite returns and positive bootstrap settings are required")
    rng = np.random.default_rng(seed)
    starts = np.arange(max(1, len(values) - block_size + 1))
    totals = np.empty(samples, dtype=float)
    blocks_needed = int(np.ceil(len(values) / block_size))
    for sample in range(samples):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        draw = np.concatenate([
            values[start:start + block_size] for start in chosen
        ])[:len(values)]
        totals[sample] = np.prod(1.0 + draw) - 1.0
    return {
        "samples": samples,
        "block_size": block_size,
        "lower_95": float(np.quantile(totals, 0.025)),
        "median": float(np.median(totals)),
        "upper_95": float(np.quantile(totals, 0.975)),
        "probability_positive": float(np.mean(totals > 0)),
    }


def trailing_performance_gate(
    prior_returns: np.ndarray,
    prospective_returns: np.ndarray,
    *,
    lookback: int = 40,
    min_observations: int = 20,
    minimum_sharpe: float = 0.0,
) -> np.ndarray:
    """Return a causal circuit-breaker gate based only on earlier shadow P&L."""
    if lookback <= 1 or min_observations <= 1 or min_observations > lookback:
        raise ValueError("circuit-breaker lookback/minimum observations are invalid")
    history = list(np.asarray(prior_returns, dtype=float))
    current = np.asarray(prospective_returns, dtype=float)
    enabled = np.zeros(len(current), dtype=bool)
    for index, value in enumerate(current):
        trailing = np.asarray(history[-lookback:], dtype=float)
        trailing = trailing[np.isfinite(trailing)]
        if len(trailing) >= min_observations:
            deviation = trailing.std(ddof=1)
            trailing_sharpe = (
                trailing.mean() / deviation * np.sqrt(252.0)
                if deviation > 0 else (np.inf if trailing.mean() > 0 else 0.0)
            )
            enabled[index] = trailing_sharpe >= minimum_sharpe
        history.append(value)
    return enabled


def correlation_aware_weights(
    features_by_ticker: dict[str, pd.DataFrame],
    *,
    maximum_weight: float = 0.25,
) -> dict[str, float]:
    """Estimate capped diversification weights from orientation-only daily returns."""
    if not features_by_ticker or maximum_weight <= 0 or maximum_weight > 1:
        raise ValueError("non-empty features and a maximum weight in (0, 1] are required")
    if len(features_by_ticker) * maximum_weight < 1.0 - 1e-12:
        raise ValueError("too few instruments for the requested maximum weight")
    daily = {}
    for ticker, frame in features_by_ticker.items():
        closes = frame.groupby("session_date", sort=False)["close"].last()
        daily[ticker] = closes.pct_change(fill_method=None)
    panel = pd.DataFrame(daily).dropna(how="all")
    correlation = panel.corr(min_periods=20).abs().fillna(1.0)
    scores = {}
    for ticker in features_by_ticker:
        others = correlation.loc[ticker].drop(ticker, errors="ignore")
        average = float(others.mean()) if len(others) else 1.0
        scores[ticker] = 1.0 / max(average, 0.05)
    weights = {ticker: 0.0 for ticker in scores}
    active = set(scores)
    remaining = 1.0
    while active:
        total_score = sum(scores[ticker] for ticker in active)
        proposed = {
            ticker: remaining * scores[ticker] / total_score for ticker in active
        }
        capped = [ticker for ticker, weight in proposed.items() if weight > maximum_weight]
        if not capped:
            for ticker, weight in proposed.items():
                weights[ticker] = weight
            break
        for ticker in capped:
            weights[ticker] = maximum_weight
            remaining -= maximum_weight
            active.remove(ticker)
    return weights


def parameter_neighborhood_support(
    ranked: pd.DataFrame,
    fields: tuple[str, ...] = (
        "breakout_atr", "min_relative_volume", "max_hold_bars",
        "take_profit_atr", "stop_loss_atr",
    ),
) -> pd.Series:
    """Count eligible candidates one grid coordinate from each candidate."""
    if missing := set(fields) - set(ranked.columns):
        raise ValueError(f"ranked candidates missing neighborhood fields: {sorted(missing)}")
    eligible = ranked[ranked["eligible"]]
    counts = pd.Series(0, index=ranked.index, dtype=int)
    if eligible.empty:
        return counts
    differences = np.zeros((len(eligible), len(eligible)), dtype=np.int16)
    for field in fields:
        values = eligible[field].astype(str).to_numpy()
        differences += values[:, None] != values[None, :]
    counts.loc[eligible.index] = (differences == 1).sum(axis=1)
    return counts


def portfolio_metrics_from_curves(
    candidates: list[dict[str, Any]],
    frames: list[pd.DataFrame],
    curves: list[np.ndarray],
    weights: np.ndarray | list[float] | None = None,
) -> pd.DataFrame:
    """Equal-weight daily-rebalanced portfolio metrics for every candidate."""
    if not frames or len(frames) != len(curves):
        raise ValueError("matching non-empty frames and curves are required")
    if len({curve.shape for curve in curves}) != 1:
        raise ValueError("portfolio curves must have identical shapes")
    instrument_returns = np.stack([
        np.vstack([
            curve[[0]] - 1.0,
            curve[1:] / curve[:-1] - 1.0,
        ])
        for curve in curves
    ])
    portfolio_weights = (
        np.full(len(curves), 1.0 / len(curves), dtype=float)
        if weights is None else np.asarray(weights, dtype=float)
    )
    if (
        portfolio_weights.shape != (len(curves),)
        or not np.isfinite(portfolio_weights).all()
        or (portfolio_weights < 0).any()
        or not np.isclose(portfolio_weights.sum(), 1.0)
    ):
        raise ValueError("portfolio weights must be finite, non-negative, and sum to one")
    portfolio_returns = np.tensordot(portfolio_weights, instrument_returns, axes=(0, 0))
    portfolio_curves = np.vstack([
        np.ones(len(candidates)),
        np.cumprod(1.0 + portfolio_returns, axis=0),
    ])
    trade_counts = np.stack([
        frame["n_trades"].to_numpy(dtype=int) for frame in frames
    ]).sum(axis=0)
    rows = []
    for index, candidate in enumerate(candidates):
        returns = pd.Series(portfolio_returns[:, index])
        curve = pd.Series(portfolio_curves[:, index])
        rows.append({
            **candidate,
            "sharpe": float(sharpe(returns)),
            "cagr": float(cagr(returns)),
            "max_drawdown": float(max_drawdown(curve)),
            "total_return": float(curve.iloc[-1] - 1.0),
            "n_trades": int(trade_counts[index]),
            "simulation_device": "cuda",
        })
    return pd.DataFrame(rows)


def describe_opening_range(candidate: dict[str, Any] | pd.Series) -> str:
    alignment = " with SPY alignment" if bool(candidate["market_alignment"]) else ""
    freshness = " on a fresh cross" if bool(candidate["fresh_breakout"]) else ""
    context = (
        "" if candidate["context_relation"] == "none" else
        f" with {candidate['context_relation']} pooled-context confirmation"
    )
    return (
        f"{candidate['direction']} {int(candidate['opening_range_bars']) * 5}-minute "
        f"opening-range breakout at {float(candidate['breakout_atr']):g} ATR, "
        f"relative volume ≥ {float(candidate['min_relative_volume']):g}×{alignment}{freshness}{context}; "
        f"{float(candidate['take_profit_atr']):g} ATR target, "
        f"{float(candidate['stop_loss_atr']):g} ATR stop, maximum "
        f"{int(candidate['max_hold_bars']) * 5} minutes and "
        f"{int(candidate['max_trades_per_session'])} trade/day."
    )
