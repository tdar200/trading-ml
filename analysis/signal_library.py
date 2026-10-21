"""Parameterized causal OHLCV indicators plus the legacy 60-signal registry."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


PARAMETRIC_INDICATORS = (
    "sma", "ema", "sma_cross", "ema_cross", "price_slope", "roc", "rsi",
    "stochastic", "stochastic_signal", "williams_r", "cci", "macd",
    "bollinger", "directional_atr", "donchian", "directional_volume", "obv",
)


@dataclass(frozen=True)
class IndicatorVariant:
    """One searchable signal slot: an indicator choice plus one value."""

    indicator: str
    value: int

    def __post_init__(self) -> None:
        if self.indicator not in PARAMETRIC_INDICATORS:
            raise ValueError(f"unsupported indicator: {self.indicator!r}")
        if self.value < 3:
            raise ValueError("indicator value must be at least 3")

    @property
    def key(self) -> str:
        value = self.value
        fast = max(2, value // 2)
        names = {
            "sma": f"sma_distance_{value}",
            "ema": f"ema_distance_{value}",
            "sma_cross": f"sma_cross_{fast}_{value}",
            "ema_cross": f"ema_cross_{fast}_{value}",
            "price_slope": f"price_slope_{value}",
            "roc": f"roc_{value}",
            "rsi": f"rsi_{value}",
            "stochastic": f"stochastic_k_{value}",
            "stochastic_signal": f"stochastic_d_delta_{value}",
            "williams_r": f"williams_r_{value}",
            "cci": f"cci_{value}",
            "macd": f"macd_{fast}_{value}",
            "bollinger": f"bollinger_z_{value}",
            "directional_atr": f"directional_atr_{value}",
            "donchian": f"donchian_position_{value}",
            "directional_volume": f"directional_volume_z_{value}",
            "obv": f"obv_slope_{value}",
        }
        return names[self.indicator]


def build_indicator_variants(
    indicators: tuple[str, ...] = PARAMETRIC_INDICATORS, *,
    value_min: int = 3, value_max: int = 50, value_step: int = 1,
) -> tuple[IndicatorVariant, ...]:
    """Build the finite indicator/value search grid used by the GPU finder."""
    if value_min < 3 or value_max < value_min or value_step <= 0:
        raise ValueError("invalid indicator value range")
    return tuple(
        IndicatorVariant(indicator, value)
        for indicator in indicators
        for value in range(value_min, value_max + 1, value_step)
    )


def _z(values: pd.Series, window: int) -> pd.Series:
    mean = values.rolling(window, min_periods=max(3, window // 2)).mean()
    std = values.rolling(window, min_periods=max(3, window // 2)).std(ddof=0)
    return (values - mean) / std.where(std > 1e-12)


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    value = 100 * gain / (gain + loss).where(gain + loss > 0)
    value = value.mask((gain == 0) & (loss == 0), 50)
    return (value - 50) / 50


def _true_range(frame: pd.DataFrame) -> pd.Series:
    previous = frame["close"].shift(1)
    return pd.concat([
        frame["high"] - frame["low"],
        (frame["high"] - previous).abs(), (frame["low"] - previous).abs(),
    ], axis=1).max(axis=1)


def compute_signal_library(frame: pd.DataFrame) -> pd.DataFrame:
    """Return exactly 60 backward-looking signals; positive always means bullish."""
    required = {"ts", "open", "high", "low", "close", "volume"}
    if missing := required - set(frame.columns):
        raise ValueError(f"signal input missing columns: {sorted(missing)}")
    data = frame.sort_values("ts").reset_index(drop=True).copy()
    close, high, low, volume = (data[name].astype(float) for name in ("close", "high", "low", "volume"))
    output: dict[str, pd.Series] = {}
    for window in (5, 10, 20, 50):
        average = close.rolling(window, min_periods=window).mean()
        output[f"sma_distance_{window}"] = close / average - 1
    for window in (5, 10, 20, 50):
        average = close.ewm(span=window, adjust=False, min_periods=window).mean()
        output[f"ema_distance_{window}"] = close / average - 1
    for fast, slow in ((5, 20), (10, 50), (20, 50)):
        output[f"sma_cross_{fast}_{slow}"] = (
            close.rolling(fast, min_periods=fast).mean()
            / close.rolling(slow, min_periods=slow).mean() - 1
        )
    for fast, slow in ((5, 20), (10, 50), (20, 50)):
        output[f"ema_cross_{fast}_{slow}"] = (
            close.ewm(span=fast, adjust=False, min_periods=fast).mean()
            / close.ewm(span=slow, adjust=False, min_periods=slow).mean() - 1
        )
    for window in (5, 10, 20, 50):
        output[f"price_slope_{window}"] = close.pct_change(window, fill_method=None)
    for window in (1, 3, 5, 10, 20):
        output[f"roc_{window}"] = close.pct_change(window, fill_method=None)
    for window in (5, 7, 14, 21):
        output[f"rsi_{window}"] = _rsi(close, window)
    for window in (5, 14, 21):
        lowest = low.rolling(window, min_periods=window).min()
        highest = high.rolling(window, min_periods=window).max()
        stochastic = (close - lowest) / (highest - lowest).where(highest > lowest)
        output[f"stochastic_k_{window}"] = stochastic - 0.5
        output[f"stochastic_d_delta_{window}"] = stochastic - stochastic.rolling(3, min_periods=3).mean()
    for window in (7, 14, 28):
        highest = high.rolling(window, min_periods=window).max()
        lowest = low.rolling(window, min_periods=window).min()
        output[f"williams_r_{window}"] = (close - lowest) / (highest - lowest).where(highest > lowest) - 0.5
    typical = (high + low + close) / 3
    for window in (10, 20, 40):
        mean = typical.rolling(window, min_periods=window).mean()
        deviation = typical.rolling(window, min_periods=window).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        output[f"cci_{window}"] = ((typical - mean) / (0.015 * deviation.where(deviation > 0))).clip(-5, 5)
    for fast, slow in ((5, 13), (12, 26), (20, 50)):
        output[f"macd_{fast}_{slow}"] = (
            close.ewm(span=fast, adjust=False, min_periods=slow).mean()
            - close.ewm(span=slow, adjust=False, min_periods=slow).mean()
        ) / close
    for window in (10, 20, 40):
        output[f"bollinger_z_{window}"] = _z(close, window)
    true_range = _true_range(data)
    direction = np.sign(close.pct_change(fill_method=None))
    for window in (7, 14, 28):
        atr = true_range.rolling(window, min_periods=window).mean() / close
        output[f"directional_atr_{window}"] = direction * atr
    for window in (10, 20, 40):
        upper = high.rolling(window, min_periods=window).max()
        lower = low.rolling(window, min_periods=window).min()
        output[f"donchian_position_{window}"] = (close - lower) / (upper - lower).where(upper > lower) - 0.5
    for window in (10, 20, 40):
        output[f"directional_volume_z_{window}"] = direction * _z(volume, window)
    obv = (direction.fillna(0) * volume).cumsum()
    for window in (5, 10, 20):
        denominator = volume.rolling(window, min_periods=window).sum()
        output[f"obv_slope_{window}"] = obv.diff(window) / denominator.where(denominator > 0)
    timestamps = pd.to_datetime(data["ts"], utc=True).dt.tz_convert("America/New_York")
    sessions = timestamps.dt.date
    cumulative_volume = volume.groupby(sessions).cumsum()
    cumulative_value = (typical * volume).groupby(sessions).cumsum()
    session_vwap = cumulative_value / cumulative_volume.where(cumulative_volume > 0)
    output["session_vwap_deviation"] = close / session_vwap - 1
    output["close_location_value"] = (close - low) / (high - low).where(high > low) - 0.5
    expanding_high = high.groupby(sessions).cummax()
    expanding_low = low.groupby(sessions).cummin()
    output["intraday_range_position"] = (
        (close - expanding_low) / (expanding_high - expanding_low).where(expanding_high > expanding_low) - 0.5
    )
    result = pd.DataFrame(output, index=data.index).replace([np.inf, -np.inf], np.nan).clip(-10, 10)
    if result.shape[1] != 60:
        raise RuntimeError(f"signal registry invariant failed: expected 60, got {result.shape[1]}")
    return result


def compute_parameterized_signal_library(
    frame: pd.DataFrame, variants: tuple[IndicatorVariant, ...],
) -> pd.DataFrame:
    """Calculate arbitrary indicator/value pairs without a hard-coded period list."""
    required = {"ts", "open", "high", "low", "close", "volume"}
    if missing := required - set(frame.columns):
        raise ValueError(f"signal input missing columns: {sorted(missing)}")
    if not variants:
        raise ValueError("at least one indicator variant is required")
    if len({variant.key for variant in variants}) != len(variants):
        raise ValueError("indicator variants must be unique")
    data = frame.sort_values("ts").reset_index(drop=True).copy()
    close, high, low, volume = (
        data[name].astype(float) for name in ("close", "high", "low", "volume")
    )
    typical = (high + low + close) / 3
    true_range = _true_range(data)
    direction = np.sign(close.pct_change(fill_method=None))
    obv = (direction.fillna(0) * volume).cumsum()
    output: dict[str, pd.Series] = {}
    for variant in variants:
        indicator, value, key = variant.indicator, variant.value, variant.key
        fast = max(2, value // 2)
        if indicator == "sma":
            output[key] = close / close.rolling(value, min_periods=value).mean() - 1
        elif indicator == "ema":
            output[key] = close / close.ewm(span=value, adjust=False, min_periods=value).mean() - 1
        elif indicator == "sma_cross":
            output[key] = (
                close.rolling(fast, min_periods=fast).mean()
                / close.rolling(value, min_periods=value).mean() - 1
            )
        elif indicator == "ema_cross":
            output[key] = (
                close.ewm(span=fast, adjust=False, min_periods=fast).mean()
                / close.ewm(span=value, adjust=False, min_periods=value).mean() - 1
            )
        elif indicator == "price_slope":
            output[key] = close.pct_change(value, fill_method=None)
        elif indicator == "roc":
            output[key] = close.pct_change(value, fill_method=None)
        elif indicator == "rsi":
            output[key] = _rsi(close, value)
        elif indicator in {"stochastic", "stochastic_signal"}:
            lowest = low.rolling(value, min_periods=value).min()
            highest = high.rolling(value, min_periods=value).max()
            stochastic = (close - lowest) / (highest - lowest).where(highest > lowest)
            output[key] = (
                stochastic - 0.5 if indicator == "stochastic"
                else stochastic - stochastic.rolling(3, min_periods=3).mean()
            )
        elif indicator == "williams_r":
            highest = high.rolling(value, min_periods=value).max()
            lowest = low.rolling(value, min_periods=value).min()
            output[key] = (close - lowest) / (highest - lowest).where(highest > lowest) - 0.5
        elif indicator == "cci":
            mean = typical.rolling(value, min_periods=value).mean()
            deviation = typical.rolling(value, min_periods=value).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
            output[key] = ((typical - mean) / (0.015 * deviation.where(deviation > 0))).clip(-5, 5)
        elif indicator == "macd":
            output[key] = (
                close.ewm(span=fast, adjust=False, min_periods=value).mean()
                - close.ewm(span=value, adjust=False, min_periods=value).mean()
            ) / close
        elif indicator == "bollinger":
            output[key] = _z(close, value)
        elif indicator == "directional_atr":
            output[key] = direction * true_range.rolling(value, min_periods=value).mean() / close
        elif indicator == "donchian":
            upper = high.rolling(value, min_periods=value).max()
            lower = low.rolling(value, min_periods=value).min()
            output[key] = (close - lower) / (upper - lower).where(upper > lower) - 0.5
        elif indicator == "directional_volume":
            output[key] = direction * _z(volume, value)
        elif indicator == "obv":
            denominator = volume.rolling(value, min_periods=value).sum()
            output[key] = obv.diff(value) / denominator.where(denominator > 0)
    return pd.DataFrame(output, index=data.index).replace(
        [np.inf, -np.inf], np.nan,
    ).clip(-10, 10)


SIGNAL_NAMES = tuple(compute_signal_library(pd.DataFrame({
    "ts": pd.date_range("2020-01-01", periods=60, freq="h", tz="UTC"),
    "open": np.arange(60) + 100, "high": np.arange(60) + 101,
    "low": np.arange(60) + 99, "close": np.arange(60) + 100.5,
    "volume": np.arange(60) + 1000,
})).columns)


def signal_category(name: str) -> str:
    if name.startswith(("sma_", "ema_", "price_slope", "macd_")):
        return "trend"
    if name.startswith(("roc_", "rsi_", "stochastic_", "williams_", "cci_")):
        return "momentum"
    if name.startswith(("bollinger_", "directional_atr_")):
        return "volatility"
    if name.startswith(("directional_volume_", "obv_")):
        return "volume"
    return "structure"


def _semanticize_signal_library(raw: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Convert indicators into causal setup states using their natural levels.

    Values are -1/0/+1. Cross rules use the immediately preceding completed bar
    and are suppressed across session boundaries.
    """
    ordered = frame.sort_values("ts").reset_index(drop=True)
    eastern = pd.to_datetime(ordered["ts"], utc=True).dt.tz_convert("America/New_York")
    same_session = pd.Series(eastern.dt.date).eq(pd.Series(eastern.dt.date).shift(1))
    semantic = pd.DataFrame(0.0, index=raw.index, columns=raw.columns)
    for name in raw:
        value, previous = raw[name], raw[name].shift(1)
        category = signal_category(name)
        if name.startswith("rsi_"):
            semantic[name] = np.select(
                [same_session & (previous <= -0.4) & (value > -0.4),
                 same_session & (previous >= 0.4) & (value < 0.4)], [1.0, -1.0], 0.0,
            )
        elif name.startswith(("stochastic_k_", "williams_r_")):
            semantic[name] = np.select(
                [same_session & (previous <= -0.3) & (value > previous),
                 same_session & (previous >= 0.3) & (value < previous)], [1.0, -1.0], 0.0,
            )
        elif name.startswith("stochastic_d_delta_"):
            semantic[name] = np.select(
                [same_session & (previous <= 0) & (value > 0),
                 same_session & (previous >= 0) & (value < 0)], [1.0, -1.0], 0.0,
            )
        elif name.startswith("cci_"):
            semantic[name] = np.where(value <= -1, 1.0, np.where(value >= 1, -1.0, 0.0))
        elif name.startswith("bollinger_z_"):
            semantic[name] = np.select(
                [same_session & (previous <= -2) & (value > -2),
                 same_session & (previous >= 2) & (value < 2)], [1.0, -1.0], 0.0,
            )
        elif name.startswith("donchian_position_"):
            semantic[name] = np.where(value >= 0.45, 1.0, np.where(value <= -0.45, -1.0, 0.0))
        elif name == "session_vwap_deviation":
            semantic[name] = np.select(
                [same_session & (previous <= 0) & (value > 0),
                 same_session & (previous >= 0) & (value < 0)], [1.0, -1.0], 0.0,
            )
        elif name == "close_location_value":
            semantic[name] = np.where(value >= 0.3, 1.0, np.where(value <= -0.3, -1.0, 0.0))
        elif name == "intraday_range_position":
            semantic[name] = np.where(value >= 0.4, 1.0, np.where(value <= -0.4, -1.0, 0.0))
        elif category == "volume":
            semantic[name] = np.where(value >= 0.5, 1.0, np.where(value <= -0.5, -1.0, 0.0))
        else:
            semantic[name] = np.select(
                [same_session & (previous <= 0) & (value > 0),
                 same_session & (previous >= 0) & (value < 0)], [1.0, -1.0], 0.0,
            )
    return semantic


def compute_semantic_signal_library(frame: pd.DataFrame) -> pd.DataFrame:
    """Return natural threshold/crossing states for the fixed compatibility library."""
    return _semanticize_signal_library(compute_signal_library(frame), frame)


def compute_parameterized_semantic_signal_library(
    frame: pd.DataFrame, variants: tuple[IndicatorVariant, ...],
) -> pd.DataFrame:
    """Return semantic states for an arbitrary indicator/value search grid."""
    return _semanticize_signal_library(
        compute_parameterized_signal_library(frame, variants), frame,
    )
