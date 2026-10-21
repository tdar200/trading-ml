"""Tests for mechanical support/resistance portfolio research."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.cross_sectional import CrossSectionalPanel
from analysis.support_resistance import (
    LEVEL_TYPES,
    SupportResistanceGrammar,
    SupportResistancePanel,
    _behavior_masks,
    generate_support_resistance_candidates,
    prepare_support_resistance_panel,
    simulate_support_resistance_candidates,
)


def _bars(slope: float, sessions: int = 10, bars_per_session: int = 30) -> pd.DataFrame:
    rows = []
    for day, date in enumerate(pd.bdate_range("2024-01-02", periods=sessions)):
        start = pd.Timestamp(date.date(), tz="America/New_York") + pd.Timedelta(hours=9, minutes=30)
        for bar in range(bars_per_session):
            close = 100 + day + slope * bar
            rows.append({
                "ts": (start + pd.Timedelta(minutes=5 * bar)).tz_convert("UTC"),
                "open": close - slope * 0.2, "high": close + 0.3,
                "low": close - 0.3, "close": close,
                "volume": 1000 + 20 * day + 5 * bar,
            })
    return pd.DataFrame(rows)


def test_default_grammar_has_all_level_and_behavior_families():
    candidates = generate_support_resistance_candidates(SupportResistanceGrammar())
    assert len(candidates) == 1680
    assert {row["level_type"] for row in candidates} == {*LEVEL_TYPES, "any_level"}
    assert {row["behavior"] for row in candidates} == {
        "breakout_retest", "rejection", "failed_breakout",
    }
    assert all(row["lookback_bars"] <= row["decision_bar"] for row in candidates)


def test_mechanical_levels_are_causal_before_future_mutation():
    stocks = {name: _bars(slope) for name, slope in {
        "A": 0.3, "B": 0.2, "C": -0.1, "D": -0.2,
    }.items()}
    original = prepare_support_resistance_panel(stocks, _bars(0.02))
    changed = {name: frame.copy() for name, frame in stocks.items()}
    changed["A"].loc[changed["A"].index[-1], ["high", "low", "close", "volume"]] *= 2
    mutated = prepare_support_resistance_panel(changed, _bars(0.02))
    for level in LEVEL_TYPES:
        for original_side, mutated_side in zip(original.levels[level], mutated.levels[level], strict=True):
            np.testing.assert_allclose(original_side[:-1], mutated_side[:-1], equal_nan=True)


def test_behavior_masks_define_rejection_and_failed_breakout_mechanically():
    torch = pytest.importorskip("torch")
    support = torch.tensor([[99.0]])
    resistance = torch.tensor([[101.0]])
    zone = torch.tensor([[0.1]])
    long_reject, short_reject = _behavior_masks(
        torch.tensor([[99.5]]), torch.tensor([[100.0]]), torch.tensor([[98.9]]),
        support, resistance, zone, torch.tensor([[99.2]]), "rejection",
    )
    assert long_reject.item() and not short_reject.item()
    long_failed, _ = _behavior_masks(
        torch.tensor([[99.2]]), torch.tensor([[99.4]]), torch.tensor([[98.7]]),
        support, resistance, zone, torch.tensor([[99.1]]), "failed_breakout",
    )
    assert long_failed.item()


def _synthetic_panel() -> SupportResistancePanel:
    sessions, bars_per_session, assets = 10, 30, 6
    total = sessions * bars_per_session
    session = np.repeat(np.arange(sessions), bars_per_session)
    bar = np.tile(np.arange(bars_per_session), sessions)
    slopes = np.asarray([0.30, 0.20, 0.10, -0.10, -0.20, -0.30])
    close = 100 + session[:, None] + bar[:, None] * slopes
    open_px = close - slopes * 0.2
    timestamps = pd.date_range("2024-01-02 14:30", periods=total, freq="5min", tz="UTC")
    dates = np.asarray([pd.Timestamp("2024-01-02").date()] * total, dtype=object)
    beta = np.broadcast_to(np.asarray([0.6, 0.8, 1.0, 1.2, 1.4, 1.6]), close.shape).copy()
    base = CrossSectionalPanel(
        tickers=tuple(f"S{i}" for i in range(assets)), timestamps=timestamps,
        session_dates=dates, sessions=session, bar_in_session=bar,
        open=open_px, close=close, relative_volume=np.full_like(close, 2.0),
        beta=beta, benchmark_open=np.full(total, 100.0),
        benchmark_close=np.full(total, 100.0),
    )
    high, low = close + 0.30, close - 0.30
    support, resistance = close - 0.20, close + 0.20
    return SupportResistancePanel(
        base=base, high=high, low=low, atr_fraction=np.full_like(close, 0.001),
        levels={name: (support.copy(), resistance.copy()) for name in LEVEL_TYPES},
    )


def test_cpu_cuda_portfolio_parity():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    candidate = {
        "level_type": "previous_day", "behavior": "rejection",
        "signal_family": "momentum", "decision_bar": 6,
        "lookback_bars": 6, "hold_bars": 12, "top_n": 2,
        "zone_atr": 0.10, "stop_atr": 0.50, "reward_risk": 2.0,
        "min_relative_volume": 1.0, "max_market_move_bps": 50.0,
    }
    cpu, _ = simulate_support_resistance_candidates(
        _synthetic_panel(), [candidate], cost_per_side_bps=1.0, device="cpu",
    )
    cuda, _ = simulate_support_resistance_candidates(
        _synthetic_panel(), [candidate], cost_per_side_bps=1.0, device="cuda",
    )
    for field in ("total_return", "sharpe", "max_drawdown", "mean_abs_beta_exposure"):
        assert cuda.iloc[0][field] == pytest.approx(cpu.iloc[0][field], abs=1e-12)
    assert cuda.iloc[0]["n_trades"] == cpu.iloc[0]["n_trades"]
