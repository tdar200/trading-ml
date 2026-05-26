"""Pure dataclass tests for RunSpec + RunResult — no Nautilus, no IO."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from analysis.runspec import RunResult, RunSpec
from strategies.momentum_topn import MomentumTopNConfig, MomentumTopNStrategy


def _basic_config() -> MomentumTopNConfig:
    return MomentumTopNConfig(universe=("XLK", "XLF"), benchmark="SPY")


def test_runspec_constructs_with_required_fields():
    spec = RunSpec(
        label="test",
        strategy_class=MomentumTopNStrategy,
        strategy_config=_basic_config(),
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 6, 1, tzinfo=timezone.utc),
    )
    assert spec.label == "test"
    assert spec.start < spec.end
    assert spec.slippage_prob == 0.25       # default
    assert spec.starting_cash == 100_000     # default
    assert spec.random_seed == 42            # default


def test_runspec_rejects_end_before_start():
    with pytest.raises(ValueError, match="end must be after start"):
        RunSpec(
            label="bad",
            strategy_class=MomentumTopNStrategy,
            strategy_config=_basic_config(),
            start=datetime(2024, 6, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )


def test_runspec_rejects_non_positive_cash():
    with pytest.raises(ValueError, match="starting_cash must be positive"):
        RunSpec(
            label="bad",
            strategy_class=MomentumTopNStrategy,
            strategy_config=_basic_config(),
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 6, 1, tzinfo=timezone.utc),
            starting_cash=0,
        )


def test_runspec_rejects_slippage_out_of_range():
    with pytest.raises(ValueError, match="slippage_prob must be in"):
        RunSpec(
            label="bad",
            strategy_class=MomentumTopNStrategy,
            strategy_config=_basic_config(),
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 6, 1, tzinfo=timezone.utc),
            slippage_prob=1.5,
        )


def test_runresult_carries_spec_metrics_and_equity():
    spec = RunSpec(
        label="ok",
        strategy_class=MomentumTopNStrategy,
        strategy_config=_basic_config(),
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 6, 1, tzinfo=timezone.utc),
    )
    equity = pd.Series(
        [100_000.0, 101_000.0],
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        name="equity",
    )
    metrics = {"cagr": 0.10, "sharpe": 1.2, "max_drawdown": -0.05,
               "total_return": 0.01, "final_equity": 101_000.0,
               "n_days": 2, "n_trades": 0}
    result = RunResult(spec=spec, metrics=metrics, equity_curve=equity)
    assert result.metrics["sharpe"] == 1.2
    assert len(result.equity_curve) == 2
    assert result.spec.label == "ok"
