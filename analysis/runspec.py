"""
Pure dataclasses for the Stage-3 harness — no Nautilus, no IO.

RunSpec describes one backtest configuration; RunResult carries its outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from nautilus_trader.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy


@dataclass(frozen=True)
class RunSpec:
    """A single backtest configuration."""
    label: str
    strategy_class: type[Strategy]
    strategy_config: StrategyConfig
    start: datetime
    end: datetime
    slippage_prob: float = 0.25
    starting_cash: int = 100_000
    random_seed: int = 42
    validate_data: bool = True
    log_level: str = "WARNING"

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("start and end must be timezone-aware")
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"unsupported log_level: {self.log_level}")
        if self.end <= self.start:
            raise ValueError(f"end must be after start (got start={self.start}, end={self.end})")
        if self.starting_cash <= 0:
            raise ValueError(f"starting_cash must be positive (got {self.starting_cash})")
        if not (0.0 <= self.slippage_prob <= 1.0):
            raise ValueError(f"slippage_prob must be in [0.0, 1.0] (got {self.slippage_prob})")


@dataclass(frozen=True)
class RunResult:
    """Output of a single backtest run."""
    spec: RunSpec
    metrics: dict[str, float]
    equity_curve: pd.Series

    @property
    def label(self) -> str:
        return self.spec.label
