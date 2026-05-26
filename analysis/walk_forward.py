"""
Walk-forward parameter sweep: cross-product of param grid × time windows.

`expand_param_grid` is a pure helper. `walk_forward` composes RunSpecs and
delegates execution to `analysis.runner`. Returns a long-format DataFrame:
one row per (combo, window) with all metrics and the parameter values used.
"""
from __future__ import annotations

from datetime import datetime
from itertools import product
from typing import Any

import pandas as pd

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar
from nautilus_trader.model.instruments import Equity
from nautilus_trader.trading.strategy import Strategy

from analysis.runner import run
from analysis.runspec import RunSpec


def expand_param_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Enumerate the cross-product of a parameter grid.

    Example: {"top_n": [2, 3], "lookback_days": [63, 126]} →
        [{"top_n": 2, "lookback_days": 63}, {"top_n": 2, "lookback_days": 126},
         {"top_n": 3, "lookback_days": 63}, {"top_n": 3, "lookback_days": 126}].
    Empty grid → [{}] so the caller still runs the base config once.
    """
    if not grid:
        return [{}]
    keys = list(grid.keys())
    return [dict(zip(keys, values, strict=True)) for values in product(*grid.values())]


def walk_forward(
    *,
    strategy_class: type[Strategy],
    config_class: type[StrategyConfig],
    base_config_kwargs: dict[str, Any],
    param_grid: dict[str, list[Any]],
    windows: list[tuple[datetime, datetime]],
    bars_by_ticker: dict[str, list[Bar]],
    instruments: list[Equity],
    slippage_prob: float = 0.25,
    starting_cash: int = 100_000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Run every (param combo) × (window) and return a long-format DataFrame.

    Each row carries: window_idx, start, end, the parameter values, plus the
    metrics from analysis.runner (cagr, sharpe, max_drawdown, total_return,
    n_days, n_trades, final_equity).
    """
    combos = expand_param_grid(param_grid)
    rows: list[dict[str, Any]] = []
    for window_idx, (start, end) in enumerate(windows):
        for combo in combos:
            config_kwargs = {**base_config_kwargs, **combo}
            config = config_class(**config_kwargs)
            label_bits = "_".join(f"{k}{v}" for k, v in combo.items()) or "base"
            spec = RunSpec(
                label=f"w{window_idx}-{label_bits}",
                strategy_class=strategy_class,
                strategy_config=config,
                start=start,
                end=end,
                slippage_prob=slippage_prob,
                starting_cash=starting_cash,
                random_seed=random_seed,
            )
            result = run(spec, bars_by_ticker=bars_by_ticker, instruments=instruments)
            row: dict[str, Any] = {
                "window_idx": window_idx,
                "start": start,
                "end": end,
                **combo,
                **result.metrics,
            }
            rows.append(row)
    return pd.DataFrame(rows)
