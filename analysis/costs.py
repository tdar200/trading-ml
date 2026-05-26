"""
Cost-stress: re-run a chosen RunSpec at multiple slippage levels and report
how the headline metrics degrade. Used after walk-forward selects a "winner".
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import pandas as pd

from nautilus_trader.model.data import Bar
from nautilus_trader.model.instruments import Equity

from analysis.runner import run
from analysis.runspec import RunSpec


def cost_stress(
    *,
    base_spec: RunSpec,
    slippage_probs: list[float],
    bars_by_ticker: dict[str, list[Bar]],
    instruments: list[Equity],
) -> pd.DataFrame:
    """Re-run `base_spec` at each slippage level; return one row per level."""
    rows: list[dict[str, Any]] = []
    for prob in slippage_probs:
        spec = replace(base_spec, slippage_prob=prob, label=f"{base_spec.label}-slip{prob:.2f}")
        result = run(spec, bars_by_ticker=bars_by_ticker, instruments=instruments)
        rows.append({"slippage_prob": prob, **result.metrics})
    return pd.DataFrame(rows)
