"""Constrained structural strategy discovery.

A StrategyStructure is a serializable choice of registered trading rules.
The finder enumerates structures plus parameter combinations, screens all of
them on tensors, and canonically validates a diverse finalist set.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Callable

import numpy as np
import pandas as pd

from analysis.optimizer import SearchSettings, evaluate_candidates, exhaustive_combinations
from analysis.structural_screen import screen_rule_candidates


@dataclass(frozen=True)
class StrategyStructure:
    signal_family: str
    trend_filter_days: int
    require_positive: bool

    @property
    def structure_id(self) -> str:
        trend = f"trend_{self.trend_filter_days}" if self.trend_filter_days else "no_trend"
        gate = "positive" if self.require_positive else "ungated"
        return f"{self.signal_family}__{trend}__{gate}"

    @property
    def complexity(self) -> int:
        return 1 + int(self.trend_filter_days > 0) + int(self.require_positive)

    def as_candidate_fields(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "structure_id": self.structure_id,
            "complexity": self.complexity,
        }


@dataclass(frozen=True)
class StrategyGrammar:
    signal_families: tuple[str, ...] = ("momentum", "mean_reversion")
    trend_filter_days: tuple[int, ...] = (0, 126, 200)
    positive_gates: tuple[bool, ...] = (False, True)
    max_complexity: int = 3

    def __post_init__(self) -> None:
        allowed = {"momentum", "mean_reversion"}
        if not self.signal_families or set(self.signal_families) - allowed:
            raise ValueError("grammar contains an unsupported signal family")
        if not self.trend_filter_days or any(days < 0 for days in self.trend_filter_days):
            raise ValueError("trend filter choices must be non-negative")
        if not self.positive_gates:
            raise ValueError("positive_gates must not be empty")
        if self.max_complexity < 1:
            raise ValueError("max_complexity must be positive")


@dataclass(frozen=True)
class FinderSettings:
    finalists: int = 24
    complexity_penalty: float = 0.03
    device: str = "auto"
    search: SearchSettings = SearchSettings(algorithm="grid")

    def __post_init__(self) -> None:
        if self.finalists <= 0:
            raise ValueError("finalists must be positive")
        if self.complexity_penalty < 0:
            raise ValueError("complexity_penalty must be non-negative")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")


def generate_structures(grammar: StrategyGrammar = StrategyGrammar()) -> list[StrategyStructure]:
    """Enumerate unique valid structures from the constrained grammar."""
    structures = [
        StrategyStructure(family, trend, gate)
        for family, trend, gate in product(
            grammar.signal_families,
            grammar.trend_filter_days,
            grammar.positive_gates,
        )
    ]
    return [structure for structure in structures if structure.complexity <= grammar.max_complexity]


def generate_candidates(
    structures: list[StrategyStructure],
    parameter_space: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Cross every structure with the existing finite parameter optimizer grid."""
    if not structures:
        raise ValueError("at least one strategy structure is required")
    parameter_sets = exhaustive_combinations(parameter_space)
    candidates = [
        {**structure.as_candidate_fields(), **params}
        for structure in structures
        for params in parameter_sets
    ]
    seen: set[tuple[tuple[str, Any], ...]] = set()
    unique: list[dict[str, Any]] = []
    for candidate in candidates:
        key = tuple(sorted(candidate.items()))
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def select_structural_finalists(screened: pd.DataFrame, finalists: int) -> pd.DataFrame:
    """Keep the best member of distinct structures, then fill by global score."""
    if finalists <= 0:
        raise ValueError("finalists must be positive")
    required = {"structure_id", "screen_score"}
    if missing := required - set(screened.columns):
        raise ValueError(f"screened frame missing columns: {sorted(missing)}")
    valid = screened.replace([np.inf, -np.inf], np.nan).dropna(subset=["screen_score"])
    valid = valid.sort_values("screen_score", ascending=False, kind="stable")
    if valid.empty:
        return valid

    representatives = valid.drop_duplicates("structure_id", keep="first")
    chosen = list(representatives.head(finalists).index)
    if len(chosen) < finalists:
        chosen.extend(
            idx for idx in valid.index
            if idx not in chosen
            for _ in [None]
            if len(chosen) < finalists
        )
    return valid.loc[chosen].sort_values(
        "screen_score", ascending=False, kind="stable",
    ).reset_index(drop=True)


def apply_complexity_penalty(
    ranked: pd.DataFrame,
    penalty: float,
) -> pd.DataFrame:
    """Penalize structural complexity after canonical performance aggregation."""
    result = ranked.copy()
    result["unpenalized_robust_score"] = result["robust_score"]
    eligible = result["eligible"]
    result.loc[eligible, "robust_score"] = (
        result.loc[eligible, "robust_score"]
        - penalty * result.loc[eligible, "complexity"].astype(float)
    )
    result = result.sort_values(
        ["eligible", "robust_score", "median_sharpe", "median_cagr"],
        ascending=[False, False, False, False],
        kind="stable",
    ).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result


def find_strategies(
    *,
    closes: pd.DataFrame,
    structures: list[StrategyStructure],
    parameter_space: dict[str, dict[str, Any]],
    windows: list[tuple],
    evaluator: Callable[[dict[str, Any], tuple], dict[str, float]],
    settings: FinderSettings = FinderSettings(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return canonical rankings, per-window evidence, and all screen results."""
    candidates = generate_candidates(structures, parameter_space)
    screened = screen_rule_candidates(closes, candidates, device=settings.device)
    finalists = select_structural_finalists(screened, min(settings.finalists, len(screened)))
    candidate_columns = [
        "signal_family", "trend_filter_days", "require_positive",
        "structure_id", "complexity", *parameter_space,
    ]
    finalist_candidates = finalists[candidate_columns].to_dict(orient="records")
    ranked, evaluations = evaluate_candidates(
        base_params={},
        candidates=finalist_candidates,
        windows=windows,
        evaluator=evaluator,
        settings=settings.search,
        param_names=candidate_columns,
    )
    ranked = apply_complexity_penalty(ranked, settings.complexity_penalty)
    return ranked, evaluations, screened


def describe_strategy(row: dict[str, Any] | pd.Series) -> str:
    """Render a discovered structure and parameters in plain English."""
    direction = "highest returns" if row["signal_family"] == "momentum" else "largest pullbacks"
    filters = []
    if int(row["trend_filter_days"]):
        filters.append(f"price above its {int(row['trend_filter_days'])}-day average")
    if bool(row["require_positive"]):
        filters.append("a positive signal")
    filter_text = f"; require {' and '.join(filters)}" if filters else ""
    return (
        f"Monthly select top {int(row['top_n'])} by {direction} over "
        f"{int(row['lookback_days'])} days, skipping {int(row['skip_days'])} days"
        f"{filter_text}."
    )
