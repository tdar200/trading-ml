"""Robust strategy parameter search over the canonical backtester.

Grid mode enumerates every finite combination, while TPE spends a fixed trial
budget on promising regions. Candidates are evaluated over multiple windows,
ranked for stability, and exposed as a Pareto frontier instead of reducing the
answer to one easily-overfit in-sample winner.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import product
import math
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd


MetricEvaluator = Callable[[dict[str, Any], tuple[datetime, datetime]], dict[str, float]]


@dataclass(frozen=True)
class SearchSettings:
    algorithm: Literal["grid", "tpe"] = "tpe"
    n_trials: int = 100
    seed: int = 42
    stability_penalty: float = 0.25
    drawdown_penalty: float = 0.10
    min_trades_per_window: int = 1
    min_positive_windows: int = 0
    min_worst_sharpe: float | None = None

    def __post_init__(self) -> None:
        if self.algorithm not in {"grid", "tpe"}:
            raise ValueError("algorithm must be 'grid' or 'tpe'")
        if self.n_trials <= 0:
            raise ValueError("n_trials must be positive")
        if self.stability_penalty < 0 or self.drawdown_penalty < 0:
            raise ValueError("penalties must be non-negative")
        if self.min_trades_per_window < 0:
            raise ValueError("min_trades_per_window must be non-negative")
        if self.min_positive_windows < 0:
            raise ValueError("min_positive_windows must be non-negative")


def build_non_overlapping_windows(
    start: datetime,
    end: datetime,
    n_windows: int,
) -> list[tuple[datetime, datetime]]:
    """Split ``[start, end)`` into contiguous, non-overlapping windows."""
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be timezone-aware")
    if end <= start:
        raise ValueError("end must be after start")
    if n_windows <= 0:
        raise ValueError("n_windows must be positive")
    span = end - start
    if span < timedelta(days=n_windows):
        raise ValueError("date range is too short for the requested windows")
    boundaries = [start + span * i / n_windows for i in range(n_windows + 1)]
    return list(zip(boundaries[:-1], boundaries[1:], strict=True))


def values_for_space(spec: dict[str, Any]) -> list[Any]:
    """Materialize one finite search-space dimension for exhaustive search."""
    kind = spec.get("type")
    if kind == "categorical":
        values = list(spec.get("choices", []))
        if not values:
            raise ValueError("categorical space requires non-empty choices")
        return values
    low, high = spec.get("low"), spec.get("high")
    if low is None or high is None or high < low:
        raise ValueError("numeric space requires low <= high")
    step = spec.get("step", 1 if kind == "int" else None)
    if step is None or step <= 0:
        raise ValueError("grid search requires a positive step")
    if kind == "int":
        return list(range(int(low), int(high) + 1, int(step)))
    if kind == "float":
        count = int(math.floor((float(high) - float(low)) / float(step) + 1e-12))
        return [float(low) + i * float(step) for i in range(count + 1)]
    raise ValueError(f"unsupported parameter type: {kind!r}")


def exhaustive_combinations(param_space: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if not param_space:
        return [{}]
    keys = list(param_space)
    dimensions = [values_for_space(param_space[key]) for key in keys]
    return [dict(zip(keys, values, strict=True)) for values in product(*dimensions)]


def _aggregate_candidate(rows: pd.DataFrame, settings: SearchSettings) -> dict[str, float | bool]:
    sharpes = rows["sharpe"].replace([np.inf, -np.inf], np.nan)
    cagrs = rows["cagr"].replace([np.inf, -np.inf], np.nan)
    drawdowns = rows["max_drawdown"].replace([np.inf, -np.inf], np.nan)
    trades = rows["n_trades"].fillna(0)
    eligible = bool(
        sharpes.notna().all()
        and cagrs.notna().all()
        and drawdowns.notna().all()
        and (trades >= settings.min_trades_per_window).all()
        and int((cagrs > 0).sum()) >= settings.min_positive_windows
        and (
            settings.min_worst_sharpe is None
            or float(sharpes.min()) > settings.min_worst_sharpe
        )
    )
    if not eligible:
        return {
            "eligible": False,
            "robust_score": float("-inf"),
            "median_sharpe": float("nan"),
            "worst_sharpe": float("nan"),
            "sharpe_iqr": float("nan"),
            "median_cagr": float("nan"),
            "worst_drawdown": float("nan"),
            "total_trades": float(trades.sum()),
            "positive_windows": int((cagrs > 0).sum()),
        }
    median_sharpe = float(sharpes.median())
    iqr = float(sharpes.quantile(0.75) - sharpes.quantile(0.25))
    worst_drawdown = float(drawdowns.min())
    score = median_sharpe - settings.stability_penalty * iqr - settings.drawdown_penalty * abs(worst_drawdown)
    return {
        "eligible": True,
        "robust_score": score,
        "median_sharpe": median_sharpe,
        "worst_sharpe": float(sharpes.min()),
        "sharpe_iqr": iqr,
        "median_cagr": float(cagrs.median()),
        "worst_drawdown": worst_drawdown,
        "total_trades": float(trades.sum()),
        "positive_windows": int((cagrs > 0).sum()),
    }


def _pareto_mask(frame: pd.DataFrame) -> pd.Series:
    """Find candidates not dominated on return, stability, and drawdown."""
    mask = pd.Series(False, index=frame.index)
    eligible = frame[frame["eligible"]]
    objectives = eligible[["median_sharpe", "worst_sharpe", "median_cagr", "worst_drawdown"]]
    for idx, row in objectives.iterrows():
        dominates = (objectives >= row).all(axis=1) & (objectives > row).any(axis=1)
        mask.loc[idx] = not bool(dominates.any())
    return mask


def rank_evaluations(
    evaluations: pd.DataFrame,
    param_names: list[str],
    settings: SearchSettings,
) -> pd.DataFrame:
    """Aggregate per-window rows into a ranked candidate table."""
    if evaluations.empty:
        raise ValueError("evaluations must be non-empty")
    required = {"sharpe", "cagr", "max_drawdown", "n_trades", *param_names}
    missing = required - set(evaluations.columns)
    if missing:
        raise ValueError(f"evaluations missing columns: {sorted(missing)}")
    records: list[dict[str, Any]] = []
    grouped = evaluations.groupby(param_names, dropna=False, sort=False) if param_names else [((), evaluations)]
    for key, rows in grouped:
        keys = key if isinstance(key, tuple) else (key,)
        records.append({**dict(zip(param_names, keys, strict=True)), **_aggregate_candidate(rows, settings)})
    ranked = pd.DataFrame(records)
    ranked["pareto_optimal"] = _pareto_mask(ranked)
    ranked = ranked.sort_values(
        ["eligible", "robust_score", "median_sharpe", "median_cagr"],
        ascending=[False, False, False, False],
        kind="stable",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def _suggest(trial: Any, name: str, spec: dict[str, Any]) -> Any:
    kind = spec.get("type")
    if kind == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1)))
    if kind == "float":
        step = spec.get("step")
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), step=float(step) if step else None)
    if kind == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    raise ValueError(f"unsupported parameter type: {kind!r}")


def evaluate_candidates(
    *,
    base_params: dict[str, Any],
    candidates: list[dict[str, Any]],
    windows: list[tuple[datetime, datetime]],
    evaluator: MetricEvaluator,
    settings: SearchSettings,
    param_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Canonically evaluate an explicit finalist list across all windows."""
    if not candidates:
        raise ValueError("candidates must be non-empty")
    if not windows:
        raise ValueError("at least one evaluation window is required")
    names = param_names or list(candidates[0])
    rows: list[dict[str, Any]] = []
    for combo in candidates:
        if set(combo) != set(names):
            raise ValueError("every candidate must have the same parameter names")
        params = {**base_params, **combo}
        for window_idx, window in enumerate(windows):
            rows.append({
                "window_idx": window_idx,
                "start": window[0],
                "end": window[1],
                **combo,
                **evaluator(params, window),
            })
    evaluations = pd.DataFrame(rows).drop_duplicates(
        subset=[*names, "window_idx"], keep="first",
    )
    return rank_evaluations(evaluations, names, settings), evaluations


def search(
    *,
    base_params: dict[str, Any],
    param_space: dict[str, dict[str, Any]],
    windows: list[tuple[datetime, datetime]],
    evaluator: MetricEvaluator,
    settings: SearchSettings = SearchSettings(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate candidates and return ``(ranked_candidates, window_rows)``."""
    if not windows:
        raise ValueError("at least one evaluation window is required")
    rows: list[dict[str, Any]] = []

    def evaluate(combo: dict[str, Any]) -> float:
        candidate_rows: list[dict[str, Any]] = []
        params = {**base_params, **combo}
        for window_idx, window in enumerate(windows):
            metrics = evaluator(params, window)
            candidate_rows.append({
                "window_idx": window_idx,
                "start": window[0],
                "end": window[1],
                **combo,
                **metrics,
            })
        rows.extend(candidate_rows)
        return float(_aggregate_candidate(pd.DataFrame(candidate_rows), settings)["robust_score"])

    if settings.algorithm == "grid":
        for combo in exhaustive_combinations(param_space):
            evaluate(combo)
    else:
        import optuna
        sampler = optuna.samplers.TPESampler(seed=settings.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial: Any) -> float:
            combo = {name: _suggest(trial, name, spec) for name, spec in param_space.items()}
            return evaluate(combo)

        study.optimize(objective, n_trials=settings.n_trials, show_progress_bar=False)

    evaluations = pd.DataFrame(rows)
    dedupe_cols = [*param_space, "window_idx"]
    evaluations = evaluations.drop_duplicates(subset=dedupe_cols, keep="first")
    ranked = rank_evaluations(evaluations, list(param_space), settings)
    return ranked, evaluations


def accelerator_status() -> dict[str, Any]:
    """Report GPU readiness without claiming the event backtester uses it."""
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        return {
            "cuda_available": cuda,
            "torch_version": torch.__version__,
            "device": torch.cuda.get_device_name(0) if cuda else None,
            "event_backtester": "cpu",
            "gpu_scope": "batched vectorized pre-screening",
        }
    except Exception as exc:
        return {
            "cuda_available": False,
            "torch_version": None,
            "device": None,
            "event_backtester": "cpu",
            "gpu_scope": "unavailable",
            "reason": str(exc),
        }
