"""Small, hypothesis-driven intraday V3 grammar with causal adaptive experts."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any
from collections.abc import Callable

import numpy as np
import pandas as pd

from analysis.metrics import cagr, max_drawdown, sharpe
from analysis.profitability import (
    DecisionDataset, _neutral_weights, causal_momentum_forecast,
)
from analysis.vectorized_screen import resolve_device

V2_PARAM_FIELDS = [
    "family", "timeframe", "decision_bar", "max_positions_per_side", "min_edge_bps",
    "edge_cost_multiple", "min_relative_volume", "event_policy",
    "maximum_spread_bps", "execution_buffer_bps",
]


@dataclass(frozen=True)
class ProfitabilityV2Grammar:
    """A deliberately small preregistered grammar (40–48 trials)."""

    decision_bars: tuple[int, ...] = (23, 47)
    max_positions_per_side: tuple[int, ...] = (2, 3)
    minimum_relative_volume: tuple[float, ...] = (1.0, 1.25)
    min_edge_bps: float = 10.0
    edge_cost_multiple: float = 2.0
    maximum_spread_bps: float = 8.0
    execution_buffer_bps: float = 0.75
    timeframe: str = "5m"


def generate_v2_candidates(
    *, quote_ready: bool, event_ready: bool,
    grammar: ProfitabilityV2Grammar = ProfitabilityV2Grammar(),
) -> list[dict[str, Any]]:
    families = [
        "structure_momentum", "structure_reversal", "pooled_structure_ridge",
        "online_expert_ensemble", "regime_expert_router", "cost_aware_asymmetric_ridge",
    ]
    if quote_ready:
        # Keep the multiple-testing budget below 50 when quote confirmation is available.
        families.remove("structure_reversal")
        families.append("quote_structure_confirmed")
    event_policy = "avoid_events" if event_ready else "all"
    rows = []
    for family, decision, positions, min_rvol in product(
        families, grammar.decision_bars, grammar.max_positions_per_side,
        grammar.minimum_relative_volume,
    ):
        rows.append({
            "family": family, "timeframe": grammar.timeframe,
            "decision_bar": int(decision),
            "max_positions_per_side": int(positions),
            "min_edge_bps": float(grammar.min_edge_bps),
            "edge_cost_multiple": float(grammar.edge_cost_multiple),
            "min_relative_volume": float(min_rvol),
            "event_policy": event_policy,
            "maximum_spread_bps": float(grammar.maximum_spread_bps),
            "execution_buffer_bps": float(grammar.execution_buffer_bps),
        })
    if not 20 <= len(rows) <= 50:
        raise ValueError(f"V3 grammar must contain 20–50 trials, got {len(rows)}")
    return rows


def _require_structure(dataset: DecisionDataset) -> None:
    required = (
        dataset.vwap_deviation, dataset.prior_range_position,
        dataset.opening_range_break, dataset.support_rejection,
    )
    if any(item is None for item in required):
        raise ValueError("V3 requires causal VWAP, prior-range, opening-range, and rejection features")


def _structure_signal(dataset: DecisionDataset, family: str) -> np.ndarray:
    _require_structure(dataset)
    momentum = (
        dataset.sector_residual if dataset.sector_residual is not None
        else dataset.first_residual
    )
    scale = np.maximum(dataset.realized_volatility, 1e-5)
    if family == "structure_reversal":
        return dataset.support_rejection * np.abs(dataset.vwap_deviation) - 0.25 * dataset.vwap_deviation
    return (
        momentum + 0.35 * dataset.vwap_deviation
        + 0.15 * dataset.opening_range_break * scale
        + 0.10 * dataset.prior_range_position * scale
        + 0.10 * dataset.support_rejection * scale
    )


def pooled_structure_ridge_predictions(
    dataset: DecisionDataset, *, train_sessions: int = 120, refit_every: int = 20,
    alpha: float = 20.0, device: str = "cuda",
) -> np.ndarray:
    """Causal pooled ridge over price structure, volume and volatility features."""
    import torch

    _require_structure(dataset)
    used_device = resolve_device(device)
    sector = dataset.sector_residual if dataset.sector_residual is not None else dataset.first_residual
    features = np.stack([
        dataset.first_residual, sector, dataset.gap_residual,
        np.log1p(np.maximum(dataset.relative_volume, 0)),
        dataset.realized_volatility, dataset.bar_pressure,
        dataset.vwap_deviation, dataset.prior_range_position,
        dataset.opening_range_break, dataset.support_rejection,
    ], axis=2)
    output = np.full(dataset.forward_return.shape, np.nan)
    for start in range(60, len(dataset.session_dates), refit_every):
        begin = max(0, start - train_sessions)
        x_train = features[begin:start].reshape(-1, features.shape[2])
        y_train = dataset.forward_return[begin:start].reshape(-1)
        valid = np.isfinite(x_train).all(1) & np.isfinite(y_train)
        if valid.sum() < 200:
            continue
        x = torch.as_tensor(x_train[valid], dtype=torch.float64, device=used_device)
        y = torch.as_tensor(y_train[valid], dtype=torch.float64, device=used_device)
        mean, std = x.mean(0), x.std(0, unbiased=False).clamp(min=1e-12)
        x = (x - mean) / std
        design = torch.cat([torch.ones((len(x), 1), dtype=torch.float64, device=used_device), x], 1)
        penalty = torch.eye(design.shape[1], dtype=torch.float64, device=used_device) * alpha
        penalty[0, 0] = 0
        coefficients = torch.linalg.solve(design.T @ design + penalty, design.T @ y)
        stop = min(len(dataset.session_dates), start + refit_every)
        test = features[start:stop].reshape(-1, features.shape[2])
        good = np.isfinite(test).all(1)
        predicted = np.full(len(test), np.nan)
        if good.any():
            values = torch.as_tensor(test[good], dtype=torch.float64, device=used_device)
            values = (values - mean) / std
            values = torch.cat([
                torch.ones((len(values), 1), dtype=torch.float64, device=used_device), values,
            ], 1)
            predicted[good] = (values @ coefficients).cpu().numpy()
        output[start:stop] = predicted.reshape(stop - start, -1)
    return output


def _cross_sectional_ic(score: np.ndarray, target: np.ndarray) -> float:
    valid = np.isfinite(score) & np.isfinite(target)
    if valid.sum() < 4:
        return np.nan
    left, right = score[valid], target[valid]
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def online_expert_predictions(
    dataset: DecisionDataset, experts: dict[str, np.ndarray], *,
    lookback: int = 60, mode: str = "ensemble",
) -> np.ndarray:
    """Combine experts using only their trailing, already-realized cross-sectional IC.

    The ensemble abstains when none of the experts has positive trailing skill.
    Router mode selects the best expert within the current causal market regime.
    """
    if mode not in {"ensemble", "router"}:
        raise ValueError("mode must be 'ensemble' or 'router'")
    names = tuple(experts)
    if not names or any(experts[name].shape != dataset.forward_return.shape for name in names):
        raise ValueError("experts must match the decision dataset shape")
    sessions = len(dataset.session_dates)
    regimes = np.where(dataset.panic_rebound, 2, np.where(dataset.high_information, 1, 0))
    daily_ic = np.full((sessions, len(names)), np.nan)
    output = np.full(dataset.forward_return.shape, np.nan)
    for index in range(sessions):
        if index >= 20:
            start = max(0, index - lookback)
            history = np.arange(start, index)
            if mode == "router":
                matching = history[regimes[history] == regimes[index]]
                if len(matching) >= 10:
                    history = matching
            quality = np.asarray([
                np.nanmean(daily_ic[history, column])
                if np.isfinite(daily_ic[history, column]).any() else np.nan
                for column in range(len(names))
            ])
            positive = np.where(np.isfinite(quality) & (quality > 0), quality, 0.0)
            if positive.sum() > 0:
                if mode == "router":
                    output[index] = experts[names[int(np.argmax(positive))]][index]
                else:
                    weights = positive / positive.sum()
                    stacked = np.stack([experts[name][index] for name in names])
                    finite = np.isfinite(stacked)
                    weighted = np.where(finite, stacked * weights[:, None], 0.0).sum(0)
                    available_weight = (finite * weights[:, None]).sum(0)
                    output[index] = np.divide(
                        weighted, available_weight,
                        out=np.full(stacked.shape[1], np.nan), where=available_weight > 0,
                    )
        for column, name in enumerate(names):
            daily_ic[index, column] = _cross_sectional_ic(
                experts[name][index], dataset.forward_return[index],
            )
    return output


def cost_aware_asymmetric_predictions(
    dataset: DecisionDataset, *, assumed_cost_per_side_bps: float = 2.5,
    confidence_z: float = 0.10, drift_lookback: int = 60,
    train_sessions: int = 120, refit_every: int = 20, alpha: float = 20.0,
    device: str = "cuda",
) -> np.ndarray:
    """Predict tradable net edge separately for longs and shorts, then abstain.

    Training labels deduct a conservative round trip before fitting. Prediction
    at session ``t`` only uses targets before ``t``. A trailing IC circuit
    breaker disables the model when its already-realized ranking skill is not
    positive, preventing persistent trading through detected signal inversion.
    """
    import torch

    _require_structure(dataset)
    used_device = resolve_device(device)
    sector = dataset.sector_residual if dataset.sector_residual is not None else dataset.first_residual
    features = np.stack([
        dataset.first_residual, sector, dataset.gap_residual,
        np.log1p(np.maximum(dataset.relative_volume, 0)),
        dataset.realized_volatility, dataset.bar_pressure,
        dataset.vwap_deviation, dataset.prior_range_position,
        dataset.opening_range_break, dataset.support_rejection,
    ], axis=2)
    impact_bps = 0.25 * np.sqrt(np.maximum(dataset.relative_volume, 1.0))
    round_trip = 2 * (assumed_cost_per_side_bps + 0.75 + impact_bps) / 10_000
    targets = np.stack([
        dataset.forward_return - round_trip,
        -dataset.forward_return - round_trip,
    ], axis=2)
    raw = np.full(dataset.forward_return.shape, np.nan)
    for start in range(60, len(dataset.session_dates), refit_every):
        begin = max(0, start - train_sessions)
        x_train = features[begin:start].reshape(-1, features.shape[2])
        y_train = targets[begin:start].reshape(-1, 2)
        valid = np.isfinite(x_train).all(1) & np.isfinite(y_train).all(1)
        if valid.sum() < 200:
            continue
        x = torch.as_tensor(x_train[valid], dtype=torch.float64, device=used_device)
        y = torch.as_tensor(y_train[valid], dtype=torch.float64, device=used_device)
        mean, std = x.mean(0), x.std(0, unbiased=False).clamp(min=1e-12)
        normalized = (x - mean) / std
        design = torch.cat([
            torch.ones((len(x), 1), dtype=torch.float64, device=used_device), normalized,
        ], 1)
        penalty = torch.eye(design.shape[1], dtype=torch.float64, device=used_device) * alpha
        penalty[0, 0] = 0
        coefficients = torch.linalg.solve(design.T @ design + penalty, design.T @ y)
        residual_scale = (y - design @ coefficients).std(0, unbiased=False).clamp(min=1e-12)
        stop = min(len(dataset.session_dates), start + refit_every)
        test = features[start:stop].reshape(-1, features.shape[2])
        good = np.isfinite(test).all(1)
        estimates = np.full((len(test), 2), np.nan)
        if good.any():
            values = torch.as_tensor(test[good], dtype=torch.float64, device=used_device)
            values = (values - mean) / std
            values = torch.cat([
                torch.ones((len(values), 1), dtype=torch.float64, device=used_device), values,
            ], 1)
            predicted = values @ coefficients
            confident = predicted.max(1).values > confidence_z * residual_scale.mean()
            predicted[~confident] = torch.nan
            estimates[good] = predicted.cpu().numpy()
        long_edge, short_edge = estimates[:, 0], estimates[:, 1]
        signed = np.where(long_edge >= short_edge, long_edge, -short_edge)
        best_edge = np.where(np.isfinite(estimates), estimates, -np.inf).max(axis=1)
        signed[best_edge <= 0] = np.nan
        raw[start:stop] = signed.reshape(stop - start, -1)
    # Causal drift circuit breaker: session t never uses its own outcome.
    daily_ic = np.asarray([
        _cross_sectional_ic(raw[index], dataset.forward_return[index])
        for index in range(len(raw))
    ])
    output = raw.copy()
    for index in range(60, len(output)):
        history = daily_ic[max(0, index - drift_lookback):index]
        if not np.isfinite(history).any() or np.nanmean(history) <= 0:
            output[index] = np.nan
    return output


def build_v2_forecasts(dataset: DecisionDataset, *, device: str = "cuda") -> dict[str, np.ndarray]:
    forecasts = {
        "structure_momentum": causal_momentum_forecast(
            _structure_signal(dataset, "structure_momentum"), dataset.forward_return, device=device,
        ),
        "structure_reversal": causal_momentum_forecast(
            _structure_signal(dataset, "structure_reversal"), dataset.forward_return, device=device,
        ),
        "pooled_structure_ridge": pooled_structure_ridge_predictions(dataset, device=device),
    }
    experts = {name: forecasts[name] for name in (
        "structure_momentum", "structure_reversal", "pooled_structure_ridge",
    )}
    forecasts["online_expert_ensemble"] = online_expert_predictions(
        dataset, experts, mode="ensemble",
    )
    forecasts["regime_expert_router"] = online_expert_predictions(
        dataset, experts, mode="router",
    )
    forecasts["cost_aware_asymmetric_ridge"] = cost_aware_asymmetric_predictions(
        dataset, device=device,
    )
    return forecasts


def _confirmation_mask(
    dataset: DecisionDataset, family: str, score: np.ndarray, index: int,
) -> np.ndarray:
    if family == "signal_consensus":
        return np.isfinite(score)
    direction = np.sign(score)
    if family == "structure_reversal":
        return np.sign(dataset.support_rejection[index]) == direction
    confirmations = (
        (np.sign(dataset.vwap_deviation[index]) == direction).astype(int)
        + (np.sign(dataset.opening_range_break[index]) == direction).astype(int)
        + (np.sign(dataset.prior_range_position[index]) == direction).astype(int)
    )
    return confirmations >= 2


def evaluate_v2_candidate(
    dataset: DecisionDataset, candidate: dict[str, Any], *, cost_per_side_bps: float,
    forecast_cache: dict[str, np.ndarray] | None = None, device: str = "cuda",
    return_ledger: bool = False,
    stream_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], np.ndarray] | tuple[dict[str, Any], np.ndarray, pd.DataFrame]:
    """Replay next-bar entry through close with spread, latency and impact costs."""
    used_device = resolve_device(device)
    family = str(candidate["family"])
    base_family = "structure_momentum" if family == "quote_structure_confirmed" else family
    scores = (forecast_cache or {}).get(base_family)
    if scores is None:
        scores = build_v2_forecasts(dataset, device=used_device)[base_family]
    daily = np.zeros(len(dataset.session_dates))
    contribution = np.zeros(len(dataset.tickers))
    ledger_rows: list[dict[str, Any]] = []
    events = trades = 0
    beta_exposure = []
    running_equity = 1.0

    def publish(index: int, session_trades: list[dict[str, Any]]) -> None:
        nonlocal running_equity
        running_equity *= 1 + daily[index]
        if stream_callback is not None:
            stream_callback({
                "session_index": index,
                "session_count": len(daily),
                "session_date": dataset.session_dates[index],
                "daily_return": float(daily[index]),
                "equity": float(running_equity),
                "cumulative_trades": trades,
                "portfolio_events": events,
                "trades": session_trades,
            })

    for index in range(len(daily)):
        session_trades: list[dict[str, Any]] = []
        score = np.where(dataset.active_mask[index], scores[index], np.nan)
        mask = _confirmation_mask(dataset, base_family, score, index)
        mask &= dataset.relative_volume[index] >= float(candidate["min_relative_volume"])
        if candidate["event_policy"] == "avoid_events":
            mask &= dataset.event_clear[index]
        if family == "quote_structure_confirmed":
            if dataset.quote_imbalance is None or dataset.spread_bps is None:
                raise ValueError("quote-confirmed V3 family requires causal SIP/NBBO coverage")
            mask &= np.sign(dataset.quote_imbalance[index]) == np.sign(score)
            mask &= dataset.spread_bps[index] <= float(candidate["maximum_spread_bps"])
        score = np.where(mask, score, np.nan)
        if not np.isfinite(score).any():
            publish(index, session_trades)
            continue
        spread_component = (
            dataset.spread_bps[index] / 2 if dataset.spread_bps is not None
            else np.full(len(score), cost_per_side_bps)
        )
        impact = 0.25 * np.sqrt(np.maximum(dataset.relative_volume[index], 1.0))
        cost = np.maximum(cost_per_side_bps, spread_component)
        cost = cost + float(candidate["execution_buffer_bps"]) + impact
        gate = np.maximum(
            float(candidate["min_edge_bps"]),
            float(candidate["edge_cost_multiple"]) * 2 * cost,
        ) / 10_000
        weights = _neutral_weights(
            score, dataset.beta[index], int(candidate["max_positions_per_side"]), gate,
        )
        if weights is None:
            publish(index, session_trades)
            continue
        selected = weights != 0
        item = np.zeros(len(weights))
        item[selected] = (
            weights[selected] * dataset.forward_return[index, selected]
            - np.abs(weights[selected]) * 2 * cost[selected] / 10_000
        )
        if return_ledger:
            for column in np.flatnonzero(selected):
                gross = float(weights[column] * dataset.forward_return[index, column])
                cost_drag = float(abs(weights[column]) * 2 * cost[column] / 10_000)
                ledger_row = {
                    "session_date": dataset.session_dates[index],
                    "ticker": dataset.tickers[column],
                    "side": "long" if weights[column] > 0 else "short",
                    "weight": float(weights[column]), "score": float(score[column]),
                    "gross_contribution": gross, "cost_contribution": cost_drag,
                    "net_contribution": gross - cost_drag,
                    "relative_volume": float(dataset.relative_volume[index, column]),
                    "high_information": bool(dataset.high_information[index]),
                    "panic_rebound": bool(dataset.panic_rebound[index]),
                }
                ledger_rows.append(ledger_row)
                session_trades.append(ledger_row)
        daily[index] = item.sum()
        contribution += item
        trades += int(selected.sum())
        events += 1
        beta_exposure.append(abs(float(weights[selected] @ dataset.beta[index, selected])))
        publish(index, session_trades)
    curve = pd.Series(np.concatenate([[1.0], np.cumprod(1 + daily)]))
    result = ({
        **candidate, "sharpe": float(sharpe(pd.Series(daily))),
        "cagr": float(cagr(pd.Series(daily))), "max_drawdown": float(max_drawdown(curve)),
        "total_return": float(curve.iloc[-1] - 1), "n_trades": trades,
        "portfolio_events": events, "positive_sessions": int((daily > 0).sum()),
        "mean_abs_beta_exposure": float(np.mean(beta_exposure)) if beta_exposure else float("nan"),
        "simulation_device": f"hybrid-{used_device}",
    }, contribution)
    if return_ledger:
        return (*result, pd.DataFrame(ledger_rows))
    return result


def describe_v2(candidate: dict[str, Any] | pd.Series) -> str:
    timeframe = str(candidate.get("timeframe", "5m"))
    bar_minutes = 60 if timeframe == "1h" else int(timeframe.removesuffix("m"))
    return (
        f"{candidate['family']} on {timeframe} bars at "
        f"{(int(candidate['decision_bar']) + 1) * bar_minutes} minutes; "
        f"next-bar entry, close exit, {int(candidate['max_positions_per_side'])} per side, "
        f"rvol >= {float(candidate['min_relative_volume']):g}, edge >= "
        f"{float(candidate['min_edge_bps']):g} bps and "
        f"{float(candidate['edge_cost_multiple']):g}x conservative round-trip cost"
    )
