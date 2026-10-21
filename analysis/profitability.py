"""Pre-registered, cost-gated intraday profitability research families."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from analysis.cross_sectional import CrossSectionalPanel
from analysis.metrics import cagr, max_drawdown, sharpe
from analysis.vectorized_screen import resolve_device
from data.quotes import quote_features_at
from data.universe_membership import members_on

FAMILIES = (
    "first_half_to_close", "seasonal_momentum", "pooled_ridge",
    "pooled_elastic_net", "quote_flow_confirmed",
)
PARAM_FIELDS = [
    "family", "decision_bar", "max_positions_per_side", "min_edge_bps",
    "edge_cost_multiple", "confidence_z", "regime", "seasonality_days", "event_policy",
]


@dataclass(frozen=True)
class ProfitabilityGrammar:
    # Alpaca timestamps bars at their start: bars 5/23/47/71 complete
    # 30/120/240/360 minutes after the 09:30 open.
    decision_bars: tuple[int, ...] = (5, 23, 47, 71)
    max_positions_per_side: tuple[int, ...] = (2, 3, 5)
    min_edge_bps: tuple[float, ...] = (0.0, 5.0, 10.0)
    edge_cost_multiple: tuple[float, ...] = (2.0, 3.0)
    confidence_z: tuple[float, ...] = (0.0, 0.5, 1.0)
    regimes: tuple[str, ...] = ("all", "high_information", "avoid_panic_rebound")
    seasonality_days: tuple[int, ...] = (20, 40)


@dataclass(frozen=True)
class DecisionDataset:
    decision_bar: int
    session_dates: np.ndarray
    tickers: tuple[str, ...]
    first_residual: np.ndarray
    gap_residual: np.ndarray
    relative_volume: np.ndarray
    realized_volatility: np.ndarray
    bar_pressure: np.ndarray
    beta: np.ndarray
    forward_return: np.ndarray
    seasonality: dict[int, np.ndarray]
    high_information: np.ndarray
    panic_rebound: np.ndarray
    active_mask: np.ndarray
    event_clear: np.ndarray
    quote_imbalance: np.ndarray | None = None
    spread_bps: np.ndarray | None = None
    sector_residual: np.ndarray | None = None
    vwap_deviation: np.ndarray | None = None
    prior_range_position: np.ndarray | None = None
    opening_range_break: np.ndarray | None = None
    support_rejection: np.ndarray | None = None
    bar_minutes: int = 5


def generate_profitability_candidates(
    *, quote_ready: bool, event_ready: bool = False,
    grammar: ProfitabilityGrammar = ProfitabilityGrammar(),
) -> list[dict[str, Any]]:
    families = FAMILIES if quote_ready else FAMILIES[:-1]
    rows = []
    for family, decision, positions, edge, multiple, confidence_z, regime in product(
        families, grammar.decision_bars, grammar.max_positions_per_side,
        grammar.min_edge_bps, grammar.edge_cost_multiple, grammar.confidence_z,
        grammar.regimes,
    ):
        seasonal_values = grammar.seasonality_days if family in {
            "seasonal_momentum", "pooled_ridge", "pooled_elastic_net",
        } else (0,)
        for seasonal_days, event_policy in product(
            seasonal_values, ("all", "avoid_events") if event_ready else ("all",),
        ):
            rows.append({
                "family": family, "decision_bar": int(decision),
                "max_positions_per_side": int(positions),
                "min_edge_bps": float(edge), "edge_cost_multiple": float(multiple),
                "confidence_z": float(confidence_z),
                "regime": regime, "seasonality_days": int(seasonal_days),
                "event_policy": event_policy,
            })
    return rows


def _row_for_bar(panel: CrossSectionalPanel, session: int, bar: int) -> int | None:
    found = np.flatnonzero((panel.sessions == session) & (panel.bar_in_session == bar))
    return int(found[0]) if len(found) == 1 else None


def prepare_decision_dataset(
    panel: CrossSectionalPanel,
    decision_bar: int,
    *,
    quotes_by_ticker: dict[str, pd.DataFrame] | None = None,
    membership: pd.DataFrame | None = None,
    event_calendar: pd.DataFrame | None = None,
) -> DecisionDataset:
    """Create one causal row per complete session; exits use the final common close."""
    records = []
    prior_last = None
    market_history = []
    for session in np.unique(panel.sessions):
        indices = np.flatnonzero(panel.sessions == session)
        signal = _row_for_bar(panel, int(session), 5)
        decision = _row_for_bar(panel, int(session), decision_bar)
        if signal is None or decision is None or decision + 1 >= len(panel.sessions):
            prior_last = int(indices[-1])
            market_history.append(panel.benchmark_close[indices[-1]])
            continue
        enter, exit_index = decision + 1, int(indices[-1])
        if panel.sessions[enter] != session or enter >= exit_index:
            prior_last = exit_index
            market_history.append(panel.benchmark_close[exit_index])
            continue
        opening = int(indices[0])
        market_first = panel.benchmark_close[signal] / panel.benchmark_open[opening] - 1
        first = panel.close[signal] / panel.open[opening] - 1 - panel.beta[signal] * market_first
        if prior_last is None:
            gap = np.full(len(panel.tickers), np.nan)
        else:
            market_gap = panel.benchmark_open[opening] / panel.benchmark_close[prior_last] - 1
            gap = panel.open[opening] / panel.close[prior_last] - 1 - panel.beta[signal] * market_gap
        returns = np.diff(np.log(panel.close[indices[indices <= decision]]), axis=0)
        realized = np.sqrt(np.mean(returns * returns, axis=0)) if len(returns) else np.full(len(panel.tickers), np.nan)
        pressure = panel.close[decision] / panel.open[decision] - 1
        future = panel.close[exit_index] / panel.open[enter] - 1
        trailing_market = (
            panel.benchmark_close[prior_last] / market_history[-5] - 1
            if prior_last is not None and len(market_history) >= 5 else 0.0
        )
        rvol_row = panel.relative_volume[decision]
        finite_rvol = rvol_row[np.isfinite(rvol_row)]
        records.append((
            session, decision, first, gap, panel.relative_volume[decision],
            realized, pressure, panel.beta[decision], future,
            bool(len(finite_rvol) and np.median(finite_rvol) >= 1.0 and np.nanstd(first) >= 0.003),
            bool(trailing_market < -0.02 and market_first > 0.002),
        ))
        prior_last = exit_index
        market_history.append(panel.benchmark_close[exit_index])
    if len(records) < 60:
        raise ValueError("fewer than 60 complete sessions for profitability research")
    sessions = np.asarray([row[0] for row in records], dtype=int)
    decision_rows = np.asarray([row[1] for row in records], dtype=int)
    target = np.stack([row[8] for row in records])
    seasonality = {
        days: pd.DataFrame(target).shift(1).rolling(days, min_periods=max(10, days // 2)).mean().to_numpy()
        for days in (20, 40)
    }
    quote_imbalance = spread = None
    if quotes_by_ticker and all(ticker in quotes_by_ticker for ticker in panel.tickers):
        quote_items = [
            quote_features_at(
                quotes_by_ticker[ticker], panel.timestamps[decision_rows] + pd.Timedelta(minutes=5),
            )
            for ticker in panel.tickers
        ]
        candidate_spread = np.column_stack([item["spread_bps"] for item in quote_items])
        if float(np.isfinite(candidate_spread).mean()) >= 0.80:
            spread = candidate_spread
            quote_imbalance = np.column_stack([item["quote_imbalance"] for item in quote_items])
    active_mask = np.ones(target.shape, dtype=bool)
    if membership is not None:
        for index, session_date in enumerate(panel.session_dates[decision_rows]):
            active = set(members_on(membership, pd.Timestamp(session_date, tz="UTC")))
            active_mask[index] = np.asarray([ticker in active for ticker in panel.tickers])
    if event_calendar is None:
        event_clear = np.ones(target.shape, dtype=bool)
    else:
        from data.event_calendar import event_clear_mask
        event_clear = event_clear_mask(
            event_calendar, panel.session_dates[decision_rows], panel.tickers,
        )
    return DecisionDataset(
        decision_bar=decision_bar,
        session_dates=panel.session_dates[decision_rows], tickers=panel.tickers,
        first_residual=np.stack([row[2] for row in records]),
        gap_residual=np.stack([row[3] for row in records]),
        relative_volume=np.stack([row[4] for row in records]),
        realized_volatility=np.stack([row[5] for row in records]),
        bar_pressure=np.stack([row[6] for row in records]),
        beta=np.stack([row[7] for row in records]), forward_return=target,
        seasonality=seasonality,
        high_information=np.asarray([row[9] for row in records], dtype=bool),
        panic_rebound=np.asarray([row[10] for row in records], dtype=bool),
        active_mask=active_mask,
        event_clear=event_clear,
        quote_imbalance=quote_imbalance, spread_bps=spread,
    )


def pooled_ridge_predictions(
    dataset: DecisionDataset, seasonality_days: int, *, train_sessions: int = 120,
    refit_every: int = 20, alpha: float = 10.0, device: str = "cuda",
) -> np.ndarray:
    """Expanding walk-forward pooled model; a prediction never trains on its target."""
    seasonal = dataset.seasonality[seasonality_days]
    sector = dataset.sector_residual if dataset.sector_residual is not None else dataset.first_residual
    features = np.stack([
        dataset.first_residual, dataset.gap_residual,
        np.log1p(np.maximum(dataset.relative_volume, 0)),
        dataset.realized_volatility, dataset.bar_pressure, seasonal, sector,
    ], axis=2)
    import torch

    used_device = resolve_device(device)
    predictions = np.full(dataset.forward_return.shape, np.nan)
    sessions = len(dataset.session_dates)
    for start in range(max(60, seasonality_days), sessions, refit_every):
        train_start = max(0, start - train_sessions)
        x_train = features[train_start:start].reshape(-1, features.shape[2])
        y_train = dataset.forward_return[train_start:start].reshape(-1)
        valid = np.isfinite(x_train).all(axis=1) & np.isfinite(y_train)
        if valid.sum() < 200:
            continue
        x_fit = torch.as_tensor(x_train[valid], dtype=torch.float64, device=used_device)
        y_fit = torch.as_tensor(y_train[valid], dtype=torch.float64, device=used_device)
        mean, scale = x_fit.mean(dim=0), x_fit.std(dim=0, unbiased=False).clamp(min=1e-12)
        standardized = (x_fit - mean) / scale
        design = torch.cat([
            torch.ones((len(standardized), 1), dtype=torch.float64, device=used_device),
            standardized,
        ], dim=1)
        penalty = torch.eye(design.shape[1], dtype=torch.float64, device=used_device) * alpha
        penalty[0, 0] = 0.0
        coefficients = torch.linalg.solve(design.T @ design + penalty, design.T @ y_fit)
        stop = min(sessions, start + refit_every)
        x_test = features[start:stop].reshape(-1, features.shape[2])
        good = np.isfinite(x_test).all(axis=1)
        predicted = np.full(len(x_test), np.nan)
        if good.any():
            x_good = torch.as_tensor(x_test[good], dtype=torch.float64, device=used_device)
            x_good = (x_good - mean) / scale
            design_good = torch.cat([
                torch.ones((len(x_good), 1), dtype=torch.float64, device=used_device), x_good,
            ], dim=1)
            predicted[good] = (design_good @ coefficients).cpu().numpy()
        predictions[start:stop] = predicted.reshape(stop - start, -1)
    return predictions


def pooled_elastic_net_predictions(
    dataset: DecisionDataset, seasonality_days: int, *, train_sessions: int = 120,
    refit_every: int = 20, alpha: float = 1e-5, l1_ratio: float = 0.5,
    iterations: int = 250, device: str = "cuda",
) -> np.ndarray:
    """Walk-forward pooled elastic net fitted by proximal gradient on CUDA."""
    import torch

    if alpha < 0 or not 0 <= l1_ratio <= 1 or iterations <= 0:
        raise ValueError("invalid elastic-net settings")
    used_device = resolve_device(device)
    seasonal = dataset.seasonality[seasonality_days]
    sector = dataset.sector_residual if dataset.sector_residual is not None else dataset.first_residual
    features = np.stack([
        dataset.first_residual, dataset.gap_residual,
        np.log1p(np.maximum(dataset.relative_volume, 0)),
        dataset.realized_volatility, dataset.bar_pressure, seasonal, sector,
    ], axis=2)
    predictions = np.full(dataset.forward_return.shape, np.nan)
    for start in range(max(60, seasonality_days), len(dataset.session_dates), refit_every):
        begin = max(0, start - train_sessions)
        x_train = features[begin:start].reshape(-1, features.shape[2])
        y_train = dataset.forward_return[begin:start].reshape(-1)
        valid = np.isfinite(x_train).all(axis=1) & np.isfinite(y_train)
        if valid.sum() < 200:
            continue
        x = torch.as_tensor(x_train[valid], dtype=torch.float64, device=used_device)
        y = torch.as_tensor(y_train[valid], dtype=torch.float64, device=used_device)
        mean, scale = x.mean(0), x.std(0, unbiased=False).clamp(min=1e-12)
        x = (x - mean) / scale
        design = torch.cat([torch.ones((len(x), 1), dtype=torch.float64, device=used_device), x], 1)
        weights = torch.zeros(design.shape[1], dtype=torch.float64, device=used_device)
        lipschitz = 2 * torch.linalg.matrix_norm(design, ord=2) ** 2 / len(design) + alpha * (1 - l1_ratio)
        step = 1.0 / lipschitz.clamp(min=1e-12)
        for _ in range(iterations):
            gradient = 2 * design.T @ (design @ weights - y) / len(design)
            gradient[1:] += alpha * (1 - l1_ratio) * weights[1:]
            updated = weights - step * gradient
            updated[1:] = torch.sign(updated[1:]) * torch.clamp(
                torch.abs(updated[1:]) - step * alpha * l1_ratio, min=0,
            )
            weights = updated
        stop = min(len(dataset.session_dates), start + refit_every)
        x_test = features[start:stop].reshape(-1, features.shape[2])
        good = np.isfinite(x_test).all(axis=1)
        output = np.full(len(x_test), np.nan)
        if good.any():
            values = torch.as_tensor(x_test[good], dtype=torch.float64, device=used_device)
            values = (values - mean) / scale
            values = torch.cat([torch.ones((len(values), 1), dtype=torch.float64, device=used_device), values], 1)
            output[good] = (values @ weights).cpu().numpy()
        predictions[start:stop] = output.reshape(stop - start, -1)
    return predictions


def causal_momentum_forecast(
    signal: np.ndarray, target: np.ndarray, *, train_sessions: int = 120,
    refit_every: int = 20, device: str = "cuda",
) -> np.ndarray:
    """Calibrate a momentum signal into expected return using only prior sessions.

    The slope is constrained non-negative so a momentum family cannot silently
    mutate into a reversal strategy after seeing a negative historical slope.
    """
    import torch

    used_device = resolve_device(device)
    output = np.full(signal.shape, np.nan)
    for start in range(60, len(signal), refit_every):
        begin = max(0, start - train_sessions)
        x = signal[begin:start].reshape(-1)
        y = target[begin:start].reshape(-1)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 200:
            continue
        x_fit = torch.as_tensor(x[valid], dtype=torch.float64, device=used_device)
        y_fit = torch.as_tensor(y[valid], dtype=torch.float64, device=used_device)
        denominator = torch.dot(x_fit, x_fit).clamp(min=1e-18)
        slope = torch.clamp(torch.dot(x_fit, y_fit) / denominator, min=0.0)
        stop = min(len(signal), start + refit_every)
        values = torch.as_tensor(signal[start:stop], dtype=torch.float64, device=used_device)
        output[start:stop] = (values * slope).cpu().numpy()
    return output


def confidence_adjusted_forecast(
    prediction: np.ndarray, target: np.ndarray, z: float, *, lookback: int = 120,
) -> np.ndarray:
    """Haircut forecast magnitude by causal trailing model error."""
    if z < 0:
        raise ValueError("confidence_z must be non-negative")
    if z == 0:
        return prediction.copy()
    adjusted = np.full(prediction.shape, np.nan)
    for index in range(len(prediction)):
        residual = target[max(0, index - lookback):index] - prediction[max(0, index - lookback):index]
        finite = residual[np.isfinite(residual)]
        if len(finite) < 200:
            continue
        uncertainty = float(np.sqrt(np.mean(finite * finite)))
        magnitude = np.maximum(np.abs(prediction[index]) - z * uncertainty, 0.0)
        adjusted[index] = np.sign(prediction[index]) * magnitude
    return adjusted


def _neutral_weights(score: np.ndarray, beta: np.ndarray, count: int, gate: np.ndarray) -> np.ndarray | None:
    long = np.flatnonzero(score > gate)
    short = np.flatnonzero(score < -gate)
    if not len(long) or not len(short):
        return None
    long = long[np.argsort(score[long])[-count:]]
    short = short[np.argsort(score[short])[:count]]
    raw = np.zeros(len(score))
    raw[long], raw[short] = 0.5 / len(long), -0.5 / len(short)
    selected = raw != 0
    design = np.column_stack([np.ones(selected.sum()), beta[selected]])
    projected = raw[selected] - design @ np.linalg.pinv(design) @ raw[selected]
    if np.abs(projected).sum() <= 1e-12:
        return None
    raw[selected] = projected / np.abs(projected).sum()
    return raw


def evaluate_profitability_candidate(
    dataset: DecisionDataset, candidate: dict[str, Any], *, cost_per_side_bps: float,
    ridge_cache: dict[int, np.ndarray] | None = None,
    forecast_cache: dict[tuple[str, int], np.ndarray] | None = None,
    confidence_cache: dict[tuple[str, int, float], np.ndarray] | None = None,
    elastic_cache: dict[int, np.ndarray] | None = None,
    device: str = "cuda",
) -> tuple[dict[str, Any], np.ndarray]:
    used_device = resolve_device(device)
    family = candidate["family"]
    momentum_signal = (
        dataset.sector_residual if dataset.sector_residual is not None
        else dataset.first_residual
    )
    if family == "first_half_to_close":
        scores = (forecast_cache or {}).get(("first", 0))
        if scores is None:
            scores = causal_momentum_forecast(momentum_signal, dataset.forward_return, device=used_device)
    elif family == "seasonal_momentum":
        days = int(candidate["seasonality_days"])
        scores = (forecast_cache or {}).get(("seasonal", days))
        if scores is None:
            raw = momentum_signal + dataset.seasonality[days]
            scores = causal_momentum_forecast(raw, dataset.forward_return, device=used_device)
    elif family == "pooled_ridge":
        days = int(candidate["seasonality_days"])
        scores = (ridge_cache or {}).get(days)
        if scores is None:
            scores = pooled_ridge_predictions(dataset, days, device=used_device)
    elif family == "pooled_elastic_net":
        days = int(candidate["seasonality_days"])
        scores = (elastic_cache or {}).get(days)
        if scores is None:
            scores = pooled_elastic_net_predictions(dataset, days, device=used_device)
    elif family == "quote_flow_confirmed":
        if dataset.quote_imbalance is None or dataset.spread_bps is None:
            raise ValueError("quote-flow family requires at least 80% causal NBBO coverage")
        scores = (forecast_cache or {}).get(("first", 0))
        if scores is None:
            scores = causal_momentum_forecast(momentum_signal, dataset.forward_return, device=used_device)
        scores = scores.copy()
        scores[np.sign(scores) != np.sign(dataset.quote_imbalance)] = np.nan
    else:
        raise ValueError("unsupported profitability family")
    confidence_key = (
        "ridge" if family == "pooled_ridge" else "elastic" if family == "pooled_elastic_net" else "first" if family in {"first_half_to_close", "quote_flow_confirmed"} else "seasonal",
        int(candidate["seasonality_days"]), float(candidate.get("confidence_z", 0.0)),
    )
    adjusted = (confidence_cache or {}).get(confidence_key)
    if adjusted is not None:
        scores = adjusted
    else:
        scores = confidence_adjusted_forecast(
            scores, dataset.forward_return, float(candidate.get("confidence_z", 0.0)),
        )
    daily = np.zeros(len(dataset.session_dates))
    contribution = np.zeros(len(dataset.tickers))
    trades = events = 0
    beta_exposure = []
    for index in range(len(daily)):
        if candidate["regime"] == "high_information" and not dataset.high_information[index]:
            continue
        if candidate["regime"] == "avoid_panic_rebound" and dataset.panic_rebound[index]:
            continue
        score = scores[index]
        score = np.where(dataset.active_mask[index], score, np.nan)
        if candidate.get("event_policy", "all") == "avoid_events":
            score = np.where(dataset.event_clear[index], score, np.nan)
        if not np.isfinite(score).any():
            continue
        cost = (
            np.maximum(cost_per_side_bps, dataset.spread_bps[index] / 2)
            if dataset.spread_bps is not None else np.full(len(score), cost_per_side_bps)
        )
        gate_bps = np.maximum(
            float(candidate["min_edge_bps"]),
            float(candidate["edge_cost_multiple"]) * 2 * cost,
        )
        weights = _neutral_weights(
            score, dataset.beta[index], int(candidate["max_positions_per_side"]),
            gate_bps / 10_000,
        )
        if weights is None:
            continue
        selected = weights != 0
        item = np.zeros(len(weights))
        item[selected] = (
            weights[selected] * dataset.forward_return[index, selected]
            - np.abs(weights[selected]) * 2 * cost[selected] / 10_000
        )
        daily[index] = item.sum()
        contribution += item
        positions = int(np.count_nonzero(weights))
        trades += positions
        events += 1
        beta_exposure.append(abs(float(weights[selected] @ dataset.beta[index, selected])))
    curve = np.concatenate([[1.0], np.cumprod(1 + daily)])
    return ({
        **candidate, "sharpe": float(sharpe(pd.Series(daily))),
        "cagr": float(cagr(pd.Series(daily))),
        "max_drawdown": float(max_drawdown(pd.Series(curve))),
        "total_return": float(curve[-1] - 1), "n_trades": trades,
        "portfolio_events": events, "positive_sessions": int((daily > 0).sum()),
        "mean_abs_beta_exposure": float(np.mean(beta_exposure)) if beta_exposure else float("nan"),
        "simulation_device": f"hybrid-{used_device}",
    }, contribution)


def slice_decision_dataset(dataset: DecisionDataset, start: int, stop: int) -> DecisionDataset:
    return take_decision_dataset(dataset, np.arange(start, stop))


def take_decision_dataset(dataset: DecisionDataset, indices: np.ndarray) -> DecisionDataset:
    indices = np.asarray(indices, dtype=int)
    return DecisionDataset(
        decision_bar=dataset.decision_bar, session_dates=dataset.session_dates[indices],
        tickers=dataset.tickers,
        first_residual=dataset.first_residual[indices], gap_residual=dataset.gap_residual[indices],
        relative_volume=dataset.relative_volume[indices], realized_volatility=dataset.realized_volatility[indices],
        bar_pressure=dataset.bar_pressure[indices], beta=dataset.beta[indices],
        forward_return=dataset.forward_return[indices],
        seasonality={days: values[indices] for days, values in dataset.seasonality.items()},
        high_information=dataset.high_information[indices], panic_rebound=dataset.panic_rebound[indices],
        active_mask=dataset.active_mask[indices],
        event_clear=dataset.event_clear[indices],
        quote_imbalance=None if dataset.quote_imbalance is None else dataset.quote_imbalance[indices],
        spread_bps=None if dataset.spread_bps is None else dataset.spread_bps[indices],
        sector_residual=None if dataset.sector_residual is None else dataset.sector_residual[indices],
        vwap_deviation=None if dataset.vwap_deviation is None else dataset.vwap_deviation[indices],
        prior_range_position=None if dataset.prior_range_position is None else dataset.prior_range_position[indices],
        opening_range_break=None if dataset.opening_range_break is None else dataset.opening_range_break[indices],
        support_rejection=None if dataset.support_rejection is None else dataset.support_rejection[indices],
        bar_minutes=dataset.bar_minutes,
    )


def align_decision_datasets(datasets: dict[int, DecisionDataset]) -> dict[int, DecisionDataset]:
    """Restrict all entry horizons to the same sessions for fair selection."""
    if not datasets:
        raise ValueError("at least one decision dataset is required")
    common = set.intersection(*(set(item.session_dates) for item in datasets.values()))
    ordered = sorted(common)
    if len(ordered) < 60:
        raise ValueError("fewer than 60 common sessions across decision horizons")
    result = {}
    for decision, dataset in datasets.items():
        lookup = {date: index for index, date in enumerate(dataset.session_dates)}
        result[decision] = take_decision_dataset(
            dataset, np.asarray([lookup[date] for date in ordered], dtype=int),
        )
    return result


def describe_profitability(row: dict[str, Any] | pd.Series) -> str:
    family = {
        "first_half_to_close": "first-half-hour residual momentum",
        "seasonal_momentum": "residual momentum plus same-horizon seasonality",
        "pooled_ridge": "pooled walk-forward ridge forecast",
        "pooled_elastic_net": "pooled walk-forward elastic-net forecast",
        "quote_flow_confirmed": "residual momentum confirmed by SIP quote imbalance",
    }[str(row["family"])]
    decision_minutes = (int(row["decision_bar"]) + 1) * 5
    return (
        f"At {decision_minutes} minutes after the open, rank {family}; hold a cash- and "
        f"beta-neutral long/short portfolio through the close, using at most "
        f"{int(row['max_positions_per_side'])} names per side and trade only when "
        f"forecast edge clears {float(row['min_edge_bps']):g} bps and "
        f"{float(row['edge_cost_multiple']):g}x estimated round-trip cost "
        f"after a {float(row.get('confidence_z', 0)):g}σ forecast-error haircut "
        f"in the {row['regime']} regime with {row.get('event_policy', 'all')} event handling."
    )
