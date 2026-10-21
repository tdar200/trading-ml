from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from analysis.profitability import (
    DecisionDataset,
    ProfitabilityGrammar,
    _neutral_weights,
    causal_momentum_forecast,
    confidence_adjusted_forecast,
    evaluate_profitability_candidate,
    generate_profitability_candidates,
    pooled_ridge_predictions,
    pooled_elastic_net_predictions,
)


def _dataset(sessions=100, assets=6, seed=7):
    rng = np.random.default_rng(seed)
    first = rng.normal(0, 0.006, (sessions, assets))
    target = 0.12 * first + rng.normal(0, 0.002, (sessions, assets))
    seasonality = {
        days: pd.DataFrame(target).shift(1).rolling(days, min_periods=10).mean().to_numpy()
        for days in (20, 40)
    }
    return DecisionDataset(
        decision_bar=5,
        session_dates=np.asarray(pd.date_range("2025-01-01", periods=sessions, freq="B").date),
        tickers=tuple(f"S{i}" for i in range(assets)), first_residual=first,
        gap_residual=rng.normal(0, 0.003, (sessions, assets)),
        relative_volume=rng.lognormal(0, 0.2, (sessions, assets)),
        realized_volatility=rng.uniform(0.001, 0.01, (sessions, assets)),
        bar_pressure=rng.normal(0, 0.002, (sessions, assets)),
        beta=rng.uniform(0.6, 1.4, (sessions, assets)), forward_return=target,
        seasonality=seasonality, high_information=np.ones(sessions, dtype=bool),
        panic_rebound=np.zeros(sessions, dtype=bool), active_mask=np.ones((sessions, assets), dtype=bool),
        event_clear=np.ones((sessions, assets), dtype=bool),
    )


def test_candidate_grammar_excludes_quote_family_without_nbbo():
    grammar = ProfitabilityGrammar(
        decision_bars=(5,), max_positions_per_side=(2,), min_edge_bps=(0,),
        edge_cost_multiple=(2,), regimes=("all",), seasonality_days=(20,),
        confidence_z=(0.0,),
    )
    without = generate_profitability_candidates(quote_ready=False, grammar=grammar)
    with_quotes = generate_profitability_candidates(quote_ready=True, grammar=grammar)
    assert {row["family"] for row in without} == {
        "first_half_to_close", "seasonal_momentum", "pooled_ridge", "pooled_elastic_net",
    }
    assert {row["family"] for row in with_quotes} == {
        "first_half_to_close", "seasonal_momentum", "pooled_ridge", "pooled_elastic_net", "quote_flow_confirmed",
    }


def test_neutral_projection_has_zero_cash_and_beta_exposure():
    score = np.asarray([0.04, 0.03, -0.03, -0.04, 0.01, -0.01])
    beta = np.asarray([0.8, 1.2, 0.7, 1.4, 1.0, 1.0])
    weights = _neutral_weights(score, beta, 2, np.zeros(6))
    assert weights is not None
    assert weights.sum() == pytest.approx(0, abs=1e-12)
    assert weights @ beta == pytest.approx(0, abs=1e-12)
    assert np.abs(weights).sum() == pytest.approx(1)


def test_execution_gate_can_produce_no_trade_sessions():
    dataset = _dataset()
    candidate = {
        "family": "first_half_to_close", "decision_bar": 5,
        "max_positions_per_side": 2, "min_edge_bps": 10_000.0,
        "edge_cost_multiple": 3.0, "regime": "all", "seasonality_days": 0,
        "confidence_z": 0.0,
        "event_policy": "all",
    }
    result, contribution = evaluate_profitability_candidate(
        dataset, candidate, cost_per_side_bps=5, device="cpu",
    )
    assert result["n_trades"] == 0
    assert result["portfolio_events"] == 0
    assert contribution.sum() == 0


def test_unavailable_nan_asset_does_not_contaminate_portfolio_return():
    dataset = _dataset()
    target = dataset.forward_return.copy()
    target[:, -1] = np.nan
    active = dataset.active_mask.copy()
    active[:, -1] = False
    dataset = replace(dataset, forward_return=target, active_mask=active)
    candidate = {
        "family": "first_half_to_close", "decision_bar": 5,
        "max_positions_per_side": 2, "min_edge_bps": 0.0,
        "edge_cost_multiple": 2.0, "confidence_z": 0.0,
        "regime": "all", "seasonality_days": 0, "event_policy": "all",
    }
    result, _ = evaluate_profitability_candidate(dataset, candidate, cost_per_side_bps=1, device="cpu")
    assert np.isfinite(result["total_return"])
    assert np.isfinite(result["mean_abs_beta_exposure"])


def test_pooled_ridge_is_causal_to_future_targets():
    dataset = _dataset(sessions=105)
    baseline = pooled_ridge_predictions(dataset, 20, refit_every=10, device="cpu")
    changed_target = dataset.forward_return.copy()
    changed_target[90:] += 100
    changed = replace(dataset, forward_return=changed_target)
    changed_prediction = pooled_ridge_predictions(changed, 20, refit_every=10, device="cpu")
    np.testing.assert_allclose(baseline[:90], changed_prediction[:90], equal_nan=True)


def test_momentum_calibration_is_causal_and_never_flips_direction():
    dataset = _dataset(sessions=105)
    baseline = causal_momentum_forecast(
        dataset.first_residual, dataset.forward_return, refit_every=10, device="cpu",
    )
    changed_target = dataset.forward_return.copy()
    changed_target[90:] += 100
    changed = causal_momentum_forecast(
        dataset.first_residual, changed_target, refit_every=10, device="cpu",
    )
    np.testing.assert_allclose(baseline[:90], changed[:90], equal_nan=True)
    valid = np.isfinite(baseline) & (dataset.first_residual != 0)
    assert np.all(np.sign(baseline[valid]) == np.sign(dataset.first_residual[valid]))


def test_confidence_haircut_is_causal_and_reduces_forecast_magnitude():
    dataset = _dataset(sessions=105)
    raw = causal_momentum_forecast(dataset.first_residual, dataset.forward_return, device="cpu")
    adjusted = confidence_adjusted_forecast(raw, dataset.forward_return, 0.5)
    assert np.nanmax(np.abs(adjusted)) <= np.nanmax(np.abs(raw))
    changed = dataset.forward_return.copy()
    changed[90:] += 100
    changed_adjusted = confidence_adjusted_forecast(raw, changed, 0.5)
    np.testing.assert_allclose(adjusted[:90], changed_adjusted[:90], equal_nan=True)


def test_pooled_ridge_cpu_cuda_parity_when_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dataset = _dataset()
    cpu = pooled_ridge_predictions(dataset, 20, device="cpu")
    gpu = pooled_ridge_predictions(dataset, 20, device="cuda")
    np.testing.assert_allclose(cpu, gpu, rtol=1e-9, atol=1e-11, equal_nan=True)


def test_elastic_net_is_causal_to_future_targets():
    dataset = _dataset(sessions=105)
    baseline = pooled_elastic_net_predictions(dataset, 20, refit_every=10, iterations=30, device="cpu")
    target = dataset.forward_return.copy()
    target[90:] += 100
    changed = pooled_elastic_net_predictions(
        replace(dataset, forward_return=target), 20, refit_every=10, iterations=30, device="cpu",
    )
    np.testing.assert_allclose(baseline[:90], changed[:90], equal_nan=True)
