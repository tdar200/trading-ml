from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from analysis.profitability import DecisionDataset
from analysis.profitability_v2 import (
    ProfitabilityV2Grammar, build_v2_forecasts, cost_aware_asymmetric_predictions,
    evaluate_v2_candidate,
    generate_v2_candidates, online_expert_predictions,
    pooled_structure_ridge_predictions,
)
from analysis.sleeves import require_equity_replay_sleeve


def _dataset(sessions=105, assets=8, seed=11):
    rng = np.random.default_rng(seed)
    first = rng.normal(0, 0.008, (sessions, assets))
    vwap = first * 0.7 + rng.normal(0, 0.001, first.shape)
    opening = np.sign(first)
    prior = np.sign(first) * rng.uniform(0.5, 1.5, first.shape)
    rejection = np.sign(first)
    target = 0.25 * first + rng.normal(0, 0.0015, first.shape)
    return DecisionDataset(
        decision_bar=23,
        session_dates=np.asarray(pd.bdate_range("2025-01-01", periods=sessions).date),
        tickers=tuple(f"S{i}" for i in range(assets)), first_residual=first,
        gap_residual=rng.normal(0, 0.002, first.shape),
        relative_volume=rng.uniform(1.1, 1.8, first.shape),
        realized_volatility=rng.uniform(0.001, 0.005, first.shape),
        bar_pressure=rng.normal(0, 0.001, first.shape),
        beta=rng.uniform(0.7, 1.3, first.shape), forward_return=target,
        seasonality={20: np.zeros_like(first), 40: np.zeros_like(first)},
        high_information=np.ones(sessions, dtype=bool),
        panic_rebound=np.zeros(sessions, dtype=bool),
        active_mask=np.ones_like(first, dtype=bool), event_clear=np.ones_like(first, dtype=bool),
        sector_residual=first, vwap_deviation=vwap,
        prior_range_position=prior, opening_range_break=opening,
        support_rejection=rejection,
    )


def _candidate():
    return generate_v2_candidates(
        quote_ready=False, event_ready=False,
        grammar=ProfitabilityV2Grammar(
            decision_bars=(23, 47), max_positions_per_side=(2, 3),
            minimum_relative_volume=(1.0, 1.25),
        ),
    )[0]


def test_v2_grammar_is_small_and_quote_family_requires_nbbo():
    without = generate_v2_candidates(quote_ready=False, event_ready=False)
    with_quotes = generate_v2_candidates(quote_ready=True, event_ready=True)
    assert len(without) == 48
    assert len(with_quotes) == 48
    assert "quote_structure_confirmed" not in {row["family"] for row in without}
    assert all(row["event_policy"] == "avoid_events" for row in with_quotes)


def test_v2_pooled_forecast_is_causal():
    dataset = _dataset()
    baseline = pooled_structure_ridge_predictions(dataset, refit_every=10, device="cpu")
    changed_target = dataset.forward_return.copy()
    changed_target[90:] += 100
    changed = pooled_structure_ridge_predictions(
        replace(dataset, forward_return=changed_target), refit_every=10, device="cpu",
    )
    np.testing.assert_allclose(baseline[:90], changed[:90], equal_nan=True)


def test_online_expert_selection_is_causal_and_can_abstain():
    dataset = _dataset(sessions=105)
    rng = np.random.default_rng(19)
    experts = {
        "good": dataset.forward_return + rng.normal(0, 0.0001, dataset.forward_return.shape),
        "bad": -dataset.forward_return,
    }
    baseline = online_expert_predictions(dataset, experts)
    changed_target = dataset.forward_return.copy()
    changed_target[90:] *= -100
    changed = online_expert_predictions(replace(dataset, forward_return=changed_target), experts)
    np.testing.assert_allclose(baseline[:91], changed[:91], equal_nan=True)
    assert np.isfinite(baseline[40:]).any()


def test_online_router_validates_inputs():
    dataset = _dataset()
    with pytest.raises(ValueError, match="mode"):
        online_expert_predictions(dataset, {"one": dataset.forward_return}, mode="future")


def test_cost_aware_asymmetric_forecast_is_causal():
    dataset = _dataset(sessions=125)
    baseline = cost_aware_asymmetric_predictions(
        dataset, confidence_z=0, refit_every=10, device="cpu",
    )
    changed_target = dataset.forward_return.copy()
    changed_target[105:] += 100
    changed = cost_aware_asymmetric_predictions(
        replace(dataset, forward_return=changed_target),
        confidence_z=0, refit_every=10, device="cpu",
    )
    np.testing.assert_allclose(baseline[:106], changed[:106], equal_nan=True)


def test_v2_conservative_cost_replay_reduces_return():
    dataset = _dataset()
    forecasts = build_v2_forecasts(dataset, device="cpu")
    candidate = _candidate()
    low, _ = evaluate_v2_candidate(
        dataset, candidate, cost_per_side_bps=1, forecast_cache=forecasts, device="cpu",
    )
    high, _ = evaluate_v2_candidate(
        dataset, candidate, cost_per_side_bps=5, forecast_cache=forecasts, device="cpu",
    )
    assert low["portfolio_events"] > 0
    assert high["total_return"] <= low["total_return"]


def test_v2_replay_streams_every_session_and_final_equity():
    dataset = _dataset(sessions=70)
    forecasts = build_v2_forecasts(dataset, device="cpu")
    events = []
    result, _ = evaluate_v2_candidate(
        dataset, _candidate(), cost_per_side_bps=2.5,
        forecast_cache=forecasts, device="cpu", stream_callback=events.append,
    )
    assert len(events) == len(dataset.session_dates)
    assert events[-1]["session_index"] == len(dataset.session_dates) - 1
    assert events[-1]["equity"] == pytest.approx(1 + result["total_return"])
    assert events[-1]["cumulative_trades"] == result["n_trades"]


def test_crypto_cannot_use_equity_session_replay():
    with pytest.raises(ValueError, match="continuous-session"):
        require_equity_replay_sleeve("crypto")
    assert require_equity_replay_sleeve("individual_stock").session_model == "us_equity_regular"
