"""Pure-math tests for the Bailey-LdP Deflated Sharpe Ratio."""
from __future__ import annotations

import math

import pytest

from analysis.deflated_sharpe import deflated_sharpe


def test_empty_sharpes_raises():
    with pytest.raises(ValueError, match="non-empty"):
        deflated_sharpe([], n_obs=252)


def test_n_obs_must_be_positive():
    with pytest.raises(ValueError, match="n_obs must be"):
        deflated_sharpe([1.0], n_obs=0)


def test_single_trial_returns_all_keys():
    out = deflated_sharpe([1.0], n_obs=252)
    for key in ("observed_sharpe", "n_trials", "expected_max_sharpe",
                "deflated_sharpe", "p_value"):
        assert key in out


def test_single_trial_no_penalty():
    """With only 1 trial there is no selection bias — expected_max_sharpe = 0."""
    out = deflated_sharpe([1.0], n_obs=252)
    assert out["observed_sharpe"] == pytest.approx(1.0)
    assert out["n_trials"] == 1
    assert out["expected_max_sharpe"] == pytest.approx(0.0)
    # With SR=1, n_obs=252: sigma_SR ≈ sqrt(1.5/251) ≈ 0.0773; z ≈ 12.94 → p ≈ 1.0
    assert out["deflated_sharpe"] > 10
    assert out["p_value"] == pytest.approx(1.0, abs=1e-6)


def test_winner_in_many_trials_with_mean_zero_depresses():
    """If 100 trial Sharpes are drawn ~N(0, 0.5), the max is much smaller after deflation."""
    import random
    random.seed(0)
    sharpes = [random.gauss(0.0, 0.5) for _ in range(100)]
    out = deflated_sharpe(sharpes, n_obs=252)
    # The observed (max) Sharpe is positive (selection bias), but DSR should be
    # near zero or negative — the multiple-testing penalty dominates.
    assert out["observed_sharpe"] > 0.5      # selection-biased high value
    assert out["deflated_sharpe"] < out["observed_sharpe"]
    assert out["p_value"] < 0.5              # cannot reject "real Sharpe is zero"


def test_all_equal_sharpes_no_variance():
    """Identical Sharpes → Var=0 → expected_max=0 → no penalty (no information about luck)."""
    out = deflated_sharpe([0.5, 0.5, 0.5, 0.5], n_obs=252)
    assert out["observed_sharpe"] == pytest.approx(0.5)
    assert out["expected_max_sharpe"] == pytest.approx(0.0)


def test_p_value_always_in_unit_interval():
    """p_value must be in [0, 1] regardless of inputs."""
    for sharpes in ([0.0], [-1.0], [5.0], [-2.0, 0.0, 2.0, 4.0]):
        for n_obs in (10, 100, 1000):
            out = deflated_sharpe(sharpes, n_obs=n_obs)
            assert 0.0 <= out["p_value"] <= 1.0


def test_higher_n_obs_tightens_p_value():
    """For a fixed positive SR, more observations → larger z-score → p closer to 1."""
    sharpes = [0.5, 0.3, 0.1]
    low = deflated_sharpe(sharpes, n_obs=50)
    high = deflated_sharpe(sharpes, n_obs=5000)
    assert high["p_value"] >= low["p_value"]
