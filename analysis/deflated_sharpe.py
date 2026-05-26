"""
Bailey & López de Prado (2014) Deflated Sharpe Ratio.

Given the Sharpe ratios from a set of `n` trial strategies / parameter combos,
return the deflated (multiple-testing-penalized) Sharpe and the implied p-value
that the winner's Sharpe is real (> 0) under the null hypothesis that all true
Sharpes are zero.

Reference: "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
Overfitting and Non-Normality" (Bailey & López de Prado, 2014).

This module uses the no-skew/no-kurt simplification of the σ_SR_obs variance
term (σ² ≈ (1 + 0.5·SR²) / (N − 1)), which the paper notes is acceptable when
return distributions are roughly normal — sufficient for daily ETF returns.
Higher-moment adjustments are deferred to Stage 5+ where ML predictions may
have fatter tails.

Pure module: no IO, no Nautilus.
"""
from __future__ import annotations

import math
from statistics import pstdev

from scipy.stats import norm

# Euler-Mascheroni constant.
_GAMMA = 0.5772156649015329


def _expected_max_sharpe(n_trials: int, sigma_sr: float) -> float:
    """E[max SR over N iid trials] under H0: all true SRs are zero."""
    if n_trials <= 1 or sigma_sr == 0.0:
        return 0.0
    # Approximation from Bailey-LdP.
    return sigma_sr * (
        (1 - _GAMMA) * norm.ppf(1 - 1.0 / n_trials)
        + _GAMMA * norm.ppf(1 - 1.0 / (n_trials * math.e))
    )


def _sigma_sr_obs(sr_obs: float, n_obs: int) -> float:
    """Standard error of the observed Sharpe (no-skew/no-kurt simplification)."""
    denom = max(n_obs - 1, 1)
    return math.sqrt((1.0 + 0.5 * sr_obs * sr_obs) / denom)


def deflated_sharpe(sharpes: list[float], n_obs: int) -> dict:
    """Compute the Deflated Sharpe Ratio.

    Args:
        sharpes: All in-sample Sharpe ratios from the trial set (e.g. all 18
            combos of a parameter grid). The "winner" is max(sharpes).
        n_obs: Number of return observations the winner's Sharpe was computed
            over (e.g. number of trading days in the full backtest).

    Returns:
        dict with:
            observed_sharpe:     max(sharpes)
            n_trials:            len(sharpes)
            expected_max_sharpe: E[max SR over n_trials] under H0
            deflated_sharpe:     z-score after subtracting selection-bias mean
                                 and dividing by σ_SR_obs
            p_value:             Φ(deflated_sharpe) — probability the winner's
                                 Sharpe is real (> 0)
    """
    if not sharpes:
        raise ValueError("sharpes must be non-empty")
    if n_obs <= 0:
        raise ValueError(f"n_obs must be positive (got {n_obs})")

    n = len(sharpes)
    sr_obs = max(sharpes)
    sigma_sr_trials = pstdev(sharpes) if n > 1 else 0.0
    expected_max = _expected_max_sharpe(n, sigma_sr_trials)
    sigma_sr_obs = _sigma_sr_obs(sr_obs, n_obs)
    deflated_z = (sr_obs - expected_max) / sigma_sr_obs if sigma_sr_obs > 0 else 0.0
    p_value = float(norm.cdf(deflated_z))

    return {
        "observed_sharpe": float(sr_obs),
        "n_trials": n,
        "expected_max_sharpe": float(expected_max),
        "deflated_sharpe": float(deflated_z),
        "p_value": p_value,
    }
