"""Pure fail-closed pre-order risk guard for a future profitability strategy."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from analysis.deployment_gate import verify_deployment_approval


@dataclass(frozen=True)
class ProfitabilityRiskLimits:
    max_gross_exposure: float = 0.25
    max_single_name_exposure: float = 0.05
    max_sector_exposure: float = 0.10
    max_daily_loss: float = 0.01
    max_data_age_seconds: float = 10.0
    minimum_quote_coverage: float = 0.80
    maximum_spread_bps: float = 10.0


@dataclass(frozen=True)
class ProfitabilityRiskSnapshot:
    proposed_weights: dict[str, float]
    sectors: dict[str, str]
    daily_pnl_fraction: float
    data_age_seconds: float
    quote_coverage: float
    maximum_spread_bps: float
    reconciliation_ok: bool
    duplicate_order_ids: bool
    model_in_distribution: bool
    kill_switch: bool = False


def assess_profitability_order(
    snapshot: ProfitabilityRiskSnapshot,
    limits: ProfitabilityRiskLimits = ProfitabilityRiskLimits(),
) -> tuple[bool, tuple[str, ...]]:
    weights = snapshot.proposed_weights
    gross = sum(abs(value) for value in weights.values())
    sector_net: dict[str, float] = {}
    for ticker, value in weights.items():
        sector = snapshot.sectors.get(ticker)
        if sector is None:
            return False, (f"missing sector for {ticker}",)
        sector_net[sector] = sector_net.get(sector, 0.0) + value
    reasons = []
    if snapshot.kill_switch: reasons.append("kill switch active")
    if gross > limits.max_gross_exposure: reasons.append("gross exposure limit")
    if weights and max(abs(value) for value in weights.values()) > limits.max_single_name_exposure:
        reasons.append("single-name exposure limit")
    if sector_net and max(abs(value) for value in sector_net.values()) > limits.max_sector_exposure:
        reasons.append("sector exposure limit")
    if snapshot.daily_pnl_fraction <= -limits.max_daily_loss: reasons.append("daily loss limit")
    if snapshot.data_age_seconds > limits.max_data_age_seconds: reasons.append("stale market data")
    if snapshot.quote_coverage < limits.minimum_quote_coverage: reasons.append("quote coverage")
    if snapshot.maximum_spread_bps > limits.maximum_spread_bps: reasons.append("spread limit")
    if not snapshot.reconciliation_ok: reasons.append("broker reconciliation")
    if snapshot.duplicate_order_ids: reasons.append("duplicate order")
    if not snapshot.model_in_distribution: reasons.append("model out of distribution")
    return not reasons, tuple(reasons)


def require_approved_profitability_order(
    run_dir: Path,
    snapshot: ProfitabilityRiskSnapshot,
    limits: ProfitabilityRiskLimits = ProfitabilityRiskLimits(),
) -> dict:
    """Validate immutable approval evidence and the current pre-order risk state.

    A caller must invoke this immediately before every real-money order batch.
    The function raises instead of returning a permissive fallback.
    """
    report = verify_deployment_approval(run_dir)
    allowed, reasons = assess_profitability_order(snapshot, limits)
    if not allowed:
        raise RuntimeError("profitability order rejected: " + ", ".join(reasons))
    return report
