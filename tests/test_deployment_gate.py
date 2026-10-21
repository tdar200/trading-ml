import json

import pandas as pd
import numpy as np
import pytest

from analysis.deployment_gate import (
    candidate_fingerprint, file_sha256, paper_qualification, paper_row_fingerprint,
    verify_deployment_approval,
)
from livesim.profitability_guard import (
    ProfitabilityRiskSnapshot, assess_profitability_order,
    require_approved_profitability_order,
)


def _paper_frame(periods=10, fingerprint="abc", evidence_digest="a" * 64):
    rows = []
    previous = "GENESIS"
    for index, date in enumerate(pd.date_range("2026-01-01", periods=periods)):
        row = {
            "session_date": date, "candidate_sha256": fingerprint,
            "predicted_gross_bps": 20.0, "realized_gross_bps": 15.0,
            "modeled_cost_bps": 2.0, "realized_cost_bps": 3.0,
            "net_return": 0.001 if index % 2 else 0.0005,
            "orders_submitted": 1, "orders_filled": 1,
            "broker_reconciled": True, "broker_evidence_path": "paper_evidence/test.json",
            "broker_evidence_sha256": evidence_digest,
            "risk_breach": False, "previous_row_sha256": previous,
        }
        row["row_sha256"] = paper_row_fingerprint(row)
        previous = row["row_sha256"]
        rows.append(row)
    return pd.DataFrame(rows)


def test_candidate_fingerprint_is_order_independent():
    assert candidate_fingerprint({"a": 1, "b": 2}) == candidate_fingerprint({"b": 2, "a": 1})
    assert candidate_fingerprint({"a": np.int64(1)}) == candidate_fingerprint({"a": 1})


def test_short_paper_observation_fails_closed():
    result = paper_qualification(_paper_frame())
    assert not result["qualified"]
    assert not result["gates"]["paper_sessions"]


def test_paper_gate_counts_fills_and_requires_broker_reconciliation(tmp_path):
    evidence = tmp_path / "paper_evidence" / "test.json"
    evidence.parent.mkdir()
    evidence.write_text("broker export")
    frame = _paper_frame(126, evidence_digest=file_sha256(evidence))
    result = paper_qualification(frame, expected_fingerprint="abc", evidence_root=tmp_path)
    assert result["qualified"]
    assert result["events"] == 126
    frame.loc[125, "broker_reconciled"] = False
    frame.loc[125, "row_sha256"] = paper_row_fingerprint(frame.loc[125].to_dict())
    result = paper_qualification(frame, expected_fingerprint="abc", evidence_root=tmp_path)
    assert not result["gates"]["broker_evidence"]


def test_risk_guard_allows_safe_order_and_rejects_stale_data():
    safe = ProfitabilityRiskSnapshot(
        proposed_weights={"A": 0.04, "B": -0.04}, sectors={"A": "x", "B": "y"},
        daily_pnl_fraction=0, data_age_seconds=1, quote_coverage=1,
        maximum_spread_bps=2, reconciliation_ok=True, duplicate_order_ids=False,
        model_in_distribution=True,
    )
    assert assess_profitability_order(safe) == (True, ())
    stale = ProfitabilityRiskSnapshot(**{**safe.__dict__, "data_age_seconds": 30})
    allowed, reasons = assess_profitability_order(stale)
    assert not allowed and "stale market data" in reasons


def test_runtime_approval_is_bound_to_artifacts(tmp_path):
    candidate = {"family": "test", "decision_bar": 5}
    fingerprint = candidate_fingerprint(candidate)
    (tmp_path / "frozen_candidate.json").write_text(json.dumps({
        "candidate": candidate, "sha256": fingerprint,
    }))
    (tmp_path / "evaluation_cost_stress.csv").write_text("cost,total_return\n1,0.1\n")
    evidence = tmp_path / "paper_evidence" / "test.json"
    evidence.parent.mkdir()
    evidence.write_text("broker export")
    _paper_frame(126, fingerprint, file_sha256(evidence)).to_csv(
        tmp_path / "paper_observations.csv", index=False,
    )
    artifacts = {
        name: file_sha256(tmp_path / name) for name in (
            "frozen_candidate.json", "evaluation_cost_stress.csv", "paper_observations.csv",
        )
    }
    (tmp_path / "deployment_qualification.json").write_text(json.dumps({
        "deployment_approved": True, "candidate_sha256": fingerprint,
        "historical": {"qualified": True}, "paper": {"qualified": True},
        "artifact_sha256": artifacts,
    }))
    safe = ProfitabilityRiskSnapshot(
        proposed_weights={"A": 0.04}, sectors={"A": "x"}, daily_pnl_fraction=0,
        data_age_seconds=1, quote_coverage=1, maximum_spread_bps=2,
        reconciliation_ok=True, duplicate_order_ids=False, model_in_distribution=True,
    )
    assert require_approved_profitability_order(tmp_path, safe)["deployment_approved"]
    (tmp_path / "evaluation_cost_stress.csv").write_text("changed")
    with pytest.raises(RuntimeError, match="changed or is missing"):
        verify_deployment_approval(tmp_path)
