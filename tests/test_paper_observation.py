import pytest

from data.paper_observation import append_paper_session


def _row(date="2026-01-02", fingerprint="abc"):
    return {
        "session_date": date, "candidate_sha256": fingerprint,
        "predicted_gross_bps": 20, "realized_gross_bps": 15,
        "modeled_cost_bps": 2, "realized_cost_bps": 3,
        "net_return": 0.001, "orders_submitted": 2, "orders_filled": 2,
        "broker_reconciled": True, "broker_evidence_path": "paper_evidence/test.json",
        "broker_evidence_sha256": "a" * 64,
        "risk_breach": False,
    }


def test_paper_journal_is_append_only_and_fingerprint_bound(tmp_path):
    path = tmp_path / "paper.csv"
    assert len(append_paper_session(path, _row())) == 1
    assert len(append_paper_session(path, _row("2026-01-03"))) == 2
    with pytest.raises(ValueError, match="fingerprint changed"):
        append_paper_session(path, _row("2026-01-04", "different"))
    with pytest.raises(ValueError, match="unique"):
        append_paper_session(path, _row("2026-01-03"))


def test_paper_journal_detects_edit(tmp_path):
    path = tmp_path / "paper.csv"
    append_paper_session(path, _row())
    frame = __import__("pandas").read_csv(path)
    frame.loc[0, "net_return"] = 0.9
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="row hash mismatch"):
        append_paper_session(path, _row("2026-01-03"))
