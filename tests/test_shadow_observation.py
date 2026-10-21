import pandas as pd
import pytest

from data.shadow_observation import append_shadow_observation, validate_shadow_log


def _row(timestamp="2026-09-14T14:30:00Z"):
    return {
        "observed_at": timestamp, "candidate_sha256": "a" * 64,
        "ticker": "AAPL", "side": "long", "quantity": 1,
        "predicted_edge_bps": 12, "intended_price": 100,
        "observed_bid": 99.99, "observed_ask": 100.01,
        "reference_exit_price": 100.20, "data_source": "manual_delayed_research",
    }


def test_shadow_journal_is_hash_chained_and_research_only(tmp_path):
    path = tmp_path / "shadow.csv"
    frame = append_shadow_observation(path, _row())
    assert len(frame) == 1
    assert frame.iloc[0].previous_row_sha256 == "GENESIS"
    stored = pd.read_csv(path)
    stored.loc[0, "observed_ask"] = 200
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_shadow_log(stored)


def test_shadow_journal_rejects_crossed_quotes(tmp_path):
    row = _row()
    row["observed_ask"] = 90
    with pytest.raises(ValueError, match="ask cannot"):
        append_shadow_observation(tmp_path / "shadow.csv", row)
