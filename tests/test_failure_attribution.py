import pandas as pd
import pytest

from analysis.failure_attribution import summarize_ledger


def test_failure_attribution_reconciles_gross_cost_and_net():
    ledger = pd.DataFrame({
        "session_date": ["2026-01-02", "2026-01-02"],
        "ticker": ["A", "B"], "side": ["long", "short"],
        "gross_contribution": [0.01, -0.002], "cost_contribution": [0.001, 0.001],
        "net_contribution": [0.009, -0.003], "score": [0.02, -0.01],
        "high_information": [True, True], "panic_rebound": [False, False],
    })
    result = summarize_ledger(ledger, ["month"])
    assert result.loc[0, "gross_return"] == pytest.approx(0.008)
    assert result.loc[0, "cost_drag"] == pytest.approx(0.002)
    assert result.loc[0, "net_return"] == pytest.approx(0.006)


def test_failure_attribution_rejects_incomplete_ledger():
    with pytest.raises(ValueError, match="missing columns"):
        summarize_ledger(pd.DataFrame({"ticker": ["A"]}), ["ticker"])
