"""Track C: accounting identities that must hold between returned features (no oracle needed), checked on the
output of the production snapshot for 400 seeded random companies. This is an independent angle on formulas:
if two features that are algebraically linked disagree, one of them is wrong regardless of any oracle.

Run: .venv/bin/python -m pytest -p no:cacheprovider -q tests/qa_reaudit/test_calc_identities.py
"""
import random
from datetime import timedelta

import pytest

from tests.qa_reaudit.calc_fixtures import new_store, snapshot
from tests.qa_reaudit.calc_gen import expected, facts_for, gen_company
from tests.qa_reaudit.test_calc_random_features import pick_T

SEEDS = list(range(400))
CHECKED = {"n": 0}


def close(a, b, rel=1e-9, abs_tol=1e-6):
    return abs(a - b) <= max(rel * max(abs(a), abs(b)), abs_tol)


@pytest.mark.parametrize("seed", SEEDS, ids=lambda s: f"seed{s}")
def test_identities_between_features(seed):
    c = gen_company(seed)
    T = pick_T(c, seed)
    snap = snapshot(new_store(facts_for(c)), "RND", T)
    if snap["status"] == "MISSING":
        return
    f, raw = snap["features"], snap["raw"]
    g = lambda k: f.get(k)
    r = raw.get("revenue")

    def have(*ks):
        return all(k is not None for k in ks)

    # FCF = OCF - CapEx exactly once; margins are fcf/ebitda over revenue
    if have(raw.get("ocf"), raw.get("capex")):
        assert close(g("fcf"), raw["ocf"] - raw["capex"]), seed
        CHECKED["n"] += 1
    if have(g("fcf"), g("fcf_margin")) and r and r > 0:
        assert close(g("fcf_margin") * r, g("fcf")), seed
    if have(g("ebitda_proxy"), g("ebitda_margin")) and r and r > 0:
        assert close(g("ebitda_margin") * r, g("ebitda_proxy")), seed
    if have(raw.get("short_debt"), raw.get("long_debt")):
        assert close(g("debt"), raw["short_debt"] + raw["long_debt"]), seed
        if have(raw.get("cash")):
            assert close(g("net_debt"), g("debt") - raw["cash"]), seed
    # leverage identities
    if have(g("debt_equity"), raw.get("equity"), g("debt")) and raw["equity"] > 0:
        assert close(g("debt_equity") * raw["equity"], g("debt")), seed
    if have(g("net_debt_ebitda"), g("ebitda_proxy"), g("net_debt")) and g("ebitda_proxy") > 0:
        assert close(g("net_debt_ebitda") * g("ebitda_proxy"), g("net_debt")), seed
    if have(g("cash_debt"), g("debt")) and g("debt") > 0:
        assert close(g("cash_debt") * g("debt"), raw["cash"]), seed
    if have(g("current_ratio"), raw.get("current_liabilities")) and raw["current_liabilities"] > 0:
        assert close(g("current_ratio") * raw["current_liabilities"], raw["current_assets"]), seed
    # earnings quality: fcf_net_income - ocf_net_income = -capex/NI
    if have(g("ocf_net_income"), g("fcf_net_income"), raw.get("capex"), raw.get("net_income")) and raw["net_income"] > 0:
        assert close(g("ocf_net_income") - g("fcf_net_income"), raw["capex"] / raw["net_income"], 1e-9, 1e-9), seed
    # turnover / days are reciprocal (same average-balance denominators): days * turnover = 365 for inventory
    if have(g("inventory_days"), g("inventory_turnover")):
        assert close(g("inventory_days") * g("inventory_turnover"), 365.0, 1e-9, 1e-9), seed
        CHECKED["n"] += 1
    # cash conversion cycle = DSO + DIO - DPO
    if have(g("receivable_days"), g("inventory_days"), g("payable_days"), g("cash_conversion_cycle")):
        assert close(g("cash_conversion_cycle"), g("receivable_days") + g("inventory_days") - g("payable_days"), 1e-9, 1e-9), seed
    # growth/acceleration algebra
    if have(g("receivables_yoy"), g("revenue_yoy"), g("receivables_vs_sales")):
        assert close(g("receivables_vs_sales"), g("receivables_yoy") - g("revenue_yoy"), 1e-12, 1e-12), seed
    if have(g("inventory_yoy"), g("revenue_yoy"), g("inventory_vs_sales")):
        assert close(g("inventory_vs_sales"), g("inventory_yoy") - g("revenue_yoy"), 1e-12, 1e-12), seed
    # ROE/ROA reconstruct net income from the average balance (current balance is in raw; prior comes from the snapshot lineage)
    if have(g("roe"), raw.get("net_income")) and g("roe") != 0:
        implied_avg_equity = raw["net_income"] / g("roe")
        assert implied_avg_equity > 0, seed                     # a defined ROE always has a positive average-equity denominator
    # flags are exact functions of the features (the oracle never needed)
    assert ("NEGATIVE_FCF" in snap["flags"]) == (g("fcf") is not None and g("fcf") < 0), seed
    assert ("FALLING_MARGINS" in snap["flags"]) == (g("gross_margin_change") is not None and g("gross_margin_change") < 0), seed
    assert ("REVENUE_DECELERATION" in snap["flags"]) == (g("revenue_acceleration") is not None and g("revenue_acceleration") < 0), seed
    assert ("RECEIVABLES_OUTPACE_SALES" in snap["flags"]) == (g("receivables_vs_sales") is not None and g("receivables_vs_sales") > 0), seed
    assert ("DILUTION" in snap["flags"]) == (g("diluted_shares_yoy") is not None and g("diluted_shares_yoy") > 0), seed


def test_identities_were_actually_exercised():
    CHECKED["n"] = 0
    for seed in SEEDS[:120]:
        test_identities_between_features(seed)
    assert CHECKED["n"] > 40, CHECKED
