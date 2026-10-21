"""Track C (calc) fixture builders: company specs -> Facts -> temp store -> snapshot.

Specs are written in 'GBP millions' (money), GBP per share (eps) and raw share counts,
exactly as a human reads a statement. Facts are emitted with units 'GBP millions' so the
store's scale normalisation is inside the loop. No calculation helper from fundamentals.*
is imported; only the public ingestion/snapshot entry points.
"""
from __future__ import annotations
from datetime import date, timedelta
from decimal import Decimal

import pandas as pd

from fundamentals.store import Fact, PointInTimeFundamentalStore
from fundamentals.features import FundamentalFeatureEngine

from tests.qa_reaudit.calc_oracle import D

FLOW = {
    "revenue", "gross_profit", "cogs", "operating_income", "net_income", "eps", "ocf", "capex",
    "interest_expense", "pretax_income", "tax_expense", "depreciation", "diluted_shares", "rd",
    "buybacks", "dividends", "issuance", "acquisitions",
}
INSTANT = {
    "cash", "current_assets", "assets", "current_liabilities", "liabilities", "short_debt", "long_debt",
    "equity", "inventory", "receivables", "payables", "prepaids", "goodwill", "retained_earnings",
    "shares_outstanding",
}
MILLIONS = Decimal(1_000_000)
SHARE_METRICS = {"diluted_shares", "shares_outstanding"}


def is_money(metric):
    return metric not in SHARE_METRICS and metric != "eps"


def to_base(metric, value):
    """Spec value -> exact Decimal in base units (money x 1e6, eps and shares unchanged)."""
    if value is None:
        return None
    return D(str(value)) * (MILLIONS if is_money(metric) else 1)


def base_dict(spec_year):
    return {m: to_base(m, v) for m, v in spec_year.items()}


def fact(symbol, metric, value, end, start, reported, *, revenue_currency="GBP", source="fixture",
         revision="r1", basis="one", observed_at=None, scaled=True, extra=None, available=None):
    """One Fact the way a human data-entry clerk would write it."""
    if metric == "eps":
        cur, units, val = revenue_currency, revenue_currency + "/shares", None if value is None else float(D(str(value)))
    elif metric in SHARE_METRICS:
        cur, units, val = "", "shares", None if value is None else float(D(str(value)))
    else:
        cur = revenue_currency
        if scaled:
            units, val = revenue_currency + " millions", None if value is None else float(D(str(value)))
        else:
            units, val = revenue_currency, None if value is None else float(D(str(value)) * MILLIONS)
    meta = {"share_basis": basis, "accounting_standard": "us-gaap"}
    if extra:
        meta.update(extra)
    kw = dict(period_start=start if metric in FLOW else "", metadata=meta)
    if observed_at:
        kw["observed_at"] = observed_at
    return Fact(symbol, metric, val, end, reported, available or reported, source, revision, cur, units, **kw)


def year_end(y, fy_end=None):
    return (fy_end or {}).get(y, f"{y}-12-31")


def year_start(y, fy_end=None):
    prev = (fy_end or {}).get(y - 1, f"{y - 1}-12-31")
    return (date.fromisoformat(prev) + timedelta(days=1)).isoformat()


def annual_facts(symbol, years, *, fy_end=None, published=None, currency="GBP", basis="one", scaled=True,
                 source="fixture", revision_prefix="FY"):
    """years: {fy: {metric: spec value}} -> list[Fact] (10-K style rows)."""
    facts = []
    for y, metrics in sorted(years.items()):
        end, start = year_end(y, fy_end), year_start(y, fy_end)
        reported = (published or {}).get(y, f"{y + 1}-02-25T21:30:00Z")
        for metric, value in metrics.items():
            facts.append(fact(symbol, metric, value, end, start, reported, revenue_currency=currency, source=source,
                              revision=f"{revision_prefix}{y}", basis=basis, scaled=scaled))
    return facts


def classification(symbol, sector, effective="2000-01-01", reported="2000-01-01T00:00:00Z", industry=None):
    meta = {"sector": sector}
    if industry:
        meta["industry"] = industry
    return Fact(symbol, "sector", None, effective, reported, reported, "fixture", "c1", "", "label",
                kind="classification", metadata=meta)


def new_store(facts=()):
    s = PointInTimeFundamentalStore()
    if facts:
        s.ingest(list(facts))
    return s


def snapshot(store, symbol, as_of, **kw):
    return FundamentalFeatureEngine(store, **kw).snapshot(symbol, as_of)


def golden_snapshot(symbol, years, as_of="2024-06-01T00:00:00Z", sector=None, **kw):
    facts = annual_facts(symbol, years, **kw)
    if sector:
        facts.append(classification(symbol, sector))
    store = new_store(facts)
    return snapshot(store, symbol, as_of), store


def utc_str(ts):
    return pd.Timestamp(ts).tz_convert("UTC").isoformat()
