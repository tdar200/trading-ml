"""Track C seeded random company generator + independent expectation (Decimal) for the annual snapshot.

gen_company(seed) -> Company with a truth table of fiscal years (nominal end, published time, values,
restatements, per-year share basis). expected(company, T) answers: what would a careful human read as the
latest annual statement, prior-year and two-years-ago statements at T, and what must every feature be?
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

from fundamentals.store import Fact
from tests.qa_reaudit import calc_oracle as O
from tests.qa_reaudit.calc_fixtures import FLOW, SHARE_METRICS, to_base

SECTORS = [None, None, None, "banks", "saas", "software", "insurance", "energy", "financials"]
FY_ENDS = [(12, 31), (6, 30), (9, 28), (3, 31)]
UNIT_STYLES = [("GBP millions", Decimal(1)), ("GBP thousands", Decimal(1000)), ("GBP", Decimal(1_000_000)), ("GBP billions", Decimal("0.001"))]
MONEY = ["revenue", "gross_profit", "cogs", "operating_income", "net_income", "pretax_income", "tax_expense", "ocf", "capex",
         "depreciation", "interest_expense", "assets", "equity", "current_assets", "current_liabilities", "short_debt", "long_debt",
         "cash", "inventory", "receivables", "payables", "prepaids", "rd", "buybacks", "dividends", "issuance", "acquisitions"]
MONEY_FEATURES = {"fcf", "ebitda_proxy", "debt", "net_debt", "debt_change", "cash_change"}
DAY_FEATURES = {"receivable_days", "inventory_days", "payable_days", "cash_conversion_cycle"}


def q2(x):
    return Decimal(x).quantize(Decimal("0.01"))


@dataclass
class YearTruth:
    idx: int
    end: str
    start: str
    published: datetime
    metrics: dict
    restated: dict = field(default_factory=dict)  # metric -> (published, value)
    basis: str | None = "one"


@dataclass
class Company:
    seed: int
    sector: str | None
    unit_style: tuple
    years: dict  # idx -> YearTruth
    distractors: list  # extra quarterly facts that must be ignored by the annual snapshot
    symbol: str = "RND"


def _values(rng, revenue):
    r = revenue
    u = rng.uniform
    assets = q2(max(r, Decimal(1)) * Decimal(str(round(u(0.3, 3.0), 3))))
    v = {"revenue": r}
    v["gross_profit"] = q2(r * Decimal(str(round(u(-0.1, 0.9), 3))))
    v["cogs"] = q2(r * Decimal(str(round(u(0.1, 1.0), 3))))
    v["operating_income"] = q2(r * Decimal(str(round(u(-0.4, 0.4), 3))))
    v["net_income"] = q2(r * Decimal(str(round(u(-0.4, 0.3), 3))))
    v["pretax_income"] = q2(r * Decimal(str(round(u(-0.4, 0.4), 3))))
    v["tax_expense"] = q2(v["pretax_income"] * Decimal(str(round(u(-0.05, 0.5), 3))))
    v["ocf"] = q2(r * Decimal(str(round(u(-0.3, 0.4), 3))))
    v["capex"] = q2(r * Decimal(str(round(u(0.0, 0.3), 3))))
    v["depreciation"] = q2(r * Decimal(str(round(u(0.0, 0.1), 3))))
    v["interest_expense"] = q2(r * Decimal(str(round(u(0.0, 0.05), 3)))) if rng.random() > 0.1 else Decimal(0)
    v["assets"] = assets
    v["equity"] = q2(assets * Decimal(str(round(u(-0.2, 0.8), 3))))
    v["current_assets"] = q2(assets * Decimal(str(round(u(0.1, 0.6), 3))))
    v["current_liabilities"] = q2(assets * Decimal(str(round(u(0.05, 0.5), 3))))
    v["short_debt"] = q2(assets * Decimal(str(round(u(0.0, 0.15), 3))))
    v["long_debt"] = q2(assets * Decimal(str(round(u(0.0, 0.3), 3))))
    v["cash"] = q2(assets * Decimal(str(round(u(0.0, 0.4), 3))))
    for m in ("inventory", "receivables", "payables", "prepaids"):
        v[m] = q2(r * Decimal(str(round(u(0.0, 0.3), 3))))
    for m in ("rd", "buybacks", "dividends", "issuance", "acquisitions"):
        v[m] = q2(r * Decimal(str(round(u(0.0, 0.1), 3))))
    v["eps"] = Decimal(str(round(u(-3, 5), 4)))
    v["diluted_shares"] = Decimal(rng.randint(1_000_000, 500_000_000))
    return v


def gen_company(seed: int) -> Company:
    rng = random.Random(seed)
    month, day = rng.choice(FY_ENDS)
    drift_mode = rng.random() < 0.4
    n = rng.randint(3, 7)
    first = rng.randint(2012, 2018)
    years = {}
    prev_end = None
    base = None
    for i in range(n):
        y = first + i
        end = date(y, month, day) + timedelta(days=rng.randint(-3, 3) if drift_mode else 0)
        start = (prev_end + timedelta(days=1)) if prev_end else (date(y - 1, month, day) + timedelta(days=1))
        rev = q2(Decimal(rng.randint(1, 500000)) / 100) if rng.random() > 0.04 else Decimal(0)
        if base is not None and rng.random() < 0.85:  # smooth-ish revenue paths so growth is often well defined
            rev = q2(base * Decimal(str(round(rng.uniform(0.7, 1.6), 3))))
        base = rev if rev > 0 else base
        metrics = _values(rng, rev)
        # random missing statement lines (None) and withdrawn lines
        for m in list(metrics):
            if m != "revenue" and rng.random() < 0.10:
                del metrics[m]
        pub = datetime(end.year, end.month, end.day, tzinfo=timezone.utc) + timedelta(days=rng.randint(35, 80), hours=rng.randint(10, 22), minutes=rng.randint(0, 59))
        restated = {}
        for m in list(metrics):
            if rng.random() < 0.08:
                newv = None if (rng.random() < 0.15 and m not in SHARE_METRICS and m not in ("eps", "revenue")) else (
                    metrics[m] * Decimal(str(round(rng.uniform(0.8, 1.25), 3))) if m not in ("diluted_shares",) else Decimal(rng.randint(1_000_000, 500_000_000)))
                if newv is not None and m == "capex" and newv < 0:
                    newv = abs(newv)
                if m == "revenue" and newv is not None:
                    newv = q2(newv)
                elif newv is not None and m not in ("eps", "diluted_shares"):
                    newv = q2(newv)
                elif newv is not None and m == "eps":
                    newv = Decimal(str(round(float(newv), 4)))
                restated[m] = (pub + timedelta(days=rng.randint(20, 400), hours=rng.randint(0, 5)), newv)
        years[i] = YearTruth(i, end.isoformat(), start.isoformat(), pub, metrics, restated, "one")
        prev_end = end
    mode = rng.choice(["all_one"] * 7 + ["mixed", "missing", "one_year_missing"])
    for i, yt in years.items():
        if mode == "mixed" and rng.random() < 0.4:
            yt.basis = "two"
        elif mode == "missing":
            yt.basis = None
        elif mode == "one_year_missing" and rng.random() < 0.4:
            yt.basis = None
    style = rng.choice(UNIT_STYLES)
    distract = []
    return Company(seed, rng.choice(SECTORS), style, years, distract)


def facts_for(company: Company, symbol="RND"):
    """All Facts (originals, restatements, distractor quarters, classification)."""
    name, mult = company.unit_style
    out = []
    meta0 = {"accounting_standard": "us-gaap"}

    def mk(metric, value, end, start, published, revision, basis):
        meta = dict(meta0)
        if basis is not None:
            meta["share_basis"] = basis
        if metric == "eps":
            cur, units, val = "GBP", "GBP/shares", None if value is None else float(value)
        elif metric in SHARE_METRICS:
            cur, units, val = "", "shares", None if value is None else float(value)
        else:
            cur, units = "GBP", name
            val = None if value is None else float(value * mult)
        ts = published.isoformat()
        return Fact(symbol, metric, val, end, ts, ts, "fixture", revision, cur, units, start if metric in FLOW else "", metadata=meta)

    for yt in company.years.values():
        for m, v in yt.metrics.items():
            out.append(mk(m, v, yt.end, yt.start, yt.published, f"FY{yt.idx}", yt.basis))
        for m, (pub2, v2) in yt.restated.items():
            out.append(mk(m, v2, yt.end, yt.start, pub2, f"FY{yt.idx}-A", yt.basis))
    if company.sector:
        out.append(Fact(symbol, "sector", None, "2000-01-01", "2000-01-01T00:00:00+00:00", "2000-01-01T00:00:00+00:00", "fixture", "c1", "", "label",
                        kind="classification", metadata={"sector": company.sector}))
    return out


def visible_metrics(yt: YearTruth, T: datetime):
    """metric -> Decimal|None for facts visible at T (latest published version wins); None value is present-but-missing."""
    seen = {}
    if yt.published <= T:
        for m, v in yt.metrics.items():
            seen[m] = v
    for m, (pub2, v2) in yt.restated.items():
        if pub2 <= T:
            if m in yt.metrics and yt.published <= T or m not in yt.metrics:
                seen[m] = v2
    return seen


def expected(company: Company, T: datetime):
    """Independent expectation of snapshot(T). Returns dict(status, features, flags, period_end, raw) or status MISSING."""
    # a fiscal year is an annual period iff its REVENUE fact is visible with a numeric value
    vis = {}
    for idx, yt in company.years.items():
        m = visible_metrics(yt, T)
        if m.get("revenue") is not None:
            vis[idx] = (yt, m)
    if not vis:
        return {"status": "MISSING"}
    cur_idx = max(vis, key=lambda i: vis[i][0].end)
    cur_yt, cur_m = vis[cur_idx]

    def year(idx):
        if idx in vis:
            return vis[idx]
        return None

    def base(idx):
        y = year(idx)
        if y is None:
            return {}
        return {k: to_base(k, v) for k, v in y[1].items() if v is not None}

    cur, prior, old = base(cur_idx), base(cur_idx - 1), base(cur_idx - 2)
    feats, flags = O.accounting(cur, prior, old, company.sector)
    for yrs in (3, 5):
        feats[f"revenue_cagr_{yrs}y"] = O.cagr(cur.get("revenue"), base(cur_idx - yrs).get("revenue"), yrs)
    for m in ("revenue", "eps", "ocf", "operating_income"):
        feats[m + "_qoq"] = None
    # per-share comparability: bases over EPS/share rows of every annual period the snapshot touched
    bases = []
    touched = [i for i in (cur_idx, cur_idx - 1, cur_idx - 2, cur_idx - 3, cur_idx - 5) if year(i)]
    for i in touched:
        yt, m = year(i)
        for k in ("eps", "diluted_shares"):
            if k in m:
                bases.append(yt.basis)
    blanked = (not bases) or (None in bases) or (len(set(bases)) != 1)
    if blanked:
        for k in ("eps_yoy", "eps_acceleration", "eps_qoq", "diluted_shares_yoy"):
            feats[k] = None
        flags = [f for f in flags if f != "DILUTION"]
    age = (T.date() - date.fromisoformat(cur_yt.end)).days
    return {"status": "STALE" if age > 550 else "AVAILABLE", "features": feats, "flags": flags, "period_end": cur_yt.end,
            "age": age, "blanked": blanked, "cur": cur}


def tolerance(name):
    if name in MONEY_FEATURES:
        return 1e-6
    if name in DAY_FEATURES:
        return 1e-9
    return 1e-13
