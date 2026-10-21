"""Track C quarterly generator + fiscal-index TTM oracle (independent of the production day-gap logic).

A company has N fiscal years of 4 quarters. 10-Q filings carry: 3M discrete rows for flow-income metrics, YTD rows
for everything (cash-flow metrics ONLY as YTD, like real filers), 10-K carries FY rows. Filings may be missing, rows may be
restated by later filings (same period, new value), EPS Q4 direct is optional, and the fiscal calendar may drift (52/53 weeks).
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pandas as pd

from fundamentals.store import Fact
from tests.qa_reaudit import calc_oracle as O

INCOME = ("revenue", "net_income")          # both 3M and YTD rows appear in a 10-Q
CASHFLOW = ("ocf", "capex")                 # YTD only in a 10-Q
PER_SHARE = ("eps",)                        # 3M direct only in a 10-Q, FY in the 10-K (+ optional Q4 direct)
METRICS = INCOME + CASHFLOW + PER_SHARE


def month_end(year, month):
    return (pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)).date()


def q2(x):
    return Decimal(x).quantize(Decimal("0.01"))


@dataclass
class Row:
    metric: str
    fy: int
    qa: int
    qb: int
    value: Decimal
    published: datetime
    revision: str
    basis: str | None


@dataclass
class QCompany:
    seed: int
    n_fy: int
    ends: dict          # (fy,k) -> end date
    starts: dict        # (fy,k) -> start date
    rows: list          # list[Row] (all versions)
    truth: dict         # (metric,fy,k) -> discrete true value (unrestated)
    unit: tuple = ("GBP millions", Decimal(1))


def gen_qcompany(seed: int, n_fy=4) -> QCompany:
    rng = random.Random(seed)
    end_month = rng.choice([12, 6, 9, 3, 1])
    drift = rng.random() < 0.4
    first_year = rng.randint(2014, 2019)
    ends, starts = {}, {}
    prev_end = None
    # quarter ends: consecutive month ends 3 months apart, optional +-2 day drift for 52/53 week calendars.
    # global quarter g = fy*4 + (k-1); Q4 of fiscal year 0 (g=3) ends at the anchor month end.
    anchor = month_end(first_year, end_month)
    all_ends = {}
    for g in range(-1, 4 * n_fy):
        e = (pd.Timestamp(anchor) + pd.DateOffset(months=3 * (g - 3)) + pd.offsets.MonthEnd(0)).date()
        if drift:
            e = e + timedelta(days=rng.randint(-2, 2))
        all_ends[g] = e
    for fy in range(n_fy):
        for k in range(1, 5):
            g = fy * 4 + (k - 1)
            ends[(fy, k)] = all_ends[g]
            starts[(fy, k)] = all_ends[g - 1] + timedelta(days=1)
    style = rng.choice([("GBP millions", Decimal(1)), ("GBP thousands", Decimal(1000)), ("GBP", Decimal(1_000_000))])
    truth = {}
    for fy in range(n_fy):
        for k in range(1, 5):
            base = Decimal(rng.randint(5_000, 90_000)) / 100
            truth[("revenue", fy, k)] = q2(base * Decimal(str(round(rng.uniform(0.6, 1.5), 3))))
            truth[("net_income", fy, k)] = q2(truth[("revenue", fy, k)] * Decimal(str(round(rng.uniform(-0.2, 0.3), 3))))
            truth[("ocf", fy, k)] = q2(truth[("revenue", fy, k)] * Decimal(str(round(rng.uniform(-0.1, 0.4), 3))))
            truth[("capex", fy, k)] = q2(truth[("revenue", fy, k)] * Decimal(str(round(rng.uniform(0.0, 0.2), 3))))
            truth[("eps", fy, k)] = Decimal(str(round(rng.uniform(-0.5, 1.5), 4)))
    rows = []
    miss_rate = 0.15
    for fy in range(n_fy):
        basis_fy = "one" if rng.random() > 0.08 else None
        for k in range(1, 5):
            if rng.random() < miss_rate and k < 4:
                continue  # this 10-Q was never ingested
            end = ends[(fy, k)]
            lag = rng.randint(20, 45) if k < 4 else rng.randint(40, 90)
            pub = datetime(end.year, end.month, end.day, tzinfo=timezone.utc) + timedelta(days=lag, hours=rng.randint(10, 22), minutes=rng.randint(0, 59))
            rev = f"{'Q' if k < 4 else 'K'}{fy}.{k}"

            def cum(metric, upto):
                return sum((truth[(metric, fy, j)] for j in range(1, upto + 1)), Decimal(0))

            for m in INCOME:
                if k < 4:
                    rows.append(Row(m, fy, k, k, truth[(m, fy, k)], pub, rev, "one"))          # 3M discrete
                if k >= 2 and rng.random() < 0.9:
                    rows.append(Row(m, fy, 1, k, cum(m, k), pub, rev, "one"))                  # YTD
                if k == 4:
                    rows.append(Row(m, fy, 1, 4, cum(m, 4), pub, rev, "one"))                  # FY (10-K)
                    if rng.random() < 0.15:
                        rows.append(Row(m, fy, 4, 4, truth[(m, fy, 4)], pub, rev, "one"))      # Q4 shown in the 10-K
            for m in CASHFLOW:
                rows.append(Row(m, fy, 1, k, cum(m, k), pub, rev, "one"))                      # YTD only (k==4: FY)
            if k < 4:
                rows.append(Row("eps", fy, k, k, truth[("eps", fy, k)], pub, rev, basis_fy))
            else:
                fyeps = sum((truth[("eps", fy, j)] for j in range(1, 5)), Decimal(0))
                rows.append(Row("eps", fy, 1, 4, fyeps, pub, rev, basis_fy))                   # FY EPS: never additive
                if rng.random() < 0.35:
                    rows.append(Row("eps", fy, 4, 4, truth[("eps", fy, 4)], pub, rev, basis_fy))
    # restatements: a LATER filing re-presents an older row with a new value (same period) with prob 0.12
    later = []
    for r in list(rows):
        if rng.random() < 0.12:
            newv = q2(r.value * Decimal(str(round(rng.uniform(0.85, 1.2), 3)))) if r.metric != "eps" else Decimal(str(round(float(r.value) * rng.uniform(0.85, 1.2), 4)))
            later.append(Row(r.metric, r.fy, r.qa, r.qb, newv, r.published + timedelta(days=rng.randint(30, 200), hours=3), r.revision + "A", r.basis))
    rows += later
    return QCompany(seed, n_fy, ends, starts, rows, truth, style)


def row_dates(c: QCompany, r: Row):
    return c.starts[(r.fy, r.qa)], c.ends[(r.fy, r.qb)]


def facts_for_q(c: QCompany, symbol="RQ", currency="GBP"):
    name, mult = c.unit
    out = []
    for r in c.rows:
        start, end = row_dates(c, r)
        meta = {"accounting_standard": "us-gaap"}
        if r.basis is not None:
            meta["share_basis"] = r.basis
        if r.metric == "eps":
            cur, units, val = currency, currency + "/shares", float(r.value)
        else:
            cur, units, val = currency, name, float(r.value * mult)
        ts = r.published.isoformat()
        out.append(Fact(symbol, r.metric, val, end.isoformat(), ts, ts, "fixture", r.revision, cur, units, start.isoformat(), metadata=meta))
    return out


def visible_rows(c: QCompany, T: datetime):
    """Latest published version per (metric, fy, qa, qb) at T."""
    best = {}
    for r in c.rows:
        if r.published <= T:
            key = (r.metric, r.fy, r.qa, r.qb)
            if key not in best or r.published > best[key].published:
                best[key] = r
    return best


def expected_ttm(c: QCompany, T: datetime):
    vis = visible_rows(c, T)
    per_metric = {}
    for m in METRICS:
        rows = [dict(fy=r.fy, qa=r.qa, qb=r.qb, value=r.value) for (mm, *_), r in vis.items() if mm == m]
        per_metric[m] = O.ttm_by_index(rows, m, additive=(m != "eps"))
    rev = per_metric["revenue"]
    if not rev:
        return {"status": "INCOMPLETE", "values": {m: None for m in (*METRICS, "fcf")}, "period_end": None, "per_metric": per_metric, "qoq": {}}
    end_key = max(rev)  # (fy,k) tuples order chronologically
    values = {}
    for m in METRICS:
        values[m] = None
        if m != "eps":   # published form first (lead re-audit), four discrete quarters as the fallback
            rows = [dict(fy=r.fy, qa=r.qa, qb=r.qb, value=r.value) for (mm, *_), r in vis.items() if mm == m]
            values[m] = O.ttm_reported_totals(rows, end_key)
        if values[m] is None:
            values[m] = O.last_four(per_metric[m], end_key)
    # eps needs one shared, explicit split basis across the four used quarters
    if values["eps"] is not None:
        fy, k = end_key
        keys = []
        for _ in range(4):
            keys.append((fy, k))
            k -= 1
            if k == 0:
                fy, k = fy - 1, 4
        bases = set()
        for (fy_, k_) in keys:
            r = vis.get(("eps", fy_, k_, k_))
            bases.add(None if r is None else r.basis)
        if len(bases) != 1 or None in bases:
            values["eps"] = None
    values["fcf"] = None if values["ocf"] is None or values["capex"] is None else values["ocf"] - values["capex"]
    status = "AVAILABLE" if all(v is not None for v in values.values()) else "INCOMPLETE"
    end_date = c.ends[end_key]
    # freshness gate added by the lead re-audit (QA-CALC-06): TTM older than 550 days is STALE, whatever its completeness
    if (T.astimezone(timezone.utc).date() - end_date).days > 550:
        status = "STALE"
    qoq = {}
    for m in ("revenue", "ocf"):
        d = per_metric[m]
        keys = sorted(d)
        q = None
        if len(keys) >= 2:
            a, b = keys[-1], keys[-2]
            prev_of_a = (a[0], a[1] - 1) if a[1] > 1 else (a[0] - 1, 4)
            if b == prev_of_a:
                q = O.gro(d[a], d[b])
        qoq[m] = q
    return {"status": status, "values": values, "period_end": end_date.isoformat(), "end_key": end_key, "per_metric": per_metric, "qoq": qoq}
