"""Fixture builders for QA track D. Only public production entry points are called."""
from __future__ import annotations

import json
import random
import sqlite3
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from fundamentals.providers import alfred_snapshot
from fundamentals.store import Fact, PointInTimeFundamentalStore

from tests.qa_reaudit.mac_oracle import SERIES, UTC, VintageArchive, month_shift

REPO = Path(__file__).resolve().parents[2]
CANARY = 123456789123


def iso(d: datetime) -> str:
    return d.astimezone(UTC).isoformat()


def month_starts(first: date, n: int) -> list[date]:
    out = []
    for i in range(n):
        y, m = month_shift(first, i)
        out.append(date(y, m, 1))
    return out


def alfred_csv(series: str, vintage: str, rows) -> bytes:
    stamp = vintage.replace("-", "")
    head = f"observation_date,{series}_{stamp}\n"
    return (head + "".join(f"{d},{v}\n" for d, v in rows)).encode()


def ingest_vintage(store, series, vintage, rows, units=None):
    units = units or ("index" if series == "CPIAUCSL" else "percent")
    return store.ingest(alfred_snapshot(alfred_csv(series, vintage, rows), series, vintage, units=units))


def simple_macro_store(vintage="2025-07-01", months=30, first=date(2023, 1, 1), cpi=None, unrate=None, ff=None, store=None):
    """One vintage of all three series with deterministic generators of month index i."""
    store = store or PointInTimeFundamentalStore()
    cpi = cpi or (lambda i: 100 + i)
    unrate = unrate or (lambda i: 4.0 + 0.1 * (i % 5))
    ff = ff or (lambda i: 5.0 - 0.01 * i)
    ms = month_starts(first, months)
    ingest_vintage(store, "CPIAUCSL", vintage, [(d.isoformat(), cpi(i)) for i, d in enumerate(ms)])
    ingest_vintage(store, "UNRATE", vintage, [(d.isoformat(), unrate(i)) for i, d in enumerate(ms)])
    ingest_vintage(store, "FEDFUNDS", vintage, [(d.isoformat(), ff(i)) for i, d in enumerate(ms)])
    return store


# ----------------------------------------------------------------- random vintage universes
def random_vintage_universe(seed: int):
    """Seeded multi-vintage history with revisions.

    Returns {series: [(vintage_date, {obs_date: Fraction})]} and the CSV bodies keyed the same way.
    Observations are released with a lag after month end; later vintages revise earlier values.
    """
    rng = random.Random(seed)
    first = date(2019, 1, 1)
    n_months = 54
    ms = month_starts(first, n_months)
    release_lag = {s: rng.randrange(8, 45) for s in SERIES}  # days after month end
    base = {}
    revisions = {}
    for s in SERIES:
        if s == "CPIAUCSL":
            v = Fraction(2500, 10)
            vals = []
            for _ in ms:
                v += Fraction(rng.randrange(-5, 18), 10)
                vals.append(v)
        elif s == "UNRATE":
            v = Fraction(rng.randrange(35, 60), 10)
            vals = []
            for _ in ms:
                v = max(Fraction(3), v + Fraction(rng.randrange(-3, 4), 10))
                vals.append(v)
        else:
            v = Fraction(rng.randrange(5, 300), 100)
            vals = []
            for _ in ms:
                v = max(Fraction(1, 100), v + Fraction(rng.randrange(-25, 30), 100))
                vals.append(v)
        base[s] = vals
        rev = {}
        for i, d in enumerate(ms):
            if s != "FEDFUNDS" and rng.random() < 0.35:
                # revised once or twice, 60-300 days after the month ends
                k = rng.randrange(1, 3)
                rev[d] = [(date(*month_shift(d, 1), 1) + timedelta(days=rng.randrange(60, 300)), Fraction(rng.randrange(-8, 9), 10 if s == "UNRATE" else 100)) for _ in range(k)]
        revisions[s] = rev
    # vintage dates: random days in 2021-01-01 .. 2024-12-31
    days = sorted({date(2021, 1, 1) + timedelta(days=rng.randrange(0, 1460)) for _ in range(rng.randrange(8, 16))})
    universe = {s: [] for s in SERIES}
    for v in days:
        for s in SERIES:
            data = {}
            for i, d in enumerate(ms):
                ey, em = month_shift(d, 1)
                released = date(ey, em, 1) + timedelta(days=release_lag[s] - 1)
                if released > v or d < date(*month_shift(v, -36), 1):
                    continue
                val = base[s][i]
                for rdate, delta in revisions[s].get(d, []):
                    if rdate <= v:
                        val += delta
                if s != "CPIAUCSL" and val < 0:
                    val = Fraction(0)
                data[d] = val
            if data:
                universe[s].append((v, data))
    return universe


def _fmt(val: Fraction) -> str:
    d = Decimal(val.numerator) / Decimal(val.denominator)
    return format(d, "f")


def store_from_universe(universe, cutoff: datetime | None = None, extra=None):
    """Ingest vintages through providers.alfred_snapshot. cutoff drops vintages public after it."""
    st = PointInTimeFundamentalStore()
    for s in SERIES:
        for v, data in universe[s]:
            if cutoff is not None and VintageArchive.public_from(v) > cutoff:
                continue
            rows = [(d.isoformat(), _fmt(val)) for d, val in sorted(data.items())]
            ingest_vintage(st, s, v.isoformat(), rows)
    for item in extra or []:
        s, v, rows = item
        ingest_vintage(st, s, v, rows)
    return st


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, allow_nan=False, default=str)


# ----------------------------------------------------------------- real DB copy
def copy_real_db(dst: Path) -> Path:
    """Backup-copy the real fundamentals DB (read-only URI) into dst; never touches the original."""
    src_path = REPO / "results/fundamentals/fundamentals.sqlite"
    src = sqlite3.connect(src_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        out = sqlite3.connect(str(dst))
        src.backup(out)
        out.close()
    finally:
        src.close()
    return dst


# ----------------------------------------------------------------- events
def add_event(store, *, symbol="XYZ", event_type="earnings", event_at="2025-05-01T13:00:00Z", reported_at=None,
              available_at="2025-05-01T14:00:00Z", source="feed", revision="1", payload=None):
    return store.add_event(symbol=symbol, event_type=event_type, event_at=event_at,
                           reported_at=reported_at or event_at, available_at=available_at, source=source,
                           revision=revision, payload=payload or {})


def fact(**kw):
    d = dict(symbol="XYZ", metric="m", value=1.0, period_end="2025-03-10", reported_at="2025-03-12T21:00:00Z",
             available_at="2025-03-12T21:05:00Z", source="SRC", revision="A", currency="USD", units="USD", kind="financial")
    d.update(kw)
    return Fact(**d)


# ----------------------------------------------------------------- physical DB copies (PHASE 98 style)
def clone_store(store) -> PointInTimeFundamentalStore:
    """Byte-level copy of an in-memory/file store via the sqlite backup API."""
    out = PointInTimeFundamentalStore()
    store.db.backup(out.db)
    return out


def truncated_store(store, cutoff: datetime) -> PointInTimeFundamentalStore:
    """A SECOND database that physically contains only observation/event rows available at or before cutoff."""
    out = PointInTimeFundamentalStore()
    cut = iso(cutoff)
    rows = store.db.execute("SELECT * FROM observations WHERE available_at<=?", (cut,)).fetchall()
    out.db.executemany("INSERT INTO observations VALUES (" + ",".join("?" * 15) + ")", [tuple(r) for r in rows])
    rows = store.db.execute("SELECT * FROM corporate_events WHERE available_at<=?", (cut,)).fetchall()
    out.db.executemany("INSERT INTO corporate_events VALUES (" + ",".join("?" * 9) + ")", [tuple(r) for r in rows])
    out.db.commit()
    return out
