"""Independent oracles for QA track D (macro, events, estimates, insiders, news timing).

Nothing here imports from fundamentals.*; only the stdlib is used. All arithmetic
is exact (Fraction) and only converted to float at the comparison boundary.
"""
from __future__ import annotations

import csv
import math
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path

UTC = timezone.utc
SERIES = ("CPIAUCSL", "UNRATE", "FEDFUNDS")
RAW_ALFRED_DIR = Path("results/fundamentals/raw")


# --------------------------------------------------------------------------- time helpers
def dt(text: str) -> datetime:
    """Parse an ISO string with explicit offset or Z into an aware UTC datetime."""
    t = text.replace("Z", "+00:00")
    d = datetime.fromisoformat(t)
    if d.tzinfo is None:
        raise ValueError("naive timestamp in oracle input: " + text)
    return d.astimezone(UTC)


def month_shift(d: date, months: int) -> tuple[int, int]:
    """Return (year, month) of d shifted by `months` calendar months."""
    idx = d.year * 12 + (d.month - 1) + months
    return idx // 12, idx % 12 + 1


def whole_days(delta: timedelta) -> int:
    """Floor of a (non-negative) timedelta in days."""
    return delta.days


# --------------------------------------------------------------------------- ALFRED archive
def parse_alfred_csv(path: Path):
    """Return (series, vintage_date, {obs_date: Fraction}) from an ALFRED graph CSV."""
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    header = rows[0]
    assert header[0] == "observation_date", header
    series, stamp = header[1].rsplit("_", 1)
    vintage = date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:]))
    data = {}
    for r in rows[1:]:
        if len(r) < 2:
            continue
        cell = r[1].strip()
        if cell in ("", "."):
            continue
        data[date.fromisoformat(r[0])] = Fraction(cell)
    return series, vintage, data


class VintageArchive:
    """Vintage snapshots keyed by series. A vintage dated D is public from D+1 00:00 UTC."""

    def __init__(self):
        self.v = {s: [] for s in SERIES}

    @classmethod
    def from_dir(cls, directory: Path = RAW_ALFRED_DIR):
        a = cls()
        for p in sorted(Path(directory).glob("*_20*.csv")):
            if p.name.startswith("alfred_"):
                continue
            series, vintage, data = parse_alfred_csv(p)
            a.v[series].append((vintage, data))
        for s in a.v:
            a.v[s].sort(key=lambda x: x[0])
        return a

    @staticmethod
    def public_from(vintage: date) -> datetime:
        return datetime(vintage.year, vintage.month, vintage.day, tzinfo=UTC) + timedelta(days=1)

    def known_at(self, series: str, ts: datetime):
        """{obs_date: (value, last_vintage_containing_it)} using only public vintages."""
        out = {}
        for vintage, data in self.v[series]:
            if self.public_from(vintage) > ts:
                continue
            for obs, val in data.items():
                out[obs] = (val, vintage)  # later vintage overwrites earlier
        return out

    @classmethod
    def from_memory(cls, data):
        """data = {series: [(vintage_date, {obs_date: Fraction})]}"""
        a = cls()
        for s, items in data.items():
            a.v[s] = sorted(((v, dict(d)) for v, d in items), key=lambda x: x[0])
        return a

    def first_vintage_values(self, series: str):
        out = {}
        for vintage, data in self.v[series]:
            for obs, val in data.items():
                out.setdefault(obs, (val, vintage))
        return out

    def final_vintage_values(self, series: str):
        out = {}
        for vintage, data in self.v[series]:
            for obs, val in data.items():
                out[obs] = (val, vintage)
        return out


# --------------------------------------------------------------------------- macro oracle
def _lookup(known, latest: date, months: int, used=None, series=None):
    y, m = month_shift(latest, -months)
    hit = [k for k in known if k.year == y and k.month == m]
    if not hit:
        return None
    k = max(hit)
    if used is not None:
        used.add((series, k, known[k][1]))
    return known[k][0]


def sign_label(x, up, down):
    if x is None:
        return "UNKNOWN"
    return up if x > 0 else down if x < 0 else "UNCHANGED"


def macro_oracle(known_by_series: dict, ts: datetime, max_age_days: int = 100):
    """Independent macro regime computation.

    known_by_series[series] = {obs_date: (Fraction value, last vintage date)}.
    Staleness: whole days between ts and the availability of the vintage that supplied the
    latest observation; strictly greater than max_age_days is STALE.
    Status precedence: STALE beats INCOMPLETE beats AVAILABLE.
    """
    feats = {}
    used = set()
    status = "AVAILABLE"
    for series in SERIES:
        known = known_by_series.get(series, {})
        if not known:
            if status != "STALE":
                status = "INCOMPLETE"
            continue
        latest = max(known)
        value, vint = known[latest]
        age = whole_days(ts - VintageArchive.public_from(vint))
        if age > max_age_days:
            status = "STALE"
            continue
        used.add((series, latest, vint))
        if series == "CPIAUCSL":
            prior = _lookup(known, latest, 12, used, series)
            three = _lookup(known, latest, 3, used, series)
            fifteen = _lookup(known, latest, 15, used, series)
            yoy = value / prior - 1 if prior is not None and prior > 0 else None
            prev = three / fifteen - 1 if three is not None and fifteen is not None and fifteen > 0 else None
            feats["inflation_yoy"] = yoy
            feats["inflation_change_3m"] = yoy - prev if yoy is not None and prev is not None else None
        else:
            name = "unemployment" if series == "UNRATE" else "policy_rate"
            prev = _lookup(known, latest, 3, used, series)
            feats[name] = value
            feats[name + "_change_3m"] = value - prev if prev is not None else None
    regime = {
        "inflation": sign_label(feats.get("inflation_change_3m"), "RISING", "FALLING"),
        "labour_market": sign_label(feats.get("unemployment_change_3m"), "WEAKENING", "STRENGTHENING"),
        "policy": sign_label(feats.get("policy_rate_change_3m"), "TIGHTENING", "EASING"),
    }
    if status != "STALE" and (len(feats) != 6 or any(v is None for v in feats.values())):
        status = "INCOMPLETE"
    return {"features": feats, "regime": regime, "status": status, "used": used}


def rel_close(a, b, tol=1e-12):
    if a is None or b is None:
        return a is None and b is None
    a, b = float(a), float(b)
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


# --------------------------------------------------------------------------- events oracle
def decay(age_days: float, half_life: float) -> float:
    """Exponential half-life decay, 2 ** (-age/half_life)."""
    return 2.0 ** (-age_days / half_life)


def eps_surprise(actual, consensus):
    """(actual - consensus) / |consensus| using exact Fractions; None when consensus is 0/None."""
    if actual is None or consensus is None:
        return None
    c = Fraction(str(consensus))
    if c == 0:
        return None
    return float((Fraction(str(actual)) - c) / abs(c))


def revision(new, old):
    """Relative estimate revision with |old| denominator; None for zero/None base."""
    if new is None or old is None:
        return None
    o = Fraction(str(old))
    if o == 0:
        return None
    return float((Fraction(str(new)) - o) / abs(o))


def finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)
