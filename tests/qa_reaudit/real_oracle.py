"""Independent parser/oracle for raw SEC companyfacts + submissions JSON (QA track B, prefix real).

Deliberately imports NOTHING from fundamentals.*: stdlib only.  It answers
"what would a human reading only filings that were public at T have seen?" for
annual revenue / net income / OCF / CapEx / assets / equity / diluted EPS and a
trailing-twelve-month figure built from four discrete quarters (Q4 = FY - 9M,
Q2/Q3 from YTD cash-flow style facts).

Availability convention used for the *exact-equality* oracle (re-implemented here,
not imported): acceptance time + 5 minutes when the submissions JSON carries an
acceptanceDateTime for the accession, otherwise 00:00 America/New_York of the
filing date + 1 calendar day.  A separate, looser "lower bound" (acceptance time
itself, or 00:00 New York of the filing date) is used to prove no value can ever be
earlier than the true dissemination time.
"""
from __future__ import annotations
import datetime as dt
import json
from pathlib import Path
from zoneinfo import ZoneInfo

NY = ZoneInfo('America/New_York')
UTC = dt.timezone.utc
PERIODIC_FORMS = ('10-K', '10-Q', '10-K/A', '10-Q/A', '20-F', '20-F/A')

REPO = Path(__file__).resolve().parents[2]
RAW8 = REPO / 'results' / 'fundamental_audit_20260930' / 'raw'
RAWNEW = REPO / 'results' / 'qa_reaudit_20261001' / 'real' / 'raw'

# Own tag knowledge (what a human analyst reads as the headline line item).
# revenue: the headline is the LARGEST revenue-like concept the same filing reports
# for the same period (totals dominate components) - deliberately different from a
# first-match priority list.
REVENUE_TAGS = {
    'us-gaap': ('Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'SalesRevenueNet',
                'SalesRevenueGoodsNet', 'RegulatedAndUnregulatedOperatingRevenue'),
    'ifrs-full': ('Revenue',),
}
# Tags a human analyst would ALSO read as revenue but that production does not map (coverage-gap probe).
EXTENDED_REVENUE_TAGS = {
    'us-gaap': REVENUE_TAGS['us-gaap'] + ('RevenueFromContractWithCustomerIncludingAssessedTax',),
    'ifrs-full': REVENUE_TAGS['ifrs-full'],
}
FLOW_TAGS = {
    'us-gaap': {
        'net_income': ('NetIncomeLoss', 'ProfitLoss'),
        'ocf': ('NetCashProvidedByUsedInOperatingActivities',),
        'capex': ('PaymentsToAcquirePropertyPlantAndEquipment',),
        'eps': ('EarningsPerShareDiluted',),
        'diluted_shares': ('WeightedAverageNumberOfDilutedSharesOutstanding',),
    },
    'ifrs-full': {
        'net_income': ('ProfitLossAttributableToOwnersOfParent',),
        'ocf': ('CashFlowsFromUsedInOperatingActivities',),
        'eps': ('DilutedEarningsLossPerShare',),
    },
}
INSTANT_TAGS = {
    'us-gaap': {
        'assets': ('Assets',),
        'equity': ('StockholdersEquity', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'),
        'cash': ('CashAndCashEquivalentsAtCarryingValue',),
    },
    'ifrs-full': {
        'assets': ('Assets',),
        'equity': ('EquityAttributableToOwnersOfParent',),
        'cash': ('CashAndCashEquivalents',),
    },
}
CURRENCIES = ('USD', 'GBP', 'EUR', 'JPY')


def parse_utc(text: str) -> dt.datetime:
    """'2024-07-30T20:11:22.000Z' or any ISO string with offset -> aware UTC datetime."""
    d = dt.datetime.fromisoformat(text.replace('Z', '+00:00'))
    if d.tzinfo is None:
        d = d.replace(tzinfo=NY)
    return d.astimezone(UTC)


def midnight_ny(day: dt.date) -> dt.datetime:
    return dt.datetime(day.year, day.month, day.day, tzinfo=NY).astimezone(UTC)


def days_inclusive(start: str, end: str) -> int:
    return (dt.date.fromisoformat(end) - dt.date.fromisoformat(start)).days + 1


class RawCompany:
    def __init__(self, symbol: str, facts_path: Path, submissions_path: Path, extended: bool = False):
        self.symbol = symbol
        self.revenue_tags = EXTENDED_REVENUE_TAGS if extended else REVENUE_TAGS
        self.facts = json.loads(Path(facts_path).read_text())
        self.subs = json.loads(Path(submissions_path).read_text())
        recent = self.subs.get('filings', {}).get('recent', {})
        self.acceptance = dict(zip(recent.get('accessionNumber', []), recent.get('acceptanceDateTime', [])))
        self.cik = self.facts['cik']
        self._rows_cache: dict[str, list[dict]] = {}
        self._pub_cache: dict[tuple, dt.datetime] = {}
        self._known_cache: dict[tuple, dict] = {}

    # ------------------------------------------------------------------ publication time
    def pub_time(self, accn: str, filed: str) -> dt.datetime:
        key = (accn, filed)
        hit = self._pub_cache.get(key)
        if hit is None:
            acc = self.acceptance.get(accn)
            if acc:
                hit = parse_utc(acc) + dt.timedelta(minutes=5)
            else:
                hit = midnight_ny(dt.date.fromisoformat(filed) + dt.timedelta(days=1))
            self._pub_cache[key] = hit
        return hit

    def lower_bound_time(self, accn: str, filed: str) -> dt.datetime:
        """Earliest conceivable public time: acceptance itself, or 00:00 New York on the filing date."""
        acc = self.acceptance.get(accn)
        return parse_utc(acc) if acc else midnight_ny(dt.date.fromisoformat(filed))

    # ------------------------------------------------------------------ raw rows
    def rows(self, standard: str) -> list[dict]:
        """Flatten periodic-form rows of one taxonomy with oracle metric names."""
        if standard in self._rows_cache:
            return self._rows_cache[standard]
        out = []
        taxonomy = self.facts['facts'].get(standard, {})

        def take(metric, tags, kind):
            for tag in tags:
                for unit, rows in taxonomy.get(tag, {}).get('units', {}).items():
                    for r in rows:
                        if r.get('form') not in PERIODIC_FORMS:
                            continue
                        if metric == 'capex' and r['val'] < 0:
                            continue   # documented policy: canonical CapEx is a positive outflow; signed provider values are quarantined
                        out.append(dict(metric=metric, tag=tag, unit=unit, start=r.get('start', ''), end=r['end'],
                                        val=r['val'], accn=r['accn'], filed=r['filed'], form=r['form'], kind=kind))
        take('revenue', self.revenue_tags[standard], 'flow')
        for metric, tags in FLOW_TAGS[standard].items():
            take(metric, tags, 'flow')
        for metric, tags in INSTANT_TAGS[standard].items():
            take(metric, tags, 'instant')
        self._rows_cache[standard] = out
        return out

    def currency(self, standard: str) -> str:
        counts: dict[str, int] = {}
        for r in self.rows(standard):
            if r['metric'] == 'revenue' and r['unit'] in CURRENCIES:
                counts[r['unit']] = counts.get(r['unit'], 0) + 1
        return max(counts, key=counts.get) if counts else 'USD'

    # ------------------------------------------------------------------ versions known at T
    def known(self, T: dt.datetime, standard: str = 'us-gaap'):
        """-> {metric: {(start,end): dict(val,pub,accn,tag)}} - latest visible version of each fact.

        Within one accession a (metric, period) with several candidate tags resolves by the
        oracle rule (revenue: max across candidates; others: first listed tag present).
        Across accessions the version with the latest publication time wins.
        """
        ckey = (T, standard)
        if ckey in self._known_cache:
            return self._known_cache[ckey]
        cur = self.currency(standard)
        per_accn: dict[tuple, dict] = {}
        for r in self.rows(standard):
            if r['kind'] == 'flow' and r['metric'] in ('eps',):
                want = cur + '/shares'
            elif r['metric'] == 'diluted_shares':
                want = 'shares'
            else:
                want = cur
            if r['unit'] != want:
                continue
            pub = self.pub_time(r['accn'], r['filed'])
            if pub > T:
                continue
            key = (r['metric'], r['accn'], r['start'], r['end'])
            tags = (self.revenue_tags[standard] if r['metric'] == 'revenue' else
                    (FLOW_TAGS[standard].get(r['metric']) or INSTANT_TAGS[standard].get(r['metric'])))
            rank = tags.index(r['tag'])
            old = per_accn.get(key)
            if old is None:
                per_accn[key] = dict(val=r['val'], pub=pub, accn=r['accn'], tag=r['tag'], rank=rank, conflict=False)
            elif r['metric'] == 'revenue':
                if r['val'] > old['val']:
                    old.update(val=r['val'], tag=r['tag'], rank=rank)
            elif rank < old['rank']:
                old.update(val=r['val'], tag=r['tag'], rank=rank, conflict=False)
            elif rank == old['rank'] and r['val'] != old['val']:
                old['conflict'] = True
        out: dict[str, dict] = {}
        for (metric, accn, start, end), v in per_accn.items():
            slot = out.setdefault(metric, {})
            cur_v = slot.get((start, end))
            if cur_v is None or (v['pub'], accn) > (cur_v['pub'], cur_v['accn']):
                slot[(start, end)] = dict(v)
        if len(self._known_cache) > 64:
            self._known_cache.clear()
        self._known_cache[ckey] = out
        return out

    # ------------------------------------------------------------------ human view
    def annual(self, T: dt.datetime, standard: str = 'us-gaap'):
        known = self.known(T, standard)
        rev = known.get('revenue', {})
        annual = {k: v for k, v in rev.items() if k[0] and 350 <= days_inclusive(*k) <= 380}
        if not annual:
            return None
        start, end = max(annual, key=lambda k: (k[1], k[0]))
        out = dict(start=start, end=end, values={}, accn={}, pub=annual[(start, end)]['pub'])
        out['values']['revenue'] = annual[(start, end)]['val']
        out['accn']['revenue'] = annual[(start, end)]['accn']
        for metric in ('net_income', 'ocf', 'capex', 'eps', 'diluted_shares'):
            v = known.get(metric, {}).get((start, end))
            if v is not None:
                out['values'][metric] = v['val']
                out['accn'][metric] = v['accn']
        for metric in ('assets', 'equity', 'cash'):
            v = known.get(metric, {}).get(('', end))
            if v is not None:
                out['values'][metric] = v['val']
                out['accn'][metric] = v['accn']
        return out

    @staticmethod
    def discrete_quarters(flows: dict, allow_derive: bool = True) -> dict:
        """{end: (start, value)} from direct ~3-month facts plus YTD differencing.

        When several versions of the same period end exist (e.g. an amendment that shifts the
        start date by a day) the version published last wins, ties by accession number.
        """
        best: dict[str, tuple] = {}
        for (s, e), v in flows.items():
            if v['val'] is not None and s and 84 <= days_inclusive(s, e) <= 100:
                rank = (v['pub'], v['accn'])
                if e not in best or rank > best[e][0]:
                    best[e] = (rank, s, v['val'], frozenset({v['tag']}))
        q = {e: (s, val, tg) for e, (_, s, val, tg) in best.items()}
        if not allow_derive:
            return q
        derived: dict[str, tuple] = {}
        for (s, e), v in flows.items():
            if not s or e in q or v['val'] is None or days_inclusive(s, e) <= 100:
                continue
            prevs = [(e2, v2['val']) for (s2, e2), v2 in flows.items()
                     if s2 == s and e2 < e and v2['val'] is not None]
            if not prevs:
                continue
            e2, val2 = max(prevs)
            gap = (dt.date.fromisoformat(e) - dt.date.fromisoformat(e2)).days
            if 84 <= gap <= 100:
                rank = (v['pub'], v['accn'])
                if e not in derived or rank > derived[e][0]:
                    derived[e] = (rank, (dt.date.fromisoformat(e2) + dt.timedelta(days=1)).isoformat(), v['val'] - val2,
                                  frozenset({v['tag'], flows[(s, e2)]['tag']}))
        for e, (_, s, val, tg) in derived.items():
            q[e] = (s, val, tg)
        return q

    def ttm(self, T: dt.datetime, standard: str = 'us-gaap'):
        known = self.known(T, standard)
        revq = self.discrete_quarters(known.get('revenue', {}))
        res = dict(end=None, values={}, reasons={}, quarters={})
        if not revq:
            return res
        first_start = None   # all metrics of one TTM must cover the identical window (documented design: no mixed fiscal durations)
        end = max(revq)
        res['end'] = end
        for metric in ('revenue', 'net_income', 'ocf', 'capex', 'eps'):
            flows = known.get(metric, {})
            totals = None if metric == 'eps' else self.ttm_from_totals(flows, end)
            if totals is not None:
                start, value, tags = totals
                if first_start is not None and start != first_start:
                    res['reasons'][metric] = 'PERIOD_MISMATCH'
                    continue
                if len(tags - {None}) > 1:
                    res['reasons'][metric] = 'MIXED_TAG'
                    continue
                first_start = start
                res['values'][metric] = value
                res['quarters'][metric] = []
                continue
            q = self.discrete_quarters(flows, allow_derive=(metric != 'eps'))
            dates = sorted(d for d in q if d <= end)[-4:]
            if len(dates) != 4 or dates[-1] != end:
                res['reasons'][metric] = 'MISSING'
                continue
            ok = all(q[b][0] == (dt.date.fromisoformat(a) + dt.timedelta(days=1)).isoformat()
                     for a, b in zip(dates, dates[1:]))
            if not ok:
                res['reasons'][metric] = 'NONCONTIGUOUS'
                continue
            span = (dt.date.fromisoformat(end) - dt.date.fromisoformat(q[dates[0]][0])).days + 1
            if not 350 <= span <= 380:
                res['reasons'][metric] = 'SPAN'
                continue
            if first_start is not None and q[dates[0]][0] != first_start:
                res['reasons'][metric] = 'PERIOD_MISMATCH'
                continue
            if len({t for d in dates for t in q[d][2]} - {None}) > 1:
                res['reasons'][metric] = 'MIXED_TAG'
                continue
            first_start = q[dates[0]][0]
            res['values'][metric] = sum(q[d][1] for d in dates)
            res['quarters'][metric] = [(d, q[d][1]) for d in dates]
        return res

    @staticmethod
    def ttm_from_totals(flows: dict, end: str):
        """Published form (lead re-audit, QA-REAL-02): TTM = FY + current YTD - prior-year YTD; at a fiscal year
        end it IS the reported annual figure. Returns (start, value, tags) or None when a total is not visible."""
        d = dt.date.fromisoformat
        cur = [(s, e, v) for (s, e), v in flows.items() if e == end and s and v['val'] is not None and 55 <= days_inclusive(s, e) <= 380]
        if not cur:
            return None
        s, e, v = max(cur, key=lambda x: days_inclusive(x[0], x[1]))
        if days_inclusive(s, e) >= 350:
            return s, v['val'], {v['tag']}
        before = (d(s) - dt.timedelta(days=1)).isoformat()
        annual = [(s2, e2, v2) for (s2, e2), v2 in flows.items() if e2 == before and s2 and v2['val'] is not None and 350 <= days_inclusive(s2, e2) <= 380]
        if len(annual) != 1:
            return None
        fy_s, fy_e, fy = annual[0]
        end_d = d(end)
        try:
            target = end_d.replace(year=end_d.year - 1)
        except ValueError:
            target = end_d.replace(year=end_d.year - 1, day=28)
        prior = [(s2, e2, v2) for (s2, e2), v2 in flows.items()
                 if s2 == fy_s and v2['val'] is not None and abs((d(e2) - target).days) <= 10 and abs(days_inclusive(s2, e2) - days_inclusive(s, e)) <= 10]
        if not prior:
            return None
        s2, e2, v2 = min(prior, key=lambda x: abs((d(x[1]) - target).days))
        return (d(e2) + dt.timedelta(days=1)).isoformat(), fy['val'] + v['val'] - v2['val'], {fy['tag'], v['tag'], v2['tag']}


def load_company(symbol: str, extended: bool = False) -> RawCompany:
    base = RAW8 if (RAW8 / f'{symbol}_facts.json').exists() else RAWNEW
    return RawCompany(symbol, base / f'{symbol}_facts.json', base / f'{symbol}_submissions.json', extended)


def iso_utc(d: dt.datetime) -> str:
    return d.astimezone(UTC).isoformat()
