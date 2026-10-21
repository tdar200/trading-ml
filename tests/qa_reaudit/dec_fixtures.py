"""Shared fixtures and an independent Fraction oracle for QA track E (prefix dec).

Nothing here imports calculation helpers from fundamentals.*; production is only used through public entry points
(Fact, PointInTimeFundamentalStore, FundamentalFeatureEngine, MacroRegimeService, EventCatalystEngine, ModelArtifact,
FundamentalScoreEngine, CombinedSignalEngine, PaperEngine).
"""
from __future__ import annotations
from fractions import Fraction
import pandas as pd

from fundamentals.store import Fact, PointInTimeFundamentalStore
from fundamentals.features import FundamentalFeatureEngine
from fundamentals.models import (ModelArtifact, FundamentalScoreEngine, MacroRegimeService, EventCatalystEngine,
                                 CombinedSignalEngine)

INSTANT = {'assets', 'equity', 'cash', 'current_assets', 'current_liabilities', 'short_debt', 'long_debt'}

# decision timestamps used across the track
T0 = '2024-03-05T04:00:00Z'          # fundamentals GOOD/BAD fresh, macro fresh (age 19 days)
T_MACRO_STALE = '2024-07-01T04:00:00Z'   # macro age > 100 days, GOOD fundamentals still AVAILABLE (age 183 days)

FY_FILED_2023 = '2024-02-20T21:00:00Z'
FY_FILED_2022 = '2023-02-21T21:00:00Z'


def _fact(sym, metric, value, end, start, reported, rev):
    currency = '' if metric in ('diluted_shares', 'shares_outstanding') else 'USD'
    units = 'shares' if not currency else 'USD/shares' if metric == 'eps' else 'USD'
    return Fact(sym, metric, value, end, reported, reported, 'fixture', rev, currency, units, start)


def company_facts(sym, *, rev_prior, rev, ocf, capex, ocf_prior=None, capex_prior=None, year=2023,
                  filed=FY_FILED_2023, filed_prior=FY_FILED_2022, extra=None, extra_prior=None):
    """Two consecutive fiscal years (calendar), flows with period_start, as the SEC adapter would produce."""
    rows = []
    for fy, rpt, vals in ((year - 1, filed_prior, dict(revenue=rev_prior, ocf=ocf_prior if ocf_prior is not None else ocf,
                                                      capex=capex_prior if capex_prior is not None else capex, **(extra_prior or {}))),
                          (year, filed, dict(revenue=rev, ocf=ocf, capex=capex, **(extra or {})))):
        for metric, value in vals.items():
            rows.append(_fact(sym, metric, value, f'{fy}-12-31', '' if metric in INSTANT else f'{fy}-01-01', rpt, f'{sym}-{fy}'))
    return rows


def macro_facts(*, cpi=True, unrate=True, fedfunds=True, last_obs='2024-01-01'):
    """Monthly vintage-style facts 2022-09 .. last_obs, each available on the 15th of the following month."""
    rows = []
    months = pd.date_range('2022-09-01', last_obs, freq='MS')
    n = len(months)
    for i, m in enumerate(months):
        avail = (m + pd.DateOffset(months=1)).replace(day=15)
        stamp = avail.strftime('%Y-%m-%dT12:00:00Z')
        end = m.strftime('%Y-%m-%d')
        if cpi:
            rows.append(Fact('MACRO:CPIAUCSL', 'CPIAUCSL', 256. + 2. * i, end, stamp, stamp, 'ALFRED/graph', stamp[:10], '', 'index', kind='macro'))
        if unrate:
            # 4.5 until two months before the end, 5.0 at the final observation
            val = 5.0 if i == n - 1 else 4.5
            rows.append(Fact('MACRO:UNRATE', 'UNRATE', val, end, stamp, stamp, 'ALFRED/graph', stamp[:10], '', 'percent', kind='macro'))
        if fedfunds:
            rows.append(Fact('MACRO:FEDFUNDS', 'FEDFUNDS', 5.0, end, stamp, stamp, 'ALFRED/graph', stamp[:10], '', 'percent', kind='macro'))
    return rows


def build_store(path=':memory:', *, macro='full', companies=True):
    s = PointInTimeFundamentalStore(path)
    if companies:
        # GOOD: revenue 128 -> 160 (growth .25 exactly), FCF 48-8=40, margin .25 exactly
        s.ingest(company_facts('GOOD', rev_prior=128., rev=160., ocf=48., capex=8.))
        # BAD: revenue 160 -> 120 (growth -.25 exactly), FCF 15-0=15, margin .125 exactly
        s.ingest(company_facts('BAD', rev_prior=160., rev=120., ocf=15., capex=0.))
        # OLD: same as GOOD but fiscal years 2020/2021 filed long ago -> STALE (age > 550 days) at T0
        s.ingest(company_facts('OLD', rev_prior=128., rev=160., ocf=48., capex=8., year=2021,
                               filed='2022-02-20T21:00:00Z', filed_prior='2021-02-21T21:00:00Z'))
    if macro != 'none':
        s.ingest(macro_facts(fedfunds=macro != 'nofed', cpi=macro != 'nocpi'))
    return s


def artifact(features, means, scales, coefs, intercept, *, trained_through='2023-12-31T00:00:00Z', horizon=5):
    return ModelArtifact(tuple(features), tuple(map(float, means)), tuple(map(float, scales)), tuple(map(float, coefs)),
                         float(intercept), trained_through, 1.0, 100, 'dec-fixture', horizon)


# Frozen synthetic models with hand-computed predictions (see oracle_score).
M_FUND = artifact(('revenue_yoy', 'fcf_margin'), (.125, .125), (.125, .125), (.5, .25), .0625)
M_JOINT = artifact(('revenue_yoy', 'technical_signal'), (.125, .5), (.125, .5), (.5, .25), .0625)
M_MACRO = artifact(('macro_unemployment', 'technical_signal'), (4.0, .5), (1.0, .5), (.25, .5), .125)
M_MISSING = artifact(('revenue_yoy', 'gross_margin_stability'), (0., 0.), (1., 1.), (1., 1.), 0.)   # needs 3 yrs of margins


def oracle_score(model, inputs):
    """score = intercept + sum_i coef_i * (x_i - mean_i) / scale_i using exact rational arithmetic."""
    total = Fraction(model.intercept)
    for name, mean, scale, coef in zip(model.features, model.means, model.scales, model.coefficients):
        total += Fraction(coef) * (Fraction(inputs[name]) - Fraction(mean)) / Fraction(scale)
    return total


# exact feature values of the fixture companies (derived by hand from company_facts arguments)
GOOD_INPUTS = {'revenue_yoy': Fraction(160, 128) - 1, 'fcf_margin': Fraction(40, 160), 'macro_unemployment': Fraction(5)}
BAD_INPUTS = {'revenue_yoy': Fraction(120, 160) - 1, 'fcf_margin': Fraction(15, 120), 'macro_unemployment': Fraction(5)}


def engines(store, model=None, *, policy='filter', threshold=0., shadow=True, macro=True, catalysts=True, half_life=None,
            max_period_age_days=550, macro_age=100):
    feat = FundamentalFeatureEngine(store, max_period_age_days=max_period_age_days)
    return CombinedSignalEngine(feat, FundamentalScoreEngine(model) if model else None,
                                MacroRegimeService(store, max_vintage_age=macro_age) if macro else None,
                                EventCatalystEngine(store, half_life_days=half_life) if catalysts else None,
                                policy=policy, threshold=threshold, research_shadow=shadow)


def add_earnings(store, sym='GOOD', *, eps=1.10, consensus=1.00, event_at='2024-02-20T21:00:00Z',
                 available_at='2024-02-20T21:05:00Z', consensus_at='2024-02-01T00:00:00Z', key='Q4-2023', source='fixture'):
    return store.add_event(symbol=sym, event_type='earnings', event_at=event_at, reported_at=event_at,
                           available_at=available_at, source=source, revision='A',
                           payload=dict(reported_eps=eps, consensus_eps=consensus, consensus_available_at=consensus_at, event_key=key))


def bars(n=60, start='2024-03-04T04:00:00Z', breakout_at=45):
    """Deterministic daily bars at 04:00 UTC. Flat noise then a decisive Donchian-20 breakout at index breakout_at."""
    idx = pd.date_range(start, periods=n, freq='D', tz='UTC')
    rows = []
    for i, ts in enumerate(idx):
        base = 100. + (0.4 if i % 2 else -0.4)
        o = base
        if i >= breakout_at:
            base = 100. + 3. * (i - breakout_at + 1)
            o = base - 1.
        rows.append(dict(ts=ts, open=o, high=base + 1., low=min(o, base) - 1., close=base, volume=1_000_000))
    return pd.DataFrame(rows)
