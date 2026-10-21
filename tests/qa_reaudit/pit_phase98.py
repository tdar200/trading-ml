"""Shared builders for the PHASE 98 tests (QA track A): full database with canary-carrying future rows vs truncated database."""
from __future__ import annotations
import json
import random
from dataclasses import replace
from datetime import timedelta

from tests.qa_reaudit import pit_oracle as O

from fundamentals.api import FundamentalAPI
from fundamentals.features import FundamentalFeatureEngine
from fundamentals.models import (CombinedSignalEngine, EventCatalystEngine, FundamentalScoreEngine, MacroRegimeService,
                                 ModelArtifact)
from fundamentals.store import Fact, PointInTimeFundamentalStore


def make_model():
    return ModelArtifact(('revenue_yoy', 'gross_margin', 'technical_signal'), (0.1, 0.4, 0.5), (0.2, 0.1, 0.5),
                         (0.01, 0.02, 0.005), 0.001, '2000-01-01T00:00:00Z', 1.0, 100, 'x' * 8, 5)


def future_canary_extra(company, T, rng):
    """Facts published AFTER T carrying the canary: restatements of every visible key + brand-new periods."""
    win = O.visible_winners(company.facts, T)
    extra = []
    for key, w in win.items():
        if rng.random() < 0.5:
            extra.append(Fact(company.symbol, w.metric, O.CANARY if w.metric not in ('diluted_shares',) else O.CANARY,
                              w.end, (T + timedelta(microseconds=rng.randint(1, 10**9))).isoformat(),
                              (T + timedelta(microseconds=rng.randint(10**9, 10**13))).isoformat(), 'SRC', 'canary-restatement',
                              '' if w.metric == 'diluted_shares' else 'USD',
                              'shares' if w.metric == 'diluted_shares' else 'USD/shares' if w.metric == 'eps' else 'USD',
                              w.start, O.OBSERVED, 'financial', {'share_basis': w.basis} if w.basis else {}))
    return extra


def build_pair(seed, T, full_path=':memory:', trunc_path=':memory:'):
    company = O.gen_company(seed)
    rng = random.Random(seed * 101 + 7)
    macro = O.gen_macro(seed)
    events = O.gen_events(seed, company.facts)
    # --- full DB: everything, with canary in every future filing/vintage/event
    future_facts = [f for f in company.facts if f.available > T]
    canary_facts = []
    for f in future_facts:
        if rng.random() < 0.6:
            canary_facts.append(replace(f, value=O.CANARY if f.metric != 'diluted_shares' else O.CANARY))
        else:
            canary_facts.append(f)
    # canary replaces values; for scaled companies keep plain-USD rendering for these (identical normalized key)
    scaled = company.scale_label
    full = PointInTimeFundamentalStore(full_path)
    r1 = random.Random(seed * 3 + 1)
    full_rows = [O.to_fact(f, company.symbol, scaled, r1) for f in company.facts if f.available <= T]
    for f in canary_facts:
        full_rows.append(O.to_fact(f, company.symbol, None if f.value == O.CANARY else scaled, r1))
    full_rows += future_canary_extra(company, T, rng)
    full_rows += O.macro_facts(macro, canary_after=T)
    r1.shuffle(full_rows)
    for i in range(3):
        full.ingest(full_rows[i::3])
    # a rejected record published after T (quarantined, must not leak into decisions)
    full.ingest_validated([Fact(company.symbol, 'revenue', O.CANARY, '2020-12-31', '2030-01-01T00:00:00Z', '2030-01-01T00:00:00Z', 'SRC', 'bad',
                                'XXX', 'XXX', '2020-01-01', O.OBSERVED, 'financial', {})])
    for e in events:
        payload = dict(e['payload'])
        if e['available_at'] > T:
            payload.update(reported_eps=O.CANARY, canary=O.CANARY)
        full.add_event(symbol=e['symbol'], event_type=e['event_type'], event_at=e['event_at'].isoformat(), reported_at=e['reported_at'].isoformat(),
                       available_at=e['available_at'].isoformat(), source=e['source'], revision=e['revision'], payload=payload)
    # --- truncated DB: physically only rows with available_at <= T
    trunc = PointInTimeFundamentalStore(trunc_path)
    r2 = random.Random(seed * 5 + 9)
    rows = [O.to_fact(f, company.symbol, scaled, r2) for f in company.facts if f.available <= T]
    rows += O.macro_facts([m for m in macro if m['available'] <= T])
    r2.shuffle(rows)
    for i in range(2):
        trunc.ingest(rows[i::2])
    for e in events:
        if e['available_at'] <= T:
            trunc.add_event(symbol=e['symbol'], event_type=e['event_type'], event_at=e['event_at'].isoformat(), reported_at=e['reported_at'].isoformat(),
                            available_at=e['available_at'].isoformat(), source=e['source'], revision=e['revision'], payload=e['payload'])
    return company, full, trunc


def outputs(store, symbol, T, policy):
    feats = FundamentalFeatureEngine(store)
    combo = CombinedSignalEngine(feats, FundamentalScoreEngine(make_model()), MacroRegimeService(store), EventCatalystEngine(store, 30.),
                                 policy=policy, threshold=0.0, research_shadow=True)
    api = FundamentalAPI(store, FundamentalScoreEngine(make_model()))
    ts = T.isoformat()
    q = {'as_of': ts}
    out = {
        'snapshot': feats.snapshot(symbol, ts),
        'decision': combo.decide(symbol, ts, {'signal': True}),
        'decision_nosignal': combo.decide(symbol, ts, {'signal': False}),
        'api_snapshot': api.get('/fundamentals/' + symbol, q),
        'api_ttm': api.get('/fundamentals/' + symbol + '/ttm', q),
        'api_history': api.get('/fundamentals/' + symbol + '/history', q),
        'api_history_rev': api.get('/fundamentals/' + symbol + '/history', {**q, 'metric': 'revenue'}),
        'api_score': api.get('/fundamentals/' + symbol + '/score', q),
        'api_events': api.get('/fundamentals/' + symbol + '/events', q),
        'api_watchlist': api.get('/watchlist/fundamental', q),
        'api_macro': api.get('/macro/regime', q),
        'api_explain': api.get('/strategies/x/explanation', {**q, 'symbol': symbol, 'asset_class': 'equity', 'technical_signal': 'true'}),
        'download_json': __import__('json').dumps(feats.snapshot(symbol, ts), indent=2, allow_nan=False),
    }
    return out


