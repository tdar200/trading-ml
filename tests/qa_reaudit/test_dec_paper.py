"""QA track E: PaperEngine + fundamental_filter + RiskEngine, end to end (phases 120-123, 156, 160-161).

Run: .venv/bin/python -m pytest -p no:cacheprovider -q tests/qa_reaudit/test_dec_paper.py
"""
from __future__ import annotations
import json
import math
import random
import sqlite3
from decimal import Decimal as D, ROUND_FLOOR

import pandas as pd
import pytest

from fundamentals.features import FundamentalFeatureEngine
from fundamentals.models import ModelArtifact, FundamentalScoreEngine, CombinedSignalEngine, MacroRegimeService, EventCatalystEngine
from fundamentals.store import PointInTimeFundamentalStore
from livesim.small_account import PaperEngine
from livesim.small_account_risk import RiskLimits
from strategies.small_account import features as tech_features, Setup
from tests.qa_reaudit import dec_fixtures as F

M_HUGE = F.artifact(('revenue_yoy',), (.25,), (.125,), (.5,), 1e6)           # score = 1,000,000 for GOOD
SETUP = Setup('donchian', 20)


@pytest.fixture(autouse=True)
def _isolated_kill_switch_environment(monkeypatch, tmp_path):
    """RiskEngine reads STOP_TRADING / STOP_TRADING_FILE / TRADING_MODE from the process environment: pin them for determinism."""
    monkeypatch.delenv('STOP_TRADING', raising=False)
    monkeypatch.delenv('TRADING_MODE', raising=False)
    monkeypatch.setenv('STOP_TRADING_FILE', str(tmp_path / 'absent-stop-file'))


def ts(s):
    return pd.Timestamp(s)


def prev_bar(t='2024-04-18T04:00:00Z', distance=2., signal=True):
    return dict(ts=ts(t), signal=signal, distance=distance, regime='trend_up')


def bar(t='2024-04-19T04:00:00Z', o=100., h=101., l=99., c=100.):
    return dict(ts=ts(t), open=o, high=h, low=l, close=c)


def mk_engine(store, model=F.M_JOINT, *, symbol='GOOD', policy='filter', threshold=0., shadow_combo=True, half_life=5.,
              database=None, setup=SETUP, macro=True, **kw):
    combo = F.engines(store, model, policy=policy, threshold=threshold, shadow=shadow_combo, half_life=half_life, macro=macro)
    kw.setdefault('shadow', True)
    return PaperEngine(setup, symbol=symbol, fundamental_filter=combo, database=database, **kw), combo


def events(e, kind):
    return [x for x in e.journal if x['event'] == kind]


def fresh_store():
    s = F.build_store()
    F.add_earnings(s)
    return s


# ---------------------------------------------------------------- independent RiskEngine arithmetic (limits read from small_account_risk.py)
RISK_PER_TRADE, MAX_EXPOSURE, STEP = D('0.005'), D('0.25'), D('0.000001')


def oracle_entry(open_, spread_bps, slip_bps):
    return D(str(open_)) * (1 + (D(str(spread_bps)) / 2 + D(str(slip_bps))) / 10000)


def oracle_qty(equity, cash, open_, distance, fee_bps, spread_bps, slip_bps, scale=1):
    fee = D(str(fee_bps)) / 10000
    entry = oracle_entry(open_, spread_bps, slip_bps)
    stop = entry - D(str(distance))
    exit_price = stop * (1 - (D(str(spread_bps)) / 2 + D(str(slip_bps))) / 10000)
    loss = entry * (1 + fee) - exit_price * (1 - fee)
    q = min(D(str(equity)) * RISK_PER_TRADE * scale / loss, D(str(cash)) / (entry * (1 + fee)), D(str(equity)) * MAX_EXPOSURE / entry)
    return (q / STEP).to_integral_value(rounding=ROUND_FLOOR) * STEP, entry, stop


# ================================================================ phase 156: paper E2E
def run_series(e, frame, setup=SETUP, close_final=True):
    recs = tech_features(frame, setup).to_dict('records')
    for i in range(1, len(recs)):
        e.step(recs[i], recs[i - 1], close_final=close_final and i == len(recs) - 1)
    return recs


def test_e2e_gbp100_strong_fundamentals_breakout_macro_catalyst_to_equity_curve():
    s = fresh_store()
    e, combo = mk_engine(s, F.M_JOINT, policy='regime', spread_bps=2., slip_bps=3., fee_bps=0.)
    recs = run_series(e, F.bars())
    entries, exits, decisions, risks = events(e, 'ENTRY'), events(e, 'EXIT'), events(e, 'FUNDAMENTAL_DECISION'), events(e, 'RISK')
    assert len(entries) == len(exits) == len(decisions) == len(risks) == 3
    # first entry: signal bar index 45 (breakout), execution bar 46 at 2024-04-19T04:00Z
    sig = recs[45]
    assert sig['signal'] is True or bool(sig['signal'])
    first = entries[0]
    assert first['timestamp'] == '2024-04-19 04:00:00+00:00' and first['signal_time'] == '2024-04-18 04:00:00+00:00'
    # independent sizing: GBP 100 equity and cash, 2bp spread + 3bp slippage, zero fee
    q, entry, stop = oracle_qty(100, 100, recs[46]['open'], sig['distance'], 0, 2., 3.)
    assert first['quantity'] == pytest.approx(float(q), abs=1e-6) and first['price'] == pytest.approx(float(entry), rel=1e-12)
    assert first['stop'] == pytest.approx(float(stop), rel=1e-12)
    assert first['target'] == pytest.approx(float(entry + 2 * D(str(sig['distance']))), rel=1e-12)
    # risk budget respected: expected_risk <= 0.5% of equity, and within one quantity step of it
    assert first['expected_risk'] <= 100 * .005 + 1e-12 and 100 * .005 - first['expected_risk'] < 5e-6 * (float(entry - stop) + 1)
    assert first['quantity'] * first['price'] <= 100 * .25 + 1e-9                        # exposure cap
    # decision comes first, then RISK, then ENTRY in the journal order
    names = [x['event'] for x in e.journal[:3]]
    assert names == ['FUNDAMENTAL_DECISION', 'RISK', 'ENTRY']
    # the decision object holds every required piece
    d = decisions[0]['decision']
    assert d['permit_signal'] is True and d['decision'] == 'RESEARCH_CANDIDATE' and d['policy'] == 'regime'
    assert d['fundamental']['score'] == .8125                                              # M_JOINT z=(1,1): .0625+.5+.25
    assert d['fundamental']['raw']['revenue'] == 160. and d['fundamental']['raw']['ocf'] == 48.
    assert d['fundamental']['features']['revenue_yoy'] == .25 and d['fundamental']['confidence'] is None
    assert d['macro']['status'] == 'AVAILABLE' and d['macro']['regime']['policy'] == 'UNCHANGED'
    assert d['catalysts']['events'][0]['eps_surprise'] == pytest.approx(.1, rel=1e-12) and d['catalysts']['events'][0]['event_type'] == 'earnings'
    assert d['technical'] == {'signal': True}
    assert d['risk'] == {'status': 'APPROVED_SHADOW', 'quantity': first['quantity'], 'flags': d['fundamental']['flags']}
    assert d['positionSize'] == first['quantity']
    assert len(d['data_snapshot_id']) == 64 and d['configuration']['policy'] == 'regime'
    assert d['timestamp'] == '2024-04-19T04:00:00+00:00'
    # paper fill, P&L and equity curve against an independent ledger
    cash = D(100)
    for en, ex in zip(entries, exits):
        cost = D(str(en['quantity'])) * D(str(en['price'])) + D(str(en['fee']))
        proceeds = D(str(ex['quantity'])) * D(str(ex['price'])) - D(str(ex['fee']))
        assert float(proceeds - cost) == pytest.approx(ex['pnl'], abs=1e-9)
        # target exit fills at target less half spread and slippage (4 bp)
        assert ex['reason'] == 'target' and ex['price'] == pytest.approx(ex['target'] * (1 - 4 / 10000), rel=1e-12)
        cash += proceeds - cost
    assert e.state['position'] is None
    assert e.state['cash'] == pytest.approx(float(cash), abs=1e-9) and e.state['equity'] == pytest.approx(float(cash), abs=1e-9)
    assert len(e.curve) == len(recs) - 1
    assert [c['ts'] for c in e.curve] == sorted(c['ts'] for c in e.curve)
    assert e.curve[-1]['equity'] == pytest.approx(float(cash), abs=1e-9)
    assert float(cash) > 100.0


def test_e2e_same_setup_but_each_hard_limit_tripped_gives_no_trade():
    """Every case: fundamentals/technicals/macro/catalyst all say go (permit True, huge score) and no order is ever placed."""
    results = {}
    frame = F.bars()
    recs = tech_features(frame, SETUP).to_dict('records')
    for name, kw in {'kill_param': dict(kill=True), 'stale_feed': dict(age_seconds=91.), 'clock_skew': dict(clock_skew=6.),
                     'unhealthy': dict(healthy=False), 'inconsistent': dict(consistent=False), 'unknown_order': dict(status_known=False)}.items():
        s = fresh_store()
        e, _ = mk_engine(s, M_HUGE, spread_bps=2., slip_bps=3.)
        for i in range(1, len(recs)):
            e.step(recs[i], recs[i - 1], **kw)
        results[name] = (len(events(e, 'ENTRY')), e.state['cash'], [x['decision']['permit_signal'] for x in events(e, 'FUNDAMENTAL_DECISION')])
    for name, (n_entries, cash, permits) in results.items():
        assert n_entries == 0 and cash == 100., name
        assert len(permits) >= 3 and all(permits), name          # the fundamental/technical/macro side said go on every breakout bar


# ================================================================ hard limits win over fundamentals
def _bar46(e, prev=None, **kw):
    return e.step(bar(), prev or prev_bar(), **kw)


@pytest.mark.parametrize('case', ['kill_param', 'stop_env', 'stop_file', 'live_mode', 'unhealthy', 'inconsistent', 'unknown_status',
                                  'stale', 'skew', 'spread', 'slippage', 'min_notional', 'not_shadow'])
def test_hard_limits_beat_permit_signal_and_a_score_of_one_million(case, monkeypatch, tmp_path):
    monkeypatch.delenv('STOP_TRADING', raising=False)
    monkeypatch.setenv('STOP_TRADING_FILE', str(tmp_path / 'absent'))
    s = fresh_store()
    ekw, skw = {}, {}
    expected = {'kill_param': 'kill switch', 'stop_env': 'kill switch', 'stop_file': 'kill switch', 'live_mode': 'live execution unavailable',
                'unhealthy': 'feed/broker unavailable', 'inconsistent': 'account inconsistent', 'unknown_status': 'unknown order status',
                'stale': 'stale data or clock skew', 'skew': 'stale data or clock skew', 'spread': 'execution cost limit',
                'slippage': 'execution cost limit', 'min_notional': 'below economical minimum', 'not_shadow': 'no validated edge'}[case]
    if case == 'kill_param': skw['kill'] = True
    if case == 'stop_env': monkeypatch.setenv('STOP_TRADING', 'true')
    if case == 'stop_file':
        (tmp_path / 'STOP').write_text('x')
        monkeypatch.setenv('STOP_TRADING_FILE', str(tmp_path / 'STOP'))
    if case == 'live_mode': monkeypatch.setenv('TRADING_MODE', 'LIVE')
    if case == 'unhealthy': skw['healthy'] = False
    if case == 'inconsistent': skw['consistent'] = False
    if case == 'unknown_status': skw['status_known'] = False
    if case == 'stale': skw['age_seconds'] = 90.001
    if case == 'skew': skw['clock_skew'] = 5.001
    if case == 'spread': ekw['spread_bps'] = 20.5
    if case == 'slippage': ekw['slip_bps'] = 20.5
    if case == 'min_notional': ekw['limits'] = RiskLimits(min_notional=50.)
    if case == 'not_shadow': ekw['shadow'] = False
    e, _ = mk_engine(s, M_HUGE, **ekw)
    out = _bar46(e, **skw)
    d = events(e, 'FUNDAMENTAL_DECISION')[0]['decision']
    assert d['permit_signal'] is True and d['fundamental']['score'] == 1e6          # fundamentals said yes, loudly
    assert e.state['position'] is None and e.state['cash'] == 100. and not events(e, 'ENTRY')
    assert expected in events(e, 'RISK')[0]['reason'] and events(e, 'RISK')[0]['quantity'] == 0.
    assert d['risk']['quantity'] == 0. and expected in d['risk']['status']
    assert out == 'RISK'


def test_boundaries_just_inside_the_limits_still_trade(monkeypatch, tmp_path):
    monkeypatch.delenv('STOP_TRADING', raising=False)
    monkeypatch.setenv('STOP_TRADING_FILE', str(tmp_path / 'absent'))
    for skw, ekw in ((dict(age_seconds=90.), {}), (dict(clock_skew=5.), {}), ({}, dict(spread_bps=20.)), ({}, dict(slip_bps=20.))):
        s = fresh_store()
        e, _ = mk_engine(s, M_HUGE, **ekw)
        _bar46(e, **skw)
        assert len(events(e, 'ENTRY')) == 1, (skw, ekw)


@pytest.mark.parametrize('limit', ['daily', 'weekly', 'drawdown'])
def test_loss_limits_block_reentry_even_when_fundamentals_permit(limit):
    """Intraday bars: first trade stops out on its entry bar (-0.5%), the next signal is vetoed by the loss limit."""
    s = fresh_store()
    lim = {'daily': RiskLimits(max_daily_loss=.004), 'weekly': RiskLimits(max_weekly_loss=.004), 'drawdown': RiskLimits(max_drawdown=.004)}[limit]
    e, _ = mk_engine(s, M_HUGE, limits=lim, spread_bps=2., slip_bps=3.)
    e.step(bar('2024-04-19T14:00:00Z', o=100., h=101., l=95., c=96.), prev_bar('2024-04-19T13:00:00Z'))
    assert len(events(e, 'EXIT')) == 1 and events(e, 'EXIT')[0]['reason'] == 'entry_bar_stop'
    loss = events(e, 'EXIT')[0]['pnl']
    assert loss < -.4 and e.state['equity'] == pytest.approx(100 + loss, abs=1e-9)
    n = len(events(e, 'ENTRY'))
    e.step(bar('2024-04-19T15:00:00Z', o=96., h=97., l=95., c=96.), prev_bar('2024-04-19T14:00:00Z'))
    d = events(e, 'FUNDAMENTAL_DECISION')[-1]['decision']
    assert d['permit_signal'] is True and d['fundamental']['score'] == 1e6
    assert len(events(e, 'ENTRY')) == n and e.state['position'] is None
    reason = events(e, 'RISK')[-1]['reason']
    assert reason.startswith('NO_TRADE: ') and ('loss limit' in reason or 'edge degradation' in reason), reason


# ================================================================ gate semantics, order of operations, fail closed
def test_gate_false_means_zero_quantity_and_decision_still_journaled():
    for sym, why in (('BAD', 'prediction below threshold'), ('OLD', 'stale'), ('NONE', 'Missing')):
        s = fresh_store()
        e, _ = mk_engine(s, F.M_FUND, symbol=sym)
        e.step(bar(), prev_bar())
        d = events(e, 'FUNDAMENTAL_DECISION')[0]['decision']
        assert d['permit_signal'] is False and d['decision'] == 'NO_TRADE' and d['positionSize'] == 0.
        assert events(e, 'RISK')[0]['reason'] == 'NO_TRADE: fundamental decision gate' and events(e, 'RISK')[0]['quantity'] == 0.
        assert not events(e, 'ENTRY') and e.state['cash'] == 100. and e.state['processed_signals'] == []
        assert why.lower() in ' '.join(d['explanation']).lower(), (sym, d['explanation'])


def test_decision_is_computed_before_risk_and_written_even_when_risk_rejects():
    s = fresh_store()
    e, _ = mk_engine(s, F.M_FUND)
    e.step(bar(), prev_bar(), kill=True)
    assert [x['event'] for x in e.journal] == ['FUNDAMENTAL_DECISION', 'RISK']
    d = e.journal[0]['decision']
    assert d['permit_signal'] is True and d['risk']['status'] == 'NO_TRADE: kill switch' and d['risk']['quantity'] == 0.


def test_hard_limit_reason_must_not_be_masked_by_fundamental_gate():
    s = fresh_store()
    e, _ = mk_engine(s, F.M_FUND, symbol='BAD')
    e.step(bar(), prev_bar(), kill=True)
    assert 'kill switch' in events(e, 'RISK')[0]['reason']


class _Boom:
    """sqlite connection stand-in that fails every statement (locked / unavailable / corrupt database)."""
    def __init__(self, real, exc):
        self.real, self.exc = real, exc
    def execute(self, *a, **k):
        raise self.exc
    def __getattr__(self, name):
        return getattr(self.real, name)


@pytest.mark.parametrize('exc', [sqlite3.OperationalError('database is locked'), sqlite3.DatabaseError('file is not a database'),
                                 sqlite3.ProgrammingError('Cannot operate on a closed database.'), ValueError('ambiguous conflicting source facts')])
@pytest.mark.parametrize('persist', [False, True])
def test_exception_inside_decide_fails_closed_state_untouched_then_recovers(exc, persist, tmp_path):
    s = fresh_store()
    e, _ = mk_engine(s, F.M_JOINT, database=(tmp_path / 'j.sqlite') if persist else None)
    snap = (json.dumps(e.state, sort_keys=True), len(e.journal), len(e.curve))
    real = s.db
    s.db = _Boom(real, exc)
    with pytest.raises(type(exc)):
        e.step(bar(), prev_bar())
    s.db = real
    assert (json.dumps(e.state, sort_keys=True), len(e.journal), len(e.curve)) == snap
    assert e.state['cash'] == 100. and e.state['position'] is None and e.state['last_ts'] is None
    if persist:
        db = sqlite3.connect(tmp_path / 'j.sqlite')
        assert db.execute('select count(*) from events').fetchone()[0] == 0 and db.execute('select count(*) from equity').fetchone()[0] == 0
        assert not e.db.in_transaction
    # the same bar can be processed after recovery: nothing was half-applied and no signal id was burned
    assert e.step(bar(), prev_bar()) == 'ENTRY' and e.state['position'] is not None


def test_closed_fundamental_database_stops_trading_instead_of_defaulting():
    s = fresh_store()
    e, _ = mk_engine(s, F.M_JOINT)
    s.close()
    with pytest.raises(sqlite3.ProgrammingError):
        e.step(bar(), prev_bar())
    assert e.state['position'] is None and e.state['cash'] == 100. and not e.journal


def test_corrupt_fundamental_database_file_cannot_become_a_default_score(tmp_path):
    p = tmp_path / 'corrupt.sqlite'
    p.write_bytes(b'this is not an sqlite database' * 200)
    with pytest.raises(sqlite3.DatabaseError):
        PointInTimeFundamentalStore(p)
    ro = PointInTimeFundamentalStore(tmp_path / 'good.sqlite')
    ro.ingest(F.company_facts('GOOD', rev_prior=128., rev=160., ocf=48., capex=8.))
    ro.close()
    PointInTimeFundamentalStore(tmp_path / 'good.sqlite', readonly=True).close()
    with pytest.raises(sqlite3.Error):
        PointInTimeFundamentalStore(tmp_path / 'missing.sqlite', readonly=True)


def test_empty_fundamental_database_means_no_trade_not_a_default_score():
    s = PointInTimeFundamentalStore()
    e, _ = mk_engine(s, F.M_FUND, macro=False)
    e.step(bar(), prev_bar())
    d = events(e, 'FUNDAMENTAL_DECISION')[0]['decision']
    assert d['fundamental']['score'] is None and d['fundamental']['status'] == 'MISSING_FEATURES' and d['permit_signal'] is False
    assert d['fundamental']['data_status'] == 'MISSING' and e.state['cash'] == 100. and not events(e, 'ENTRY')


def test_model_trained_after_decision_time_raises_and_never_trades():
    s = fresh_store()
    late = F.artifact(('revenue_yoy',), (.25,), (.125,), (.5,), .5, trained_through='2025-01-01T00:00:00Z')
    e, _ = mk_engine(s, late)
    with pytest.raises(ValueError, match='historical'):
        e.step(bar(), prev_bar())
    assert e.state['cash'] == 100. and not e.journal


def test_observe_policy_is_trade_for_trade_identical_to_no_filter():
    frame = F.bars()
    out = []
    for use_filter in (False, True):
        s = fresh_store()
        if use_filter:
            e, _ = mk_engine(s, None, policy='observe', shadow_combo=False, spread_bps=2., slip_bps=3.)
        else:
            e = PaperEngine(SETUP, symbol='GOOD', shadow=True, spread_bps=2., slip_bps=3.)
        run_series(e, frame)
        out.append([(x['event'], x['timestamp'], x.get('price'), x.get('quantity'), x.get('pnl')) for x in e.journal if x['event'] in ('ENTRY', 'EXIT', 'RISK')])
        out[-1].append(e.state['equity'])
    assert out[0] == out[1] and len(out[0]) > 5


def test_fundamental_risk_flags_are_recorded_but_do_not_change_size():
    """LIMITATION: NEGATIVE_FCF etc. are descriptive only. A negative-FCF company with a high score gets the same size."""
    s = fresh_store()
    s.ingest(F.company_facts('NEG', rev_prior=128., rev=160., ocf=5., capex=8.))          # FCF = -3
    sizes = {}
    for sym in ('GOOD', 'NEG'):
        e, _ = mk_engine(s, F.M_JOINT, symbol=sym)
        e.step(bar(), prev_bar())
        d = events(e, 'FUNDAMENTAL_DECISION')[0]['decision']
        sizes[sym] = (events(e, 'ENTRY')[0]['quantity'], d['risk']['flags'], d['fundamental']['features']['fcf'])
    assert sizes['GOOD'][1] == [] and sizes['NEG'][1] == ['NEGATIVE_FCF'] and sizes['NEG'][2] == -3.
    assert sizes['GOOD'][0] == sizes['NEG'][0] > 0


def test_decision_position_size_is_the_approved_size_not_the_filled_size():
    """LIMITATION: FUNDAMENTAL_DECISION is written before fills; partial IOC / venue rejection appear only in later events."""
    s = fresh_store()
    e, _ = mk_engine(s, F.M_JOINT)
    e.step(bar(), prev_bar(), fill_fraction=.5)
    d = events(e, 'FUNDAMENTAL_DECISION')[0]['decision']
    filled = events(e, 'ENTRY')[0]['quantity']
    assert d['positionSize'] == events(e, 'RISK')[0]['quantity'] > filled > 0
    assert events(e, 'PARTIAL_IOC')[0]['filled_quantity'] == filled
    s2 = fresh_store()
    e2, _ = mk_engine(s2, F.M_JOINT)
    e2.step(bar(), prev_bar(), order_rejected=True)
    assert e2.journal[0]['decision']['positionSize'] > 0 and not events(e2, 'ENTRY') and events(e2, 'ORDER_REJECTED')


def test_two_shadow_flags_are_independent_and_both_required():
    s = fresh_store()
    e, _ = mk_engine(s, F.M_JOINT, shadow_combo=True, shadow=False)           # combo shadow, engine not
    e.step(bar(), prev_bar())
    assert events(e, 'FUNDAMENTAL_DECISION')[0]['decision']['permit_signal'] is True and 'no validated edge' in events(e, 'RISK')[0]['reason']
    e2, _ = mk_engine(s, F.M_JOINT, shadow_combo=False, shadow=True)          # engine shadow, combo not
    e2.step(bar(), prev_bar())
    assert events(e2, 'FUNDAMENTAL_DECISION')[0]['decision']['permit_signal'] is False
    assert events(e2, 'FUNDAMENTAL_DECISION')[0]['decision']['explanation'] == ['No fundamental model approved for deployment']
    assert not events(e, 'ENTRY') and not events(e2, 'ENTRY')


# ================================================================ same-timestamp filing and candle
def test_filing_at_exactly_the_bar_timestamp_is_visible_one_microsecond_later_is_not():
    """Bar ts = 04:00:00 UTC (decision as_of). A filing available_at == ts is used (<=), one published a microsecond later is not."""
    for avail, expect_new in (('2024-04-19T04:00:00.000000Z', True), ('2024-04-19T04:00:00.000001Z', False)):
        s = fresh_store()
        # restatement: FY2023 revenue 160 -> 100 (growth -22%), model score would flip sign
        s.ingest([F._fact('GOOD', 'revenue', 100., '2023-12-31', '2023-01-01', avail, 'GOOD-2023-restated')])
        e, _ = mk_engine(s, F.M_JOINT)
        e.step(bar(), prev_bar())
        d = events(e, 'FUNDAMENTAL_DECISION')[0]['decision']
        want = 100. / 128. - 1 if expect_new else .25
        assert d['fundamental']['features']['revenue_yoy'] == pytest.approx(want, rel=1e-12), avail
        assert d['permit_signal'] is (not expect_new)


def test_future_canary_filing_never_reaches_an_earlier_decision():
    canary = 123456789123.
    s = fresh_store()
    s.ingest([F._fact('GOOD', 'revenue', canary, '2023-12-31', '2023-01-01', '2024-04-19T04:00:00.000001Z', 'canary')])
    s.add_event(symbol='GOOD', event_type='earnings', event_at='2024-04-20T00:00:00Z', reported_at='2024-04-20T00:00:00Z',
                available_at='2024-04-20T00:00:05Z', source='fixture', revision='B',
                payload=dict(reported_eps=canary, consensus_eps=1., consensus_available_at='2024-04-01T00:00:00Z', event_key='canary'))
    e, _ = mk_engine(s, F.M_JOINT)
    e.step(bar(), prev_bar())
    blob = json.dumps(e.journal, default=str)
    assert '123456789123' not in blob and 'canary' not in blob
    # and the same canary appears once its availability time is reached
    later = F.engines(s, F.M_JOINT, policy='filter').decide('GOOD', '2024-04-21T04:00:00Z', {'signal': True})
    assert '123456789123' in json.dumps(later, default=str)


# ================================================================ audit trail from the journal alone, versions, restart
def _persisted_run(tmp_path, **kw):
    s = fresh_store()
    e, combo = mk_engine(s, F.M_JOINT, policy='regime', database=tmp_path / 'paper.sqlite', spread_bps=2., slip_bps=3., **kw)
    run_series(e, F.bars())
    e.db.close()
    return s, combo


def test_audit_trail_reconstructs_why_we_bought_from_the_journal_alone(tmp_path):
    s, combo = _persisted_run(tmp_path)
    db = sqlite3.connect(f'file:{tmp_path / "paper.sqlite"}?mode=ro', uri=True)           # no access to the fundamental store
    rows = [json.loads(p) for (p,) in db.execute('SELECT payload FROM events ORDER BY seq')]
    entry = next(r for r in rows if r['event'] == 'ENTRY')
    t = entry['timestamp']
    chain = [r for r in rows if r['timestamp'] == t]
    assert [r['event'] for r in chain] == ['FUNDAMENTAL_DECISION', 'RISK', 'ENTRY']
    d, risk = chain[0]['decision'], chain[1]
    # why: data known, score, events, technical signal, risk decision
    assert d['fundamental']['period_end'] == '2023-12-31' and d['fundamental']['available_at'] == '2024-02-20T21:00:00+00:00'
    assert pd.Timestamp(d['fundamental']['available_at']) <= pd.Timestamp(d['timestamp'])
    assert d['fundamental']['raw']['revenue'] == 160. and d['fundamental']['features']['revenue_yoy'] == .25 and d['fundamental']['lineage']
    assert d['fundamental']['score'] == .8125 and d['configuration']['threshold'] == 0. and d['fundamental']['score'] > d['configuration']['threshold']
    assert d['technical']['signal'] is True and risk['signal_time'] == entry['signal_time']
    assert d['catalysts']['events'][0]['eps_surprise'] == pytest.approx(.1, rel=1e-12) and d['macro']['status'] == 'AVAILABLE'
    assert d['risk']['status'] == risk['reason'] == 'APPROVED_SHADOW' and d['risk']['quantity'] == risk['quantity'] == entry['quantity']
    # explanatory text agrees with the machine state it describes
    assert d['permit_signal'] is True and d['decision'] == 'RESEARCH_CANDIDATE'
    assert d['explanation'] == ['Fitted research prediction exceeds frozen threshold']
    for r in rows:
        if r['event'] == 'FUNDAMENTAL_DECISION':
            x = r['decision']
            above = x['fundamental']['score'] is not None and x['fundamental']['score'] > x['configuration']['threshold']
            says_exceeds = any('exceeds' in line for line in x['explanation'])
            assert says_exceeds == (x['permit_signal'] and above and x['technical']['signal'])
    # every entry has exactly one preceding decision with identical quantity
    for r in (r for r in rows if r['event'] == 'ENTRY'):
        pre = [c for c in rows if c['event'] == 'FUNDAMENTAL_DECISION' and c['timestamp'] == r['timestamp']]
        assert len(pre) == 1 and pre[0]['decision']['risk']['quantity'] == r['quantity']


def test_decision_reproducible_from_recorded_configuration_and_snapshot_id(tmp_path):
    s, _ = _persisted_run(tmp_path)
    db = sqlite3.connect(f'file:{tmp_path / "paper.sqlite"}?mode=ro', uri=True)
    decisions = [json.loads(p)['decision'] for (p,) in db.execute("SELECT payload FROM events WHERE payload LIKE '%FUNDAMENTAL_DECISION%' ORDER BY seq")]
    assert len(decisions) == 3
    for d in decisions:
        cfg = d['configuration']
        model = ModelArtifact(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg['model'].items()})
        combo = CombinedSignalEngine(FundamentalFeatureEngine(s, max_period_age_days=cfg['max_period_age_days']), FundamentalScoreEngine(model),
                                     MacroRegimeService(s, max_vintage_age=cfg['macro_max_age']), EventCatalystEngine(s, half_life_days=cfg['catalyst_half_life']),
                                     policy=cfg['policy'], threshold=cfg['threshold'], research_shadow=cfg['research_shadow'])
        again = combo.decide('GOOD', d['timestamp'], {'signal': d['technical']['signal']})
        assert again['data_snapshot_id'] == d['data_snapshot_id']
        assert again['fundamental']['score'] == d['fundamental']['score'] and again['permit_signal'] == d['permit_signal']
        assert again['configuration'] == json.loads(json.dumps(cfg)) or json.loads(json.dumps(again['configuration'])) == cfg


def test_restart_restores_state_and_rejects_every_changed_configuration(tmp_path):
    s, combo = _persisted_run(tmp_path)
    path = tmp_path / 'paper.sqlite'
    kwargs = dict(symbol='GOOD', shadow=True, spread_bps=2., slip_bps=3., database=path)
    base = PaperEngine(SETUP, fundamental_filter=combo, **kwargs)
    assert len(events(base, 'FUNDAMENTAL_DECISION')) == 3 and base.state['equity'] > 100
    base.db.close()
    changed = {
        'threshold': F.engines(s, F.M_JOINT, policy='regime', threshold=.1, half_life=5.),
        'policy': F.engines(s, F.M_JOINT, policy='filter', half_life=5.),
        'half_life': F.engines(s, F.M_JOINT, policy='regime', half_life=6.),
        'macro_age': F.engines(s, F.M_JOINT, policy='regime', half_life=5., macro_age=99),
        'max_age': F.engines(s, F.M_JOINT, policy='regime', half_life=5., max_period_age_days=500),
        'coefficient': F.engines(s, F.artifact(('revenue_yoy', 'technical_signal'), (.125, .5), (.125, .5), (.5, .2500001), .0625), policy='regime', half_life=5.),
        'intercept': F.engines(s, F.artifact(('revenue_yoy', 'technical_signal'), (.125, .5), (.125, .5), (.5, .25), .0626), policy='regime', half_life=5.),
        'no_model': F.engines(s, None, policy='regime', half_life=5.),
        'no_macro': F.engines(s, F.M_JOINT, policy='regime', half_life=5., macro=False),
        'shadow': F.engines(s, F.M_JOINT, policy='regime', half_life=5., shadow=False),
    }
    for name, other in changed.items():
        with pytest.raises(ValueError, match='configuration differs'):
            PaperEngine(SETUP, fundamental_filter=other, **kwargs)
    with pytest.raises(ValueError, match='configuration differs'):
        PaperEngine(SETUP, **kwargs)                                                    # filter removed
    with pytest.raises(ValueError, match='configuration differs'):
        PaperEngine(SETUP, fundamental_filter=combo, **{**kwargs, 'fee_bps': 1.})
    with pytest.raises(ValueError, match='configuration differs'):
        PaperEngine(Setup('donchian', 21), fundamental_filter=combo, **kwargs)
    with pytest.raises(ValueError, match='configuration differs'):
        PaperEngine(SETUP, fundamental_filter=combo, **{**kwargs, 'limits': RiskLimits(risk_per_trade=.004)})


def test_persisted_decisions_are_strict_json_without_nan(tmp_path):
    _persisted_run(tmp_path)
    db = sqlite3.connect(tmp_path / 'paper.sqlite')
    for (p,) in db.execute('SELECT payload FROM events'):
        json.loads(p, parse_constant=lambda c: (_ for _ in ()).throw(AssertionError('non-finite token ' + c)))


# ================================================================ sizing oracle, randomized
@pytest.mark.parametrize('seed', range(120))
def test_position_size_matches_independent_decimal_oracle(seed):
    rng = random.Random(seed)
    price = round(rng.choice([rng.uniform(.5, 5), rng.uniform(5, 500), rng.uniform(500, 5000)]), 4)
    distance = round(price * rng.uniform(.002, .4), 4) or .0001
    fee_bps, spread, slip = rng.choice([0., 1., 10., 25.]), rng.choice([0., 2., 10., 20.]), rng.choice([0., 3., 15., 20.])
    s = fresh_store()
    e, _ = mk_engine(s, M_HUGE, fee_bps=fee_bps, spread_bps=spread, slip_bps=slip)
    # open == price, wide bar so neither stop nor target can fill on the entry bar
    out = e.step(bar(o=price, h=price * 3, l=price * .999, c=price), prev_bar(distance=distance))
    q, entry, stop = oracle_qty(100, 100, price, distance, fee_bps, spread, slip)
    if q * entry < 1 or stop <= 0:
        assert not events(e, 'ENTRY'), f'seed {seed}'
        return
    ents = events(e, 'ENTRY')
    if not ents:      # an unlucky entry-bar stop/target is still an ENTRY event first; no entry means a risk refusal
        pytest.fail(f'seed {seed}: no entry but oracle size {q}')
    got = ents[0]['quantity']
    assert abs(got - float(q)) <= 1e-12, f'seed {seed}: prod {got} oracle {q}'   # exact step agreement; float repr noise only
    assert ents[0]['price'] == pytest.approx(float(entry), rel=1e-12), f'seed {seed}'
    # hard budget: loss at the stop never exceeds 0.5% of equity (plus one quantity step of rounding)
    assert ents[0]['expected_risk'] <= 0.5 + 1e-9, f'seed {seed}'
    assert got * ents[0]['price'] * (1 + fee_bps / 10000) <= 100 + 1e-9, f'seed {seed}'
    assert got * ents[0]['price'] <= 25. + 1e-6, f'seed {seed}'


def test_losing_streak_halves_size_independent_oracle():
    """Three consecutive losers halve the risk budget (scale .5) in RiskEngine; verify via the paper path with intraday bars."""
    s = fresh_store()
    e, _ = mk_engine(s, M_HUGE, limits=RiskLimits(max_daily_loss=.5, max_weekly_loss=.5, max_drawdown=.5), spread_bps=2., slip_bps=3.)
    sizes = []
    for k in range(4):
        h = 13 + 2 * k
        e.step(bar(f'2024-04-19T{h + 1:02d}:00:00Z', o=100., h=101., l=95., c=96.), prev_bar(f'2024-04-19T{h:02d}:00:00Z'))
        sizes.append(events(e, 'ENTRY')[-1]['quantity'])
        # a non-signal bar between trades so the exited state is clean
        e.step(bar(f'2024-04-19T{h + 2:02d}:00:00Z', o=100., h=100.5, l=99.5, c=100.), dict(ts=ts(f'2024-04-19T{h + 1:02d}:00:00Z'), signal=False, distance=2., regime='x'))
    assert e.state['streak'] >= 3
    q_full = oracle_qty(100, 100, 100., 2., 0, 2., 3.)[0]
    assert sizes[0] == pytest.approx(float(q_full), abs=1e-6)
    equity_before_4th = 100 + sum(events(e, 'EXIT')[i]['pnl'] for i in range(3))
    q_half = oracle_qty(equity_before_4th, equity_before_4th, 100., 2., 0, 2., 3., scale=D('0.5'))[0]
    assert sizes[3] == pytest.approx(float(q_half), abs=1e-6)


# ================================================================ remaining hard limits: duplicate signal, journal database failure
def test_duplicate_signal_is_vetoed_even_with_permit_and_huge_score():
    s = fresh_store()
    e, _ = mk_engine(s, M_HUGE)
    e.step(bar(), prev_bar())                                               # entry from signal 2024-04-18
    assert len(events(e, 'ENTRY')) == 1
    e.step(bar('2024-04-20T04:00:00Z', o=100., h=130., l=99., c=125.), None)   # target exit
    assert e.state['position'] is None and e.state['processed_signals'] == ['2024-04-18 04:00:00+00:00']
    e.step(bar('2024-04-21T04:00:00Z'), prev_bar())                         # the SAME completed-bar signal presented again
    assert len(events(e, 'ENTRY')) == 1
    d = events(e, 'FUNDAMENTAL_DECISION')[-1]['decision']
    assert d['permit_signal'] is True and 'duplicate' in events(e, 'RISK')[-1]['reason']


def test_locked_journal_database_means_no_trade_and_unchanged_state(tmp_path):
    s = fresh_store()
    path = tmp_path / 'paper.sqlite'
    e, _ = mk_engine(s, F.M_JOINT, database=path)
    blocker = sqlite3.connect(path, isolation_level=None)
    blocker.execute('BEGIN EXCLUSIVE')
    before = json.dumps(e.state, sort_keys=True)
    with pytest.raises(sqlite3.OperationalError):
        e.step(bar(), prev_bar())
    blocker.execute('ROLLBACK')
    blocker.close()
    assert json.dumps(e.state, sort_keys=True) == before and not e.journal and not e.curve and e.state['cash'] == 100.
    assert e.step(bar(), prev_bar()) == 'ENTRY'


def test_risk_snapshot_database_ok_is_never_wired_from_the_paper_engine():
    """LIMITATION: the RiskEngine 'database unavailable' veto exists but PaperEngine never sets it; DB failures surface as exceptions."""
    import pathlib
    src = pathlib.Path('livesim/small_account.py').read_text()
    assert 'database_ok' not in src and 'qualified' not in src
    # consequence: a non-shadow PaperEngine can never open a position ('no validated edge')
    s = fresh_store()
    e, _ = mk_engine(s, M_HUGE, shadow=False)
    e.step(bar(), prev_bar())
    assert 'no validated edge' in events(e, 'RISK')[0]['reason'] and not events(e, 'ENTRY')


def test_etf_bar_uses_only_macro_and_technical_inputs_through_the_paper_path():
    s = fresh_store()
    e, _ = mk_engine(s, F.M_MACRO, symbol='GOOD', policy='weighted')
    e.step(bar(), dict(prev_bar(), asset_class='etf'))
    d = events(e, 'FUNDAMENTAL_DECISION')[0]['decision']
    assert d['fundamental']['data_status'] == 'REQUIRES_DATED_HOLDINGS' and d['fundamental']['score'] == .875    # .125 + .25*1 + .5*1
    assert d['permit_signal'] is True and len(events(e, 'ENTRY')) == 1 and d['policy'] == 'weighted'
    # the same bar for an equity symbol with a fundamentals-only model is not interchangeable
    e2, _ = mk_engine(fresh_store(), F.M_FUND, symbol='GOOD')
    e2.step(bar(), dict(prev_bar(), asset_class='etf'))
    assert events(e2, 'FUNDAMENTAL_DECISION')[0]['decision']['permit_signal'] is False and not events(e2, 'ENTRY')


def test_restart_does_not_detect_a_different_fundamental_database(tmp_path):
    """LIMITATION: the config fingerprint covers policy/model/thresholds/ages but not the identity or content of the fundamental
    data source, so a journal can be resumed on an empty/different database (decisions then differ; data_snapshot_id shows it)."""
    s, combo = _persisted_run(tmp_path)
    empty = PointInTimeFundamentalStore()
    other = F.engines(empty, F.M_JOINT, policy='regime', half_life=5.)
    e = PaperEngine(SETUP, symbol='GOOD', shadow=True, spread_bps=2., slip_bps=3., fundamental_filter=other, database=tmp_path / 'paper.sqlite')
    assert len(events(e, 'FUNDAMENTAL_DECISION')) == 3                                  # accepted: same fingerprint
    e.db.close()


@pytest.mark.parametrize('fee_bps,spread,slip', [(0., 2., 3.), (10., 2., 3.), (25., 10., 15.), (15., 0., 0.)])
def test_stop_out_realises_exactly_the_risk_budget_that_sized_the_order(fee_bps, spread, slip):
    """Sizing arithmetic closes the loop: a fill at the stop loses qty*(entry*(1+fee) - stop_fill*(1-fee)) = expected_risk <= 0.5."""
    s = fresh_store()
    e, _ = mk_engine(s, M_HUGE, fee_bps=fee_bps, spread_bps=spread, slip_bps=slip)
    e.step(bar(o=100., h=101., l=90., c=92.), prev_bar(distance=3.))
    [en], [ex] = events(e, 'ENTRY'), events(e, 'EXIT')
    assert ex['reason'] == 'entry_bar_stop'
    assert ex['pnl'] == pytest.approx(-en['expected_risk'], abs=1e-9)
    assert 0.5 - 1e-3 < en['expected_risk'] <= 0.5 + 1e-12
    assert e.state['equity'] == pytest.approx(100 + ex['pnl'], abs=1e-9) and e.curve[-1]['equity'] == pytest.approx(e.state['equity'])
    fee = fee_bps / 10000
    assert en['fee'] == pytest.approx(en['quantity'] * en['price'] * fee, rel=1e-12) and ex['fee'] == pytest.approx(ex['quantity'] * ex['price'] * fee, rel=1e-12)
