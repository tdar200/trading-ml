"""Track G (phases 92-94, 113-119): universe, survivorship, terminal-loss and corporate-action behaviour.

Synthetic series are driven through the real PaperEngine; membership through data.universe_membership.
Run: .venv/bin/python -m pytest -p no:cacheprovider -q tests/qa_reaudit/test_bt_survivorship.py
"""
import sqlite3

import pandas as pd
import pytest

from tests.qa_reaudit import bt_runner as run

START = pd.Timestamp('2025-01-02 05:00', tz='UTC')


def bar(day, o, h, l, c):
    return {'ts': START + pd.Timedelta(days=day), 'open': o, 'high': h, 'low': l, 'close': c, 'volume': 1e6}


def sig(day, close=100., dist=2.):
    b = bar(day, close, close + 1, close - 1, close)
    b.update(signal=True, distance=dist, regime='trend_up')
    return b


def engine(**kw):
    from livesim.small_account import PaperEngine
    from strategies.small_account import Setup
    return PaperEngine(Setup('donchian', 20), symbol='XYZ', fee_bps=15, shadow=True, **kw)


def enter(e, price=100.):
    assert e.step(bar(1, price, price + 1, price - 1, price), sig(0, price)) == 'ENTRY'
    return e.state['position']


# ------------------------------------------------------------------------------------------ terminal loss / gap rules
def test_bt_bankruptcy_gap_is_realised_at_the_gap_open_not_at_the_stop():
    e = engine()
    p = enter(e)
    assert e.step(bar(2, 100, 100.5, 99.5, 100), None) == 'HOLD'
    assert e.step(bar(3, 0.5, 0.6, 0.01, 0.02), None) == 'EXIT'
    ex = [x for x in e.journal if x['event'] == 'EXIT'][0]
    assert ex['reason'] == 'gap_stop'
    assert ex['price'] == pytest.approx(0.5 * (1 - 4 / 10000))  # gap open less spread/slippage, NOT the 98.04 stop
    loss_fraction = -ex['pnl'] / 100.0
    assert 0.20 < loss_fraction < 0.22  # 21% of the account vs the 0.5% risk budget: gap risk is not bounded by sizing
    assert e.state['position'] is None and e.state['halted'] is True  # drawdown >= 10% halts further entries
    assert e.state['equity'] == pytest.approx(e.state['cash'])


def test_bt_exposure_cap_bounds_the_worst_bankruptcy_loss_to_25_percent():
    e = engine()
    enter(e)
    cost = e.state['position']['cost']
    assert cost <= 25.0 + 1e-9  # 25% max exposure
    e.step(bar(2, 1e-6, 1e-6, 1e-6, 1e-6), None)
    assert e.state['equity'] > 75.0


def test_bt_price_of_exactly_zero_is_refused_and_position_stays_open():
    """A delisted stock quoted at 0 cannot be processed: the engine fails closed (ValueError), the loss is never booked."""
    import json
    e = engine()
    enter(e)
    before, journal = json.dumps(e.state), list(e.journal)
    with pytest.raises(ValueError, match='invalid market event'):
        e.step(bar(2, 0.0, 0.0, 0.0, 0.0), None)
    assert json.dumps(e.state) == before and e.journal == journal  # nothing booked, nothing changed
    assert e.state['position'] is not None


def test_bt_series_that_ends_while_long_is_closed_at_the_last_close_with_no_delisting_loss():
    """LIMITATION (no delisting-return model): if the data feed simply stops (delisting/acquisition), the replay's
    evaluation_end exit happens at the last observed close; the terminal collapse is never in the P&L."""
    e = engine()
    enter(e)
    e.step(bar(2, 100, 101, 99.5, 100.5), None)
    assert e.step(bar(3, 100.5, 101, 100, 100.8), None, close_final=True) == 'EXIT'
    ex = [x for x in e.journal if x['event'] == 'EXIT'][0]
    assert ex['reason'] == 'evaluation_end' and ex['price'] == pytest.approx(100.8 * (1 - 4 / 10000))
    assert e.state['equity'] > 99.9  # a stock that then went to zero would have lost ~21%


def test_bt_live_stream_without_new_bars_keeps_a_stale_mark_and_raises_no_alarm():
    """LIMITATION: a halted/suspended instrument leaves the position marked at the last close; only an external
    age_seconds argument could flag it (the engine itself has no per-instrument staleness)."""
    e = engine()
    enter(e)
    before = e.state['equity']
    assert e.curve[-1]['equity'] == before
    assert len(e.curve) == 1  # nothing else is written until a new bar arrives


def test_bt_corporate_action_split_is_not_adjusted_phantom_loss_on_unadjusted_data():
    """NOT IMPLEMENTED: no split handling for an open position. A 2-for-1 split on raw prices is a -50% 'gap stop'."""
    e = engine()
    enter(e)
    assert e.step(bar(2, 50.0, 50.4, 49.6, 50.0), None) == 'EXIT'
    ex = [x for x in e.journal if x['event'] == 'EXIT'][0]
    assert ex['reason'] == 'gap_stop' and ex['pnl'] < -10  # an economically flat event booked as a 10%+ account loss


def test_bt_reverse_split_is_booked_as_a_target_win_at_the_target_price():
    e = engine()
    enter(e)
    assert e.step(bar(2, 200.0, 201.0, 199.0, 200.0), None) == 'EXIT'
    ex = [x for x in e.journal if x['event'] == 'EXIT'][0]
    assert ex['reason'] == 'target' and ex['price'] == pytest.approx(104.04 * (1 - 4 / 10000))  # fills at target, not the 200 open


def test_bt_dividend_not_credited_so_adjusted_bars_are_the_only_total_return_path():
    """Dividends: the engine has no cash-dividend event. On RAW bars the ex-date drop is booked as loss and the
    dividend is lost; on ADJUSTED bars (what the data pipeline fetches: Adjustment.ALL) the drop is absent. No double
    counting is possible because there is no dividend input."""
    def run_path(drop):
        e = engine()
        enter(e)
        qty = e.state['position']['qty']
        e.step(bar(2, 100, 100.5, 99.8, 100.2), None)
        o = 100.2 * (1 - drop)
        e.step(bar(3, o, o + .3, o - .3, o), None)  # ex-dividend morning
        return e.state['equity'], qty
    (raw, qty), (adjusted, _) = run_path(0.01), run_path(0.0)
    assert raw < adjusted  # raw loses the dividend as a price drop, nothing credits it back
    assert adjusted - raw == pytest.approx(qty * 100.2 * 0.01, rel=1e-6)


def test_bt_ambiguous_bar_is_resolved_stop_first_and_entry_bar_stop_is_executable():
    e = engine()
    enter(e)
    assert e.step(bar(2, 100, 106, 97, 101), None) == 'EXIT'  # both 104.04 target and 98.04 stop inside the bar
    ex = [x for x in e.journal if x['event'] == 'EXIT'][0]
    assert ex['reason'] == 'stop'
    e2 = engine()
    # wide entry bar that includes the stop: filled at entry then stopped in the same bar
    assert e2.step(bar(1, 100, 100.5, 97.0, 99.0), sig(0)) == 'EXIT'
    assert [x['event'] for x in e2.journal if x['event'] in ('ENTRY', 'EXIT')] == ['ENTRY', 'EXIT']


def test_bt_time_exit_after_ten_calendar_days_fills_at_the_open():
    e = engine()
    enter(e)
    for d in range(2, 11):
        assert e.step(bar(d, 100.1, 100.4, 99.9, 100.2), None) == 'HOLD'
    assert e.step(bar(11, 100.7, 101.0, 100.5, 100.9), None) == 'EXIT'
    ex = [x for x in e.journal if x['event'] == 'EXIT'][0]
    assert ex['reason'] == 'time' and ex['price'] == pytest.approx(100.7 * (1 - 4 / 10000))


# ------------------------------------------------------------------------------------------ membership / universe
def _membership(tmp_path, text):
    from data.universe_membership import load_membership
    p = tmp_path / 'm.csv'
    p.write_text(text)
    return load_membership(p)


def test_bt_membership_boundaries_start_inclusive_end_exclusive_and_missing_file_fails_closed(tmp_path):
    from data.universe_membership import members_on, load_membership
    f = _membership(tmp_path, 'ticker,start,end\nOLD,2019-01-01,2020-06-01\nNEW,2020-06-01,\nFUT,2030-01-01,\n')
    ts = lambda s: pd.Timestamp(s, tz='UTC')
    assert members_on(f, ts('2020-05-31 23:59:59')) == ('OLD',)
    assert members_on(f, ts('2020-06-01')) == ('NEW',)  # end exclusive, start inclusive: no overlap, no gap
    assert 'FUT' not in members_on(f, ts('2025-01-01'))
    with pytest.raises(FileNotFoundError):
        load_membership(tmp_path / 'absent.csv')
    with pytest.raises(ValueError):
        _membership(tmp_path, 'ticker,start,end\nBAD,2021-01-01,2020-01-01\n')
    with pytest.raises(ValueError):
        _membership(tmp_path, 'ticker,start,end\nBAD,not-a-date,\n')


def test_bt_membership_malformed_end_date_must_not_mean_member_forever(tmp_path):
    from data.universe_membership import members_on
    try:
        f = _membership(tmp_path, 'ticker,start,end\nDEAD,2020-01-01,2021-13-45\nLIVE,2020-01-01,\n')
    except ValueError:
        return  # raising is the correct fail-closed behaviour
    assert 'DEAD' not in members_on(f, pd.Timestamp('2025-01-01', tz='UTC'))


def test_bt_universe_modules_are_survivor_lists_and_membership_is_optional():
    """Evidence for the survivorship limitation: universes are fixed present-day lists; PIT membership is an optional
    CLI argument; the 30 'scalping stocks' contain post-2020 listings and recent winners."""
    from data import tickers
    assert tickers.UNIVERSE == ('XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC')
    names = set(tickers.SCALPING_STOCKS)
    assert {'PLTR', 'COIN', 'SMCI', 'MSTR', 'NVDA'} <= names  # IPO'd 2020/2021 or 2023-24 outperformers
    import pathlib
    text = (run.ROOT / 'examples' / 'find_profitable_intraday_v2.py').read_text()
    assert 'membership = load_membership(args.membership) if args.membership else None' in text  # optional, not required
    text16 = (run.ROOT / 'analysis' / 'dynamic_profitability.py').read_text()
    assert 'available = np.isfinite(matrices["future"]) & np.isfinite(matrices["first"])' in text16  # eligibility uses FUTURE availability


def test_bt_fundamental_store_symbols_filter_by_publication_not_by_period(tmp_path):
    """Watchlist/universe: symbols(as_of) lists a company only after its first financial fact was published."""
    from fundamentals.store import PointInTimeFundamentalStore, Fact
    s = PointInTimeFundamentalStore(str(tmp_path / 'u.sqlite'))
    def fact(sym, avail):
        return Fact(sym, 'revenue', 1.0, '2024-12-31', avail, avail, 'T', 'a', 'USD', 'USD', '2024-01-01', observed_at=avail)
    s.ingest([fact('OLDCO', '2025-02-01T00:00:00+00:00'), fact('NEWCO', '2025-08-01T00:00:00+00:00')])
    assert s.symbols('2025-03-01T00:00:00+00:00') == ['OLDCO']
    assert s.symbols('2025-09-01T00:00:00+00:00') == ['NEWCO', 'OLDCO']
    s.close()


# ------------------------------------------------------------------------------------------ coverage report
def test_bt_coverage_by_symbol_sector_year_is_written_and_shows_large_cap_survivor_bias():
    """Phase 113-119: coverage of the stored research DB and of the eight-company audit DB, by symbol/year/metric."""
    out = run.OUT
    rows = []
    sector = {'XOM': 'Energy', 'MSFT': 'Technology', 'JPM': 'Financials', 'WMT': 'Consumer staples/retail', 'JNJ': 'Health care',
              'CAT': 'Industrials', 'BA': 'Industrials (loss-making)', 'SONY': 'Consumer electronics (JP, IFRS since FY2019)'}
    for label, path in (('research_db', run.ensure_db_copy()), ('audit_db_v3', out / 'dbcopies' / 'real_ingestion_v3.sqlite')):
        if not path.exists():
            pytest.skip('audit db copy missing: copy results/fundamental_closure_20260930/real_ingestion_v3.sqlite into bt/dbcopies first')
        db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        for sym, year, metrics, n_facts, first_av, last_av in db.execute(
                "SELECT symbol, substr(period_end,1,4) y, count(distinct metric), count(*), min(available_at), max(available_at) "
                "FROM observations WHERE kind='financial' GROUP BY symbol, y ORDER BY symbol, y"):
            rows.append({'db': label, 'symbol': sym, 'sector': sector.get(sym, ''), 'fiscal_year': year, 'distinct_metrics': metrics,
                         'facts': n_facts, 'first_available_at': first_av, 'last_available_at': last_av})
        for sym, series, y, n in db.execute("SELECT symbol, metric, substr(period_end,1,4), count(*) FROM observations WHERE kind='macro' GROUP BY symbol, metric, substr(period_end,1,4)"):
            rows.append({'db': label, 'symbol': sym, 'sector': 'macro series', 'fiscal_year': y, 'distinct_metrics': 1, 'facts': n,
                         'first_available_at': '', 'last_available_at': ''})
        db.close()
    df = pd.DataFrame(rows)
    df.to_csv(out / 'coverage_by_symbol_sector_year.csv', index=False)
    fin = df[df.sector != 'macro series']
    research_symbols = set(fin[fin.db == 'research_db'].symbol)
    audit_symbols = set(fin[fin.db == 'audit_db_v3'].symbol)
    assert research_symbols == {'XOM'}  # the stored research database covers exactly ONE company
    assert audit_symbols == {'XOM', 'MSFT', 'JPM', 'WMT', 'JNJ', 'CAT', 'BA', 'SONY'}  # all mega-cap survivors, zero small caps or delistings
    assert fin.fiscal_year.astype(int).min() <= 2007 and fin.fiscal_year.astype(int).max() >= 2026
