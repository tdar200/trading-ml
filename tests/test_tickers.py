from data.tickers import (
    ALL_TICKERS, BENCHMARK, SCALPING_COMMODITY_PROXIES, SCALPING_CRYPTO,
    SCALPING_GROUPS, SCALPING_OTHER, SCALPING_SECURITIES, SCALPING_STOCKS,
    SCALPING_UNIVERSE,
    UNIVERSE, scalping_asset_group,
)


def test_universe_has_eleven_sector_etfs():
    assert len(UNIVERSE) == 11


def test_benchmark_is_spy():
    assert BENCHMARK == "SPY"


def test_all_tickers_includes_benchmark():
    assert BENCHMARK in ALL_TICKERS
    assert set(UNIVERSE).issubset(ALL_TICKERS)
    assert len(ALL_TICKERS) == 12


def test_tickers_uppercase_no_duplicates():
    for t in ALL_TICKERS:
        assert t == t.upper()
        assert " " not in t
    assert len(ALL_TICKERS) == len(set(ALL_TICKERS))


def test_universe_covers_all_sectors():
    # The 11 SPDR sector ETFs as of 2026 (XLC added in 2018, XLRE in 2015)
    expected = {"XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"}
    assert set(UNIVERSE) == expected


def test_benchmark_is_last_in_all_tickers():
    """Stage-4.5 follow-up invariant. The rebalance trigger in
    strategies/momentum_topn.py fires on the BENCHMARK bar, which must be
    the LAST bar in each day's group. Bar order = ALL_TICKERS iteration
    order; if a future refactor ever appends another ticker after
    BENCHMARK, the stale-close decision-time bug from Stage-4.5 silently
    returns. Also asserted at module-import time in data/tickers.py."""
    assert ALL_TICKERS[-1] == BENCHMARK
    assert ALL_TICKERS[:-1] == UNIVERSE


def test_scalping_universe_has_four_explicit_sleeves():
    assert len(SCALPING_STOCKS) == 30
    assert len(SCALPING_COMMODITY_PROXIES) == 10
    assert len(SCALPING_CRYPTO) == 10
    assert len(SCALPING_OTHER) == 10
    assert len(SCALPING_UNIVERSE) == 60
    assert len(SCALPING_SECURITIES) == 50
    assert SCALPING_UNIVERSE == SCALPING_SECURITIES + SCALPING_CRYPTO
    assert len(set(SCALPING_UNIVERSE)) == 60
    assert BENCHMARK not in SCALPING_UNIVERSE
    for name, symbols in SCALPING_GROUPS.items():
        assert symbols
        assert all(scalping_asset_group(symbol) == name for symbol in symbols)


def test_unknown_scalping_symbol_is_custom():
    assert scalping_asset_group("CUSTOM") == "custom"
