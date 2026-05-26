from data.tickers import UNIVERSE, BENCHMARK, ALL_TICKERS


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
