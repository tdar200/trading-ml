from nautilus_trader.model.data import Bar, BarType

from data.bar_converter import alpaca_df_to_bars


def test_converter_returns_one_bar_per_row(sample_alpaca_bars_df):
    bars = alpaca_df_to_bars(sample_alpaca_bars_df, "SPY")
    assert len(bars) == 3
    assert all(isinstance(b, Bar) for b in bars)


def test_converter_preserves_ohlcv_values(sample_alpaca_bars_df):
    bars = alpaca_df_to_bars(sample_alpaca_bars_df, "SPY")
    first = bars[0]
    assert str(first.open) == "470.50"
    assert str(first.high) == "472.00"
    assert str(first.low) == "469.80"
    assert str(first.close) == "471.10"
    assert int(first.volume) == 50_000_000


def test_converter_sets_correct_bar_type(sample_alpaca_bars_df):
    bars = alpaca_df_to_bars(sample_alpaca_bars_df, "SPY")
    expected_str = "SPY.ARCX-1-DAY-LAST-EXTERNAL"
    assert str(bars[0].bar_type) == expected_str


def test_converter_timestamps_monotonic_and_nanos(sample_alpaca_bars_df):
    bars = alpaca_df_to_bars(sample_alpaca_bars_df, "SPY")
    ts = [b.ts_event for b in bars]
    assert ts == sorted(ts)
    assert all(isinstance(t, int) for t in ts)
    assert all(t > 10**18 for t in ts)


def test_converter_ts_init_equals_ts_event(sample_alpaca_bars_df):
    bars = alpaca_df_to_bars(sample_alpaca_bars_df, "SPY")
    for b in bars:
        assert b.ts_init == b.ts_event


def test_converter_filters_by_symbol_in_multi_symbol_df(sample_alpaca_bars_df):
    other = sample_alpaca_bars_df.copy()
    other.index = other.index.set_levels(["XLF"], level=0)
    import pandas as pd
    combined = pd.concat([sample_alpaca_bars_df, other])
    bars = alpaca_df_to_bars(combined, "SPY")
    assert len(bars) == 3
    assert all(str(b.bar_type).startswith("SPY") for b in bars)


def test_converter_empty_df_returns_empty_list():
    import pandas as pd
    empty = pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"],
        index=pd.MultiIndex.from_tuples([], names=["symbol", "timestamp"]),
    )
    assert alpaca_df_to_bars(empty, "SPY") == []
