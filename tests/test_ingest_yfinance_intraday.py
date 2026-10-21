import pandas as pd

from data.ingest_yfinance_intraday import (
    normalize_yfinance_frame, provider_symbol, write_free_shards,
)


def test_provider_symbol_maps_crypto_without_changing_stocks():
    assert provider_symbol("BTC/USD") == "BTC-USD"
    assert provider_symbol("AAPL") == "AAPL"


def test_normalize_yfinance_frame_and_isolated_shards(tmp_path):
    index = pd.date_range("2026-01-02 09:30", periods=2, freq="5min", tz="America/New_York")
    columns = pd.MultiIndex.from_product([
        ["Open", "High", "Low", "Close", "Volume"], ["AAPL"],
    ])
    raw = pd.DataFrame([
        [100, 101, 99, 100.5, 1000], [101, 102, 100, 101.5, 1100],
    ], index=index, columns=columns)
    result = normalize_yfinance_frame(raw, "AAPL", asset_class="stock")
    assert result.ts.dt.tz is not None
    assert set(result.source) == {"yfinance_research_only"}
    assert write_free_shards(result, "5m", tmp_path) == 2
    stored = pd.read_parquet(tmp_path / "AAPL" / "5m" / "2026-01.parquet")
    assert len(stored) == 2
    assert stored.source.iloc[0] == "yfinance_research_only"
