import pandas as pd

from data.ingest_alpaca_quotes import normalize_quote_frame, write_quote_shards


def _response_frame():
    index = pd.MultiIndex.from_arrays([
        ["AAPL", "AAPL"], pd.to_datetime(["2026-01-02 14:30Z", "2026-02-02 14:30Z"]),
    ], names=["symbol", "timestamp"])
    return pd.DataFrame({
        "bid_price": [100.0, 101.0], "ask_price": [100.1, 101.1],
        "bid_size": [10, 12], "ask_size": [11, 13],
    }, index=index)


def test_normalize_and_monthly_shard(tmp_path):
    result = normalize_quote_frame(_response_frame())
    assert list(result["symbol"].unique()) == ["AAPL"]
    assert write_quote_shards(_response_frame(), root=tmp_path) == 2
    assert (tmp_path / "AAPL" / "2026-01.parquet").exists()
    assert (tmp_path / "AAPL" / "2026-02.parquet").exists()
