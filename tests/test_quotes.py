import pandas as pd
import pytest

from data.quotes import quote_coverage, quote_features_at, validate_quotes


def _quotes():
    return pd.DataFrame({
        "ts": pd.to_datetime(["2026-01-02 14:30:00Z", "2026-01-02 14:35:00Z"]),
        "bid_price": [99.9, 100.0], "ask_price": [100.1, 100.2],
        "bid_size": [300, 100], "ask_size": [100, 300],
    })


def test_validate_quotes_rejects_crossed_market():
    frame = _quotes()
    frame.loc[0, "ask_price"] = 99.0
    with pytest.raises(ValueError, match="bid <= ask"):
        validate_quotes(frame)


def test_quote_features_are_backward_looking_and_tolerance_bounded():
    times = pd.DatetimeIndex([
        "2026-01-02 14:34:59Z", "2026-01-02 14:35:00Z", "2026-01-02 14:41:00Z",
    ])
    features = quote_features_at(_quotes(), times)
    assert features.iloc[0]["quote_imbalance"] == pytest.approx(0.5)
    assert features.iloc[1]["quote_imbalance"] == pytest.approx(-0.5)
    assert pd.isna(features.iloc[2]["spread_bps"])
    assert quote_coverage(features) == pytest.approx(2 / 3)
