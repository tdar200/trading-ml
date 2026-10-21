import pandas as pd
import pytest

from data.event_calendar import event_clear_mask, load_event_calendar


def test_event_calendar_blocks_symbol_and_market_dates(tmp_path):
    path = tmp_path / "events.csv"
    pd.DataFrame({
        "date": ["2026-01-02", "2026-01-03"],
        "ticker": ["AAPL", "*"], "event": ["earnings", "FOMC"],
    }).to_csv(path, index=False)
    frame = load_event_calendar(path)
    mask = event_clear_mask(
        frame, [pd.Timestamp("2026-01-02").date(), pd.Timestamp("2026-01-03").date()],
        ("AAPL", "MSFT"),
    )
    assert mask.tolist() == [[False, True], [False, False]]


def test_event_calendar_missing_columns_rejected(tmp_path):
    path = tmp_path / "events.csv"
    pd.DataFrame({"date": ["2026-01-02"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_event_calendar(path)
