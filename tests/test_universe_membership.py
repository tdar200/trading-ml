import pandas as pd
import pytest

from data.universe_membership import load_membership, members_on


def test_missing_membership_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError, match="must not be backfilled"):
        load_membership(tmp_path / "missing.csv")


def test_membership_is_point_in_time(tmp_path):
    path = tmp_path / "members.csv"
    pd.DataFrame({
        "ticker": ["OLD", "NEW"], "start": ["2020-01-01", "2022-01-01"],
        "end": ["2022-01-01", None],
    }).to_csv(path, index=False)
    frame = load_membership(path)
    assert members_on(frame, pd.Timestamp("2021-06-01", tz="UTC")) == ("OLD",)
    assert members_on(frame, pd.Timestamp("2023-06-01", tz="UTC")) == ("NEW",)
