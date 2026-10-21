import numpy as np
import pandas as pd

from analysis.dynamic_profitability import prepare_dynamic_decision_dataset


def _bars(multiplier, sessions=70):
    rows = []
    for day, date in enumerate(pd.bdate_range("2025-01-02", periods=sessions)):
        start = pd.Timestamp(date.date(), tz="America/New_York") + pd.Timedelta(hours=9, minutes=30)
        for bar in range(78):
            price = 100 + day * 0.03 + multiplier * bar
            rows.append({
                "ts": (start + pd.Timedelta(minutes=5 * bar)).tz_convert("UTC"),
                "open": price, "high": price + 0.1, "low": price - 0.1,
                "close": price + multiplier * 0.2, "volume": 1000 + day * 5 + bar,
            })
    return pd.DataFrame(rows)


def _hourly_bars(multiplier, sessions=70):
    rows = []
    for day, date in enumerate(pd.bdate_range("2025-01-02", periods=sessions)):
        start = pd.Timestamp(date.date(), tz="America/New_York") + pd.Timedelta(hours=9, minutes=30)
        for bar in range(7):
            price = 100 + day * 0.03 + multiplier * bar
            rows.append({
                "ts": (start + pd.Timedelta(hours=bar)).tz_convert("UTC"),
                "open": price, "high": price + 0.1, "low": price - 0.1,
                "close": price + multiplier * 0.2, "volume": 1000 + day * 5 + bar,
            })
    return pd.DataFrame(rows)


def test_dynamic_universe_keeps_day_when_one_stock_bar_is_missing():
    stocks = {f"S{i}": _bars((i - 2.5) * 0.01) for i in range(6)}
    missing_ts = stocks["S0"].loc[(10 * 78) + 5, "ts"]
    stocks["S0"] = stocks["S0"][stocks["S0"].ts != missing_ts]
    result = prepare_dynamic_decision_dataset(stocks, _bars(0.002), 5, min_assets=4)
    date = missing_ts.tz_convert("America/New_York").date()
    row = list(result.session_dates).index(date)
    assert len(result.session_dates) == 70
    assert not result.active_mask[row, 0]
    assert result.active_mask[row].sum() == 5


def test_dynamic_features_do_not_look_past_decision_bar():
    stocks = {f"S{i}": _bars((i - 2.5) * 0.01) for i in range(6)}
    baseline = prepare_dynamic_decision_dataset(stocks, _bars(0.002), 5, min_assets=4)
    changed = {ticker: frame.copy() for ticker, frame in stocks.items()}
    changed["S0"].loc[changed["S0"].index[-1], "close"] *= 2
    mutated = prepare_dynamic_decision_dataset(changed, _bars(0.002), 5, min_assets=4)
    np.testing.assert_allclose(
        baseline.first_residual, mutated.first_residual, equal_nan=True,
    )
    np.testing.assert_allclose(baseline.beta, mutated.beta, equal_nan=True)
    np.testing.assert_allclose(baseline.vwap_deviation, mutated.vwap_deviation, equal_nan=True)
    np.testing.assert_allclose(baseline.prior_range_position, mutated.prior_range_position, equal_nan=True)
    np.testing.assert_allclose(baseline.opening_range_break, mutated.opening_range_break, equal_nan=True)
    np.testing.assert_allclose(baseline.support_rejection, mutated.support_rejection, equal_nan=True)
    assert baseline.forward_return[-1, 0] != mutated.forward_return[-1, 0]


def test_dynamic_hourly_dataset_uses_hourly_session_boundaries():
    stocks = {f"S{i}": _hourly_bars((i - 2.5) * 0.1) for i in range(6)}
    result = prepare_dynamic_decision_dataset(
        stocks, _hourly_bars(0.02), 1, min_assets=4, bar_minutes=60,
    )
    assert len(result.session_dates) == 70
    assert result.bar_minutes == 60
    assert result.vwap_deviation.shape == (70, 6)


def test_dynamic_hourly_dataset_supports_one_bar_exit():
    stocks = {f"S{i}": _hourly_bars((i - 2.5) * 0.1) for i in range(6)}
    one_bar = prepare_dynamic_decision_dataset(
        stocks, _hourly_bars(0.02), 1, min_assets=4,
        bar_minutes=60, exit_after_bars=1,
    )
    close_exit = prepare_dynamic_decision_dataset(
        stocks, _hourly_bars(0.02), 1, min_assets=4, bar_minutes=60,
    )
    assert not np.allclose(one_bar.forward_return, close_exit.forward_return)
