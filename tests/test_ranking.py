"""Pure ranker tests — no Nautilus, no IO."""
from strategies.ranking import rank_by_momentum


def _series(start: float, daily_return: float, n: int) -> list[float]:
    """Build a close series that grows by `daily_return` per step from `start`."""
    out = [start]
    for _ in range(n - 1):
        out.append(out[-1] * (1.0 + daily_return))
    return out


def test_known_rank_order_descending_by_lookback_return():
    closes = {
        "HIGH": _series(100.0, 0.002, 200),   # +0.2%/day
        "MID":  _series(100.0, 0.001, 200),   # +0.1%/day
        "LOW":  _series(100.0, 0.0,   200),   # flat
    }
    ranked = rank_by_momentum(closes, lookback_days=126, skip_days=21)
    assert ranked == ["HIGH", "MID", "LOW"]


def test_skip_days_actually_skips_recent_window():
    # A: flat for the lookback window, then big +50% spike in the last skip_days.
    # B: steady +0.1%/day for the whole window.
    # The skip should ignore A's spike → B should rank ABOVE A.
    a = [100.0] * 200
    a[-10:] = [a[-10] * 1.05 ** (i + 1) for i in range(10)]  # spike in last 10 days
    b = _series(100.0, 0.001, 200)
    ranked = rank_by_momentum({"A": a, "B": b}, lookback_days=126, skip_days=21)
    assert ranked == ["B", "A"], "skip_days must exclude the recent spike"


def test_insufficient_history_excludes_ticker():
    closes = {
        "OK":    _series(100.0, 0.001, 200),
        "SHORT": _series(100.0, 0.001, 50),   # too few bars
    }
    ranked = rank_by_momentum(closes, lookback_days=126, skip_days=21)
    assert ranked == ["OK"]
    assert "SHORT" not in ranked


def test_empty_input_returns_empty_list():
    assert rank_by_momentum({}, lookback_days=126, skip_days=21) == []


def test_all_tickers_too_short_returns_empty():
    closes = {"A": [100.0] * 10, "B": [100.0] * 5}
    assert rank_by_momentum(closes, lookback_days=126, skip_days=21) == []


def test_ties_do_not_raise():
    closes = {
        "A": _series(100.0, 0.001, 200),
        "B": _series(100.0, 0.001, 200),  # identical returns
    }
    ranked = rank_by_momentum(closes, lookback_days=126, skip_days=21)
    assert set(ranked) == {"A", "B"}
    assert len(ranked) == 2
