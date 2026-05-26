"""
Pure cross-sectional momentum ranker.

`rank_by_momentum` returns tickers ordered by total return over a `lookback_days`
window ending `skip_days` ago. This is the classic momentum-skip-recent pattern:
it captures the medium-horizon trend while avoiding the short-term mean-reversion
effect that contaminates raw 6-month returns.

No Nautilus, no IO. Fully unit-testable.
"""
from __future__ import annotations


def rank_by_momentum(
    closes_by_ticker: dict[str, list[float]],
    lookback_days: int = 126,
    skip_days: int = 21,
) -> list[str]:
    """Return tickers ranked descending by lookback return ending skip_days ago.

    A ticker is included only if it has at least `lookback_days + skip_days + 1` closes.
    Returns are computed as (close[-1-skip_days] / close[-1-skip_days-lookback_days]) - 1.
    Tickers without enough history are silently excluded (not ranked last).
    """
    needed = lookback_days + skip_days
    scored: list[tuple[float, str]] = []
    for ticker, closes in closes_by_ticker.items():
        if len(closes) < needed + 1:
            continue
        end = closes[-1 - skip_days]
        start = closes[-1 - skip_days - lookback_days]
        if start <= 0.0:
            continue
        ret = end / start - 1.0
        scored.append((ret, ticker))
    # Sort descending by return; Python's sort is stable, so equal returns
    # preserve insertion order (good enough for the spec's tie-handling).
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored]
