"""
Stage-2 Nautilus Strategy: monthly top-N momentum rotator on US sector ETFs.

The Strategy maintains an in-memory ring buffer of recent closes per ticker
and rebalances on the first bar of each new calendar month, going long the
top-N tickers (equal-weight by current equity) per the pure momentum ranker
in `strategies.ranking`.

SPY is subscribed as a benchmark — its bars feed into the buffer like the
tradeable tickers do, but the Strategy never places an order on it.
"""
from __future__ import annotations

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.trading.strategy import Strategy

from data.instruments import VENUE_MIC
from strategies.ranking import rank_by_momentum


class MomentumTopNConfig(StrategyConfig, frozen=True):
    """Configuration for MomentumTopNStrategy."""
    universe: tuple[str, ...]              # tradeable tickers (long-only)
    benchmark: str = "SPY"                 # subscribed but not traded
    top_n: int = 3
    lookback_days: int = 126
    skip_days: int = 21


class MomentumTopNStrategy(Strategy):
    """Monthly top-N momentum rotator. See module docstring."""

    def __init__(self, config: MomentumTopNConfig) -> None:
        super().__init__(config)
        self._closes: dict[str, list[float]] = {}
        self._last_rebalance_month: tuple[int, int] | None = None
        self._rebalance_count: int = 0  # observable for tests

    # ---- lifecycle ----

    def on_start(self) -> None:
        # Subscribe to bars for every ticker (tradeable + benchmark).
        for ticker in self._all_tickers():
            bar_type = BarType.from_str(
                f"{ticker}.{VENUE_MIC}-1-DAY-LAST-EXTERNAL"
            )
            self.subscribe_bars(bar_type)

    def on_bar(self, bar: Bar) -> None:
        ticker = self._ticker_for(bar)
        history = self._closes.setdefault(ticker, [])
        history.append(float(bar.close))
        cap = self.config.lookback_days + self.config.skip_days + 5
        if len(history) > cap:
            del history[0 : len(history) - cap]

        # Rebalance check — only on tradeable-universe bars, only after warm-up,
        # only when this bar's month differs from the last rebalance month.
        if ticker not in self.config.universe:
            return
        if not self._is_warmed_up():
            return

        year, month = self._year_month(bar.ts_event)
        if self._last_rebalance_month == (year, month):
            return
        # Either first bar ever or first bar of a new month.
        self._last_rebalance_month = (year, month)
        self._rebalance(bar.ts_event)

    # ---- helpers ----

    def _all_tickers(self) -> tuple[str, ...]:
        return self.config.universe + (self.config.benchmark,)

    def _ticker_for(self, bar: Bar) -> str:
        # BarType serializes as "<TICKER>.<VENUE>-1-DAY-...". Pull the ticker out.
        return str(bar.bar_type).split(".")[0]

    def _is_warmed_up(self) -> bool:
        needed = self.config.lookback_days + self.config.skip_days + 1
        return all(
            len(self._closes.get(t, [])) >= needed
            for t in self.config.universe
        )

    @staticmethod
    def _year_month(ts_ns: int) -> tuple[int, int]:
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc)
        return dt.year, dt.month

    def _rebalance(self, now_ts_ns: int) -> None:
        # Task 7 fills this in; for now we only count.
        self._rebalance_count += 1
