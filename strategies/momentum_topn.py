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
        self._last_rebalance_month: tuple[int, int] | None = None  # (year, month)

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
        # Keep just enough history for the ranker plus a small slack.
        cap = self.config.lookback_days + self.config.skip_days + 5
        if len(history) > cap:
            del history[0 : len(history) - cap]

    # ---- helpers ----

    def _all_tickers(self) -> tuple[str, ...]:
        return self.config.universe + (self.config.benchmark,)

    def _ticker_for(self, bar: Bar) -> str:
        # BarType serializes as "<TICKER>.<VENUE>-1-DAY-...". Pull the ticker out.
        return str(bar.bar_type).split(".")[0]
