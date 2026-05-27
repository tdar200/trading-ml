"""
Stage-2 Nautilus Strategy: monthly top-N momentum rotator on US sector ETFs.

The Strategy maintains an in-memory ring buffer of recent closes per ticker
and rebalances on the first bar of each new calendar month, going long the
top-N tickers (equal-weight by current equity) per the pure momentum ranker
in `strategies.ranking`.

SPY is subscribed as a benchmark — its bars feed into the buffer like the
tradeable tickers do, but the Strategy never places an order on it.

Stage-4.5 follow-up: the rebalance fires on the **benchmark** (SPY) bar each
new month, not the first tradeable bar. Reason: bars arrive interleaved
across tickers within each day, ordered by ticker index in
``data.tickers.ALL_TICKERS = UNIVERSE + (BENCHMARK,)``. If the rebalance
triggered on the first universe bar (e.g. XLK), the strategy's
``self._closes[t][-1]`` for tickers later in the iteration order (XLF, XLE,
XLY, ...) would still hold YESTERDAY's close — today's bars for those
tickers had not arrived yet. The rebalance budgets ``target_shares =
int(per_name_usd / last_close)`` using that stale price, and the livesim
exec client (which buffers fills against the most-recently-seen bar's
close) ended up filling BUYs at TODAY's close after their bars streamed in
mid-cmd-drain. On an overnight gap-up day (e.g. 2022-10-03), the strategy
under-budgeted the BUY notional → actual cash debit exceeded available cash
→ ``AccountBalanceNegative`` on the full-catalog replay. Anchoring the
trigger to the benchmark (always the LAST bar of each day's group) ensures
all universe-ticker closes are TODAY's at decision time, so budgeting and
fills both reference the same prices.
"""
from __future__ import annotations

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.model.objects import Quantity
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
        # Equity snapshots — one per calendar day, taken on benchmark bars.
        # Stored as list of (ts_ns: int, equity_usd: float). Public-ish for
        # downstream tools (examples/backtest_momentum.py reads this).
        self._equity_history: list[tuple[int, float]] = []
        self._last_equity_snapshot_date: tuple[int, int, int] | None = None  # (year, month, day)

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
        cap = self._history_cap()
        if len(history) > cap:
            del history[0 : len(history) - cap]

        # Daily equity snapshot — only on benchmark bars to dedupe across the
        # 12-ticker universe (every business day sees one bar per ticker).
        if ticker == self.config.benchmark:
            year, month, day = self._year_month_day(bar.ts_event)
            if self._last_equity_snapshot_date != (year, month, day):
                self._last_equity_snapshot_date = (year, month, day)
                try:
                    venue = Venue(VENUE_MIC)
                    account = self.portfolio.account(venue)
                    if account is not None:
                        eq_money = self.portfolio.equity(venue)[account.base_currency]
                        self._equity_history.append((bar.ts_event, float(eq_money.as_decimal())))
                except (KeyError, AttributeError, TypeError):
                    # account / equity may be unavailable in very early bars before init
                    pass

        # Rebalance check — only on BENCHMARK bars, only after warm-up, only
        # when this bar's month differs from the last rebalance month. The
        # benchmark bar is the LAST bar in each day's bar group (it follows
        # all universe tickers in ``ALL_TICKERS = UNIVERSE + (BENCHMARK,)``),
        # so by the time we get here ``self._closes[t][-1]`` for every
        # universe ticker has been updated with TODAY's close. See module
        # docstring for the parity-bug history that motivates this choice.
        if ticker != self.config.benchmark:
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

    def _history_cap(self) -> int:
        """Maximum length of the per-ticker close ring buffer. Subclasses may widen for richer features."""
        return self.config.lookback_days + self.config.skip_days + 5

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

    @staticmethod
    def _year_month_day(ts_ns: int) -> tuple[int, int, int]:
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc)
        return dt.year, dt.month, dt.day

    def _rebalance(self, now_ts_ns: int) -> None:
        """Reconcile current holdings to the ranker's top-N target (equal-weight)."""
        self._rebalance_count += 1

        # 1. Ranker's verdict over the tradeable universe.
        ranked = rank_by_momentum(
            {t: self._closes.get(t, []) for t in self.config.universe},
            lookback_days=self.config.lookback_days,
            skip_days=self.config.skip_days,
        )
        if not ranked:
            return  # nothing to do (shouldn't happen post-warmup, but defensive)
        target_tickers = set(ranked[: self.config.top_n])

        # 2. Equity → per-name target USD. Use `portfolio.equity(venue)` which
        # includes the mark value of open positions (not just cash), so the
        # equal-weight target survives the cash-account "spent most of it on
        # stock" cycle. For cash accounts: equity = balance.total + Σ mark_value.
        venue = Venue(VENUE_MIC)
        account = self.portfolio.account(venue)
        equity_money = self.portfolio.equity(venue)[account.base_currency]
        equity_usd = float(equity_money.as_decimal())
        # Conservative 2% cash buffer: avoid AccountBalanceNegative on slippage + integer
        # rounding when deploying ~100% of equity each rebalance.
        per_name_usd = (equity_usd * 0.98) / max(self.config.top_n, 1)

        # 3. For each universe ticker, compute current signed shares + target shares + delta.
        # Collect orders first so SELLs can be submitted before BUYs. On a CASH
        # account the buy leg can only spend settled cash; if a rotation's BUY for a
        # new name is submitted alongside (or before) the SELL that frees the cash,
        # the account momentarily goes negative and the engine halts with
        # AccountBalanceNegative. Submitting all SELLs first releases that cash for
        # the BUYs. (A 2% buffer alone is insufficient here — the overrun is roughly
        # one full position's notional, not a slippage-sized sliver.)
        sells: list[tuple[InstrumentId, int]] = []
        buys: list[tuple[InstrumentId, int]] = []
        for t in self.config.universe:
            instr_id = InstrumentId.from_str(f"{t}.{VENUE_MIC}")
            current = self._current_position_qty(instr_id)
            if t in target_tickers:
                last_close = self._closes[t][-1]
                target = int(per_name_usd / last_close) if last_close > 0 else 0
            else:
                target = 0
            delta = target - current
            if delta == 0:
                continue
            if delta > 0:
                buys.append((instr_id, delta))
            else:
                sells.append((instr_id, -delta))

        for instr_id, qty in sells:
            order = self.order_factory.market(
                instrument_id=instr_id,
                order_side=OrderSide.SELL,
                quantity=Quantity.from_int(qty),
            )
            self.submit_order(order)
        for instr_id, qty in buys:
            order = self.order_factory.market(
                instrument_id=instr_id,
                order_side=OrderSide.BUY,
                quantity=Quantity.from_int(qty),
            )
            self.submit_order(order)

    def _current_position_qty(self, instrument_id: InstrumentId) -> int:
        """Return current signed position in shares for the given instrument."""
        positions = self.cache.positions_open(instrument_id=instrument_id)
        if not positions:
            return 0
        # Sum signed quantities across any open positions for this instrument.
        return int(sum(float(p.signed_qty) for p in positions))
