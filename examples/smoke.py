"""
Stage-0 smoke test for trading-ml.

Builds a BacktestEngine, feeds it a handful of synthetic daily bars for SPY,
runs a do-nothing Strategy that logs each bar received, and prints the final
account balance. If this script exits 0 and prints bars, the Nautilus install
and the API paths in docs/nautilus_api_notes.md are correct.

All API paths/signatures here match docs/nautilus_api_notes.md (verified
against nautilus_trader==1.227.0), not memory.
"""

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig, StrategyConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, TraderId, Venue
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.trading.strategy import Strategy


class LogBarsConfig(StrategyConfig, frozen=True):
    bar_type: BarType


class LogBars(Strategy):
    def __init__(self, config: LogBarsConfig) -> None:
        super().__init__(config)

    def on_start(self) -> None:
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        self.log.info(f"bar: {bar}")


def build_spy_instrument() -> Equity:
    # Only the 8 required positional args (notes §5); fees/margins/isin are
    # optional and default to 0/None.
    ts = 1_700_000_000_000_000_000  # arbitrary UNIX nanoseconds
    return Equity(
        instrument_id=InstrumentId.from_str("SPY.ARCX"),
        raw_symbol=Symbol("SPY"),
        currency=USD,
        price_precision=2,
        price_increment=Price.from_str("0.01"),
        lot_size=Quantity.from_int(1),
        ts_event=ts,
        ts_init=ts,
    )


def build_synthetic_bars(bar_type: BarType, n: int = 5) -> list[Bar]:
    bars: list[Bar] = []
    base_ts = 1_700_000_000_000_000_000  # arbitrary nanos
    for i in range(n):
        ts = base_ts + i * 86_400_000_000_000  # +1 day per bar
        bars.append(
            Bar(
                bar_type=bar_type,
                open=Price.from_str(f"{500 + i}.00"),
                high=Price.from_str(f"{500 + i}.50"),
                low=Price.from_str(f"{499 + i}.50"),
                close=Price.from_str(f"{500 + i}.25"),
                volume=Quantity.from_int(1_000_000),
                ts_event=ts,
                ts_init=ts,
            )
        )
    return bars


def main() -> None:
    instrument = build_spy_instrument()
    bar_type = BarType.from_str(f"{instrument.id}-1-DAY-LAST-EXTERNAL")

    config = BacktestEngineConfig(
        trader_id=TraderId("SMOKE-001"),
        logging=LoggingConfig(log_level="INFO"),
    )
    engine = BacktestEngine(config=config)

    venue = Venue("ARCX")
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        starting_balances=[Money(100_000, USD)],
        base_currency=USD,
    )
    engine.add_instrument(instrument)
    engine.add_data(build_synthetic_bars(bar_type))
    engine.add_strategy(LogBars(config=LogBarsConfig(bar_type=bar_type)))
    engine.run()

    # Balance read path verified in notes §8 (NOT cache.account_for_venue).
    account = engine.portfolio.account(venue)
    print(f"final balance: {account.balance_total(USD)}")

    engine.dispose()


if __name__ == "__main__":
    main()
