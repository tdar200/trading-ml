"""
Nautilus Equity instrument builders for the trading-ml universe.

All 12 ETFs are constructed identically (USD, 2dp prices, $0.01 increment).
`build_universe_instruments()` returns one Equity per ticker in ALL_TICKERS.
"""
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.identifiers import InstrumentId, Symbol
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Price, Quantity

from data.tickers import ALL_TICKERS

VENUE_MIC: str = "ARCX"  # NYSE Arca — all 12 ETFs are Arca-listed


def build_equity(ticker: str) -> Equity:
    """Build a Nautilus Equity instrument for a single ticker."""
    return Equity(
        instrument_id=InstrumentId.from_str(f"{ticker}.{VENUE_MIC}"),
        raw_symbol=Symbol(ticker),
        currency=USD,
        price_precision=2,
        price_increment=Price.from_str("0.01"),
        lot_size=Quantity.from_int(1),
        ts_event=0,
        ts_init=0,
    )


def build_universe_instruments() -> list[Equity]:
    """Build Equity instruments for every ticker in ALL_TICKERS."""
    return [build_equity(t) for t in ALL_TICKERS]
