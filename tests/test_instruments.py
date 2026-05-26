from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Equity

from data.instruments import build_equity, build_universe_instruments
from data.tickers import ALL_TICKERS


def test_build_equity_returns_equity_with_correct_id():
    eq = build_equity("SPY")
    assert isinstance(eq, Equity)
    assert eq.id == InstrumentId.from_str("SPY.ARCX")
    assert str(eq.raw_symbol) == "SPY"


def test_build_equity_price_precision_is_two():
    eq = build_equity("XLF")
    assert eq.price_precision == 2


def test_build_universe_instruments_returns_twelve():
    instruments = build_universe_instruments()
    assert len(instruments) == len(ALL_TICKERS)
    ids = {str(eq.id) for eq in instruments}
    assert ids == {f"{t}.ARCX" for t in ALL_TICKERS}
