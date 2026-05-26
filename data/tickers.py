"""
The 12-ticker universe for trading-ml.

UNIVERSE: the 11 S&P sector SPDR ETFs (long/trade candidates).
BENCHMARK: SPY (used for performance comparison, not trading).
ALL_TICKERS: union of the above — what the data pipeline ingests.

All 12 tickers are NYSE-Arca-listed and existed for the entire 2016→ window
chosen in the design spec, so survivorship bias is sidestepped at this stage.
"""

UNIVERSE: tuple[str, ...] = (
    "XLK",   # Technology
    "XLF",   # Financials
    "XLE",   # Energy
    "XLV",   # Health Care
    "XLI",   # Industrials
    "XLY",   # Consumer Discretionary
    "XLP",   # Consumer Staples
    "XLU",   # Utilities
    "XLB",   # Materials
    "XLRE",  # Real Estate
    "XLC",   # Communication Services
)

BENCHMARK: str = "SPY"

ALL_TICKERS: tuple[str, ...] = UNIVERSE + (BENCHMARK,)
