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

# Dedicated, configurable Stage-12 scalping research basket. These constants
# intentionally do not alter ALL_TICKERS or the daily sector-rotation universe.
SCALPING_STOCKS: tuple[str, ...] = (
    "AAPL", "MSFT", "NVDA", "AMD", "AVGO", "INTC", "MU", "QCOM",
    "TSLA", "AMZN", "META", "GOOGL", "NFLX", "PLTR", "COIN",
    "MSTR", "SMCI", "JPM", "BAC", "GS", "XOM", "CVX", "OXY",
    "LLY", "UNH", "BA", "CAT", "GE", "DIS", "NKE",
)

SCALPING_STOCK_SECTORS: dict[str, str] = {
    "AAPL": "technology", "MSFT": "technology", "NVDA": "technology",
    "AMD": "technology", "AVGO": "technology", "INTC": "technology",
    "MU": "technology", "QCOM": "technology", "PLTR": "technology",
    "SMCI": "technology", "TSLA": "consumer", "AMZN": "consumer",
    "NKE": "consumer", "META": "communication", "GOOGL": "communication",
    "NFLX": "communication", "DIS": "communication", "COIN": "financial",
    "MSTR": "financial", "JPM": "financial", "BAC": "financial",
    "GS": "financial", "XOM": "energy", "CVX": "energy", "OXY": "energy",
    "LLY": "healthcare", "UNH": "healthcare", "BA": "industrial",
    "CAT": "industrial", "GE": "industrial",
}

# Alpaca does not offer commodity futures. These exchange-traded securities
# are liquid commodity or commodity-producer proxies, not futures contracts.
SCALPING_COMMODITY_PROXIES: tuple[str, ...] = (
    "GLD", "SLV", "USO", "UNG", "DBA", "CPER", "PPLT", "PALL", "URA", "WEAT",
)

SCALPING_CRYPTO: tuple[str, ...] = (
    "BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "DOGE/USD",
    "AAVE/USD", "AVAX/USD", "LINK/USD", "LTC/USD", "BCH/USD",
)

SCALPING_OTHER: tuple[str, ...] = (
    "QQQ", "IWM", "DIA", "TQQQ", "SQQQ", "SOXL", "SOXS", "ARKK", "SMH", "XBI",
)

SCALPING_SECURITIES: tuple[str, ...] = (
    SCALPING_STOCKS + SCALPING_COMMODITY_PROXIES + SCALPING_OTHER
)

SCALPING_UNIVERSE: tuple[str, ...] = (
    SCALPING_SECURITIES + SCALPING_CRYPTO
)

SCALPING_GROUPS: dict[str, tuple[str, ...]] = {
    "individual_stock": SCALPING_STOCKS,
    "commodity_proxy": SCALPING_COMMODITY_PROXIES,
    "crypto": SCALPING_CRYPTO,
    "other_contender": SCALPING_OTHER,
}


def scalping_asset_group(symbol: str) -> str:
    """Return the explicit research sleeve for a scalping symbol."""
    matches = [name for name, symbols in SCALPING_GROUPS.items() if symbol in symbols]
    if not matches:
        return "custom"
    if len(matches) > 1:
        raise ValueError(f"symbol {symbol!r} has multiple sleeve assignments")
    return matches[0]

BENCHMARK: str = "SPY"

ALL_TICKERS: tuple[str, ...] = UNIVERSE + (BENCHMARK,)

# Stage-4.5 follow-up invariant: the BENCHMARK must be the LAST ticker in
# ALL_TICKERS. The bar provider in examples/run_trading_node.py emits bars
# in ALL_TICKERS iteration order; within each day's group the strategy
# triggers its monthly rebalance on the BENCHMARK bar (so every
# universe-ticker's _closes[t][-1] is today's close at decision time).
# If a future refactor appends another ticker after BENCHMARK or sources
# the iteration order elsewhere, the stale-close decision-time bug from
# Stage-4.5 silently returns. tests/test_tickers.py asserts this.
assert ALL_TICKERS[-1] == BENCHMARK, (
    "BENCHMARK must be the LAST entry in ALL_TICKERS — see Stage-4.5 fix in "
    "strategies/momentum_topn.py:on_bar."
)
