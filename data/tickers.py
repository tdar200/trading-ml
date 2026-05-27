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
