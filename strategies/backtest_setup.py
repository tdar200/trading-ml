"""
Shared backtest-venue configuration helpers.

Centralizes the FillModel + venue setup used by the live backtest entry point
(examples/backtest_momentum.py) and any future stress-test variants. The
synthetic strategy tests use their own (deterministic, slippage-off) setup —
this helper is for the REAL backtest where realistic cost-modeling matters.

Slippage model:
    NautilusTrader v1.227's FillModel slippage is BINARY one-tick — magnitude is NOT
    configurable. For an ETF priced ~$50 with a $0.01 tick, one tick of slippage
    is ~2 bps. The parent design spec (§5) targets 0.5 bps per fill, so we use
    prob_slippage=0.25: each fill has a 25% chance of a one-tick slip, averaging
    ~0.5 bps over many fills. The random_seed makes the slip pattern reproducible
    so the backtest is deterministic across runs.

    See docs/nautilus_api_notes.md §9.8 for the empirical verification of this
    behavior. To upgrade to deterministic-one-tick (~2 bps) for stress-testing,
    pass slippage_prob=1.0; to disable slippage entirely (sanity-check runs),
    pass slippage_prob=0.0.
"""
from __future__ import annotations

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.models import FillModel
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money

from data.instruments import VENUE_MIC

# Per parent design spec §5: ~0.5 bps slippage on liquid US ETFs.
# Approximated by a 25% probability of one-tick slip per fill.
SLIPPAGE_PROB_DEFAULT: float = 0.25


def add_default_backtest_venue(
    engine: BacktestEngine,
    starting_cash: int = 100_000,
    slippage_prob: float = SLIPPAGE_PROB_DEFAULT,
    random_seed: int = 42,
) -> Venue:
    """Add a standard ARCX cash venue with the project's default FillModel.

    Args:
        engine: the BacktestEngine to attach the venue to.
        starting_cash: USD account balance at backtest start.
        slippage_prob: probability per fill of a one-tick slip. Defaults to 0.25,
            approximating the parent spec's 0.5 bps slippage target on ~$50 ETFs.
        random_seed: seed for FillModel's RNG so backtests are reproducible.

    Returns:
        The Venue object (so callers can pass it to subsequent engine APIs).
    """
    venue = Venue(VENUE_MIC)
    fill_model = FillModel(
        prob_fill_on_limit=1.0,
        prob_slippage=slippage_prob,
        random_seed=random_seed,
    )
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        base_currency=USD,
        starting_balances=[Money(starting_cash, USD)],
        fill_model=fill_model,
    )
    return venue
