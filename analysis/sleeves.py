"""Execution-homogeneous research sleeves; never mix incompatible sessions/costs."""
from __future__ import annotations

from dataclasses import dataclass

from data.tickers import (
    BENCHMARK, SCALPING_COMMODITY_PROXIES, SCALPING_CRYPTO,
    SCALPING_OTHER, SCALPING_STOCKS,
)


@dataclass(frozen=True)
class SleeveSpec:
    name: str
    symbols: tuple[str, ...]
    benchmark: str
    session_model: str
    deployable_with_equity_replay: bool


SLEEVES = {
    "individual_stock": SleeveSpec(
        "individual_stock", SCALPING_STOCKS, BENCHMARK, "us_equity_regular", True,
    ),
    "commodity_proxy": SleeveSpec(
        "commodity_proxy", SCALPING_COMMODITY_PROXIES, BENCHMARK,
        "us_equity_regular", True,
    ),
    "other_contender": SleeveSpec(
        "other_contender", SCALPING_OTHER, BENCHMARK, "us_equity_regular", True,
    ),
    "crypto": SleeveSpec(
        "crypto", SCALPING_CRYPTO, "BTC/USD", "continuous_24_7", False,
    ),
}


def require_equity_replay_sleeve(name: str) -> SleeveSpec:
    if name not in SLEEVES:
        raise ValueError(f"unknown sleeve {name!r}; choose from {tuple(SLEEVES)}")
    sleeve = SLEEVES[name]
    if not sleeve.deployable_with_equity_replay:
        raise ValueError(
            f"{name} uses {sleeve.session_model}; a dedicated continuous-session "
            "dataset and crypto fee/fill model are required"
        )
    return sleeve
