"""
Factory classes that wire CatalogLiveDataClient + SimulatedLiveExecutionClient
into TradingNodeConfig. Per §12.6 API notes: classes with @staticmethod create()
that returns the instantiated client.

Note: deliberately not named `SandboxLiveExecClientFactory` — the Nautilus
builder injects an extra `portfolio` kwarg into that specific class name
(node_builder.py:246). Naming ours `SimulatedExecClientFactory` avoids it.
"""
from __future__ import annotations

from typing import Any

from nautilus_trader.live.factories import (
    LiveDataClientFactory,
    LiveExecClientFactory,
)

from livesim.data_client import CatalogLiveDataClient
from livesim.exec_client import SimulatedLiveExecutionClient


class CatalogDataClientFactory(LiveDataClientFactory):
    """Factory for CatalogLiveDataClient.

    The `bar_provider` (and optionally `poll_seconds`) are passed in via the
    factory's config dict at TradingNodeConfig construction time.
    """

    @staticmethod
    def create(
        loop,
        name: str,
        config: dict[str, Any],
        msgbus,
        cache,
        clock,
    ) -> CatalogLiveDataClient:
        return CatalogLiveDataClient(
            bar_provider=config["bar_provider"],
            poll_seconds=config.get("poll_seconds"),
            loop=loop,
            client_id=name,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
        )


class SimulatedExecClientFactory(LiveExecClientFactory):
    """Factory for SimulatedLiveExecutionClient.

    Reads slippage_prob + random_seed + (eventually) any other sim-exec
    config from the factory config dict. The Nautilus base clients flow
    through (loop, name, msgbus, cache, clock); the engine wires the rest
    (oms_type, account_type, base_currency, instrument_provider).
    """

    @staticmethod
    def create(
        loop,
        name: str,
        config: dict[str, Any],
        msgbus,
        cache,
        clock,
    ) -> SimulatedLiveExecutionClient:
        return SimulatedLiveExecutionClient(
            slippage_prob=config.get("slippage_prob", 0.25),
            random_seed=config.get("random_seed", 42),
            loop=loop,
            client_id=name,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
            # oms_type, account_type, base_currency, instrument_provider are
            # supplied by Nautilus's exec-engine wiring or the config dict.
            # If they are required at this layer, they should also be read
            # from config — verify the exact path against §12.5.
            **{k: config[k] for k in ("oms_type", "account_type", "base_currency", "instrument_provider") if k in config},
        )
