"""
Factory classes that wire ``CatalogLiveDataClient`` + ``SimulatedLiveExecutionClient``
into ``TradingNodeConfig``. Per §12.6 of the API notes: classes with
``@staticmethod create()`` returning the instantiated client.

Note: deliberately not named ``SandboxLiveExecClientFactory`` — the Nautilus
builder injects an extra ``portfolio`` kwarg into that specific class name
(``node_builder.py:246``). Naming ours ``SimulatedExecClientFactory`` avoids it.

The ``config`` argument the builder passes to ``create()`` is a typed
msgspec Struct (``LiveDataClientConfig`` / ``LiveExecClientConfig``), NOT a
dict. Those structs in v1.227 have no freeform ``options``/``params`` dict
to thread our extra kwargs (``bar_provider``, ``poll_seconds``,
``slippage_prob``, ``random_seed``) through. We therefore read them from the
module-level registry in ``livesim.node_config``, keyed by the client name
the builder dispatches with.
"""
from __future__ import annotations

from nautilus_trader.live.factories import (
    LiveDataClientFactory,
    LiveExecClientFactory,
)

from livesim.data_client import CatalogLiveDataClient
from livesim.exec_client import SimulatedLiveExecutionClient
from livesim.node_config import get_data_client_params, get_exec_client_params


class CatalogDataClientFactory(LiveDataClientFactory):
    """Factory for ``CatalogLiveDataClient``.

    ``bar_provider`` (and optionally ``poll_seconds``) are read from the
    module-level registry in ``livesim.node_config`` rather than from the
    ``config`` arg, because ``LiveDataClientConfig`` is a typed struct with
    no freeform dict (§12.6).
    """

    @staticmethod
    def create(
        loop,
        name: str,
        config,
        msgbus,
        cache,
        clock,
    ) -> CatalogLiveDataClient:
        params = get_data_client_params(name)
        return CatalogLiveDataClient(
            bar_provider=params.bar_provider,
            poll_seconds=params.poll_seconds,
            loop=loop,
            client_id=name,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
        )


class SimulatedExecClientFactory(LiveExecClientFactory):
    """Factory for ``SimulatedLiveExecutionClient``.

    ``slippage_prob`` / ``random_seed`` come from the module-level registry
    in ``livesim.node_config`` (same rationale as ``CatalogDataClientFactory``
    — ``LiveExecClientConfig`` is a typed struct, not a dict). If no entry is
    registered, defaults from ``ExecClientParams()`` are used.
    """

    @staticmethod
    def create(
        loop,
        name: str,
        config,
        msgbus,
        cache,
        clock,
    ) -> SimulatedLiveExecutionClient:
        params = get_exec_client_params(name)
        return SimulatedLiveExecutionClient(
            slippage_prob=params.slippage_prob,
            random_seed=params.random_seed,
            loop=loop,
            client_id=name,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
            # oms_type, account_type, base_currency, instrument_provider are
            # supplied by Nautilus's exec-engine wiring at the kernel layer;
            # if a future SimulatedLiveExecutionClient grows required ctor
            # args, plumb them through ExecClientParams.
        )
