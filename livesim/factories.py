"""
Factory classes that wire ``CatalogLiveDataClient`` + ``SimulatedLiveExecutionClient``
into ``TradingNodeConfig``. Per §12.6 of the API notes: classes with
``@staticmethod create()`` returning the instantiated client.

Each factory also stores the created instance in a module-level registry
(``_LAST_DATA_CLIENT`` / ``_LAST_EXEC_CLIENT``) keyed by client name so the
integration test can reach the underlying client without grovelling through
``node.kernel.data_engine``'s private ``_clients`` dict (Cython-private).

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

from nautilus_trader.common.providers import InstrumentProvider
from nautilus_trader.live.factories import (
    LiveDataClientFactory,
    LiveExecClientFactory,
)
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import ClientId, Venue

from livesim.alpaca_exec_client import AlpacaLiveExecutionClient
from livesim.data_client import CatalogLiveDataClient
from livesim.exec_client import SimulatedLiveExecutionClient
from livesim.node_config import get_data_client_params, get_exec_client_params


# Side-channel for the integration test to fetch the most-recently-created
# client by name without depending on ``DataEngine._clients`` internals.
_LAST_DATA_CLIENT: dict[str, CatalogLiveDataClient] = {}
_LAST_EXEC_CLIENT: dict[str, SimulatedLiveExecutionClient] = {}


def get_last_data_client(name: str) -> CatalogLiveDataClient:
    """Look up the most recently created data client by name (test helper)."""
    return _LAST_DATA_CLIENT[name]


def get_last_exec_client(name: str) -> SimulatedLiveExecutionClient:
    """Look up the most recently created exec client by name (test helper)."""
    return _LAST_EXEC_CLIENT[name]


def reset_client_registry() -> None:
    """Clear both client registries (use between tests)."""
    _LAST_DATA_CLIENT.clear()
    _LAST_EXEC_CLIENT.clear()


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
        # The dispatch name is the venue MIC (see ``build_trading_node_config``).
        # ``LiveMarketDataClient.__init__`` requires both ``venue`` and
        # ``instrument_provider`` — supply them here rather than in the
        # subclass constructor so the latter stays focused on livesim params.
        client = CatalogLiveDataClient(
            bar_provider=params.bar_provider,
            poll_seconds=params.poll_seconds,
            loop=loop,
            client_id=ClientId(name),
            venue=Venue(name),
            msgbus=msgbus,
            cache=cache,
            clock=clock,
            instrument_provider=InstrumentProvider(),
        )
        _LAST_DATA_CLIENT[name] = client
        return client


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
        # The dispatch name is the venue MIC (see ``build_trading_node_config``).
        # ``LiveExecutionClient.__init__`` requires venue/oms_type/account_type/
        # base_currency/instrument_provider — supply USD-cash defaults here
        # so the subclass constructor stays focused on livesim params. These
        # match ``strategies.backtest_setup.add_default_backtest_venue`` so the
        # parity comparison vs ``BacktestEngine`` is apples-to-apples.
        client = SimulatedLiveExecutionClient(
            slippage_prob=params.slippage_prob,
            random_seed=params.random_seed,
            starting_balance_usd=params.starting_balance_usd,
            loop=loop,
            client_id=ClientId(name),
            venue=Venue(name),
            oms_type=OmsType.NETTING,
            account_type=AccountType.CASH,
            base_currency=USD,
            instrument_provider=InstrumentProvider(),
            msgbus=msgbus,
            cache=cache,
            clock=clock,
        )
        _LAST_EXEC_CLIENT[name] = client
        return client


class AlpacaPaperExecClientFactory(LiveExecClientFactory):
    """Factory for ``AlpacaLiveExecutionClient`` — Stage-8 paper-account adapter.

    Credentials come from ``ALPACA_API_KEY`` / ``ALPACA_API_SECRET`` env vars
    (loaded from .env). Real-money rollout is gated by the deploy checklist;
    this factory always passes ``paper=True``.
    """

    @staticmethod
    def create(
        loop,
        name: str,
        config,
        msgbus,
        cache,
        clock,
    ) -> AlpacaLiveExecutionClient:
        params = get_exec_client_params(name)
        client = AlpacaLiveExecutionClient(
            paper=True,
            starting_balance_usd=params.starting_balance_usd,
            loop=loop,
            client_id=ClientId(name),
            venue=Venue(name),
            oms_type=OmsType.NETTING,
            account_type=AccountType.CASH,
            base_currency=USD,
            instrument_provider=InstrumentProvider(),
            msgbus=msgbus,
            cache=cache,
            clock=clock,
        )
        _LAST_EXEC_CLIENT[name] = client
        return client
