"""
Helper that builds a complete ``TradingNodeConfig`` for the Stage-2 strategy
running through livesim's ``CatalogLiveDataClient`` + ``SimulatedLiveExecutionClient``.

Per §12 of ``docs/nautilus_api_notes.md``:

- ``cache=None`` (in-memory only). ``DatabaseConfig.type`` only supports ``'redis'``
  in v1.227 — there is no local-Parquet Cache state backend (§12.7). The
  fallback for Stage 4 is re-replay on startup.
- ``exec_engine=LiveExecEngineConfig(reconciliation=False)``. Default
  ``reconciliation=True`` causes a 10-second startup hang because stub/sim
  exec clients return no mass-status (§12.3 — verified empirically in §12.9).
- ``strategies`` are declared as ``ImportableStrategyConfig`` — factory-built
  by the kernel during ``node.build()``. (Alternative: build the strategy
  instance directly and attach via ``node.kernel.trader.add_strategy()`` after
  ``node.build()``. Both work — this helper uses the importable path so the
  whole config is declarative.)
- ``LiveDataClientConfig`` and ``LiveExecClientConfig`` are msgspec Structs
  with *typed* fields only — there is no freeform ``options``/``params`` dict
  to thread our extra kwargs (``bar_provider``, ``poll_seconds``,
  ``slippage_prob``, ``random_seed``) through. We therefore use a
  module-level registry keyed by client_id so the factories can look up
  their per-client params at ``node.build()`` time. The ``TradingNodeConfig``
  dict registers the *Nautilus* config (instrument_provider / routing);
  this module registers the *livesim* config separately. Callers must do
  both before ``node.build()``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from nautilus_trader.config import ImportableStrategyConfig
from nautilus_trader.live.config import (
    LiveDataClientConfig,
    LiveExecClientConfig,
    LiveExecEngineConfig,
    TradingNodeConfig,
)
from nautilus_trader.model.data import Bar

from data.instruments import VENUE_MIC


BarProvider = Callable[[], dict[str, list[Bar]]]


@dataclass(frozen=True)
class DataClientParams:
    """Per-data-client params not expressible through ``LiveDataClientConfig``."""

    bar_provider: BarProvider
    poll_seconds: float | None = None


@dataclass(frozen=True)
class ExecClientParams:
    """Per-exec-client params not expressible through ``LiveExecClientConfig``."""

    slippage_prob: float = 0.25
    random_seed: int = 42


# Module-level registries keyed by client_id. The factories read from these
# at ``node.build()`` time (see ``livesim.factories``). Callers must populate
# them BEFORE calling ``node.build()`` — most naturally right after
# ``build_trading_node_config()`` returns and before adding the factories.
_DATA_CLIENT_PARAMS: dict[str, DataClientParams] = {}
_EXEC_CLIENT_PARAMS: dict[str, ExecClientParams] = {}


def register_data_client_params(client_id: str, params: DataClientParams) -> None:
    """Register the runtime params for a data-client name (called before build)."""
    _DATA_CLIENT_PARAMS[client_id] = params


def register_exec_client_params(client_id: str, params: ExecClientParams) -> None:
    """Register the runtime params for an exec-client name (called before build)."""
    _EXEC_CLIENT_PARAMS[client_id] = params


def register_bar_provider(client_id: str, provider: BarProvider) -> None:
    """Convenience: register a bar_provider (without poll_seconds) for a name.

    This is the most common path — fast-replay mode where ``poll_seconds`` is
    ``None``. For continuous-mode tests/runnable, build a ``DataClientParams``
    directly and call ``register_data_client_params``.
    """
    _DATA_CLIENT_PARAMS[client_id] = DataClientParams(bar_provider=provider)


def get_data_client_params(client_id: str) -> DataClientParams:
    """Factory hook: look up a data-client's runtime params by name."""
    if client_id not in _DATA_CLIENT_PARAMS:
        raise KeyError(
            f"no DataClientParams registered for client_id={client_id!r}; "
            "call register_bar_provider(...) or register_data_client_params(...) "
            "before node.build()"
        )
    return _DATA_CLIENT_PARAMS[client_id]


def get_exec_client_params(client_id: str) -> ExecClientParams:
    """Factory hook: look up an exec-client's runtime params by name.

    Falls back to ``ExecClientParams()`` defaults if nothing is registered;
    callers can rely on the default slippage/seed without an explicit setup.
    """
    return _EXEC_CLIENT_PARAMS.get(client_id, ExecClientParams())


def get_bar_provider(client_id: str) -> BarProvider:
    """Convenience accessor used by ``CatalogDataClientFactory.create()``."""
    return get_data_client_params(client_id).bar_provider


def reset_registries() -> None:
    """Clear both registries (use between tests to avoid cross-talk)."""
    _DATA_CLIENT_PARAMS.clear()
    _EXEC_CLIENT_PARAMS.clear()


def build_trading_node_config(
    *,
    strategy_path: str,
    strategy_config_path: str,
    strategy_config_kwargs: dict[str, Any],
    trader_id: str = "STAGE4-001",
    cache_dir: Path | str = "livesim/cache",
) -> TradingNodeConfig:
    """Build a ``TradingNodeConfig`` wired with the strategy + livesim clients.

    The livesim-specific kwargs (``bar_provider``, ``poll_seconds``,
    ``slippage_prob``, ``random_seed``) are NOT parameters of this function —
    register them separately via ``register_data_client_params`` /
    ``register_exec_client_params`` (or the ``register_bar_provider``
    convenience) keyed by ``VENUE_MIC``. This indirection is required
    because ``LiveDataClientConfig`` / ``LiveExecClientConfig`` are typed
    msgspec Structs with no freeform dict field for opaque values. The
    factories look the params up at ``node.build()`` time.

    Args:
        strategy_path: ``'module:ClassName'``
            (e.g. ``'strategies.momentum_topn:MomentumTopNStrategy'``).
        strategy_config_path: ``'module:ConfigClassName'``.
        strategy_config_kwargs: kwargs passed to the StrategyConfig constructor.
        trader_id: TraderId for the node (must contain a hyphen).
        cache_dir: kept as a parameter for future use; v1.227 doesn't support
            local-Parquet Cache persistence (§12.7) so it is currently unused
            but the directory is still created so downstream tooling can
            scratch in it.

    Returns:
        ``TradingNodeConfig`` ready for ``TradingNode(config=...)``.
    """
    # Ensure cache_dir exists for any downstream tooling (snapshot dumps,
    # etc.). v1.227 doesn't wire this into Nautilus's Cache itself.
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    return TradingNodeConfig(
        trader_id=trader_id,
        strategies=[
            ImportableStrategyConfig(
                strategy_path=strategy_path,
                config_path=strategy_config_path,
                config=strategy_config_kwargs,
            ),
        ],
        # The dict key MUST equal the name passed to add_data_client_factory.
        # We use VENUE_MIC (ARCX) so the factory lookup, the registry lookup,
        # and the venue stay consistent.
        data_clients={VENUE_MIC: LiveDataClientConfig()},
        exec_clients={VENUE_MIC: LiveExecClientConfig()},
        # §12.3 / §12.9: must be False for stub/sim exec clients or startup
        # gates on a 10-second reconciliation delay.
        exec_engine=LiveExecEngineConfig(reconciliation=False),
        # §12.7: no local-Parquet Cache backend in v1.227.
        cache=None,
    )
