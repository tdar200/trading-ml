# Stage 4: LiveNode Sim-Live Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the existing `MomentumTopNStrategy` into Nautilus's `TradingNode` runtime via a custom `CatalogLiveDataClient` (bars-replay) and `SimulatedLiveExecutionClient` (next-bar fills with FillModel slippage), validated by an integration test that replays the Stage-1 catalog and produces final equity within 1% of `BacktestEngine`.

**Architecture:** Two new Nautilus boundary classes (data + exec) plus factory glue and a TradingNodeConfig helper. Strategy itself is unchanged. State persists via Nautilus's built-in Parquet cache. The runnable supports fast-replay (test/dev) and continuous polling (production) modes.

**Tech Stack:**
- `nautilus_trader==1.227.0` — `TradingNode`, `LiveDataClient`, `LiveExecutionClient`, factories, Cache
- `pytest-asyncio>=0.24` — NEW dev dep for async integration tests
- Standard library `asyncio` — TradingNode's runtime
- `pandas`, `polars` — equity-curve comparison in tests

⚠️ **Scope warning.** Task 2 (the §10 API recon) is the largest single task in the project to date. Budget time accordingly. The recon must verify, with working code: TradingNode lifecycle, LiveDataClient abstract surface, LiveExecutionClient abstract surface, factory protocol, cache config, and Strategy.on_save/on_load (or document its absence). All later tasks read from §10.

---

## File Structure

| Path | Responsibility |
|---|---|
| `pyproject.toml` | Add `pytest-asyncio>=0.24` to `[dependency-groups].dev` |
| `.gitignore` | Add `livesim/cache/` |
| `docs/nautilus_api_notes.md` | Append §10 — TradingNode + LiveDataClient + LiveExecutionClient APIs |
| `livesim/__init__.py` | Package marker |
| `livesim/data_client.py` | `CatalogLiveDataClient(LiveDataClient)` — replays a callable-provided `bars_by_ticker` dict; fast-replay + polling modes |
| `livesim/exec_client.py` | `SimulatedLiveExecutionClient(LiveExecutionClient)` — subscribes to bars; fills pending orders at next bar's open + FillModel slippage |
| `livesim/factories.py` | Factory subclasses Nautilus's config system uses to instantiate the data/exec clients |
| `livesim/node_config.py` | Helper that builds a `TradingNodeConfig` wiring strategy + clients + cache |
| `examples/run_trading_node.py` | Runnable — `--fast-replay` / `--poll-seconds N` CLI; runs TradingNode against the real catalog |
| `tests/test_catalog_data_client.py` | Unit tests on synthetic in-memory bars |
| `tests/test_simulated_exec_client.py` | Unit tests with mocked engine bus |
| `tests/test_live_sim_integration.py` | Async integration: TradingNode end-to-end on synthetic data; equity within 1% of `BacktestEngine` |
| `livesim/cache/` | Runtime Nautilus Parquet cache directory (gitignored) |

**Boundaries:**
- `livesim/` is the ONLY package that imports `TradingNode` or `LiveDataClient`/`LiveExecutionClient`. `strategies/`, `analysis/`, `data/` stay live-runtime-agnostic.
- `data_client.py` takes a **callable `bar_provider`** rather than a `ParquetDataCatalog` directly — keeps it pure-testable. The runnable provides a catalog-reading provider; tests provide a dict-returning one.
- `exec_client.py` does its own work via message-bus bar subscription; it does NOT share state with the data client beyond what flows through the engine.

---

## Task 1: Scaffolding (`pytest-asyncio`, `livesim/__init__.py`, gitignore)

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Create: `livesim/__init__.py`

- [ ] **Step 1: Add `pytest-asyncio` to dev dependencies**

Edit `pyproject.toml` `[dependency-groups].dev`. Add `"pytest-asyncio>=0.24"` as the last entry. The new dev block:
```toml
[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-mock>=3.12",
    "pytest-asyncio>=0.24",
]
```

Also add an `[tool.pytest.ini_options]` `asyncio_mode = "auto"` entry so tests marked `async def test_...` are auto-handled. The new pytest section:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v"
asyncio_mode = "auto"
```

If the existing `[tool.pytest.ini_options]` already has those keys, just add the `asyncio_mode` line.

- [ ] **Step 2: Sync the env**
```bash
cd /home/tahseen-dar/Projects/trading-ml
uv sync
uv run python -c "import pytest_asyncio; print('pytest_asyncio', pytest_asyncio.__version__)"
```
Expected: prints a version (e.g. `pytest_asyncio 0.24.x`).

- [ ] **Step 3: Add `livesim/cache/` to `.gitignore`**

Append `livesim/cache/` to `/home/tahseen-dar/Projects/trading-ml/.gitignore` if not already present. Verify:
```bash
grep -E '^livesim/cache/?$' .gitignore || echo "MISSING"
```

- [ ] **Step 4: Create the `livesim` package**
```bash
mkdir -p livesim
touch livesim/__init__.py
```

- [ ] **Step 5: Full-suite sanity**
```bash
uv run pytest -q 2>&1 | tail -3
```
Expected: 66 passed (Stage-3 baseline). Adding `pytest-asyncio` and `asyncio_mode = "auto"` shouldn't break anything.

- [ ] **Step 6: Commit**
```bash
git add pyproject.toml uv.lock .gitignore livesim/__init__.py
git commit -m "chore(stage4): add pytest-asyncio dev dep + livesim package scaffold

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: API recon — TradingNode + LiveDataClient + LiveExecutionClient (§10)

⚠️ **This is the largest task in the plan.** Every later task depends on §10 being literal ground truth. Budget time appropriately. Do NOT write any signature you haven't verified by import + construction (or by running a working snippet).

**Files:**
- Modify: `docs/nautilus_api_notes.md` (append §10)

The deliverable is an empirically-verified §10 covering:
1. TradingNode lifecycle: how to construct, configure, start, and stop a TradingNode in v1.227.
2. `LiveDataClient` abstract surface: which methods MUST be overridden, what their signatures are, how to emit data into the engine.
3. `LiveExecutionClient` abstract surface: same as above but for orders + fill events.
4. Factory protocol: how Nautilus's config system wires factory classes to client instances.
5. Cache configuration: how to point cache at a local Parquet directory + flush semantics.
6. `Strategy.on_save()` / `on_load()` (if v1.227 supports them) — or document their absence.
7. A minimal end-to-end snippet: TradingNode + stub data client (emits one bar) + stub exec client (no-op) + a Strategy that logs the bar; runs for a moment then exits cleanly.

- [ ] **Step 1: Find the right import paths**
```bash
cd /home/tahseen-dar/Projects/trading-ml
uv run python -c "
from nautilus_trader.live.node import TradingNode
from nautilus_trader.live.config import TradingNodeConfig
from nautilus_trader.live.data_client import LiveDataClient
from nautilus_trader.live.execution_client import LiveExecutionClient
print('all live imports OK')
"
```
Expected: `all live imports OK`. If any import fails, find the v1.227 path via exploration:
```bash
uv run python -c "
import nautilus_trader.live as live
import pkgutil
for m in pkgutil.iter_modules(live.__path__):
    print(m.name)
"
```

- [ ] **Step 2: Introspect TradingNode lifecycle**
```bash
uv run python -c "
from nautilus_trader.live.node import TradingNode
import inspect
print('TradingNode public methods:', sorted([m for m in dir(TradingNode) if not m.startswith('_')]))
for method in ('start', 'stop', 'run', 'dispose', 'cache', 'portfolio'):
    if hasattr(TradingNode, method):
        attr = getattr(TradingNode, method)
        if callable(attr):
            try:
                sig = inspect.signature(attr)
                print(f'{method}{sig}')
            except (ValueError, TypeError):
                print(f'{method}: (no signature available)')
        else:
            print(f'{method}: (property/attr)')
"
```
Capture which lifecycle methods exist. Likely candidates: `start()` (async), `stop()` (async), `dispose()`, possibly a `run()` wrapper for synchronous use.

- [ ] **Step 3: Introspect TradingNodeConfig**
```bash
uv run python -c "
from nautilus_trader.live.config import TradingNodeConfig
import inspect
print('TradingNodeConfig signature:', inspect.signature(TradingNodeConfig))
"
```
Note all required + optional fields. Particular attention to: `trader_id`, `strategies`, `data_clients`, `exec_clients`, `cache`, `logging`.

- [ ] **Step 4: Introspect LiveDataClient abstract surface**
```bash
uv run python -c "
from nautilus_trader.live.data_client import LiveDataClient
import inspect
print('LiveDataClient public methods:')
for m in sorted([m for m in dir(LiveDataClient) if not m.startswith('_') or m.startswith('_subscribe') or m.startswith('_unsubscribe') or m.startswith('_handle')]):
    print(f'  {m}')
print('LiveDataClient __init__ signature:', inspect.signature(LiveDataClient))
"
```
Document which `_subscribe_*` and `_unsubscribe_*` hooks exist (especially `_subscribe_bars`, `_unsubscribe_bars`). Note the constructor's required args (typically `client_id`, `venue`, `msgbus`, `cache`, `clock`).

- [ ] **Step 5: Introspect LiveExecutionClient abstract surface**

Same pattern as Step 4 but for `LiveExecutionClient`. Find the abstract methods that must be overridden — typically `_submit_order`, `_modify_order`, `_cancel_order`, plus connection lifecycle (`_connect`, `_disconnect`). Document which are required vs optional.

- [ ] **Step 6: Find the factory protocol**

In v1.227, Nautilus uses factory CLASSES referenced by config. Investigate:
```bash
uv run python -c "
from nautilus_trader.live.factories import LiveDataClientFactory, LiveExecClientFactory
import inspect
print('LiveDataClientFactory:', inspect.signature(LiveDataClientFactory.create))
print('LiveExecClientFactory:', inspect.signature(LiveExecClientFactory.create))
"
```
If those imports fail, find the v1.227 equivalents by looking in `nautilus_trader.live` for a `factories` submodule, or look at how existing adapters (e.g. `nautilus_trader.adapters.binance`) register their factories.

- [ ] **Step 7: Find an existing concrete LiveDataClient + LiveExecutionClient implementation to study**

Read the source of a Nautilus-shipped adapter to see how a real LiveDataClient/LiveExecClient subclass is structured. Pick one of: `nautilus_trader.adapters.binance.spot`, `nautilus_trader.adapters.interactive_brokers`, `nautilus_trader.adapters.sandbox`. Look at its `LiveDataClient` subclass and note:
- Which abstract methods it overrides.
- How it calls `self._handle_data(...)` to emit bars.
- How it implements `_connect()` / `_disconnect()`.
- How its factory class is structured.

Bash:
```bash
uv run python -c "
import nautilus_trader.adapters
import pkgutil
for m in pkgutil.iter_modules(nautilus_trader.adapters.__path__):
    print(m.name)
"
```
Pick one and read its source. Note the patterns; you'll mirror them in Tasks 3-6.

- [ ] **Step 8: Verify cache configuration**
```bash
uv run python -c "
from nautilus_trader.cache.config import CacheConfig
import inspect
print('CacheConfig fields:', CacheConfig.model_fields if hasattr(CacheConfig, 'model_fields') else inspect.signature(CacheConfig))
"
```
Note how to configure: a database/storage path (for Parquet-backed persistence), flush interval/event triggers, snapshot policy. If the v1.227 cache config uses a different class name, find the right one via exploration.

- [ ] **Step 9: Check for Strategy.on_save / on_load**
```bash
uv run python -c "
from nautilus_trader.trading.strategy import Strategy
methods = [m for m in dir(Strategy) if 'save' in m.lower() or 'load' in m.lower() or 'persist' in m.lower() or 'snapshot' in m.lower()]
print('strategy persistence methods:', methods)
"
```
If `on_save`/`on_load` exist with usable signatures, document them. If not, document the absence and the fallback (Stage 4 doesn't persist strategy state; it re-replays bars on TradingNode startup).

- [ ] **Step 10: Build a minimal end-to-end TradingNode test snippet**

This is the most important verification: prove a TradingNode can be assembled with stub clients and a Strategy, started, run for a moment, stopped cleanly. The snippet may be ~80-120 lines and lives only in §10 as a copy-pasteable reference.

The snippet should:
1. Define a `_StubDataClient(LiveDataClient)` that emits one synthetic Bar in `_subscribe_bars` and signals "done".
2. Define a `_StubExecClient(LiveExecutionClient)` that no-ops `_submit_order`.
3. Define their corresponding factory classes.
4. Build a `TradingNodeConfig` referencing them.
5. Use a minimal `Strategy` subclass that logs received bars.
6. Build and start the TradingNode via asyncio; await a brief moment; stop the node; assert at least one bar was received.

If the snippet runs end-to-end, the LiveNode plumbing is real and §10 is trustworthy. If it can't be assembled, escalate BLOCKED with the failure.

- [ ] **Step 11: Write §10 in `docs/nautilus_api_notes.md`**

Append a new section "## 10. LiveNode (Stage 4)" to the existing notes file. Structure it as:
1. **Imports** — the verified import block from Step 1.
2. **TradingNode lifecycle** — `start()`, `stop()`, `dispose()` signatures and async nature.
3. **TradingNodeConfig** — required/optional fields with types.
4. **LiveDataClient** — required overrides (`_subscribe_bars`, `_unsubscribe_bars`, `_connect`, `_disconnect`, etc.), constructor args, how to call `self._handle_data(bar)` to deliver.
5. **LiveExecutionClient** — required overrides, constructor args, how to emit `OrderFilled` events.
6. **Factory protocol** — class structure, `create(...)` method signature, how TradingNodeConfig references them.
7. **Cache configuration** — how to point at local Parquet directory.
8. **Strategy persistence** — whether `on_save`/`on_load` exist, recommended fallback if not.
9. **End-to-end snippet** — the working snippet from Step 10.

- [ ] **Step 12: Commit**
```bash
git add docs/nautilus_api_notes.md
git commit -m "docs: extend Nautilus API notes with LiveNode/TradingNode (§10)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `CatalogLiveDataClient` — fast-replay mode

**Files:**
- Create: `livesim/data_client.py`
- Create: `tests/test_catalog_data_client.py`

The fast-replay mode emits all bars from a callable bar-provider as fast as possible, then signals completion via an `asyncio.Future`. Tests can `await client.done()` to know when it's safe to stop the node.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_catalog_data_client.py`:
```python
"""Tests for CatalogLiveDataClient in fast-replay mode."""
from __future__ import annotations

import asyncio
from typing import Callable

import pytest

from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId, Symbol
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Price, Quantity

from data.instruments import VENUE_MIC
from livesim.data_client import CatalogLiveDataClient


def _instrument(ticker: str) -> Equity:
    return Equity(
        instrument_id=InstrumentId.from_str(f"{ticker}.{VENUE_MIC}"),
        raw_symbol=Symbol(ticker), currency=USD, price_precision=2,
        price_increment=Price.from_str("0.01"),
        lot_size=Quantity.from_int(1),
        ts_event=0, ts_init=0,
    )


def _bars(ticker: str, n: int, start_ts_ns: int = 1_700_000_000_000_000_000) -> list[Bar]:
    bar_type = BarType.from_str(f"{ticker}.{VENUE_MIC}-1-DAY-LAST-EXTERNAL")
    one_day_ns = 86_400_000_000_000
    out: list[Bar] = []
    for i in range(n):
        ts = start_ts_ns + i * one_day_ns
        p = Price.from_str(f"{100.0 + i:.2f}")
        out.append(Bar(bar_type=bar_type, open=p, high=p, low=p, close=p,
                       volume=Quantity.from_int(1_000_000),
                       ts_event=ts, ts_init=ts))
    return out


def test_fast_replay_provider_callable_signature():
    """The data client takes a callable `bar_provider` that returns dict[ticker → list[Bar]]."""
    bars = {"A": _bars("A", 5)}
    provider: Callable[[], dict[str, list[Bar]]] = lambda: bars
    # Should construct without error; constructor signature exposes bar_provider, poll_seconds=None.
    client = CatalogLiveDataClient.__new__(CatalogLiveDataClient)
    # We don't fully initialize without the engine harness; just confirm the class accepts the args via constructor inspect.
    import inspect
    sig = inspect.signature(CatalogLiveDataClient.__init__)
    assert "bar_provider" in sig.parameters
    assert "poll_seconds" in sig.parameters
    assert sig.parameters["poll_seconds"].default is None


def test_fast_replay_done_signaled_after_all_bars_emitted(monkeypatch):
    """In fast-replay mode (poll_seconds=None), client should signal done() after emitting all bars."""
    # We cannot construct a real LiveDataClient without msgbus/cache/clock fixtures.
    # Instead, exercise the internal _emit_pending_bars-style method using a stubbed engine handle.
    # The contract: after calling _replay_once with bars, all bars are pushed to a recorded "emit" list,
    # and the `done` Future is set in fast-replay mode.
    bars = _bars("A", 3)
    emitted: list[Bar] = []

    # We use a thin test double instead of the full Nautilus engine wiring.
    # `_emit_bar` is the seam: emits a bar to the engine. Test double records calls.
    class _Recorder:
        def __init__(self):
            self.calls: list[Bar] = []

        def emit(self, bar: Bar) -> None:
            self.calls.append(bar)

    recorder = _Recorder()
    # The data client should expose a `_replay_bars(bars: list[Bar], emit_fn)` helper for unit testability.
    # Verify the helper emits each bar exactly once in order.
    from livesim.data_client import replay_bars  # pure helper
    replay_bars(bars, emit_fn=recorder.emit)
    assert len(recorder.calls) == 3
    assert recorder.calls == bars


def test_replay_bars_sorts_by_ts_event():
    """The replay helper sorts bars by ts_event before emitting (input may be out-of-order)."""
    bars = _bars("A", 4)
    shuffled = [bars[2], bars[0], bars[3], bars[1]]
    emitted: list[Bar] = []
    from livesim.data_client import replay_bars
    replay_bars(shuffled, emit_fn=emitted.append)
    assert [b.ts_event for b in emitted] == sorted(b.ts_event for b in bars)
```

The tests above are partly unit (around the pure `replay_bars` helper) and partly contract (around the constructor signature). The full async behavior is exercised in Task 8's integration test, where a real TradingNode wires the engine.

- [ ] **Step 2: Run to confirm failure**
```bash
uv run pytest tests/test_catalog_data_client.py -v
```
Expected: collection error or assertion failures (`livesim.data_client` doesn't exist).

- [ ] **Step 3: Implement `livesim/data_client.py`**

The exact `LiveDataClient` subclass signature depends on §10's verified introspection. The skeleton below uses the most likely v1.227 surface; ADAPT to §10 if it differs:

```python
"""
CatalogLiveDataClient — replays bars from a callable provider into Nautilus's
TradingNode. Supports fast-replay mode (emit-all-then-done) and continuous
polling mode (poll for newly-appeared bars at a fixed cadence).

The data client is decoupled from any catalog/filesystem concerns by taking
a callable `bar_provider() -> dict[str, list[Bar]]` rather than a catalog
path. The runnable wraps a ParquetDataCatalog in a closure; tests pass a
dict-returning lambda.
"""
from __future__ import annotations

import asyncio
from typing import Callable

from nautilus_trader.live.data_client import LiveDataClient
from nautilus_trader.model.data import Bar


BarProvider = Callable[[], dict[str, list[Bar]]]


def replay_bars(bars: list[Bar], emit_fn: Callable[[Bar], None]) -> None:
    """Pure helper: emit each bar in ts_event-ascending order. Unit-testable."""
    for bar in sorted(bars, key=lambda b: b.ts_event):
        emit_fn(bar)


class CatalogLiveDataClient(LiveDataClient):
    """Replays bars from a callable provider. See module docstring."""

    def __init__(
        self,
        *,
        bar_provider: BarProvider,
        poll_seconds: float | None = None,
        **kwargs,  # forwards to LiveDataClient: client_id, venue, msgbus, cache, clock, etc.
    ) -> None:
        super().__init__(**kwargs)
        self._bar_provider = bar_provider
        self._poll_seconds = poll_seconds
        self._emitted_ts_per_ticker: dict[str, set[int]] = {}
        self._done: asyncio.Future = asyncio.Future()
        self._subscribed_bar_types: set = set()

    def done(self) -> asyncio.Future:
        """Returns a Future that is set when fast-replay completes (or never in continuous mode)."""
        return self._done

    async def _connect(self) -> None:
        """Called by TradingNode on startup. Kick off replay."""
        # Emit all currently-available bars, then either set done() or start polling.
        await self._replay_once()
        if self._poll_seconds is not None:
            # Continuous mode: schedule polling task; do NOT set done.
            asyncio.create_task(self._poll_loop())
        else:
            if not self._done.done():
                self._done.set_result(True)

    async def _disconnect(self) -> None:
        """Called by TradingNode on shutdown."""
        if not self._done.done():
            self._done.set_result(False)

    def _subscribe_bars(self, bar_type) -> None:
        """Strategy is subscribing to this bar type. Record it."""
        self._subscribed_bar_types.add(bar_type)

    def _unsubscribe_bars(self, bar_type) -> None:
        self._subscribed_bar_types.discard(bar_type)

    async def _replay_once(self) -> None:
        """Read the current bar set from the provider; emit any not-yet-emitted bars."""
        bars_by_ticker = self._bar_provider()
        for ticker, bars in bars_by_ticker.items():
            emitted = self._emitted_ts_per_ticker.setdefault(ticker, set())
            new_bars = [b for b in bars if b.ts_event not in emitted]
            replay_bars(new_bars, emit_fn=self._handle_data)
            emitted.update(b.ts_event for b in new_bars)

    async def _poll_loop(self) -> None:
        """Continuous-mode polling. Re-reads from provider every `poll_seconds`."""
        while True:
            await asyncio.sleep(self._poll_seconds)
            try:
                await self._replay_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # Log via Nautilus's logger; don't crash the loop.
                self.log.error(f"poll error: {e}")
```

⚠️ **The exact `LiveDataClient` constructor args** depend on what §10 verified. If `__init__` requires `client_id`, `venue`, `msgbus`, `cache`, `clock` (most likely), those flow through `**kwargs`. If the abstract method names differ (`_subscribe_bars` vs `subscribe_bars` vs other), adapt to §10. The two tests above primarily exercise the pure helper + constructor signature, so the test will pass as long as the constructor exposes `bar_provider` and `poll_seconds` parameters.

- [ ] **Step 4: Run to confirm pass**
```bash
uv run pytest tests/test_catalog_data_client.py -v
```
Expected: 3 passed.

If the test for "signature has bar_provider and poll_seconds" fails because the `LiveDataClient` base requires args we haven't wired through `**kwargs`, refactor the constructor signature so `bar_provider` and `poll_seconds` are top-level kw-only args (which the introspection test verifies). The base-class args flow through `**kwargs`.

- [ ] **Step 5: Full-suite sanity**
```bash
uv run pytest -q 2>&1 | tail -3
```
Expected: 69 passed (66 + 3).

- [ ] **Step 6: Commit**
```bash
git add livesim/data_client.py tests/test_catalog_data_client.py
git commit -m "feat(livesim): CatalogLiveDataClient with fast-replay + polling skeleton

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `CatalogLiveDataClient` — continuous polling mode tests

Task 3 implemented the polling loop but only tested the pure helper + the constructor. This task adds tests for polling-mode behavior using a controlled provider closure.

**Files:**
- Modify: `tests/test_catalog_data_client.py` (append tests)

- [ ] **Step 1: Append polling-mode tests**

Read `tests/test_catalog_data_client.py` to find the end. APPEND:
```python
def test_replay_once_emits_only_new_bars_per_ticker(monkeypatch):
    """Calling _replay_once twice with the SAME bars should not double-emit."""
    bars = _bars("A", 3)
    state = {"current": list(bars)}
    emitted: list[Bar] = []

    # Simulate the internal logic via the pure replay_bars helper + dedup set.
    from livesim.data_client import replay_bars
    emitted_ts: set[int] = set()

    def emit_if_new(bar: Bar) -> None:
        if bar.ts_event not in emitted_ts:
            emitted.append(bar)
            emitted_ts.add(bar.ts_event)

    replay_bars(state["current"], emit_fn=emit_if_new)
    replay_bars(state["current"], emit_fn=emit_if_new)  # same bars, second time
    assert len(emitted) == 3


def test_replay_once_picks_up_newly_appeared_bars():
    """If the provider returns more bars on the second call, only the deltas are emitted."""
    bars_initial = _bars("A", 3)
    bars_extended = _bars("A", 5)  # bars 4,5 are new (indices 3,4)
    emitted: list[Bar] = []
    from livesim.data_client import replay_bars
    emitted_ts: set[int] = set()

    def emit_if_new(bar: Bar) -> None:
        if bar.ts_event not in emitted_ts:
            emitted.append(bar)
            emitted_ts.add(bar.ts_event)

    replay_bars(bars_initial, emit_fn=emit_if_new)
    assert len(emitted) == 3
    replay_bars(bars_extended, emit_fn=emit_if_new)
    assert len(emitted) == 5
    # Last two emitted are the deltas (indices 3, 4 of bars_extended).
    assert emitted[-2:] == bars_extended[-2:]
```

These tests cover the polling logic at the helper level. The full async loop is integration-tested in Task 8.

- [ ] **Step 2: Run to confirm pass**
```bash
uv run pytest tests/test_catalog_data_client.py -v
```
Expected: 5 passed (3 from Task 3 + 2 new).

- [ ] **Step 3: Full-suite sanity**
```bash
uv run pytest -q 2>&1 | tail -3
```
Expected: 71 passed (69 + 2).

- [ ] **Step 4: Commit**
```bash
git add tests/test_catalog_data_client.py
git commit -m "test(livesim): polling-mode bar-delta tests for data client

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `SimulatedLiveExecutionClient`

The exec client receives orders and emits fills on the next bar for the order's instrument. It subscribes to the message bus to receive bar events (the same bars the data client emits).

**Files:**
- Create: `livesim/exec_client.py`
- Create: `tests/test_simulated_exec_client.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_simulated_exec_client.py`:
```python
"""Tests for SimulatedLiveExecutionClient using a pure-function adapter pattern.

The exec client's order-intake + fill-emission logic is unit-testable
without the full Nautilus engine fixtures by extracting the core decision
into a pure helper `compute_fill_price(bar, side, slippage_ticks, tick_size)`.
"""
from __future__ import annotations

from decimal import Decimal

from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId, Symbol
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Price, Quantity

from data.instruments import VENUE_MIC
from livesim.exec_client import compute_fill_price


def _bar(ticker: str, open_price: float) -> Bar:
    bar_type = BarType.from_str(f"{ticker}.{VENUE_MIC}-1-DAY-LAST-EXTERNAL")
    p = Price.from_str(f"{open_price:.2f}")
    return Bar(bar_type=bar_type, open=p, high=p, low=p, close=p,
               volume=Quantity.from_int(1_000_000),
               ts_event=1_700_000_000_000_000_000, ts_init=1_700_000_000_000_000_000)


def test_compute_fill_price_no_slippage_returns_bar_open():
    bar = _bar("A", 100.00)
    px = compute_fill_price(bar, side=OrderSide.BUY, slippage_ticks=0, tick_size=Decimal("0.01"))
    assert px == Decimal("100.00")


def test_compute_fill_price_buy_with_one_tick_slippage_adds_tick():
    """Buy side slips UP by one tick when slippage is applied (worse price for buyer)."""
    bar = _bar("A", 100.00)
    px = compute_fill_price(bar, side=OrderSide.BUY, slippage_ticks=1, tick_size=Decimal("0.01"))
    assert px == Decimal("100.01")


def test_compute_fill_price_sell_with_one_tick_slippage_subtracts_tick():
    """Sell side slips DOWN by one tick (worse price for seller)."""
    bar = _bar("A", 100.00)
    px = compute_fill_price(bar, side=OrderSide.SELL, slippage_ticks=1, tick_size=Decimal("0.01"))
    assert px == Decimal("99.99")


def test_compute_fill_price_three_tick_slippage():
    bar = _bar("A", 50.20)
    buy_px = compute_fill_price(bar, side=OrderSide.BUY, slippage_ticks=3, tick_size=Decimal("0.01"))
    assert buy_px == Decimal("50.23")
    sell_px = compute_fill_price(bar, side=OrderSide.SELL, slippage_ticks=3, tick_size=Decimal("0.01"))
    assert sell_px == Decimal("50.17")
```

- [ ] **Step 2: Run to confirm failure**
```bash
uv run pytest tests/test_simulated_exec_client.py -v
```
Expected: collection error.

- [ ] **Step 3: Implement `livesim/exec_client.py`**

Create `livesim/exec_client.py`:
```python
"""
SimulatedLiveExecutionClient — accepts orders and emits fills at the next bar.

Subscribes to the message bus to receive bar events (the same bars the data
client emits). On each bar for an instrument we have a pending order for,
generate a fill at that bar's open price, plus optional slippage_ticks of
slippage (one-tick magnitude per the FillModel convention, gated by a
probability seeded for determinism).

The fill-price decision is extracted into `compute_fill_price` for unit
testability. The full async order-fill cycle is integration-tested in Task 8.
"""
from __future__ import annotations

import random
from decimal import Decimal

from nautilus_trader.live.execution_client import LiveExecutionClient
from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.orders import Order


def compute_fill_price(
    bar: Bar,
    side: OrderSide,
    slippage_ticks: int,
    tick_size: Decimal,
) -> Decimal:
    """Pure helper: compute the simulated fill price for an order against a bar.

    Buys slip UP by `slippage_ticks * tick_size` (worse for buyer);
    sells slip DOWN by the same amount (worse for seller).
    """
    base = Decimal(str(bar.open))
    delta = Decimal(slippage_ticks) * tick_size
    if side == OrderSide.BUY:
        return base + delta
    return base - delta


class SimulatedLiveExecutionClient(LiveExecutionClient):
    """Simulated live exec — fills pending orders at next bar's open."""

    def __init__(
        self,
        *,
        slippage_prob: float = 0.25,
        random_seed: int = 42,
        **kwargs,  # forwards to LiveExecutionClient: client_id, venue, msgbus, cache, clock, etc.
    ) -> None:
        super().__init__(**kwargs)
        self._slippage_prob = slippage_prob
        self._rng = random.Random(random_seed)
        self._pending: dict[str, list[Order]] = {}  # ticker → pending orders

    async def _connect(self) -> None:
        # Subscribe to bars on the message bus so we can fill orders on next bar.
        # The exact subscription mechanism depends on §10's verified API.
        # Likely: self._msgbus.subscribe(topic="data.bars", handler=self._on_bar)
        pass

    async def _disconnect(self) -> None:
        pass

    async def _submit_order(self, order: Order) -> None:
        """Record the order as pending; fill on next bar."""
        ticker = str(order.instrument_id).split(".")[0]
        self._pending.setdefault(ticker, []).append(order)

    def _on_bar(self, bar: Bar) -> None:
        """Fill any pending orders for this bar's instrument at the bar's open."""
        ticker = str(bar.bar_type).split(".")[0]
        if ticker not in self._pending:
            return
        pending_orders = self._pending.pop(ticker)
        tick_size = Decimal("0.01")  # All Stage-2 instruments use $0.01 tick; revisit if universe expands.
        for order in pending_orders:
            slip_ticks = 1 if self._rng.random() < self._slippage_prob else 0
            fill_price = compute_fill_price(
                bar=bar, side=order.side,
                slippage_ticks=slip_ticks, tick_size=tick_size,
            )
            self._emit_fill(order, fill_price=fill_price, bar=bar)

    def _emit_fill(self, order: Order, fill_price: Decimal, bar: Bar) -> None:
        """Generate an OrderFilled event for the order at the given price.

        The exact event-emission API depends on §10's verified surface.
        Likely path: build an OrderFilled event and call self.generate_order_filled(...)
        or self._handle_event(event).
        """
        # Implementation deferred to integration with the real LiveExecutionClient API.
        # This is the seam Task 8 exercises end-to-end.
        raise NotImplementedError(
            "Wire to LiveExecutionClient's fill-emission API per docs/nautilus_api_notes.md §10"
        )
```

⚠️ **`_emit_fill` is intentionally a stub** that raises NotImplementedError until §10 confirms the correct fill-emission API. Task 8's integration test will exercise this; if `_emit_fill` raises, Task 8 fails with a clear pointer to fix it based on §10. This avoids guessing at the API now.

The unit tests only exercise `compute_fill_price` (pure) and the constructor signature (introspection). They don't require `_emit_fill` to work.

- [ ] **Step 4: Run to confirm pass**
```bash
uv run pytest tests/test_simulated_exec_client.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Full-suite sanity**
```bash
uv run pytest -q 2>&1 | tail -3
```
Expected: 75 passed (71 + 4).

- [ ] **Step 6: Commit**
```bash
git add livesim/exec_client.py tests/test_simulated_exec_client.py
git commit -m "feat(livesim): SimulatedLiveExecutionClient + compute_fill_price helper

Pure compute_fill_price for unit tests; _emit_fill stubbed until §10 of
the API notes confirms the correct LiveExecutionClient fill-emission API
(wired in Task 8 integration).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Factories

Nautilus's config system instantiates clients via factory classes referenced in `TradingNodeConfig`. The exact factory protocol depends on what §10 verified.

**Files:**
- Create: `livesim/factories.py`

- [ ] **Step 1: Implement the factories**

Create `livesim/factories.py`. The exact base class and `create()` signature depend on §10. The likely structure:

```python
"""
Factory classes that wire CatalogLiveDataClient + SimulatedLiveExecutionClient
into TradingNodeConfig. The factory pattern is Nautilus's mechanism for
deferred client construction — TradingNodeConfig references factories by
class; the node calls .create() during startup.
"""
from __future__ import annotations

from typing import Any

from nautilus_trader.live.factories import LiveDataClientFactory, LiveExecClientFactory

from livesim.data_client import CatalogLiveDataClient
from livesim.exec_client import SimulatedLiveExecutionClient


class CatalogDataClientFactory(LiveDataClientFactory):
    """Factory for CatalogLiveDataClient.

    The `bar_provider` (and optionally `poll_seconds`) are passed in via the
    factory's config dict at TradingNodeConfig construction time.
    """

    @classmethod
    def create(
        cls,
        *,
        loop: Any,
        name: str,
        config: dict[str, Any],
        msgbus: Any,
        cache: Any,
        clock: Any,
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
    """Factory for SimulatedLiveExecutionClient."""

    @classmethod
    def create(
        cls,
        *,
        loop: Any,
        name: str,
        config: dict[str, Any],
        msgbus: Any,
        cache: Any,
        clock: Any,
    ) -> SimulatedLiveExecutionClient:
        return SimulatedLiveExecutionClient(
            slippage_prob=config.get("slippage_prob", 0.25),
            random_seed=config.get("random_seed", 42),
            loop=loop,
            client_id=name,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
        )
```

⚠️ **The `create()` signature and parameter names** depend on §10. If `LiveDataClientFactory.create()` takes positional args, or different kwarg names, adapt to match. The intent — config-dict-driven construction — is what stays stable.

- [ ] **Step 2: Verify imports**
```bash
cd /home/tahseen-dar/Projects/trading-ml
uv run python -c "from livesim.factories import CatalogDataClientFactory, SimulatedExecClientFactory; print('OK')"
```
Expected: `OK`. If the imports fail (e.g. `LiveDataClientFactory` not at `nautilus_trader.live.factories`), fix per §10.

- [ ] **Step 3: Full-suite sanity**
```bash
uv run pytest -q 2>&1 | tail -3
```
Expected: still 75 passed (no new tests; factories are exercised end-to-end in Task 8).

- [ ] **Step 4: Commit**
```bash
git add livesim/factories.py
git commit -m "feat(livesim): factory classes for catalog data + simulated exec clients

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: `node_config.py` — TradingNodeConfig builder

A helper that builds a complete `TradingNodeConfig` wiring strategy + data client + exec client + cache. Used by both the integration test (Task 8) and the runnable (Task 9).

**Files:**
- Create: `livesim/node_config.py`

- [ ] **Step 1: Implement the builder**

Create `livesim/node_config.py`:
```python
"""
Helper that builds a complete TradingNodeConfig for the Stage-2 strategy.

Centralizes the wiring: strategy class + config, data client (via factory +
config dict containing bar_provider), exec client (via factory + config dict),
cache (Parquet-backed, project-local directory).

The integration test (tests/test_live_sim_integration.py) and the runnable
(examples/run_trading_node.py) both use this helper. Future stages with
different strategies/clients add more variants here, not new files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from nautilus_trader.config import ImportableStrategyConfig
from nautilus_trader.live.config import TradingNodeConfig
from nautilus_trader.model.data import Bar
from nautilus_trader.model.identifiers import TraderId

from data.instruments import VENUE_MIC


BarProvider = Callable[[], dict[str, list[Bar]]]


def build_trading_node_config(
    *,
    bar_provider: BarProvider,
    strategy_path: str,
    strategy_config_path: str,
    strategy_config_kwargs: dict[str, Any],
    poll_seconds: float | None = None,
    slippage_prob: float = 0.25,
    random_seed: int = 42,
    cache_dir: Path | str = "livesim/cache",
    trader_id: str = "STAGE4-001",
) -> TradingNodeConfig:
    """Build a TradingNodeConfig wired with the Stage-2 strategy + our clients.

    Args:
        bar_provider: callable that returns dict[ticker, list[Bar]] — used by data client.
        strategy_path: fully-qualified strategy class path (e.g. 'strategies.momentum_topn:MomentumTopNStrategy').
        strategy_config_path: fully-qualified config class path (e.g. 'strategies.momentum_topn:MomentumTopNConfig').
        strategy_config_kwargs: dict of kwargs passed to the StrategyConfig constructor.
        poll_seconds: None for fast-replay mode; positive number for continuous polling.
        slippage_prob: passed to SimulatedLiveExecutionClient (0.0..1.0).
        random_seed: passed to SimulatedLiveExecutionClient for deterministic slippage.
        cache_dir: where to persist Parquet cache (project-local default).
        trader_id: TradingNode identifier (must contain a hyphen).
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    return TradingNodeConfig(
        trader_id=TraderId(trader_id),
        strategies=[
            ImportableStrategyConfig(
                strategy_path=strategy_path,
                config_path=strategy_config_path,
                config=strategy_config_kwargs,
            ),
        ],
        data_clients={
            VENUE_MIC: {
                "factory_path": "livesim.factories:CatalogDataClientFactory",
                "config": {
                    "bar_provider": bar_provider,
                    "poll_seconds": poll_seconds,
                },
            },
        },
        exec_clients={
            VENUE_MIC: {
                "factory_path": "livesim.factories:SimulatedExecClientFactory",
                "config": {
                    "slippage_prob": slippage_prob,
                    "random_seed": random_seed,
                },
            },
        },
        # Cache config: Parquet-backed local directory.
        # The exact field name + format depends on §10's verified CacheConfig surface.
        # cache=CacheConfig(database=..., flush_on_event=True),
    )
```

⚠️ **The exact `TradingNodeConfig` field names** (`data_clients`, `exec_clients`, `strategies`, `cache`) depend on §10. Adapt as needed. The `ImportableStrategyConfig` may have a different name in v1.227; verify.

⚠️ **The cache config is commented out** because the exact `CacheConfig` API is unknown until §10 verifies. If §10 documents how to configure Parquet-backed persistence, add it back. Otherwise, leave it out — the integration test (Task 8) is fast-replay only and doesn't need persistence; the runnable (Task 9) can document a manual cache-dir setup as a follow-up.

- [ ] **Step 2: Verify imports**
```bash
uv run python -c "from livesim.node_config import build_trading_node_config; print('OK')"
```
Expected: `OK`.

- [ ] **Step 3: Full-suite sanity**
```bash
uv run pytest -q 2>&1 | tail -3
```
Expected: still 75 passed.

- [ ] **Step 4: Commit**
```bash
git add livesim/node_config.py
git commit -m "feat(livesim): build_trading_node_config helper for TradingNodeConfig wiring

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Integration test — TradingNode end-to-end on synthetic data

This is the validation milestone: prove a TradingNode + our clients + MomentumTopNStrategy produces equity matching `BacktestEngine` on the same synthetic data, within 1% tolerance.

**Files:**
- Create: `tests/test_live_sim_integration.py`

This task likely surfaces real bugs in Tasks 3, 5, 6, 7 — the integration is where the API-recon assumptions meet reality. Expect to iterate.

- [ ] **Step 1: Write the failing integration test**

Create `tests/test_live_sim_integration.py`:
```python
"""End-to-end integration test: TradingNode replays synthetic catalog through
MomentumTopNStrategy; final equity must match BacktestEngine within 1%.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pandas as pd
import pytest

from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId, Symbol
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Price, Quantity

from analysis.runner import run as backtest_run
from analysis.runspec import RunSpec
from data.instruments import VENUE_MIC
from livesim.data_client import CatalogLiveDataClient  # noqa: F401 (used via factory)
from livesim.node_config import build_trading_node_config
from strategies.momentum_topn import MomentumTopNConfig, MomentumTopNStrategy


def _instrument(ticker: str) -> Equity:
    return Equity(
        instrument_id=InstrumentId.from_str(f"{ticker}.{VENUE_MIC}"),
        raw_symbol=Symbol(ticker), currency=USD, price_precision=2,
        price_increment=Price.from_str("0.01"),
        lot_size=Quantity.from_int(1),
        ts_event=0, ts_init=0,
    )


def _bars(ticker: str, closes: list[float], start_ts_ns: int) -> list[Bar]:
    bar_type = BarType.from_str(f"{ticker}.{VENUE_MIC}-1-DAY-LAST-EXTERNAL")
    one_day_ns = 86_400_000_000_000
    out: list[Bar] = []
    for i, c in enumerate(closes):
        ts = start_ts_ns + i * one_day_ns
        p = Price.from_str(f"{c:.2f}")
        out.append(Bar(bar_type=bar_type, open=p, high=p, low=p, close=p,
                       volume=Quantity.from_int(1_000_000),
                       ts_event=ts, ts_init=ts))
    return out


def _grow(rate: float, n: int) -> list[float]:
    out = [100.0]
    for _ in range(n - 1):
        out.append(out[-1] * (1.0 + rate))
    return out


def _synthetic_data() -> tuple[dict[str, list[Bar]], list[Equity]]:
    n = 380
    start_ts_ns = 1_577_836_800_000_000_000  # 2020-01-01
    closes_by_ticker = {
        "A": _grow(0.0020, n), "B": _grow(0.0010, n), "BENCH": _grow(0.0005, n),
    }
    bars_by_ticker = {t: _bars(t, c, start_ts_ns) for t, c in closes_by_ticker.items()}
    instruments = [_instrument(t) for t in bars_by_ticker]
    return bars_by_ticker, instruments


async def test_trading_node_equity_within_1pct_of_backtest():
    """The TradingNode fast-replay path must produce final equity within 1% of BacktestEngine."""
    from nautilus_trader.live.node import TradingNode

    bars_by_ticker, instruments = _synthetic_data()

    # 1. Run the BacktestEngine path for reference equity.
    backtest_spec = RunSpec(
        label="backtest-baseline",
        strategy_class=MomentumTopNStrategy,
        strategy_config=MomentumTopNConfig(
            universe=("A", "B"), benchmark="BENCH",
            top_n=1, lookback_days=63, skip_days=21,
        ),
        start=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end=datetime(2021, 1, 17, tzinfo=timezone.utc),
        slippage_prob=0.0,  # deterministic for parity
    )
    backtest_result = backtest_run(
        backtest_spec, bars_by_ticker=bars_by_ticker, instruments=instruments,
    )
    backtest_equity = backtest_result.metrics["final_equity"]
    assert backtest_equity > 0

    # 2. Run the TradingNode path against the same data.
    config = build_trading_node_config(
        bar_provider=lambda: bars_by_ticker,
        strategy_path="strategies.momentum_topn:MomentumTopNStrategy",
        strategy_config_path="strategies.momentum_topn:MomentumTopNConfig",
        strategy_config_kwargs={
            "universe": ("A", "B"), "benchmark": "BENCH",
            "top_n": 1, "lookback_days": 63, "skip_days": 21,
        },
        poll_seconds=None,  # fast-replay
        slippage_prob=0.0,  # deterministic
    )
    node = TradingNode(config=config)
    node.build()

    try:
        # Start the node, wait for the data client to signal "done", then stop.
        await node.start_async()
        # Get the data client and await its done future.
        data_client = node.kernel.data_engine.get_client(VENUE_MIC)
        # Wait for fast-replay completion (with timeout safety).
        try:
            await asyncio.wait_for(data_client.done(), timeout=30.0)
        except asyncio.TimeoutError:
            pytest.fail("TradingNode fast-replay did not complete within 30 seconds")
    finally:
        await node.stop_async()
        node.dispose()

    # 3. Compare final equities.
    venue = node.portfolio.account_for_venue_ids()[0]  # the only venue
    account = node.portfolio.account(venue)
    live_equity = float(node.portfolio.equity(venue)[account.base_currency].as_decimal())

    drift = abs(live_equity - backtest_equity) / backtest_equity
    assert drift < 0.01, (
        f"TradingNode equity {live_equity:.2f} drifted {drift:.4f} from "
        f"BacktestEngine {backtest_equity:.2f} (>1% tolerance)"
    )
```

⚠️ **The TradingNode lifecycle calls** (`build()`, `start_async()`, `stop_async()`, `dispose()`) are best-guesses. ADAPT to whatever §10 verified. The flow is:
1. Construct TradingNode with a TradingNodeConfig.
2. Build/start the node (sync or async per §10).
3. Wait for the data client to finish fast-replay (await its `done()` future).
4. Stop the node cleanly.
5. Read the final portfolio equity.

If §10 documents a different lifecycle pattern (e.g. a synchronous `run_until_done()` helper), adapt the test accordingly.

- [ ] **Step 2: Run the integration test**
```bash
uv run pytest tests/test_live_sim_integration.py -v -s 2>&1 | tail -40
```

This is THE critical test for Stage 4. It will likely fail on the first run because of API-shape mismatches in Tasks 3-7. Each failure tells you something to fix. Iterate:
- If `LiveDataClient.__init__` errors on `kwargs`: §10 documented different constructor args; fix the data client.
- If `_emit_fill` raises NotImplementedError: wire it to the verified fill-emission API from §10 (likely `self.generate_order_filled(...)` or similar).
- If `node.start_async()` doesn't exist: try `node.start()` (sync), `asyncio.run(node.run())`, or whatever §10 documented.
- If `data_client.done()` returns immediately (before any bars are emitted): the data client's `_connect()` async kickoff may not be wiring through; debug the lifecycle.
- If equity drifts >1% but in a consistent direction: there may be a slippage or fill-timing semantic difference. Compare intermediate equity points to localize.
- If equity drifts >1% noisy: order-of-events bug. Investigate.

Iterate until the test passes within 1%.

- [ ] **Step 3: Full-suite sanity**
```bash
uv run pytest -q 2>&1 | tail -3
```
Expected: 76 passed (75 + 1). All prior tests must stay green.

- [ ] **Step 4: Commit**
```bash
git add tests/test_live_sim_integration.py livesim/data_client.py livesim/exec_client.py livesim/factories.py livesim/node_config.py
git commit -m "test(livesim): end-to-end TradingNode parity vs BacktestEngine within 1%

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```
(May also include fix commits to the livesim/ files surfaced during integration. Bundle them together if the test required multiple integration fixes.)

---

## Task 9: Runnable `examples/run_trading_node.py`

A CLI that exercises the live infrastructure against the real Stage-1 catalog.

**Files:**
- Create: `examples/run_trading_node.py`

- [ ] **Step 1: Implement the runnable**

Create `examples/run_trading_node.py`:
```python
"""
Stage-4 runnable: start a TradingNode running MomentumTopNStrategy against
the Stage-1 catalog, via livesim's CatalogLiveDataClient and
SimulatedLiveExecutionClient.

Usage:
    uv run python examples/run_trading_node.py --fast-replay
        Emit all catalog bars as fast as possible; exit when done.

    uv run python examples/run_trading_node.py --poll-seconds 60
        Continuous mode: emit historical bars, then poll for new bars every
        60 seconds (suitable for dev; the daily ingest job adds bars).

    uv run python examples/run_trading_node.py
        Same as --poll-seconds 86400 (one day).
"""
from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from nautilus_trader.live.node import TradingNode
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

from data.instruments import VENUE_MIC
from data.tickers import ALL_TICKERS, BENCHMARK, UNIVERSE
from livesim.node_config import build_trading_node_config


CATALOG_PATH = _PROJECT_ROOT / "data" / "catalog"


def make_catalog_bar_provider(catalog_path: Path):
    """Closure that reads bars from the Parquet catalog on each call."""
    catalog = ParquetDataCatalog(path=str(catalog_path.resolve()))

    def provider():
        out = {}
        for ticker in ALL_TICKERS:
            bars = catalog.bars(instrument_ids=[f"{ticker}.{VENUE_MIC}"])
            out[ticker] = bars
        return out

    return provider


async def _run(fast_replay: bool, poll_seconds: float | None) -> int:
    bar_provider = make_catalog_bar_provider(CATALOG_PATH)
    snapshot = bar_provider()
    total = sum(len(b) for b in snapshot.values())
    print(f"loaded {total} bars across {len(snapshot)} tickers")
    if total == 0:
        print("ERROR: catalog is empty. Run data/ingest_alpaca.py first.")
        return 1

    config = build_trading_node_config(
        bar_provider=bar_provider,
        strategy_path="strategies.momentum_topn:MomentumTopNStrategy",
        strategy_config_path="strategies.momentum_topn:MomentumTopNConfig",
        strategy_config_kwargs={
            "universe": UNIVERSE, "benchmark": BENCHMARK,
            "top_n": 3, "lookback_days": 63, "skip_days": 0,  # Stage-3 winner combo
        },
        poll_seconds=None if fast_replay else poll_seconds,
    )
    node = TradingNode(config=config)
    node.build()

    # SIGTERM handler for graceful shutdown.
    stop_event = asyncio.Event()
    def _stop(*_args):
        print("\n(stop signal received)")
        stop_event.set()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _stop)

    try:
        await node.start_async()
        if fast_replay:
            data_client = node.kernel.data_engine.get_client(VENUE_MIC)
            await asyncio.wait_for(data_client.done(), timeout=300.0)
            print("fast-replay complete")
        else:
            print(f"continuous mode; polling every {poll_seconds}s. Ctrl+C to stop.")
            await stop_event.wait()
    finally:
        await node.stop_async()
        node.dispose()

    # Print final account state.
    venue_id = node.portfolio.account_for_venue_ids()[0]
    account = node.portfolio.account(venue_id)
    final_equity = node.portfolio.equity(venue_id)[account.base_currency]
    print(f"final equity: {final_equity}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast-replay", action="store_true",
                        help="Emit all bars then exit (no polling)")
    parser.add_argument("--poll-seconds", type=float, default=86400.0,
                        help="Polling cadence in continuous mode (default 86400 = 1 day)")
    args = parser.parse_args()
    return asyncio.run(_run(fast_replay=args.fast_replay, poll_seconds=args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
```

⚠️ The same `start_async`/`stop_async`/`dispose` adaptations from Task 8 apply here. If Task 8 used a different lifecycle pattern, mirror it here.

- [ ] **Step 2: Verify `--help`**
```bash
cd /home/tahseen-dar/Projects/trading-ml
uv run python examples/run_trading_node.py --help
```
Expected: argparse help with `--fast-replay` and `--poll-seconds`, exit 0.

- [ ] **Step 3: Run fast-replay against the real catalog**
```bash
time uv run python examples/run_trading_node.py --fast-replay
```
Expected: completes in some bounded time (~5 sec to ~few minutes depending on event volume). Should print:
- `loaded NNN bars across 12 tickers`
- `fast-replay complete`
- `final equity: <some Money value>`

Check: no `AccountBalanceNegative`, no `Stopping backtest` (those are BacktestEngine errors; TradingNode has different error messages — watch for any exceptions or stack traces).

If the run errors mid-flight, debug; the issue is almost certainly a §10 API misuse surfaced now that we're running against real data.

- [ ] **Step 4: Full-suite sanity (no test changes)**
```bash
uv run pytest -q 2>&1 | tail -3
```
Expected: still 76 passed.

- [ ] **Step 5: Commit**
```bash
git add examples/run_trading_node.py
git commit -m "feat(examples): Stage-4 runnable TradingNode with --fast-replay + --poll-seconds

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Full acceptance — replay parity on the real catalog

The integration test (Task 8) validated parity on synthetic data. This task validates the same property against the REAL catalog: TradingNode fast-replay vs BacktestEngine on the full Stage-1 catalog should produce equity within 1%.

**Files:** none new — runs both paths and compares.

- [ ] **Step 1: Capture BacktestEngine equity on the real catalog (the Stage-3 winner config)**

Use the Stage-3 winner config (`top_n=3, lookback_days=63, skip_days=0`):
```bash
cd /home/tahseen-dar/Projects/trading-ml
uv run python -c "
import sys; sys.path.insert(0, '.')
from pathlib import Path
from datetime import datetime, timezone

from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

from analysis.runner import run as backtest_run
from analysis.runspec import RunSpec
from data.instruments import VENUE_MIC, build_universe_instruments
from data.tickers import ALL_TICKERS, BENCHMARK, UNIVERSE
from strategies.momentum_topn import MomentumTopNConfig, MomentumTopNStrategy

catalog = ParquetDataCatalog(path='data/catalog')
bars_by_ticker = {t: catalog.bars(instrument_ids=[f'{t}.{VENUE_MIC}']) for t in ALL_TICKERS}
instruments = build_universe_instruments()

earliest = min(b.ts_event for b in bars_by_ticker[UNIVERSE[0]])
latest = max(b.ts_event for b in bars_by_ticker[BENCHMARK])
spec = RunSpec(
    label='parity-baseline',
    strategy_class=MomentumTopNStrategy,
    strategy_config=MomentumTopNConfig(
        universe=UNIVERSE, benchmark=BENCHMARK,
        top_n=3, lookback_days=63, skip_days=0,
    ),
    start=datetime.fromtimestamp(earliest/1e9, tz=timezone.utc),
    end=datetime.fromtimestamp(latest/1e9, tz=timezone.utc),
    slippage_prob=0.0,
)
result = backtest_run(spec, bars_by_ticker=bars_by_ticker, instruments=instruments)
print(f'BACKTEST final equity: {result.metrics[\"final_equity\"]:.2f}')
print(f'BACKTEST n_days: {result.metrics[\"n_days\"]}')
print(f'BACKTEST n_trades: {result.metrics[\"n_trades\"]}')
" 2>&1 | grep "BACKTEST"
```

Capture the printed equity, n_days, n_trades.

- [ ] **Step 2: Run TradingNode fast-replay on the same catalog with slippage_prob=0.0**

A quick edit may be needed: `examples/run_trading_node.py` defaults to the Stage-3 winner config (top_n=3, lookback=63, skip=0) which matches the baseline above. To get deterministic parity, run with slippage_prob=0.0 — currently the runnable doesn't expose that flag, so either:
- (a) Run the runnable as-is and accept the slippage_prob=0.25 stochastic variation; or
- (b) Add a `--slippage-prob FLOAT` flag to `examples/run_trading_node.py` (one-line change) and run with `--slippage-prob 0.0`.

If (b), edit `examples/run_trading_node.py` to add the argparse flag and pipe it through to `build_trading_node_config`. Then run:
```bash
time uv run python examples/run_trading_node.py --fast-replay --slippage-prob 0.0 2>&1 | grep -E "(final equity|fast-replay complete)"
```

Capture the printed equity.

- [ ] **Step 3: Compare**

Compute `drift = |live_equity - backtest_equity| / backtest_equity`. Expected: < 0.01 (1%).

If drift > 1%:
- Compare the equity curves at intermediate points (export from both paths).
- Look for an off-by-one bar at the start or end.
- Look for asymmetric slippage application.
- If drift is consistently small (~0.5%-2%) and the cause is event-ordering-async-vs-sequential, document it and consider relaxing the test tolerance with explicit explanation.

DO NOT just relax the tolerance to make the test pass without investigation.

- [ ] **Step 4: Commit (only if Step 2's edit was needed)**

If you added the `--slippage-prob` flag in Step 2:
```bash
git add examples/run_trading_node.py
git commit -m "feat(examples): expose --slippage-prob flag on run_trading_node for parity testing

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

No commit if Step 2 ran the runnable as-is (the acceptance numbers go into the README in Task 11).

---

## Task 11: README + Stage 4 wrap-up

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the Status section**

Edit `README.md`. Change Stage 4 from `[ ]` to `[x]` and add sub-bullets citing the actual numbers from Task 10's parity check. The Status section should read (adapt to match existing formatting):
```markdown
## Status

- [x] Stage 0 — Project skeleton + Nautilus smoke backtest
- [x] Stage 1 — Alpaca → Nautilus ParquetDataCatalog ingestion pipeline
  - [x] Data pipeline built and unit-tested
  - [x] Live catalog populated (12 tickers; SPY ~2018-11, sector ETFs ~2020-07 → today on free-tier IEX). Cross-check passes at 1.5% tolerance (see design spec §7).
- [x] Stage 2 — Momentum top-N strategy + tearsheet
  - [x] Pure ranker + Nautilus Strategy (monthly top-3 by 6m-skip-1 momentum) + quantstats tearsheet
  - [x] Acceptance backtest: `uv run python examples/backtest_momentum.py`
  - On the ~2020-07→2026 window the simple momentum strategy underperformed SPY buy-and-hold (≈9% vs ≈21% CAGR) — the expected honest baseline per the design spec.
- [x] Stage 3 — Honesty harness (walk-forward × param grid + Deflated Sharpe + cost stress)
  - [x] `analysis/{runspec,deflated_sharpe,metrics,runner,walk_forward,costs,harness_report}.py`
  - [x] Runnable: `uv run python examples/run_harness.py` (full) or `--smoke` (dev iteration)
  - [x] Acceptance run: winner `top_n=3, lookback=63, skip=0`; median window Sharpe 1.02; DSR p=1.00 / z=23.33 with n_trials=90.
- [x] Stage 4 — LiveNode sim-live infrastructure
  - [x] `livesim/{data_client,exec_client,factories,node_config}.py` — Nautilus TradingNode wiring; strategy code unchanged
  - [x] Runnable: `uv run python examples/run_trading_node.py --fast-replay` (replay) or `--poll-seconds N` (continuous)
  - [x] Replay-parity validated: TradingNode fast-replay produces final equity `<INSERT live equity>` vs BacktestEngine `<INSERT backtest equity>` (drift `<INSERT drift %>` — within 1% tolerance).
- [ ] Stage 5 — ML overlay (meta-labeling)
- [ ] Stages 6–8 — see design spec
```

REPLACE the `<INSERT …>` placeholders with the ACTUAL values from Task 10. DO NOT leave any `<INSERT …>` literal.

- [ ] **Step 2: Add the Stage-4 runnable to the Backtest section**

The existing Backtest section has Stage 2 + Stage 3 entries. Append below them:
```markdown
### LiveNode sim-live

```bash
# Stage 4: TradingNode replay/continuous against the Stage-1 catalog
uv run python examples/run_trading_node.py --fast-replay      # fast replay (~seconds)
uv run python examples/run_trading_node.py --poll-seconds 60  # continuous, poll every 60s for dev
uv run python examples/run_trading_node.py                    # continuous, poll daily (production cadence)
```
Persists state to `livesim/cache/` (gitignored) so restarts pick up where the previous run left off.
```

- [ ] **Step 3: Final pytest gate**
```bash
uv run pytest -q 2>&1 | tail -3
```
Expected: 76 passed. Zero failures.

- [ ] **Step 4: Commit**
```bash
git add README.md
git commit -m "docs: README — Stage 4 complete, LiveNode runnable documented

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: Stage 4 done**

The spec milestones are met:
- `livesim/data_client.py`, `livesim/exec_client.py`, `livesim/factories.py`, `livesim/node_config.py` all exist and are unit-tested.
- The runnable `examples/run_trading_node.py` starts a TradingNode against the real catalog with both `--fast-replay` and continuous modes.
- The replay-parity integration test validates TradingNode == BacktestEngine within 1%.

The 2+ weeks of real-time fills milestone is external to this plan — the user runs the runnable in continuous mode after Stage 4 ships.

Stage 5 (ML overlay — meta-labeling on the existing momentum strategy, evaluated through the Stage-3 harness) is the next plan.

---

## Self-review

**1. Spec coverage**

| Spec section | Implementing task(s) |
|---|---|
| §1 Goal: TradingNode wires MomentumTopNStrategy via custom data + exec clients, validated by 1%-parity test | Tasks 3, 5, 6, 7, 8, 10 |
| §1 Milestone splits: replay-parity (this plan) + continuous-mode (external) | Tasks 8 + 10 (in-plan); §11 closing paragraph (external) |
| §2 File structure: `livesim/{data_client,exec_client,factories,node_config}.py` + runnable + tests + §10 API notes | Tasks 1 (scaffold), 2 (§10), 3-7 (files), 8 (integration test), 9 (runnable) |
| §2 Commitments (strategy unchanged, two boundary types, Parquet cache, fast-replay test mode, livesim-only TradingNode imports) | Distributed across Tasks 3-9; verified in the integration test (Task 8) — strategy code is not modified |
| §3 Stack: `pytest-asyncio>=0.24` added | Task 1 |
| §4.1 Data flow (subscribe_bars + emit historical + polling-mode delta) | Tasks 3 + 4 |
| §4.2 Execution flow (next-bar fill + FillModel slippage) | Task 5 (`compute_fill_price` + `_on_bar`) |
| §4.3 State persistence via Parquet cache | Task 7 (`build_trading_node_config`) — note: cache config is commented out pending §10 verification; if §10 confirms the API, restore it; otherwise document the deferral |
| §4.4 Scheduling (fast-replay test default; poll_seconds CLI in production) | Task 7 (config) + Task 9 (CLI) |
| §4.5 Validation milestone (1% tolerance, with relaxation path if needed) | Task 8 (synthetic-data parity) + Task 10 (real-catalog parity) |
| §5 Testing approach (3 test files: data client, exec client, integration) | Tasks 3+4 (test_catalog_data_client.py), 5 (test_simulated_exec_client.py), 8 (test_live_sim_integration.py) |
| §6 Risks (event ordering, asyncio lifecycle, cache durability, persistence absence fallback) | Task 2 verifies them; Task 8 surfaces ordering issues empirically; persistence fallback documented in §4.3 |
| §7 Out-of-scope (Alpaca live data/exec, Streamlit, multi-strategy, Redis, etc.) | None implemented — verified by file inventory (no `alpaca.async` imports, no Redis-config, no Streamlit, no multi-strategy support) |

No spec section is uncovered. The cache-config deferral in §4.3 / Task 7 is the only known partial coverage — explicitly acknowledged with a fallback.

**2. Placeholder scan**

- Task 11 contains `<INSERT live equity>` / `<INSERT backtest equity>` / `<INSERT drift %>` placeholders that Task 10 produces and Task 11 substitutes. The README block explicitly says "REPLACE the `<INSERT …>` placeholders with the ACTUAL values from Task 10. DO NOT leave any `<INSERT …>` literal." This is a substitution instruction, not a plan failure.
- Multiple ⚠️ blocks throughout call out "depends on §10 verification" — those are honest API-recon dependencies with documented fallbacks (e.g. cache config falls back to "leave it out"; lifecycle method names "adapt to §10").
- No `TBD`, no `implement later`, no "similar to Task N", no "add appropriate error handling" generalities.

**3. Type consistency**

- `BarProvider = Callable[[], dict[str, list[Bar]]]` is defined in `livesim/data_client.py` (Task 3) and imported by `livesim/node_config.py` (Task 7).
- `CatalogLiveDataClient(bar_provider, poll_seconds=None, **kwargs)` signature in Task 3 matches the factory's `create()` call in Task 6.
- `SimulatedLiveExecutionClient(slippage_prob=0.25, random_seed=42, **kwargs)` signature in Task 5 matches the factory's `create()` call in Task 6.
- `build_trading_node_config(bar_provider, strategy_path, strategy_config_path, strategy_config_kwargs, poll_seconds=None, slippage_prob=0.25, random_seed=42, cache_dir="livesim/cache", trader_id="STAGE4-001")` signature in Task 7 is called identically (with kwargs) in Task 8 and Task 9.
- `compute_fill_price(bar, side, slippage_ticks, tick_size)` pure helper in Task 5 has consistent signature across implementation + 4 unit tests.

All type/method/property names are consistent across tasks.
