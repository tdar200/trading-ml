# Stage 4 — LiveNode Sim-Live: Design Spec

**Date:** 2026-05-26
**Status:** Approved through brainstorming; pending user spec-review before implementation planning.
**Parent spec:** `2026-05-26-trading-ml-design.md` (architecture, philosophy, full roadmap)
**Stage:** 4 of 8 — wire the Stage-2 strategy into Nautilus's `TradingNode` runtime against a simulated execution venue with a catalog-replay data source.

## 1. Context & goal

Stages 0-3 produced: a Stage-1 catalog (~17.5k daily bars across 12 ETFs), a Stage-2 momentum strategy with a credible backtest, and a Stage-3 harness that lets us evaluate it honestly. Everything so far has been **batch** — `BacktestEngine.run()` replays a fixed window and exits.

Stage 4 wires the *same* `MomentumTopNStrategy` into Nautilus's `TradingNode` — the runtime designed for **continuous, event-driven live trading** — against:
- A custom `CatalogLiveDataClient` that streams bars from the Stage-1 `ParquetDataCatalog`.
- A custom `SimulatedLiveExecutionClient` that accepts orders and produces deterministic fills at the next bar's open.

**Milestone (per parent spec §5):** "2+ weeks of sim-live fills logged, matching backtest within tolerance." We split this into:
1. **Replay parity (this plan)** — an integration test replays the full catalog through TradingNode in fast-replay mode; final equity matches `BacktestEngine` within **1%** tolerance.
2. **Continuous-mode acceptance (external)** — the user runs `examples/run_trading_node.py` for 2+ weeks of real-time use after Stage 4 ships. Outside this plan's scope.

**This stage is INFRASTRUCTURE, not research.** The strategy doesn't change; only the runtime around it. The honest value: Stage 8 (real Alpaca paper trading) will plug a different DataClient and ExecClient into the same TradingNode and reuse everything else unchanged. This stage builds the half of that future infrastructure that doesn't depend on having an Alpaca-Nautilus adapter.

### Honest scope flag

This is the most Nautilus-API-heavy stage by a wide margin. Task 2 alone (the §10 API recon) will likely take longer than any single task in Stages 0-3. `LiveDataClient`, `LiveExecutionClient`, factory protocols, `TradingNodeConfig`, async lifecycle, cache persistence — all need empirical verification. Plan accordingly.

## 2. Architecture

```
livesim/
  __init__.py
  data_client.py            # CatalogLiveDataClient(LiveDataClient)
  exec_client.py            # SimulatedLiveExecutionClient(LiveExecutionClient)
  factories.py              # LiveDataClientFactory + LiveExecClientFactory subclasses
  node_config.py            # Helper: build a TradingNodeConfig wiring all of above

examples/
  run_trading_node.py       # Starts TradingNode; runs strategy in continuous loop

tests/
  test_catalog_data_client.py     # Unit: subscribe → bars emitted → unsubscribe
  test_simulated_exec_client.py   # Unit: submit_order → next-bar fill → events emitted
  test_live_sim_integration.py    # Full-catalog replay through TradingNode == BacktestEngine

docs/nautilus_api_notes.md       # Appended §10 — TradingNode + LiveDataClient + LiveExecClient
```

### Architectural commitments

1. **Strategy unchanged.** `strategies/momentum_topn.py` is not modified by Stage 4. The whole point of Nautilus's research→live parity is that the same `Strategy` subclass works in both `BacktestEngine` and `TradingNode`. If we have to modify the strategy to make Stage 4 work, that's a smell to investigate.

2. **Two new boundary types only.** Stage 4 implements exactly two new Nautilus surface classes: a `LiveDataClient` subclass and a `LiveExecutionClient` subclass. Plus the factory glue Nautilus's config system requires. No other Nautilus API gets re-implemented; everything else is configured-not-coded.

3. **State persistence via Nautilus's built-in Cache.** Parquet-backed (no Redis dependency). The TradingNode writes positions, orders, account events, and strategy state on a configurable interval; on restart the cache reloads and the strategy resumes.

4. **No real wall-clock pacing in tests.** Integration tests run a fast-replay mode (emit all bars as fast as possible, exit cleanly) for offline validation. Real-time pacing — sleeping until 16:30 ET, polling the catalog for new bars — is a runnable-only behavior, not a test concern.

5. **`livesim/` is the only place that touches `TradingNode`.** `analysis/` and `strategies/` stay engine-agnostic. The architecture's purity from Stages 1-3 is preserved.

## 3. Stack additions

No new dependencies. Stage 4 uses:
- `nautilus_trader==1.227.0` (already pinned) — `TradingNode`, `LiveDataClient`, `LiveExecutionClient` from existing install.
- `pandas`, `polars` (already present) — for equity-curve comparison in tests.
- Standard library `asyncio` — TradingNode's runtime.
- `pytest`, `pytest-asyncio` (NEW DEV DEPENDENCY) — needed because TradingNode integration tests run async code.

`pyproject.toml`'s `[dependency-groups].dev` gets `pytest-asyncio>=0.24` added in Task 1.

## 4. Key behaviors

### 4.1 Data flow

- Strategy subscribes to bars in `on_start` via `self.subscribe_bars(bar_type)` (unchanged from Stage 2).
- `CatalogLiveDataClient` overrides `_subscribe_bars(bar_type)` (Nautilus's hook). On the first subscription, it kicks off an async task that reads all matching bars from the `ParquetDataCatalog` and emits them through `self._handle_data(...)` (the standard LiveDataClient delivery path).
- In **fast-replay mode** (test default): emits all bars immediately, then declares "no more data" so TradingNode can stop cleanly.
- In **continuous mode** (runnable default): emits all historical bars on startup, then polls the catalog every `poll_interval_seconds` (CLI flag, default `86400` = once per day) for any newly-appeared bars and emits the delta. Sleeps with `asyncio.sleep` between polls.

### 4.2 Execution flow

- Strategy calls `self.submit_order(order)` (unchanged from Stage 2). Order routes to the venue assigned to the order's instrument.
- `SimulatedLiveExecutionClient` overrides `_submit_order(order)`. Records the order as pending. When the NEXT bar for that instrument arrives at the engine (one of the data client's emissions), the exec client generates an `OrderFilled` event at that bar's open price.
- Slippage: applied via the same `FillModel` mechanism as `BacktestEngine` (`prob_slippage=0.25` by default; same one-tick semantics documented in `strategies/backtest_setup.py` and §9.8 of the API notes). Configurable per-client.
- No partial fills, no rejection paths, no margin/short-selling — same simplifications as Stage 2.

### 4.3 State persistence

- `TradingNodeConfig.cache.database` configured to point at a local Parquet directory (default `~/.nautilus/cache/trading-ml/` or a project-local `livesim/cache/`).
- Configure the cache to flush on every fill event (`flush_on_event=True` if that's the actual Nautilus knob — verified during Task 2 recon).
- On TradingNode startup, the cache reloads: positions, orders, account events, and (critically) the strategy's persisted state.
- The strategy's `_closes` ring buffer and `_last_rebalance_month` are **instance state**, not cache state. To survive a restart, the strategy needs to either (a) reconstruct from re-replayed bars, or (b) implement Nautilus's `Strategy.on_save()`/`on_load()` hooks if v1.227 exposes them. **Task 2 must verify which approach is supported**; the spec doesn't commit to either yet. Fallback (always available): re-replay all historical bars on startup, accepting the rebuild cost.

### 4.4 Scheduling

- The TradingNode's asyncio loop runs continuously after `await node.start()`.
- The data client's polling cadence determines when work happens. For a *catalog* data source, "new data" only appears when someone separately runs `data/ingest_alpaca.py` (a Stage-1 runnable). Suggested deployment: a cron job that runs `data/ingest_alpaca.py` after market close, then the running TradingNode picks up the new bars on its next poll.
- Runnable supports `--fast-replay` (emit everything, exit) and `--poll-seconds N` (continuous mode polling cadence) flags.

### 4.5 Validation milestone

The replay-parity integration test (`tests/test_live_sim_integration.py`):
1. Loads the same catalog data used by `examples/run_harness.py`.
2. Runs the **Stage-2 backtest path** (`BacktestEngine` + `MomentumTopNStrategy`) and captures final equity.
3. Runs the **TradingNode path** (fast-replay mode, same strategy, `prob_slippage=0.0` for determinism) and captures final equity.
4. Asserts `abs(live_equity - backtest_equity) / backtest_equity < 0.01`.

If the tolerance fails, the most likely cause is async event ordering — the TradingNode's data → strategy → exec event chain may interleave differently than `BacktestEngine`'s sequential loop. Diagnose by comparing intermediate equity-curve points, not just final equity. If a real ordering difference exists, document it and either (a) tighten the exec client to match BacktestEngine's "fill at next bar's open" semantics exactly, or (b) loosen the tolerance to 2-3% with an explicit comment.

The "2+ weeks of real-time fills" milestone is outside this plan — the user runs the runnable after Stage 4 ships.

## 5. Testing approach

| Test file | Coverage | Approach |
|---|---|---|
| `tests/test_catalog_data_client.py` | LiveDataClient methods: subscribe/unsubscribe, fast-replay emission, polling-mode delta detection | Unit tests on a tmp-path catalog written with synthetic Bar objects |
| `tests/test_simulated_exec_client.py` | LiveExecutionClient: order intake, next-bar fill, fill-event emission, slippage application | Unit tests with mocked engine bus |
| `tests/test_live_sim_integration.py` | End-to-end: TradingNode + both clients + MomentumTopNStrategy on synthetic 3-ticker / 380-day data; equity matches BacktestEngine within 1% | Async integration test using `pytest-asyncio`; uses the same synthetic-data helpers from `tests/test_momentum_strategy.py` (extracted into a shared fixture in `tests/conftest.py` if duplication is bad) |

**No tests for `examples/run_trading_node.py`.** Same convention as `examples/{smoke,backtest_momentum,run_harness}.py` — runnables are exercised manually for the acceptance run.

**Total new tests:** ~10-15. Suite goal post-Stage-4: ~76-80 passing.

## 6. Risks & open items

1. **`Strategy.on_save()` / `on_load()` may not exist in v1.227** as cleanly as the design assumes. Task 2 must verify; if absent, the strategy persists by re-replaying historical bars on each TradingNode startup. Cost: ~5 seconds of warm-up per restart (acceptable).

2. **Event-ordering parity with BacktestEngine** is non-trivial. TradingNode is async; BacktestEngine is sequential. The replay-parity tolerance of 1% acknowledges this; we may need to relax it to 2-3% after empirical observation.

3. **The "live" abstraction is leaky.** Without a real exchange we can't fully validate the "would this work in production" question. Stage 4 validates the *plumbing*, not the live-trading reality. Stage 8 closes that loop.

4. **`asyncio` lifecycle bugs** are easy to introduce. The runnable must handle SIGTERM cleanly so cache state isn't corrupted on shutdown. Tests should explicitly cover graceful-shutdown behavior.

5. **TradingNode + Parquet cache may have performance characteristics** we haven't measured — disk I/O on every fill, etc. For daily-bar trading (1-2 fills per month) this is fine; for higher-frequency strategies it'd matter. Not a Stage 4 concern.

6. **`pytest-asyncio` adds a dev dependency.** Small footprint. Documented in Task 1.

7. **State persistence durability after crash.** If the process is killed mid-write to Parquet cache, recovery semantics depend on Nautilus's internals. Task 2 should verify what "graceful shutdown" requires.

8. **A `livesim/cache/` directory will grow** as the cache accumulates events. Gitignore it. Documented in Task 1.

## 7. Out of scope (deliberately, do not add)

- **Real Alpaca live data ingest** (LiveDataClient that calls `alpaca-py`). Stage 8.
- **Real Alpaca paper trading execution** (LiveExecutionClient that submits to Alpaca's REST API). Stage 8.
- **Streamlit / web monitoring dashboard** of the running TradingNode. Stage 7.
- **Multi-strategy concurrent running.** Out of project scope.
- **Multi-venue order routing.** Out of project scope.
- **Redis-backed cache backend** (Parquet is sufficient for daily-bar volumes).
- **Order modification or cancellation paths** in the simulated exec client. Strategy is long-only equal-weight; submitted-and-immediately-filled is enough. Modify/cancel handlers can no-op or raise.
- **Margin, shorting, fractional shares** in the simulated exec client. Same simplifications as Stage 2.
- **Multi-asset / non-equity venues** in the data/exec clients.
- **Real-time WebSocket reconnection logic, heartbeat handling.** Catalog-only design has none of these concerns.

## 8. Bottom-line expected outcome

Stage 4 ships a working TradingNode-based simulator that:
- Replays the Stage-1 catalog through the Stage-2 strategy with **equity matching the backtest within 1%**.
- Persists state to local Parquet between runs.
- Can be run as a long-lived process that polls for new catalog bars daily.
- Has 10-15 new tests covering the data client, exec client, and end-to-end integration.

The strategy still underperforms SPY in CAGR (~16% vs ~21% per Stage 3). The Stage-4 milestone is not about strategy performance — it's about proving the production runtime works. **Stage 5 (the ML overlay) is where research resumes.**

---

*This spec is the Stage-4 contract. Implementation plan to be written next via `superpowers:writing-plans`.*
