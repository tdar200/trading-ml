# Stage 8 — Alpaca Paper Trading Execution Adapter (code-only)

## 1. Context & honest scope

The parent design spec §5 Stage 8 deliverable is *"Custom Nautilus execution adapter for Alpaca paper trading, then real $"* with the milestone *"Live paper Sharpe ≥ 50% of backtest Sharpe over ≥3 months before any real capital"*.

The **3-month paper-trading observation** is the gate to real money; that part cannot be compressed in this session. Stage 8 *in this session* means: build the Alpaca paper execution adapter, wire it into the TradingNode, smoke against the paper account, ship a deploy-readiness checklist that enumerates the observation that must precede real-money rollout. Real-money rollout is an explicit human decision after the checklist is satisfied.

## 2. Architecture

### New code

| Path | Responsibility |
|---|---|
| `livesim/alpaca_exec_client.py` | `AlpacaLiveExecutionClient(LiveExecutionClient)` — `_submit_order` translates Nautilus market orders → `alpaca.trading.requests.MarketOrderRequest` and POSTs via `TradingClient.submit_order`. A polling task calls `get_orders(status="all", after=last_poll_ts)` each tick and emits Nautilus `OrderFilled` / `OrderRejected` events for state transitions. Order-id mapping: `client_order_id` (Nautilus) ↔ Alpaca's `id` (UUID). |
| `livesim/factories.py` (extension) | `AlpacaExecClientFactory` — sibling of the existing `SimulatedExecClientFactory`; instantiates the new exec client with paper-account credentials from env. |
| `examples/run_trading_node.py` (extension) | `--exec-mode {simulated, alpaca-paper}` flag — picks the factory. Simulated remains default. |
| `examples/preflight_alpaca.py` | Stand-alone smoke runnable. Connects to paper account, prints balance + open positions, exits 0 on success. Verifies credentials + connectivity without committing the whole node. |
| `docs/superpowers/specs/2026-05-27-stage-8-deploy-checklist.md` | Human-action checklist: what to monitor, what halts the experiment, what gates real-money rollout. |
| `tests/test_alpaca_exec_client.py` | Mocked `TradingClient`; verifies request shape, id round-trip, fill emission, and SELL-before-BUY pacing. |

### Touched (additive)

- `livesim/node_config.py` — accept an optional `exec_client_factory_name` so `run_trading_node.py` can switch between simulated and alpaca-paper without breaking the v1 path.
- `README.md` — Stage 8 entry, run command, checklist pointer, explicit observation-gate disclaimer.

### Untouched

- All strategy code (`strategies/`) — the whole point of Nautilus's research→live parity.
- `livesim/exec_client.py` (simulated client is unchanged; still the default).
- `livesim/data_client.py` (`CatalogLiveDataClient` continues feeding bars from the catalog — the daily ingest cadence pairs with daily rebalance).

### Boundaries

- `livesim/alpaca_exec_client.py` is the **only** module that imports from `alpaca.trading`. The rest of the project remains alpaca-data-API only (used since Stage 1 for bars + since Stage 6 for news).
- `_submit_order` and the fill-poll loop are async; a small helper (`_translate_alpaca_status`) is pure-functional and trivially testable.

## 3. Risks

1. **Paper-account quirks**: fractional fills, partial fills, weekend submits, halts. The exec client logs unknown statuses but doesn't crash. The dashboard's Models/Equity tab + the human's daily check is the safety net during observation.
2. **Credentials**: `.env` already has paper-account keys from Stage 1; whether they remain valid is unknown. The preflight runnable surfaces auth errors clearly.
3. **No real-money path**: adapter only points at `https://paper-api.alpaca.markets`. A Stage 8.1 follow-up swaps the base URL after the deploy checklist is satisfied.
4. **Fast-replay against a live broker**: Alpaca will REJECT orders for symbols when the market is closed. The smoke run's expected outcome is `REJECTED` events, not fills — this is itself a valid adapter test.
5. **Polling cadence**: 1 Hz works for daily-bar strategies but uses ~3600 API calls/hour. Alpaca paper free tier allows 200 req/min — well within. If we ever switch to intraday strategies, move to the trade-updates WebSocket.

## 4. Acceptance — Stage 8 (code-only portion) is done when

- `uv run python examples/preflight_alpaca.py` connects to paper, prints balance + open positions, exits 0 (or prints a clear auth error message and exits 1).
- `uv run python examples/run_trading_node.py --exec-mode alpaca-paper --fast-replay --slippage-prob 0.0` runs without uncaught exceptions. Fills may be REJECTED if market is closed — the adapter logs this cleanly.
- `tests/test_alpaca_exec_client.py` covers request-shape, id round-trip, fill emission, SELL/BUY pacing. ≥5 tests.
- README Stage 8 entry exists, explains the run command, links to the deploy checklist, and includes the **3-month observation gate** disclaimer in the status line.

## 5. Out of scope

- Real-money execution (gated by 3-month paper observation).
- WebSocket trade-updates stream (polling is sufficient for daily-bar; cite as Stage 8.1 lever).
- Multi-account routing.
- Order types beyond MARKET (the entire strategy uses MARKET only).
- Slippage / commission modeling — Alpaca paper handles its own.
