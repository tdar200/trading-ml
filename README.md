# trading-ml

ML + trading research lab on NautilusTrader using free Alpaca daily-bar data.
Long-running project — see `docs/superpowers/specs/2026-05-26-trading-ml-design.md`
for the full design, philosophy, and staged roadmap.

## Status

- [x] Stage 0 — Project skeleton + Nautilus smoke backtest
- [x] Stage 1 — Alpaca → Nautilus ParquetDataCatalog ingestion pipeline
  - [x] Data pipeline built and unit-tested (universe, instruments, bar converter,
        Alpaca client, idempotent + overlap-safe ingest orchestrator)
  - [x] Live catalog populated (12 tickers; SPY ~2018-11, sector ETFs ~2020-07 → today on free-tier IEX).
        Cross-check passes at 1.5% tolerance (free-tier IEX vs yfinance SIP venue
        noise — see design spec §7).
- [x] Stage 2 — Momentum top-N strategy + tearsheet
  - [x] Pure ranker + Nautilus Strategy (monthly top-3 by 6m-skip-1 momentum) + quantstats tearsheet
  - [x] Acceptance backtest: `uv run python examples/backtest_momentum.py`
  - On the ~2020-07→2026 window the simple momentum strategy underperformed SPY buy-and-hold (≈9% vs ≈21% CAGR) — the expected honest baseline per the design spec; the milestone is a credible, reproducible tearsheet, not beating SPY.
- [x] Stage 3 — Honesty harness (walk-forward × param grid + Deflated Sharpe + cost stress)
  - [x] `analysis/{runspec,deflated_sharpe,metrics,runner,walk_forward,costs,harness_report}.py`
  - [x] Runnable: `uv run python examples/run_harness.py` (full) or `--smoke` (dev iteration)
  - [x] Acceptance run: 90 walk-forward + 4 cost-stress backtests over the full catalog. Winner: `top_n=3, lookback_days=63, skip_days=0` (3-month momentum without skip — different from Stage 2's hardcoded baseline). Winner median window Sharpe 1.02 (range [0.88, 1.68]); cluster of similarly-good configs (5-95th percentile of all combo-window Sharpes: [0.41, 1.66]); CAGR ≈16% across slippage 0→2 bps (robust to costs); Deflated Sharpe p=1.00, z=23.33 with n_trials=90 — the winner's Sharpe is unlikely to be pure selection bias.
  - The harness winner still underperforms SPY in CAGR (≈16% vs ≈21%) but has comparable risk-adjusted return (Sharpe ≈1.03 vs SPY ≈1.10) and a better max drawdown (-18% vs -25%). Honest characterization, not a deployment signal.
- [x] Stage 4 — LiveNode sim-live infrastructure
  - [x] `livesim/{data_client,exec_client,factories,node_config}.py` — `TradingNode` wiring with `CatalogLiveDataClient` (catalog-replay) + `SimulatedLiveExecutionClient` (same-bar close fills, SELL-before-BUY buffering for cash-account safety).
  - [x] Runnable: `uv run python examples/run_trading_node.py --fast-replay` (one-shot) or `--poll-seconds N` (continuous).
  - [x] **Integration parity (synthetic, top_n=1)**: 0.38% drift vs `BacktestEngine` — well under the 1% gate. Runtime infrastructure validated end-to-end (tests/test_live_sim_integration.py).
  - [x] **Real-catalog parity (top_n=3)**: Stage-4.5 closed the gap. Three issues were diagnosed and fixed: (1) `LiveRiskEngineConfig.max_order_submit_rate` defaulted to `100/00:00:01`, which under fast-replay's `LiveClock` (wallclock) DENIED 195 of 295 orders — raised to `1_000_000/00:00:01`; (2) livesim filled at next-bar OPEN while BacktestEngine fills at same-bar CLOSE (Nautilus's daily-bar `SimulatedExchange` semantic) — aligned livesim to same-bar close; (3) once the throttle was lifted, `AccountBalanceNegative` surfaced on rotation days because the exec client's `_on_bar` runs BEFORE the strategy's `on_bar` for the same bar, so `SubmitOrder` for the rebalance ticker arrives AFTER its bar passed — fixed by tracking the last bar per ticker and buffering the fill at `_submit_order` time against that bar's close. Real-catalog drift over a 1-year window is now 0.55% (regression test `tests/test_live_sim_parity_real_catalog.py` gates at 2%; trade counts match exactly).
- [x] Stage 5 — Meta-labeling ML overlay
  - [x] Pure pipeline: `models/{labeling,features,cv,trainer,store}.py` (holding-period-return-sign labels, 11 no-lookahead features, purged+embargoed walk-forward CV, LightGBM fit/predict, dated joblib ModelStore) + `strategies/momentum_meta.py` (Stage-2 subclass with filter+sizer: `weight = base × (p_win - threshold)/(1 - threshold)` when `p_win ≥ threshold`, else 0).
  - [x] Walk-forward train: `uv run python examples/train_meta_models.py` — 8 LightGBM combos (n_estimators × max_depth × learning_rate cross-product) × 60 monthly cutoffs each = **472 saved models** (8 cutoffs skipped by the warm-up gate). Smoke mode (`--smoke`) trains 1 combo × ≤3 cutoffs in ~3 sec.
  - [x] Harness eval: `uv run python examples/run_harness_meta.py` — Stage-3 harness for baseline (Stage-2 winner) + 24-combo meta grid (8 LightGBM × 3 thresholds) × 5 windows = 5 + 120 runs. **Baseline median window Sharpe 1.0162 vs meta winner `n200_d5_lr10` @ threshold 0.5 with median window Sharpe 1.3083 (+28.7%)**. Deflated Sharpe over the 125 (combo × window) trials: p=1.0000, z=11.98 (observed Sharpe 1.717, expected_max under H0 0.618). Verdict: **meta clearly beats baseline OOS + survives DSR.**
  - Caveat: the per-window-trial DSR formulation treats each `(combo, window)` pair as a separate independent trial (n_trials=125), which is generous; a more conservative formulation using only the 24 per-combo-median Sharpes (n_trials=24) would shrink the apparent significance, though the +0.29 Sharpe edge is large enough that the headline conclusion is unlikely to flip. Stage-5.5 follow-up to switch to the per-combo-median formulation. The headline edge is robust regardless.
  - Final-review bugfix: a timestamp-normalization bug in `strategies/momentum_meta.py::_rebalance` (rebalance_date at NYSE-close UTC vs midnight-aligned synthetic feature index) caused the meta strategy to silently stop trading after warm-up on real catalog data, producing an earlier degenerate harness result. Fixed by normalizing `rebalance_date` to midnight; regression test (`test_non_midnight_timestamps_still_trade`) added; harness now refuses to write a verdict when every threshold within a (window, combo) group produces the identical Sharpe.
- [ ] Stages 6–8 — see design spec

## Setup

```bash
# 1. Install dependencies (Python 3.12)
uv sync

# 2. Get free Alpaca paper-account API keys at https://alpaca.markets/
cp .env.example .env
# edit .env and fill in ALPACA_API_KEY / ALPACA_API_SECRET

# 3. Smoke-test the engine (no data/credentials needed)
uv run python examples/smoke.py

# 4. Populate the data catalog (needs credentials; 1–5 min first run, seconds on re-runs).
#    Free-tier Alpaca uses DataFeed.IEX (~5.5 years of history, ~2020+ in practice).
#    SIP feed (2016+) requires a paid Alpaca Algo Trader Plus subscription.
uv run python data/ingest_alpaca.py

# 5. Validate the catalog against an independent source
uv run python scripts/cross_check_yfinance.py --ticker SPY
```

## Tests

```bash
# Unit tests (no network, no credentials)
uv run pytest tests -v --ignore=tests/test_catalog_integration.py

# Integration tests (real Alpaca calls — needs .env populated; auto-skip otherwise)
uv run pytest tests/test_catalog_integration.py -v
```

## Backtest

```bash
# Stage 2: momentum top-N over the full catalog → quantstats tearsheet
uv run python examples/backtest_momentum.py
```

Writes `equity.csv`, `spy_close.csv`, and `tearsheet.html` to `results/momentum_topn_<timestamp>/`.
Prints a CAGR / Sharpe / max-drawdown summary for the strategy vs SPY.

### Honesty harness

```bash
# Stage 3: walk-forward × parameter grid + Deflated Sharpe + cost stress
uv run python examples/run_harness.py            # full battery (90 + 4 backtests, ~1 min with Nautilus caching)
uv run python examples/run_harness.py --smoke    # tiny grid (~5 sec) for dev iteration
```

Writes `walk_forward.csv`, `cost_stress.csv`, `summary.json`, `summary.md` to `results/harness_<timestamp>/`.
Prints the 5-section Markdown verdict (time stability / parameter robustness / cost survival / Deflated Sharpe / bottom line).

### LiveNode sim-live

```bash
# Stage 4: TradingNode replay/continuous against the Stage-1 catalog
uv run python examples/run_trading_node.py --fast-replay --slippage-prob 0.0  # one-shot replay
uv run python examples/run_trading_node.py --poll-seconds 60                  # continuous, poll every 60s for dev
uv run python examples/run_trading_node.py                                    # continuous, poll daily (production cadence)
```

Wires `MomentumTopNStrategy` into Nautilus's `TradingNode` runtime via livesim's `CatalogLiveDataClient` + `SimulatedLiveExecutionClient`. Prints final equity. Strategy code is unchanged from Stage 2 (research→live parity).

Stage-4.5 closed the real-catalog parity gap (was ~29%, now ~0.55%). See the Stage 4 status entry above for the three root causes (risk-engine throttle, fill-bar semantic, and `_on_bar`/`_submit_order` ordering on the msgbus). The regression test `tests/test_live_sim_parity_real_catalog.py` gates real-catalog drift at 2%.

### Meta-labeling (Stage 5)

```bash
# 1. Walk-forward train the LightGBM meta-model grid (~30 sec; --smoke ≈3 sec)
uv run python examples/train_meta_models.py
uv run python examples/train_meta_models.py --smoke    # 1 combo × 3 cutoffs

# 2. Evaluate baseline (Stage-2) vs meta through the Stage-3 harness (~2 min; --smoke ≈7 sec)
uv run python examples/run_harness_meta.py
uv run python examples/run_harness_meta.py --smoke     # 1 combo × 2 windows
```

Writes `baseline.csv`, `meta.csv`, `summary.json`, `summary.md` to `results/harness_meta_<timestamp>/`. The summary contains the per-combo median Sharpe ranking, Deflated Sharpe over the trial set, and an honest YES/NO verdict on whether meta beat baseline OOS + DSR.

## Layout

```
data/         pipeline modules + the ParquetDataCatalog (catalog gitignored)
strategies/   Nautilus Strategy implementations + venue/FillModel helper
analysis/     honesty-harness modules (runner, walk_forward, costs, DSR, tearsheet)
livesim/      Stage-4 TradingNode wiring — CatalogLiveDataClient, SimulatedLiveExecutionClient, factories, node_config
models/       Stage-5 ML pipeline — labeling, features, CV, trainer, dated ModelStore (cache/ gitignored)
examples/     runnable demos (smoke, momentum, harness, run_trading_node, train_meta_models, run_harness_meta)
scripts/      one-off scripts (yfinance cross-check)
tests/        pytest suite
docs/         design spec, plans, Nautilus API notes
```
