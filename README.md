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
  - [x] Acceptance run (post-Stage-4.5-followup, with corrected strategy): 90 walk-forward + 4 cost-stress backtests over the full catalog. Winner: `top_n=2, lookback_days=63, skip_days=0`. Winner median window Sharpe **0.9541** (range [0.875, 1.847]); winner rank 1/18; 5–95th percentile of all (combo, window) Sharpes [0.362, 1.582]; CAGR **17.5%** across slippage 0→2 bps (robust to costs); Deflated Sharpe p=1.00, z=**23.96** with n_trials=90.
  - **Strategy-correction note:** the pre-Stage-4.5-followup README claimed winner `top_n=3` at Sharpe 1.02. The strategy fix (anchor rebalance to benchmark bar so all universe closes are today's at decision time) shifted the optimum to top_n=2 and dropped the absolute Sharpe slightly. Stage 4/5 entries below still use top_n=3 — that is the *previously-selected* winner used as a consistent baseline across Stages 4/5, not the new harness optimum. Re-baselining Stages 4/5 onto top_n=2 is a deferred follow-up.
- [x] Stage 4 — LiveNode sim-live infrastructure
  - [x] `livesim/{data_client,exec_client,factories,node_config}.py` — `TradingNode` wiring with `CatalogLiveDataClient` (catalog-replay) + `SimulatedLiveExecutionClient` (same-bar close fills, SELL-before-BUY buffering for cash-account safety).
  - [x] Runnable: `uv run python examples/run_trading_node.py --fast-replay` (one-shot) or `--poll-seconds N` (continuous).
  - [x] **Integration parity (synthetic, top_n=1)**: 0.38% drift vs `BacktestEngine` — well under the 1% gate. Runtime infrastructure validated end-to-end (tests/test_live_sim_integration.py).
  - [x] **Real-catalog parity (top_n=3)**: Stage-4.5 (+ follow-up) closed the gap. Four issues were diagnosed and fixed: (1) `LiveRiskEngineConfig.max_order_submit_rate` defaulted to `100/00:00:01`, which under fast-replay's `LiveClock` (wallclock) DENIED 195 of 295 orders — raised to `1_000_000/00:00:01`; (2) livesim filled at next-bar OPEN while BacktestEngine fills at same-bar CLOSE (Nautilus's daily-bar `SimulatedExchange` semantic) — aligned livesim to same-bar close; (3) once the throttle was lifted, `AccountBalanceNegative` surfaced on rotation days because the exec client's `_on_bar` runs BEFORE the strategy's `on_bar` for the same bar, so `SubmitOrder` for the rebalance ticker arrives AFTER its bar passed — fixed by tracking the last bar per ticker and buffering the fill at `_submit_order` time against that bar's close; (4) follow-up (2026-05-27): the full-catalog replay still crashed on 2022-10-03 with `AccountBalanceNegative(-812.19)`. Root cause: the strategy fired the monthly rebalance on the FIRST tradeable universe bar of the new month (XLK, index 0 in `ALL_TICKERS`); bars within a day's group arrive in `ALL_TICKERS` order, so `self._closes[t][-1]` for later tickers (XLF, XLE, ...) still held YESTERDAY's close at decision time. The exec client (whose `_last_bar_by_ticker` updates as bars stream in during cmd-queue drain) buffered BUY fills at TODAY's close. On an overnight gap-up day the strategy under-budgeted by ~$3k and overdrew cash. Fixed by anchoring the rebalance trigger to the BENCHMARK bar (SPY, always the LAST bar of each day's group), so every universe ticker's `_closes` is TODAY's at decision time. Real-catalog parity over the widened 2-year window (2021→2023) is **drift=0.000000** (perfect equality, 88 trades match exactly); full-catalog drift over the entire ~5.8-year window is **0.78%** (BT $206,263.81 vs ND $207,875.87, 289 BT trades; ND completes without overdraft). Regression test `tests/test_live_sim_parity_real_catalog.py` gates at 2%. **Note:** the cause-4 strategy fix (benchmark-bar rebalance trigger) is a correctness improvement that retroactively shifts Stage 2/3/5 baseline numbers vs the pre-fix values — re-running the Stage-3 harness and Stage-5 meta-eval post-fix is required to update those entries (see updated numbers below).
- [x] Stage 5 — Meta-labeling ML overlay
  - [x] Pure pipeline: `models/{labeling,features,cv,trainer,store}.py` (holding-period-return-sign labels, 11 no-lookahead features, purged+embargoed walk-forward CV, LightGBM fit/predict, dated joblib ModelStore) + `strategies/momentum_meta.py` (Stage-2 subclass with filter+sizer: `weight = base × (p_win - threshold)/(1 - threshold)` when `p_win ≥ threshold`, else 0).
  - [x] Walk-forward train: `uv run python examples/train_meta_models.py` — 8 LightGBM combos (n_estimators × max_depth × learning_rate cross-product) × 60 monthly cutoffs each = **472 saved models** (8 cutoffs skipped by the warm-up gate). Smoke mode (`--smoke`) trains 1 combo × ≤3 cutoffs in ~3 sec.
  - [x] Harness eval (post-Stage-4.5-followup, re-baselined onto top_n=2): `uv run python examples/run_harness_meta.py` — Stage-3 harness for baseline (top_n=2, the new Stage-3 winner) + 24-combo meta grid (8 LightGBM × 3 thresholds) × 5 windows = 5 + 120 runs. **Baseline median window Sharpe 0.9541 vs meta winner `n100_d3_lr05` @ threshold 0.5 with median window Sharpe 1.2319 (+29.1%)**. Deflated Sharpe over 25 per-combo-median trials (Stage-5.5 conservative formulation): p=1.0000, z=**25.14** (observed Sharpe 1.232, expected_max under H0 0.106). Verdict: **meta clearly beats baseline OOS + survives DSR.**
  - **History:** pre-Stage-4.5-followup numbers were baseline 1.0162 / meta winner 1.3083 (+28.7%) on top_n=3 with the buggy strategy. Post-Stage-4.5-followup on the same top_n=3 baseline (with strategy fix): baseline 0.7393 / meta winner 1.1848 (+60.3%). Re-baselining to top_n=2 (the new Stage-3 winner) raises baseline to 0.9541 and meta winner to 1.2319 (+29.1%). The verdict (meta beats baseline) holds across all three formulations; the magnitude of the edge depends on baseline quality.
  - Stage-5.5 (DSR formulation tightened): switched DSR from per-window-trial (n_trials=125, generous) to **per-combo-median** (n_trials=25, conservative — 1 baseline + 24 meta combos). New numbers: observed Sharpe **1.1848**, expected_max under H0 **0.1591**, deflated **z=23.28**, p=1.0000. The conservative formulation actually *strengthens* the deflated z (observed is the stable per-combo median while expected_max scales sub-linearly with n_trials). Verdict unchanged: **meta clearly beats baseline + survives DSR**.
  - Final-review bugfix: a timestamp-normalization bug in `strategies/momentum_meta.py::_rebalance` (rebalance_date at NYSE-close UTC vs midnight-aligned synthetic feature index) caused the meta strategy to silently stop trading after warm-up on real catalog data, producing an earlier degenerate harness result. Fixed by normalizing `rebalance_date` to midnight; regression test (`test_non_midnight_timestamps_still_trade`) added; harness now refuses to write a verdict when every threshold within a (window, combo) group produces the identical Sharpe.
- [x] Stage 6 — NLP sentiment overlay (clean rejection)
  - [x] News ingest: `uv run python data/ingest_alpaca_news.py` — Alpaca free-tier `/v1beta1/news`; headlines + per-ticker tags; written to `data/news/<ticker>/<YYYY-MM>.parquet`. Acceptance run: **4251 (ticker, article) rows** across 12 tickers / 2020-07 → 2026-05. Free tier caps responses at 50 items per 30-day window, so per-ticker coverage is sparse (~4 articles/ticker/month on average).
  - [x] Sentiment scoring: `uv run python examples/score_news.py` → `nlp/cache/finbert-v1/<ticker>.parquet`. **Implementation deviation:** original spec called for FinBERT, but anonymous HuggingFace Hub rate-limit prevented the 440MB model from completing on this machine (multiple `snapshot_download` attempts stalled mid-download). Substituted **VADER** (`vaderSentiment`, lexicon-only, ~100KB, zero downloads, instant inference). FinBERT restoration is a drop-in replacement inside `nlp/sentiment.py::_load_pipeline` when HF Hub is reachable / `HF_TOKEN` is set; rationale + trade-offs documented in that module's docstring. Acceptance: 3600 unique articles scored in **1.3 seconds**; mean=0.01, range [-0.61, +0.54].
  - [x] Feature integration: 3 sentiment features (`sentiment_7d_mean`, `sentiment_7d_count`, `sentiment_30d_mean`) added to `models/features.py` behind an optional `news_provider` seam. v2 (price + sentiment) trains into `models/cache_v2/`. `MomentumMetaConfig.news_provider_path` wires the SentimentStore through to the strategy at inference time.
  - [x] Ablation verdict: `uv run python examples/ablate_sentiment.py`. **Verdict: REJECT — sentiment is noise within permutation tolerance.** Median ΔIC=+0.0809, permutation p-value=0.8260 (one-sided, n_perm=1000). Per-fold IC is very noisy (range -1.00 to +0.72) because the v2 training set is only **111 rows after sentiment-NaN drop** (vs v1's 693 rows — 84% row loss from sparse coverage), which limits how many cutoffs can train (64 v2 models vs 472 v1) and how many test folds the ablation can evaluate (6 folds vs the theoretical ~50). The v2 harness corroborates: baseline 0.9541 vs meta winner `n200_d3_lr10 @ 0.5` median Sharpe 0.9476 (meta *underperforms* by -0.7%, vs v1's +29% meta win). The realized cause is exactly Spec §7's "sparse news per ETF" risk: sector ETFs (not individual stocks) attract limited per-ticker coverage. Future paths if revisited as Stage 6.5: (a) aggregate constituent-stock news per sector ETF (much denser); (b) FinBERT restoration when HF Hub is reachable (may sharpen the signal but won't fix sparsity); (c) accept that sector-rotation sentiment doesn't add to a meta-labeling overlay on top of price momentum.
- [x] Stage 7 — Streamlit research dashboard
  - [x] `dashboard/app.py` wires 5 tabs over typed read-only loaders in `dashboard/data_sources.py`:
    - **Equity** — Stage-2 momentum vs SPY (normalized) overlaid, rolling 60-day Sharpe, drawdown
    - **Harness Verdicts** — Stage-3 winner JSON + Stage-5/6 baseline-vs-meta cards (v1 + v2 metrics with %-delta)
    - **Sentiment** — Stage-6 ablation verdict (REJECT) + per-fold ΔIC bar chart + per-ticker headline browser
    - **Models** — `models/cache/` + `cache_v2/` coverage pivot + LightGBM feature importances (latest cutoff, first combo)
    - **Journal** — all `docs/superpowers/specs/*.md` + `plans/*.md` rendered inline, newest first
  - [x] Run: `uv run streamlit run dashboard/app.py` → opens on http://localhost:8501. Every loader is `@st.cache_data(ttl=300)` wrapped; views degrade gracefully (empty-state messages with the command to produce missing data).
- [~] Stage 8 — Alpaca paper-trading execution adapter (code complete; **3-month observation gate uncompressable per design spec §5**)
  - [x] `livesim/alpaca_exec_client.py` — `AlpacaLiveExecutionClient` implements the Nautilus `LiveExecutionClient` interface against alpaca-py's `TradingClient`. MARKET orders, paper-account only. Polling loop (1 Hz) emits Nautilus `OrderFilled` events from Alpaca fills. SELL-before-BUY pacing (0.5s) preserves the cash-account safety pattern. **Only runtime-trading module that imports `alpaca.trading`** — data side stays catalog-driven. (Test + preflight scripts also import `alpaca.trading`, by necessity.)
  - [x] `livesim/factories.py::AlpacaPaperExecClientFactory` + `examples/run_trading_node.py` `--exec-mode {simulated, alpaca-paper}` CLI flag. Simulated remains default. **The deployed paper-trade strategy is the Stage-3 baseline winner (`top_n=2, lookback=63, skip=0`), not the Stage-5 meta winner.** The meta overlay's +29% research edge has its own observation requirement; paper-trading meta would need a separate Stage-8.5 observation window and re-training cadence not yet built.
  - [x] `examples/preflight_alpaca.py` — stand-alone smoke confirms connectivity (account id, equity, buying power, positions, recent orders). **Verified against the user's real paper account: equity $100k, status ACTIVE.**
  - [x] `docs/superpowers/specs/2026-05-27-stage-8-deploy-checklist.md` — the human-action checklist for the 3-month paper observation that must precede real-money rollout. Real-money rollout is a 1-line config change (`paper=False`) gated entirely by this checklist, not by code complexity. **Defense-in-depth (Stage-8.5):** `AlpacaLiveExecutionClient.__init__` now raises `RuntimeError` if `paper=False` is passed, requiring a deliberate commit that references the checklist sign-off — so flipping to real money is no longer a single-config-knob change.
  - [ ] **3-month paper observation** (the parent spec gate). I cannot complete this in-session — it requires daily runs over ≥3 months with the metrics in checklist §C-E. Real-money rollout follows §F-G only if all gates pass.
- [x] Stage 9 — Multi-timeframe TradingView-style dashboard + self-service Backtest Lab
  - [x] `data/ingest_alpaca_intraday.py` — 1m / 5m / 15m / 30m / 1h / 4h bar ingest from Alpaca's free-tier IEX feed (sibling of Stage 1's daily catalog). Idempotent per-(ticker, timeframe, month) parquet shards under `data/intraday/<ticker>/<tf>/<YYYY-MM>.parquet`.
  - [x] `data/multitf.py` — single `get_bars(ticker, timeframe, start, end)` entry. Three branches: intraday parquet / Stage-1 daily catalog / pandas OHLC resample for 1w / 1M / 1Y. Uniform output schema [ts, open, high, low, close, volume] across all timeframes.
  - [x] `dashboard/views/equity.py` — rewritten to use `streamlit-lightweight-charts` (Streamlit-native wrapper around TradingView's open-source Lightweight Charts, Apache 2.0). Ticker selector + TradingView-style timeframe segmented control (1m / 5m / 15m / 30m / 1h / 4h / 1d / 1w / 1M / 1Y) + candlestick + volume subchart with up/down color coding. Per-timeframe sensible-default lookback (1m: 5 days → 1Y: 20 years). Falls back to legacy Stage-7 equity overlay when intraday cache empty.
  - [x] `dashboard/views/backtest_lab.py` — self-service strategy tester. Auto-discovers `(Strategy, StrategyConfig)` pairs via `pkgutil` walk over `strategies/` (new strategy = new dropdown entry, no manual registry). Auto-renders parameter form via msgspec `__struct_fields__` introspection (subclass-only fields; the 16+ inherited base fields stay hidden). Three modes: **Single** (one backtest via `analysis.runner.run`), **Grid** (sweep one numeric param ≤200 cells via `analysis.walk_forward.walk_forward` + sortable table + bar chart), **Optuna** (n_trials hyperparameter optimization writing to `results/optuna.db` + shows the launch command for the rich `optuna-dashboard` visualization).
  - [x] `dashboard/views/strategy_runner.py` — form-driven launcher for `examples/run_trading_node.py` (exec mode / fast-replay / poll-seconds / slippage); streams subprocess stdout via `st.write_stream`. For quick smoke checks only — the actual 3-month paper observation is terminal-based.
  - [x] `dashboard/app.py` — 7 tabs wired (Equity, Backtest Lab, Strategy Runner, Harness Verdicts, Sentiment, Models, Journal). All on the free-tier Alpaca IEX feed. 170 tests pass.
  - **Costs**: $0/mo as designed. Optional Tiingo Power ($30/mo) gives 30+ years of daily/weekly history if you outgrow the ~5.5yr Alpaca IEX window. Live websocket tick streaming is also free on Alpaca; historical tick playback requires Alpaca Plus ($99/mo) or Polygon (~$99-199/mo) and is **not** built — most retail dashboards show live ticks, not historical.

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

Stage-4.5 and its 2026-05-27 follow-up closed the real-catalog parity gap (was ~29%, now drift=0.000000 over the 2-year regression window; full-catalog run completes without overdraft). See the Stage 4 status entry above for the four root causes (risk-engine throttle, fill-bar semantic, `_on_bar`/`_submit_order` ordering on the msgbus, and the strategy's stale-close decision-time references). The regression test `tests/test_live_sim_parity_real_catalog.py` widened its window to 2021→2023 to cover the 2022-10-03 rebalance that exposed cause 4 and gates real-catalog drift at 2%.

### Alpaca paper trading (Stage 8 — code-complete; multi-month observation gate before real money)

```bash
# 1. Preflight — verify paper-account credentials + connectivity
uv run python examples/preflight_alpaca.py

# 2. Run the live node against Alpaca paper (orders go to the real broker)
uv run python examples/run_trading_node.py --exec-mode alpaca-paper --poll-seconds 60
```

`paper=True` is hardcoded; real-money rollout is a 1-line config change gated by `docs/superpowers/specs/2026-05-27-stage-8-deploy-checklist.md`. Read that checklist end-to-end before considering real capital.

### Dashboard (Stage 7 + Stage 9 — 7 tabs, TradingView-style multi-timeframe)

```bash
# 1. (optional, recommended) Populate intraday cache from Alpaca free-tier IEX
uv run python data/ingest_alpaca_intraday.py --timeframe 5m
uv run python data/ingest_alpaca_intraday.py --timeframe 1h

# 2. Launch dashboard
uv run streamlit run dashboard/app.py
# -> opens on http://localhost:8501
```

**Tabs** (Stage 7 + Stage 9 combined):

| Tab | What you see |
|---|---|
| Equity | TradingView-style candlestick + volume chart with timeframe selector: 1m / 5m / 15m / 30m / 1h / 4h / 1d / 1w / 1M / 1Y. Ticker selector over the 12-ETF universe. |
| Backtest Lab | Auto-discovered strategy + auto-rendered param form. Three modes: Single backtest / Grid sweep (≤200 cells) / Optuna optimization. Same engine as the canonical Stage-3 harness. |
| Strategy Runner | Form-driven launcher for `run_trading_node.py` (exec-mode + fast-replay or poll cadence). Streams subprocess output inline. |
| Harness Verdicts | Stage-3 winner JSON + Stage-5/6 baseline-vs-meta cards with %-delta. |
| Sentiment | Stage-6 ablation verdict + per-fold ΔIC + per-ticker headline browser. |
| Models | `models/cache/` + `cache_v2/` coverage pivots + LightGBM feature importances. |
| Journal | All `docs/superpowers/specs/*.md` + `plans/*.md` rendered inline, newest first. |

Built on `streamlit-lightweight-charts` (MIT, 223⭐ Streamlit-native wrapper around TradingView's open-source `lightweight-charts` JS library, Apache 2.0). Strategy auto-discovery via `pkgutil`; param-form auto-render via msgspec `__struct_fields__` introspection. Optuna mode hands off to `optuna-dashboard` (MIT, separate browser tab) for parallel-coords / hyperparameter-importance / contour plots — don't re-implement what's already free.

**Costs**: $0/mo as designed. All deps MIT / Apache. Alpaca free-tier IEX gives ~5 years of 1m → 1h history + ~5.5 years daily. Optional Tiingo Power ($30/mo) gives 30+ years of daily history if needed.

### Sentiment ablation (Stage 6)

```bash
# 1. Ingest news from Alpaca free-tier (~5 min for full 2020-07 -> today)
uv run python data/ingest_alpaca_news.py
uv run python data/ingest_alpaca_news.py --since 2024-01-01    # incremental

# 2. Score with VADER (~1 sec; FinBERT swap-in is a drop-in change in nlp/sentiment.py)
uv run python examples/score_news.py
uv run python examples/score_news.py --smoke                    # ~100 headlines

# 3. Train v2 model grid (price + sentiment features)
uv run python examples/train_meta_models.py --features-version v2

# 4. Harness eval on v2 (for the baseline-vs-meta comparison)
uv run python examples/run_harness_meta.py --features-version v2

# 5. Ablation verdict — does sentiment add signal?
uv run python examples/ablate_sentiment.py
```
Writes `summary.md` + `per_fold.csv` to `results/sentiment_ablation_<timestamp>/`. The current verdict is **REJECT** (sentiment is noise within permutation tolerance); see Stage 6 status entry above for the honest interpretation.

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
              + Stage-6 news shards under data/news/<ticker>/<YYYY-MM>.parquet (gitignored)
strategies/   Nautilus Strategy implementations + venue/FillModel helper
analysis/     honesty-harness modules (runner, walk_forward, costs, DSR, tearsheet)
livesim/      Stage-4 TradingNode wiring — CatalogLiveDataClient, SimulatedLiveExecutionClient, factories, node_config
models/       Stage-5 ML pipeline — labeling, features, CV, trainer, dated ModelStore (cache/ gitignored; cache_v2/ for v2 price+sentiment models)
nlp/          Stage-6 NLP — VADER (FinBERT-ready) scorer + persistent SentimentStore (cache/ gitignored)
dashboard/    Stage-7 Streamlit dashboard — typed data loaders + 5 view modules
examples/     runnable demos (smoke, momentum, harness, run_trading_node, train_meta_models, run_harness_meta, ingest news, score_news, ablate_sentiment)
scripts/      one-off scripts (yfinance cross-check)
tests/        pytest suite
docs/         design spec, plans, Nautilus API notes
```
