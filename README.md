## Financial QA audit — 30 September 2026

See [the independent audit](docs/qa-financial-audit.md) before interpreting trading results. The local £100 replay/paper path has an independent Decimal oracle, a permanent 30-trade golden fixture and numerical browser/export tests. The legacy Alpaca adapter still fails partial-fill accounting and is not certified. Live money remains disabled.

Current-source historical outputs are in `results/qa_audit_20260930/corrected_research`; old `results/small_account` source fingerprints intentionally no longer match after repairs. Legacy CCI/RSI and multi-asset CV outputs require regeneration. No strategy qualified for deployment.

## Re-audit corrections, 1 October 2026

An independent re-audit (see [docs/fundamental-reaudit-2026-10-01.md](docs/fundamental-reaudit-2026-10-01.md)) recomputed the headline research claims. These statements below are **withdrawn or qualified**:

- **Stage 5 "meta clearly beats baseline OOS, +29.1% median window Sharpe" is withdrawn.** The harness "walk-forward" windows are nested and expanding, so their median is not an out-of-sample statistic. On the full 1,463-session window the best meta combination has Sharpe 0.9916 against 0.9867 for the baseline (+0.5%), only 1 of 24 combinations beats the baseline, total return is 68.8% against 153.7%, and meta wins 1 of 5 disjoint sub-periods.
- **The Deflated Sharpe values (z = 23.96, 25.14, 23.28) are invalid.** They combine an annualised Sharpe with a per-day standard error. Units-consistent z is about 2.5 (Stage 3), 2.1 and 1.9 (Stage 5). Pairing the best-window Sharpe with its own 295-session sample gives z = 1.11, p = 0.87, below the 0.95 gate, so "survives DSR" does not hold.
- **The macro filter is not shown to add value.** It was no better than random trade skipping in a 24-seed placebo (all p > 0.2); it is inert for XLE (score above zero on every day) and blocks every XLP and XLV trade. The XLE zero increment is therefore not a robust null.
- **Selection bias:** the "leading XLE candidate" is the top-ranked of 144 candidates with no candidate qualifying, and all 144 diagnostic outcomes had already been inspected.
- **Known limitation (QA-BT-02):** Stage 5 features use the close of the decision date, and backtests fill at that same close, so decision information is not strictly earlier than the fill. Not fixed here because it would invalidate the stored models.

# trading-ml

> Research audit (2026-09-30): historical DSR confidence values below were
> computed with inconsistent annualized/per-day units and are invalid as
> deployment evidence. The scalar calculation is now corrected, but older
> reports have not been regenerated; medians of window Sharpes still do not
> satisfy its IID sampling assumptions. See [£100 research](docs/small-account-research.md).


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
  - [x] `dashboard/views/backtest_lab.py` — self-service research lab with five modes: **Single**, **Grid**, **Parameter Optimizer**, structural **Strategy Finder**, and intraday **Scalping Finder**.
  - [x] `dashboard/views/strategy_runner.py` — form-driven launcher for `examples/run_trading_node.py` (exec mode / fast-replay / poll-seconds / slippage); streams subprocess stdout via `st.write_stream`. For quick smoke checks only — the actual 3-month paper observation is terminal-based.
  - [x] `dashboard/app.py` — 7 tabs wired (Equity, Backtest Lab, Strategy Runner, Harness Verdicts, Sentiment, Models, Journal). All on the free-tier Alpaca IEX feed.
  - **Costs**: $0/mo as designed. Optional Tiingo Power ($30/mo) gives 30+ years of daily/weekly history if you outgrow the ~5.5yr Alpaca IEX window. Live websocket tick streaming is also free on Alpaca; historical tick playback requires Alpaca Plus ($99/mo) or Polygon (~$99-199/mo) and is **not** built — most retail dashboards show live ticks, not historical.
- [x] Stage 10 — Robust Parameter Optimizer + hardened backtest validation
  - [x] cTrader-style exhaustive grid mode enumerates every finite parameter combination.
  - [x] Seeded adaptive TPE mode searches large spaces within a trial budget.
  - [x] Hybrid tensor mode batches the complete grid on CPU/CUDA, keeps strong candidates from diverse parameter regions, then validates every finalist through Nautilus.
  - [x] Every candidate is evaluated on non-overlapping windows and ranked by median Sharpe with stability/drawdown penalties; no-trade and invalid candidates are disqualified.
  - [x] Full ranked results, per-window evidence, and Pareto-optimal candidates are retained—not only one winner.
  - [x] Canonical runner rejects ambiguous datetimes, missing data, duplicate/unsorted bars, ticker mismatches, and missing instrument definitions.
  - [x] CLI: `uv run python examples/optimize_parameters.py --smoke`; dashboard: Backtest Lab → Parameter Optimizer.
  - [x] Real-catalog acceptance smoke: 2 candidates × 2 independent windows; top_n=2 ranked first with robust score 0.9878. **This smoke result is validation evidence, not a trading recommendation.**
  - [x] 228 offline tests pass, including optimizer repeatability, full-grid coverage, tensor batching, CUDA/CPU execution parity, synthetic Nautilus winner parity, Pareto ranking, data integrity, event/live parity, structural-search, and multi-instrument scalping suites.
  - [x] CUDA pre-screen acceptance passed on the local RTX 3050: all 165 default combinations were tensor-screened on GPU, then 20 finalists were validated across 4 windows (80 Nautilus runs). The lockfile remains CPU-portable, so use the activation instructions in `docs/parameter-optimizer.md`; Nautilus finalist validation remains CPU-bound by design.

- [x] Stage 11 — Structural Strategy Finder
  - [x] Serializable rule grammar searches momentum vs mean reversion, optional 126/200-day trend filters, and optional positive-signal gates.
  - [x] Crosses each rule structure with the existing top-N/lookback/skip parameter grid; default search covers 12 structures and 2,160 combined candidates.
  - [x] GPU/CPU tensor screen preserves representatives from distinct structures; canonical Nautilus validation and an explicit complexity penalty determine final rankings.
  - [x] Plain-English descriptions plus rankings, per-window evidence, and structural pre-screen CSV exports.
  - [x] CLI: `.venv/bin/python examples/find_strategies.py --smoke --device cuda`; dashboard: Backtest Lab → Rotation Rule Finder.
  - [x] The grammar is constrained and serializable—no generated Python is executed. See `docs/strategy-finder.md`.

- [x] Stage 12 — Intraday Scalping Finder
  - [x] Searches mean-reversion, momentum, and breakout structures across 60 instruments on 5-minute bars: 30 stocks, 10 commodity proxies, 10 crypto pairs, and 10 other contenders; SPY is benchmark-only.
  - [x] CUDA generates signals, screens all 108 entry rules independently per instrument, and executes the stateful validation kernel; 18 cross-instrument finalists expand into 162 candidates evaluated across 240 instrument-window cells.
  - [x] Full CUDA-first acceptance completes in 170.24 seconds (including data I/O and reports), with all rankings and holdouts matching the CPU reference within FP64 tolerance.
  - [x] Uses completed-bar signals, next-bar-open fills, regular-session entries, mandatory end-of-day flattening, common-session alignment, and explicit per-side cost stress.
  - [x] Preserves an untouched final 20% holdout for every instrument and reports aggregate, sleeve-level, and per-instrument evidence against passive SPY.
  - [x] Full acceptance run used 2,125,227 traded bars / 490 common sessions. Best discovery candidate held at most 30 minutes but failed holdout: -10.80% average instrument return, -2.209 median Sharpe, 1/60 instruments positive, 0/60 beating SPY; median return at 5 bp/side was -34.76%.
  - [x] **Verdict: REJECT — no tested scalping candidate is deployable.**
  - [x] CLI: `.venv/bin/python examples/find_scalping_strategies.py --device cuda`; dashboard: Backtest Lab → Scalping Finder.
  - [x] Five-minute IEX bars are not tick/quote/order-book data; scope and evidence are documented in `docs/scalping-finder.md`.

- [x] Stage 13 — Selective Opening Range Finder
  - [x] CUDA-only long/short 15/30/60-minute opening-range structures across the 30-stock basket.
  - [x] Causal relative-volume, ATR, and optional SPY-alignment filters; exact-clock complete-range enforcement and one trade per stock/day.
  - [x] Shared-capital equal-weight portfolio ranking across four discovery windows, followed by an untouched final 20% holdout and 1/2.5/5 bp cost stress.
  - [x] Searches 162 entry structures and validates 648 fully specified candidates on 1,538,750 bars.
  - [x] Acceptance winner: short 15-minute range, 0.5 ATR breakout, 1× relative volume, SPY alignment, 1.5 ATR target, 0.5 ATR stop, 30-minute hold.
  - [x] **Verdict: REJECT** — untouched portfolio -2.92%, Sharpe -4.780, -2.94% maximum drawdown, 1,563 trades; -6.87% at 5 bp/side.
  - [x] CLI: `.venv/bin/python examples/find_opening_range_strategies.py --device cuda`; dashboard: Backtest Lab → Opening Range Finder.
  - [x] See `docs/opening-range-finder.md` and `results/opening_range_finder_20260912T135021Z/`.

- [x] Stage 14 — Cross-Sectional Market-Neutral Finder
  - [x] CUDA search over 972 momentum/reversal portfolio candidates across 30 individual stocks, with fixed intraday decision times, beta-residual ranking, relative-volume and dispersion gates, and at most one portfolio entry per day.
  - [x] Next-bar-open entry/exit, explicit two-sided costs, causal rolling SPY beta, and weight projection to approximately zero dollars and zero market beta.
  - [x] Four discovery windows plus a sealed final 20% holdout, 1/2.5/5 bp cost stress, contribution concentration, combinatorial PBO, and Deflated Sharpe diagnostics.
  - [x] Best research candidate: 10:00 ET momentum ranking over 30 minutes, long/short five, held 60 minutes. Untouched holdout at 1 bp/side: +5.83%, Sharpe 1.540, -4.21% maximum drawdown, 1,140 stock round trips, mean absolute beta 0.003.
  - [x] **Verdict: REJECT** — return falls to -3.19% at 5 bp/side, combinatorial PBO is 66.7%, and multiple-testing-adjusted Deflated Sharpe confidence is effectively 0%; quote-aware validation and a new future holdout are still required.
  - [x] CLI: `.venv/bin/python examples/find_cross_sectional_strategies.py`; dashboard: Backtest Lab → Cross-Sectional Finder.
  - [x] See `docs/cross-sectional-finder.md`.

- [x] Stage 15 — Support/Resistance Confluence Finder
  - [x] Six mechanical causal level families: previous day, previous week, available IEX premarket proxy, 30-minute opening range, 24-bar swing range, and expanding VWAP bands, plus an any-level ensemble.
  - [x] Three-bar breakout/retest, rejection, and failed-breakout confirmations combined with momentum/reversal beta-residual ranking across 30 stocks.
  - [x] CUDA evaluates 1,680 candidates with next-open entries, beta-neutral long/short weights, 0.5 ATR stop, 2R target, stop-first same-bar ambiguity, and 1/2.5/5 bp cost stress.
  - [x] Discovery winner: any-level failed-breakout reversal at 10:00 ET, 30-minute lookback, long/short two, 120-minute maximum hold.
  - [x] **Verdict: REJECT** — reused evaluation period returned -2.73% at 1 bp/side and -9.53% at 5 bp/side, Sharpe -1.440. It did not improve the plain cross-sectional winner.
  - [x] Evaluation dates are explicitly marked as previously exposed and can never produce a deployment PASS. A new future period plus SIP/NBBO replay is mandatory.
  - [x] CLI: `.venv/bin/python examples/find_support_resistance_strategies.py`; dashboard: Backtest Lab → Support/Resistance Finder.
  - [x] See `docs/support-resistance-finder.md` and `results/support_resistance_finder_20260912T152530Z/`.

- [x] Stage 16 — Cost-Gated Profitability Finder
  - [x] Four pre-registered families: first-half-hour sector/beta-residual momentum, causal intraday seasonality, pooled walk-forward ridge, and CUDA elastic net; SIP quote-flow confirmation activates only with sufficient NBBO coverage.
  - [x] Dynamic per-session universe recovered 667 sessions while requiring ≥15 eligible stocks; CUDA FP64 forecast fitting, causal expected-return/error calibration, variable breadth, cash/beta-neutral projection, regime filters, and explicit no-trade gates.
  - [x] Fail-closed SIP quote ingest plus optional point-in-time membership and earnings/macro blackout inputs. Current IEX bars are never mislabeled as NBBO data.
  - [x] Full run searched 4,536 candidates: 60 warm-up + 485 discovery + 122 evaluation sessions. Winner returned +7.94% at 1 bp/side, Sharpe 1.620, max drawdown -6.51%, 60 events.
  - [x] Frozen-candidate SHA-256 manifests, append-only fingerprint-bound paper journal, strict deployment qualifier, and fail-closed pre-order exposure/data/spread/reconciliation/duplicate/daily-loss/kill-switch guard.
  - [x] **Verdict: REJECT** — -0.33% at 2.5 bp/side, no trades at 5 bp, PBO 50.0%, effectively zero DSR confidence, insufficient evaluation/events, and required SIP/membership/events/new-forward data are absent.
  - [x] The 60-signal combinatorial engine is exposed as Backtest Lab → Strategy Finder; the older two-family ETF search is labelled Rotation Rule Finder.
  - [x] See `docs/profitability-finder.md` and `results/profitability_finder_20260912T164940Z/`.
  - [x] V3 limits the search to 40–48 preregistered configurations and adds
    causal VWAP, opening/prior-range support/resistance, rejection, relative-volume,
    causal online expert weighting, regime routing, event, and SIP confirmation
    with conservative latency/impact costs.
  - [x] Keyless hourly acceptance used 723 common sessions with 252 held out.
    The adaptive families did not displace pooled ridge; holdout return was
    -4.39% at 2.5 bp/side, Sharpe -1.439, 29 events: **REJECT**.
  - [x] V4 primitive-signal finder provides exactly 60 causal indicators and
    constructs standalone plus 2–4 signal consensus strategies. The acceptance
    run GPU-screened 1,586 combinations, canonically replayed 48 finalists, and
    found EMA(5) distance + stochastic(5) + ROC(3). It failed the exposed
    evaluation at -28.99%, Sharpe -2.826, PBO 66.7%, DSR confidence 0.04%:
    **REJECT**. Future runs now require positive worst-window discovery results
    before the holdout can be opened.
  - [x] V5 replaces continuous indicator voting with natural threshold/crossing
    setups and searches individual indicator/time variants without family or
    correlation pruning, separates 280 orientation sessions from 126 nested
    validation sessions with a 5-session purge, and uses one-bar versus
    session-close targets. The unrestricted 60-indicator, one-to-four-way space
    contains 523,685 combinations per decision time and is GPU-screened in
    batches before canonical finalist validation.
  - [x] Separate equity, commodity-proxy, other-security, and crypto sleeves prevent
    incompatible session and fee models from being mixed. Crypto fails closed until
    its dedicated continuous-session replay exists.
  - [x] Data/deployment preflights and operator runbook:
    `examples/preflight_profitability_data.py`,
    `examples/preflight_profitability_deployment.py`, and
    `docs/profitability-deployment-runbook.md`.
  - [x] Keyless Yahoo research ingest is isolated under `data/free_intraday`,
    carries explicit `yfinance_research_only` provenance, and cannot satisfy
    SIP/NBBO or broker-evidence approval gates.

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
| Backtest Lab | Single, Grid, Parameter Optimizer, structural Strategy Finder, and intraday Scalping Finder; GPU/CPU screening plus robust multi-window/Pareto rankings and strict holdout rejection. |
| Strategy Runner | Form-driven launcher for `run_trading_node.py` (exec-mode + fast-replay or poll cadence). Streams subprocess output inline. |
| Harness Verdicts | Stage-3 winner JSON + Stage-5/6 baseline-vs-meta cards with %-delta. |
| Sentiment | Stage-6 ablation verdict + per-fold ΔIC + per-ticker headline browser. |
| Models | `models/cache/` + `cache_v2/` coverage pivots + LightGBM feature importances. |
| Journal | All `docs/superpowers/specs/*.md` + `plans/*.md` rendered inline, newest first. |

Built on `streamlit-lightweight-charts` (MIT, 223⭐ Streamlit-native wrapper around TradingView's open-source `lightweight-charts` JS library, Apache 2.0). Strategy auto-discovery via `pkgutil`; param-form auto-render via msgspec `__struct_fields__` introspection. Parameter Optimizer uses Optuna TPE or exhaustive grids; Strategy Finder searches constrained serialized rule structures.

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

## Rule Strategy Finder

The main dashboard now includes a **Rule Strategy Finder** tab for instrument ×
timeframe × indicator-role searches, chronological validation, saved input
snapshots and exact trade replay. Launch it separately with:

```bash
.venv/bin/python -m streamlit run dashboard/finder_app.py
```

Use **Synthetic demo** to exercise the complete workflow, or select existing
local market data. See [the workflow and testing guide](docs/rule-strategy-finder.md)
for CLI commands, execution assumptions and browser E2E tests.

## Project directories

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

## £100 account research and paper framework

The 2026-09-30 audit and 144-candidate comparison produced **NO_TRADE**: no
strategy qualified for real capital. The 10× target is analyzed, never used
to raise risk. See [research and runbook](docs/small-account-research.md),
[markets and evidence](docs/small-account-market-evidence.md), and
[architecture audit](docs/small-account-architecture.md).

```bash
.venv/bin/python examples/research_small_account.py
.venv/bin/python examples/replay_small_account.py
.venv/bin/python examples/paper_small_account.py --once
.venv/bin/python -m streamlit run dashboard/small_account_app.py
```

Public live GBP crypto quotes feed a local paper journal. Real-money orders
are unavailable. The default evidence gate rejects entries; `--shadow` is
an explicitly hypothetical paper experiment. Create
`results/small_account/STOP_TRADING` to block new entries in a running process.
