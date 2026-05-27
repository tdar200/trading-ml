# trading-ml — Project Completion Summary (2026-05-27)

## What this is

The 8-stage research-and-deployment roadmap from `docs/superpowers/specs/2026-05-26-trading-ml-design.md` is fully addressed. Stages 0–7 are merged, tested, and acceptance-verified. Stage 8 is code-complete with a documented multi-month observation gate before real-money rollout. This document is the closing artifact: what shipped, what the verdicts were, and what's intentionally left for a future session.

## Headline verdicts (post-Stage-4.5-followup, top_n=2 baseline)

| Stage | Verdict |
|---|---|
| 2 — Momentum top-N | Tearsheet vs SPY produced. The simple momentum strategy underperforms SPY in CAGR (≈9% vs ≈21%) but exists as an honest baseline. |
| 3 — Honesty harness | Winner `top_n=2, lookback_days=63, skip_days=0`. Median window Sharpe **0.9541**, range [0.875, 1.847], DSR z=23.96 (n_trials=90), CAGR 0.175 robust across slippage 0→2 bps. |
| 4 — LiveNode infra | Real-catalog parity vs `BacktestEngine`: **0.78% drift** over the full ~5.8-year window (was 29% before the Stage-4.5 follow-up). Regression test (`tests/test_live_sim_parity_real_catalog.py`) gates at 2%. |
| 5 — Meta-labeling | Meta winner `n100_d3_lr05 @ threshold 0.5`: median window Sharpe **1.2319** vs baseline 0.9541 (**+29.1%**). DSR z=25.14, p=1.0000 (per-combo-median formulation, n_trials=25). Verdict: **meta clearly beats baseline + survives DSR**. |
| 6 — Sentiment overlay | **REJECT — sentiment is noise within permutation tolerance.** Median ΔIC +0.0809, permutation p=0.8260, n_folds=6. v2 training set only 111 rows (vs v1's 693) due to sparse free-tier ETF news; VADER substituted for FinBERT due to HF Hub rate-limit. |
| 7 — Dashboard | Streamlit, 5 tabs (Equity, Harness Verdicts, Sentiment, Models, Journal), read-only over existing artifacts. Smoke verified end-to-end. |
| 8 — Alpaca paper adapter | Code complete. Preflight verified against real paper account ($100k equity, status ACTIVE). 3-month observation gate before real money is the binding constraint. |

## What 4 root-cause bugs got fixed

1. **Stage-5 timestamp normalization** — meta strategy silently stopped trading after warm-up on real catalog data because `rebalance_date` wasn't normalized to midnight, missing the synthetic feature index. Fix: one `.normalize()` call + non-midnight regression test + harness-level "all-thresholds-identical" guard.
2. **Stage-4.5 throttle** — `LiveRiskEngineConfig.max_order_submit_rate` defaulted to `100/00:00:01`, which under fast-replay's `LiveClock` denied 195 of 295 orders. Raised to `1_000_000/00:00:01` in `livesim/node_config.py`.
3. **Stage-4.5 fill-bar semantic** — livesim filled at next-bar OPEN; `BacktestEngine` fills at decision-bar CLOSE (Nautilus daily-bar `SimulatedExchange` semantic). Aligned livesim to BT.
4. **Stage-4.5 cause-3 (msgbus race)** — exec client's `_on_bar` fires BEFORE strategy's `on_bar` for the same bar, so `SubmitOrder` couldn't find a pending entry. Switched to submit-time fill buffering against `_last_bar_by_ticker`.
5. **Stage-4.5 follow-up (strategy fix)** — rebalance triggered on first universe bar, but bars later in `ALL_TICKERS` iteration order still held YESTERDAY's close at decision time. Fix: anchor trigger to the BENCHMARK bar (always last in `ALL_TICKERS`). This retroactively shifted Stage 2/3/5 numbers; the README documents the before/after honestly.

## Deferred follow-ups (tracked here for the next session, not blockers)

These were surfaced by the post-Stage-8 whole-project review and the Stage-8.5 cleanup pass. None block walking away.

### Code-quality
1. **Alpaca exec adapter integration test.** `tests/test_alpaca_exec_client.py` reimplements `_handle_alpaca_order_update`'s dispatch logic inline rather than driving the real `AlpacaLiveExecutionClient` instance (Cython base class is impractical for unit-test instantiation). A mock-`TradingClient` integration test that boots the `TradingNode` end-to-end would cover `_emit_fill_from_alpaca` and protect against Nautilus 1.227 API drift.
2. **Stage-5 meta v2 inference path test.** No test exercises a `MomentumMetaStrategy` instance with `news_provider_path` wired through `MomentumMetaConfig`. The v2 path at inference time is currently single-sourced from `examples/run_harness_meta.py --features-version v2`.
3. **`ALL_TICKERS[-1] == BENCHMARK` invariant.** Now asserted at module import in `data/tickers.py` and tested in `tests/test_tickers.py::test_benchmark_is_last_in_all_tickers`. Landed as part of Stage-8.5 cleanup.

### Documentation polish
4. **Stale "FinBERT" mentions** in `examples/score_news.py` docstring — fixed in Stage-8.5 cleanup; the active implementation is VADER per `nlp/sentiment.py`'s explicit substitution note.
5. **`pyproject.toml` has `torch>=2.2` + `transformers>=4.40`** with no runtime importers (post-VADER swap). Justifiable as "FinBERT-ready" but worth a clearer comment than the current docstring claim that "FinBERT inference on CPU is fast enough."
6. **DSR `observed_sharpe` label** in `analysis/harness_report.py` for Stage-3 reports the max single-trial Sharpe (1.8472), not the winner's median (0.9541). Per-run summary.md should rename or recompute.
7. **`backtest_momentum.py` defaults** are the Stage-2 milestone settings (top_n=3, lookback=126, skip=21), not the Stage-3 winner (top_n=2). A docstring note prevents the dashboard from being misread as "what's being deployed."

### Strategy-deployment honesty
8. **Stage-8 deploys baseline, not meta.** Documented in Stage-8.5 README update. The +29% meta edge has its own observation requirement and re-training cadence; paper-trading meta is a separate Stage-8.5 effort, not a 1-line config change.

### Stage-8.5 defenses landed
- `AlpacaLiveExecutionClient.__init__` now raises `RuntimeError` if `paper=False` is passed — flipping to real money is no longer a silent config-knob change. Removing the guard requires a deliberate commit that should reference checklist sign-off.
- README's "only module that imports alpaca.trading" wording corrected to "only runtime-trading module" (preflight + test also import it).
- Module-level `assert ALL_TICKERS[-1] == BENCHMARK` in `data/tickers.py` prevents future refactors from silently re-introducing the Stage-4.5 stale-close bug.

## What only a human can do now

1. **The 3-month paper observation** per `docs/superpowers/specs/2026-05-27-stage-8-deploy-checklist.md` §B-E.
2. **Real-money rollout** per checklist §F-G — only after paper Sharpe ≥ 50% of backtest Sharpe over the full window AND no uncaught exceptions for 4 weeks AND slippage within 2× cost-stress range AND all of §F.
3. **Rotate the Alpaca paper credentials** in `.env`. They've been in the working directory across many sessions and the conversation transcript; rotating is cheap and resets the clock.
4. **Decide whether to chase Stage 6.5** (constituent-stock news aggregation; the cleanest path to a non-REJECT sentiment verdict) or to accept that sector-rotation sentiment doesn't add to a meta-labeling overlay on top of price momentum.

## Test count
- **152 passing** (151 from the 8-stage merge + 1 from the Stage-8.5 ticker invariant test).
- The 2 integration tests gated by `.env` (`tests/test_catalog_integration.py`) auto-skip when credentials are absent.

## Files added in Stage 8.5
- This document (`docs/superpowers/specs/2026-05-27-project-completion-summary.md`).
- Module-level `assert` + comment in `data/tickers.py`.
- `test_benchmark_is_last_in_all_tickers` in `tests/test_tickers.py`.
- `paper=False` guard in `livesim/alpaca_exec_client.py::__init__`.
- README polish on Stage 8 (deploys-baseline-not-meta note + corrected "only runtime-trading module" wording).
- Stage-6 score_news.py docstring updated to reflect the VADER substitution.
