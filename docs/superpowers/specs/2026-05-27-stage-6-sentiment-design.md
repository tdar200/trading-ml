# Stage 6 — NLP Sentiment Overlay Design

## 1. Context & goal

The parent project's Stage-5 meta-labeling overlay (LightGBM filter + sizer on top of Stage-2 momentum) is validated end-to-end on the 11-feature price-only feature set (`models/features.py`): with the post-Stage-4.5 strategy and top_n=2 baseline, meta beats baseline 1.23 vs 0.95 (+29%) median window Sharpe; conservative per-combo-median DSR p=1.00, z=25.14.

Stage 6's question: **does news sentiment add incremental signal to the meta-model, or not?** Both outcomes are valid milestones per the parent spec §5. The goal is a *credible verdict* — not a particular sign.

## 2. Architecture

Three new isolated units, one extension point, two re-runs.

### New code

| Path | Responsibility |
|---|---|
| `data/ingest_alpaca_news.py` | Pull historical news from Alpaca's free `/v1beta1/news` endpoint for the 12-ticker universe over the Stage-1 catalog window. Idempotent (re-runs only fetch new days). Writes `data/news/<ticker>/<YYYY-MM>.parquet`. |
| `nlp/__init__.py` | Package marker. |
| `nlp/sentiment.py` | FinBERT inference. `score_headlines(headlines: list[str]) -> np.ndarray` batched, returns net sentiment ∈ [-1, +1] (= P(pos) − P(neg)). Loads model lazily. |
| `nlp/sentiment_store.py` | `SentimentStore(cache_dir, model_version)` — persistent cache keyed by article hash so we never re-infer. Exposes `score_or_compute(headlines)` and `get_scored_per_ticker(ticker, start, end) -> pd.DataFrame[date, score]`. |
| `examples/score_news.py` | Runnable: bulk FinBERT inference over ingested news → write to sentiment_store. Smoke mode for ~100 headlines. |

### Touched (additive only)

| Path | Change |
|---|---|
| `models/features.py` | Add 3 sentiment features (`sentiment_7d_mean`, `sentiment_7d_count`, `sentiment_30d_mean`) to `FEATURE_COLUMNS`. New optional `news_provider` parameter on `extract_features_for_date` / `make_features`. When `news_provider=None`, sentiment features are NaN (and `make_features` drops the row, same as for missing price features) — preserves Stage-5 behaviour. |
| `examples/train_meta_models.py` | Accept `--cache-dir` override + `--features-version` flag. Default `v1` (price-only, no sentiment) preserves Stage-5 cache; `v2` enables sentiment features + writes to `models/cache_v2/`. |
| `examples/run_harness_meta.py` | Accept `--features-version v1|v2` to switch model_store_path between `models/cache/` and `models/cache_v2/`. Writes per-side `summary.md`. |

### Untouched

`strategies/`, `livesim/`, `analysis/`, all existing tests. Stage-5 models in `models/cache/` remain authoritative for the v1 baseline.

### Boundaries

- `nlp/` is the **only** package that imports `transformers` / `torch`.
- `data/ingest_alpaca_news.py` is the **only** module that touches Alpaca's news API.
- `models/features.py` reads sentiment from `nlp/sentiment_store.py` only — never imports `transformers` directly.
- The `news_provider` parameter on `extract_features_for_date` is the seam between price features (catalog-driven, fast) and sentiment features (cache-driven, optional).

## 3. Stack additions

| Component | Library | Why |
|---|---|---|
| Transformer model | `transformers>=4.40`, `torch>=2.2` | FinBERT (`ProsusAI/finbert`) — standard finance-news sentiment baseline |
| Tokenizer + model files | downloaded at first run via HF Hub | Pinned model_version string (`"ProsusAI/finbert@<commit-sha>"`) so cache keys are reproducible |

CPU inference is acceptable for this scale (~30k historical headlines × ~10 ms/headline ≈ 5 min one-time). GPU optional; supported via `device="auto"` fallback.

## 4. Data scope

- **Tickers**: same 12 as existing universe (`data.tickers.ALL_TICKERS`).
- **Time window**: catalog window (~2020-07 → today on free tier).
- **Source**: Alpaca free-tier `/v1beta1/news`. Free tier returns headlines + `created_at` + `symbols` tags historically (no body text); recent days include benzinga body content. We use headlines uniformly across the entire window — that's what FinBERT is trained on, and it keeps the historical and recent paths consistent.
- **Per-ticker volume (empirical, from probe)**: low — sector ETFs are tagged in ~1-5 headlines/day on average. SPY higher. Acceptable: we aggregate over 7- and 30-day trailing windows to densify the per-(ticker, date) signal.

## 5. Feature definitions

For each (ticker, rebalance_date), the three new features are computed from all articles in `sentiment_store` whose `symbols` list includes `ticker` and whose `created_at < rebalance_date`:

- `sentiment_7d_mean`: average net sentiment of qualifying articles in `[rebalance_date - 7d, rebalance_date)`. NaN if zero articles.
- `sentiment_7d_count`: count of articles in the same window (intensity proxy). NaN if zero articles.
- `sentiment_30d_mean`: average net sentiment over `[rebalance_date - 30d, rebalance_date)`. NaN if zero articles.

Same no-lookahead guarantee as the existing 11 features: `created_at < rebalance_date` is strict.

NaN handling matches `models/features.py` convention: any NaN in the feature row drops the entire row in `make_features`. This is honest — if a ticker has zero news in the last 30 days, the model just doesn't trade it that month.

## 6. Ablation methodology

The acceptance gate ("measurable incremental IC or a clean rejection") needs a defined test.

1. Train **two sets** of LightGBM combos via `train_meta_models.py`:
   - **price-only** (v1): existing 11 features, existing `models/cache/`. *(Already trained; reuse.)*
   - **price+sentiment** (v2): 11 + 3 = 14 features, new `models/cache_v2/`.
2. Run `run_harness_meta.py --features-version v1` and `--v2`. Produce two `summary.md`s with the same metrics: baseline Sharpe, meta winner Sharpe, DSR z + p, verdict.
3. Compute **per-window IC** = Spearman correlation between meta-model's `P(win)` and label, on each held-out test fold (window). Report:
   - mean IC per fold, both sides
   - ΔIC per fold = IC_v2 − IC_v1
   - median ΔIC across the 5 windows
4. Run a **permutation test**: shuffle the sentiment columns 1000 times, retrain (or re-predict using the already-trained v2 model on shuffled features), recompute median ΔIC. The empirical p-value of the observed ΔIC under this null distribution must be < 0.05 for a positive verdict. (Permutation is cheaper than retraining; use predict-with-shuffled-features as the lightweight null.)

### Verdict criteria

| Result | Interpretation |
|---|---|
| median ΔIC > 0 AND permutation p < 0.05 AND v2 baseline-vs-meta verdict = YES | **Sentiment adds real signal** — record as Stage 6 win |
| median ΔIC ≈ 0 (within permutation noise) OR p ≥ 0.05 | **Clean rejection** — record as honest negative, document why (likely: ETF news too sparse / sector-level signal too coarse) |
| median ΔIC < 0 with p < 0.05 | **Sentiment hurts** — possible if FinBERT mis-scores macro-tagged sentiment for sectors; document, retain v1 |

All three are valid Stage-6 outcomes.

## 7. Risks & open items

1. **Sparse news per ETF** (most likely outcome driver): sector ETFs aren't covered as heavily as individual stocks. The 7- and 30-day aggregation smooths some of this, but if median article-count is < 3/window for most (ticker, date) pairs, the sentiment signal will be noisy. Mitigation if discovered: aggregate constituent news per sector ETF (Stage-6.5, explicitly out of scope here).
2. **Headlines-only on free tier historically**: no body text pre-recent. FinBERT-on-headlines is the standard finance-NLP pattern (FinBERT was fine-tuned on headlines + short snippets); loss is acceptable but the model can't disambiguate complex sentiment that requires body context.
3. **FinBERT inference cost**: first pass over ~30k historical headlines is the slow step (CPU: ~5 min batched, GPU: seconds). Cached after first run.
4. **Multi-ticker articles**: many articles tag 5-10 tickers. Each tagged ticker gets the same article's score — appropriate for sector-rotation signal, may bias toward "macro mood" rather than ticker-specific.
5. **Permutation test cost**: 1000× re-prediction over the held-out folds is cheap (no retraining). If retraining is required for an unbiased null, that's 1000 × 60 cutoffs × 8 combos = expensive. The lightweight predict-with-shuffled-features null is the documented default; flag the limitation in the verdict.

## 8. Out of scope (explicitly)

- Per-constituent-stock news aggregation (Stage 6.5 if needed)
- Real-time news ingestion (live trading concern — Stage 8)
- Larger sentiment models (LLM-based, etc.) — FinBERT is the cited literature baseline
- Sentiment as a primary signal (this is a feature, not a strategy)
- Body-text NLP — headlines-only by design + free-tier constraint

## 9. Acceptance — Stage 6 is done when

- `data/ingest_alpaca_news.py` populates `data/news/` for the full universe + catalog window.
- `examples/score_news.py --smoke` runs FinBERT on ≥100 cached headlines and writes scores to `nlp/cache/`.
- `examples/score_news.py` (full mode) scores the entire ingested news corpus.
- `examples/train_meta_models.py --features-version v2` produces `models/cache_v2/` with 472 models (same count as v1).
- `examples/run_harness_meta.py --features-version v2` runs without errors, writes a v2 `summary.md`.
- An ablation script compares v1 vs v2 IC + DSR + verdict, writes `results/sentiment_ablation_<timestamp>/summary.md` with the verdict per §6.
- README Stage 6 entry updated with the actual verdict numbers.
