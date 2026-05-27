# Stage 7 — Streamlit Dashboard Design

## 1. Context & goal

Stages 0–6 produce verdicts as Markdown / CSV / joblib under `results/`, `models/cache*/`, `nlp/cache/`, `data/news/`, `data/catalog/`. Reading these by hand is fine for a one-time check but doesn't answer the recurring research-loop question quickly. Stage 7's goal is the screen the parent design spec §5 specifies: **"Can read 'is the strategy still working or just noisy?' off the screen"** — a Streamlit dashboard that pulls every existing artifact into one place with the equity curves, harness verdicts, sentiment ablation, model coverage, and research journal in five tabs.

The dashboard is read-only — it never writes to `models/` or `results/`. It runs locally (`uv run streamlit run dashboard/app.py`); there's no deployment concern.

## 2. Architecture

```
dashboard/
├── __init__.py
├── app.py               # Streamlit entry; wires the 5 tabs
├── data_sources.py      # Typed loaders for each artifact type
└── views/
    ├── __init__.py
    ├── equity.py        # 3 overlaid equity curves vs SPY + rolling Sharpe + drawdown
    ├── harness_verdict.py  # Stage-3 + Stage-5 verdict cards
    ├── sentiment.py     # Stage-6 REJECT verdict + ΔIC chart + per-ticker timeseries
    ├── models.py        # cache coverage map + per-combo feature importances
    └── journal.py       # markdown render of specs + summaries, newest-first
```

### Data sources (`dashboard/data_sources.py`)

Pure typed loaders, each cached via `@st.cache_data(ttl=300)`:

| Function | Returns | Source |
|---|---|---|
| `load_latest_tearsheet()` | `Path \| None` | `results/momentum_topn_*/tearsheet.html` (newest) |
| `load_latest_harness()` | `dict \| None` | Latest `results/harness_*/summary.json` (excludes harness_meta) |
| `load_latest_meta_harness(version: str)` | `dict \| None` | Latest `results/harness_meta_<v>_*/summary.json` or `results/harness_meta_*/summary.json` if unversioned |
| `load_latest_ablation()` | `dict \| None` | Latest `results/sentiment_ablation_*/summary.json` + `per_fold.csv` |
| `load_equity_curves()` | `dict[label, pd.Series]` | From the latest `equity.csv` per `results/momentum_topn_*/` and `results/harness_*/walk_forward.csv` |
| `load_news_for_ticker(ticker, window_days=180)` | `pd.DataFrame` | Latest N days from `data/news/<ticker>/*.parquet` + scores from `nlp/cache/finbert-v1/<ticker>.parquet` (left-joined) |
| `load_model_coverage(version: str)` | `pd.DataFrame[combo, cutoff_date]` | `models/cache/` or `models/cache_v2/` joblib filenames |
| `load_feature_importances(version, combo, cutoff)` | `pd.Series \| None` | Loads a single joblib, extracts `LGBMClassifier.feature_importances_` keyed by `feature_names` saved at training time |
| `load_journal_entries()` | `list[Path]` | All `docs/superpowers/specs/*.md` + `docs/superpowers/plans/*.md` + `results/**/summary.md`, newest first by mtime |

Every loader returns `None` (or empty container) when the artifact doesn't exist. Views render an empty-state message + the command to produce the data.

### Views

Each view is a function `def render(data_sources)` invoked by `app.py` under its tab. Views may use `st.metric`, `st.line_chart`, `st.dataframe`, `st.markdown`. Standard Streamlit, no custom components.

### Acceptance — Stage 7 is done when

- `uv run streamlit run dashboard/app.py` launches on `http://localhost:8501` without errors.
- Equity tab shows ≥1 equity series (Stage-2 raw) overlaid with SPY; ≥2 series with Stage-3 + Stage-5 results present.
- Harness Verdict tab shows the Stage-3 winner + Stage-5 baseline-vs-meta with current numbers (Sharpe 0.9541 vs 1.2319, +29% per current `main`).
- Sentiment tab shows the Stage-6 REJECT verdict + per-fold ΔIC bar chart.
- Models tab shows cache coverage maps for v1 + v2 + a feature importance plot.
- Journal tab renders specs + plans + summaries inline.
- Each view degrades gracefully when its data is missing (shows the runnable command).
- README + Stage 7 entry updated.

## 3. Risks & open items

1. **Streamlit cache invalidation**: 5-min TTL is generous for research. If the user wants to see new results from a just-finished run, they hit "Rerun" in the Streamlit UI.
2. **Empty-state UX**: on a fresh clone with no `results/`, every tab needs the empty-state message.
3. **Tests**: only `data_sources.py` is unit-tested (with `tmp_path` fixtures). The Streamlit UI itself isn't tested — `st.testing.v1.AppTest` exists but is brittle for our scope.
4. **No live refresh**: a future Stage-8 LiveNode would publish status snapshots; the data_sources layer is designed so a `load_live_status()` loader can be added without touching views. Out of scope for Stage 7.

## 4. Out of scope

- Live position updates from a running TradingNode (Stage 8).
- Deploying the dashboard publicly.
- Auth / multi-user.
- Editing strategy parameters from the UI (read-only dashboard by design).
- Custom Streamlit components / heavy interactivity beyond standard widgets.
