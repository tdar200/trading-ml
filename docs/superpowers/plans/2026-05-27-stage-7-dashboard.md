# Stage 7: Streamlit Dashboard Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans to implement task-by-task. Steps use checkbox (`- [ ]`).

**Goal:** A 5-tab Streamlit dashboard reading from existing `results/` / `models/cache*/` / `nlp/cache/` / `data/news/` artifacts; answers "is the strategy still working or just noisy?" off the screen.

**Architecture:** `dashboard/app.py` wires 5 view modules. All data flows through `dashboard/data_sources.py` (typed, `@st.cache_data(ttl=300)`, gracefully returns None on missing data). Views render via standard Streamlit widgets.

**Tech Stack:** `streamlit>=1.40` (NEW). Reuses existing `pandas`, `joblib`, `pyarrow`.

---

## File Structure

| Path | Responsibility |
|---|---|
| `pyproject.toml` | Add `streamlit>=1.40` |
| `.gitignore` | Add `.streamlit/` and `~/.streamlit/` (unlikely but defensive) |
| `dashboard/__init__.py` | Package marker |
| `dashboard/app.py` | Streamlit entry; wires 5 tabs |
| `dashboard/data_sources.py` | Typed loaders (cached) |
| `dashboard/views/__init__.py` | Package marker |
| `dashboard/views/equity.py` | Equity curves + rolling Sharpe + drawdown |
| `dashboard/views/harness_verdict.py` | Stage-3 + Stage-5 verdict cards |
| `dashboard/views/sentiment.py` | Stage-6 ablation + per-fold ΔIC + per-ticker timeseries |
| `dashboard/views/models.py` | Cache coverage maps + feature importances |
| `dashboard/views/journal.py` | Markdown render of specs/plans/summaries |
| `tests/test_dashboard_data_sources.py` | Tests for loaders (5 tests with `tmp_path` fixtures) |

---

## Task 1: Scaffolding (streamlit dep, dashboard package)

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Create: `dashboard/__init__.py`, `dashboard/views/__init__.py`

- [ ] **Step 1:** Add `"streamlit>=1.40"` to `pyproject.toml` dependencies.
- [ ] **Step 2:** `uv sync`. Verify import: `uv run python -c "import streamlit; print(streamlit.__version__)"`.
- [ ] **Step 3:** Add `.streamlit/` to `.gitignore`.
- [ ] **Step 4:** `mkdir -p dashboard/views && touch dashboard/__init__.py dashboard/views/__init__.py`.
- [ ] **Step 5:** Full pytest sanity: `uv run pytest -q` should still show 132 passed.
- [ ] **Step 6:** Commit: `chore(stage7): add streamlit dep + dashboard package scaffold`.

---

## Task 2: `dashboard/data_sources.py` + tests

**Files:**
- Create: `dashboard/data_sources.py`
- Create: `tests/test_dashboard_data_sources.py`

The module exposes pure loader functions. Each returns `None` / empty container when the artifact doesn't exist.

- [ ] **Step 1:** Write `tests/test_dashboard_data_sources.py` with 5 tests:
  - `test_load_latest_harness_returns_newest_summary` — write 3 fake harness dirs to `tmp_path`, assert newest is returned.
  - `test_load_latest_harness_returns_none_when_empty` — no dirs → None.
  - `test_load_latest_meta_harness_filters_by_version` — write `harness_meta_v1_X` and `harness_meta_v2_Y`; v1 query returns X, v2 returns Y.
  - `test_load_model_coverage_extracts_combo_and_cutoff_from_filenames` — write empty `.joblib` files matching `nXXX_dY_lrZZ_YYYY-MM-DD.joblib`; loader returns DataFrame with combo + cutoff columns.
  - `test_load_journal_entries_sorts_newest_first` — write 3 markdown files with controlled mtimes; result is reverse-chrono.

```python
"""Tests for dashboard.data_sources — typed loaders for the Streamlit dashboard."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import pytest

from dashboard.data_sources import (
    load_journal_entries,
    load_latest_harness,
    load_latest_meta_harness,
    load_model_coverage,
)


def _write_summary(dir_path: Path, sharpe: float = 1.0) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "summary.json").write_text(json.dumps({"sharpe": sharpe}))


def test_load_latest_harness_returns_newest_summary(tmp_path: Path):
    results = tmp_path / "results"
    for ts, sh in [("20260101T000000Z", 0.5), ("20260301T000000Z", 1.2), ("20260201T000000Z", 0.8)]:
        d = results / f"harness_{ts}"
        _write_summary(d, sharpe=sh)
        # Make mtime track the timestamp so ordering is deterministic.
        ts_unix = pd.Timestamp(ts.replace("T", " ").replace("Z", ""), tz="UTC").timestamp()
        import os
        os.utime(d / "summary.json", (ts_unix, ts_unix))
    out = load_latest_harness(results_root=results)
    assert out is not None
    assert out["sharpe"] == 1.2  # 2026-03 is newest


def test_load_latest_harness_returns_none_when_empty(tmp_path: Path):
    out = load_latest_harness(results_root=tmp_path / "no-such-dir")
    assert out is None


def test_load_latest_meta_harness_filters_by_version(tmp_path: Path):
    results = tmp_path / "results"
    _write_summary(results / "harness_meta_v1_20260101T000000Z", sharpe=0.7)
    _write_summary(results / "harness_meta_v2_20260101T000000Z", sharpe=0.9)
    v1 = load_latest_meta_harness("v1", results_root=results)
    v2 = load_latest_meta_harness("v2", results_root=results)
    assert v1 is not None and v1["sharpe"] == 0.7
    assert v2 is not None and v2["sharpe"] == 0.9


def test_load_model_coverage_extracts_combo_and_cutoff(tmp_path: Path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "n100_d3_lr05_2021-06-01.joblib").write_bytes(b"")
    (cache / "n100_d3_lr05_2021-07-01.joblib").write_bytes(b"")
    (cache / "n200_d5_lr10_2021-06-01.joblib").write_bytes(b"")
    df = load_model_coverage(cache_dir=cache)
    assert len(df) == 3
    assert set(df["combo"]) == {"n100_d3_lr05", "n200_d5_lr10"}
    assert pd.Timestamp("2021-06-01") in df["cutoff"].tolist()


def test_load_journal_entries_sorts_newest_first(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    paths = []
    for i, name in enumerate(["old.md", "newer.md", "newest.md"]):
        p = docs / name
        p.write_text(f"# {name}\n")
        # Increasing mtimes.
        import os
        os.utime(p, (1_700_000_000 + i * 1000, 1_700_000_000 + i * 1000))
        paths.append(p)
    entries = load_journal_entries(roots=[docs])
    assert [p.name for p in entries[:3]] == ["newest.md", "newer.md", "old.md"]
```

- [ ] **Step 2:** Run — confirms collection error.
- [ ] **Step 3:** Implement `dashboard/data_sources.py`:

```python
"""
Typed loaders for the Stage-7 Streamlit dashboard.

Every function returns `None` (or an empty container) when the artifact
doesn't exist — views render an empty-state message in that case.

Streamlit cache: each loader is wrapped with `@st.cache_data(ttl=300)`
in `app.py` rather than here so this module remains importable + testable
without Streamlit context.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


_PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_RESULTS_ROOT = _PROJECT_ROOT / "results"
DEFAULT_CACHE_V1 = _PROJECT_ROOT / "models" / "cache"
DEFAULT_CACHE_V2 = _PROJECT_ROOT / "models" / "cache_v2"

_FILENAME_RE = re.compile(r"^(?P<combo>n\d+_d\d+_lr\d+)_(?P<cutoff>\d{4}-\d{2}-\d{2})\.joblib$")


def _newest_dir(roots: Path, prefix: str) -> Path | None:
    if not roots.exists():
        return None
    candidates = [p for p in roots.glob(f"{prefix}*") if p.is_dir() and (p / "summary.json").exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p / "summary.json").stat().st_mtime)


def _load_summary_json(d: Path | None) -> dict | None:
    if d is None:
        return None
    sj = d / "summary.json"
    if not sj.exists():
        return None
    try:
        return json.loads(sj.read_text())
    except Exception:
        return None


def load_latest_harness(*, results_root: Path = DEFAULT_RESULTS_ROOT) -> dict | None:
    """Latest Stage-3 harness summary (excludes harness_meta)."""
    if not results_root.exists():
        return None
    candidates = [
        p for p in results_root.glob("harness_*")
        if p.is_dir()
        and not p.name.startswith("harness_meta")
        and (p / "summary.json").exists()
    ]
    if not candidates:
        return None
    newest = max(candidates, key=lambda p: (p / "summary.json").stat().st_mtime)
    return _load_summary_json(newest)


def load_latest_meta_harness(version: str, *, results_root: Path = DEFAULT_RESULTS_ROOT) -> dict | None:
    """Latest Stage-5/6 meta harness summary for the given version (v1 / v2).

    Also accepts the legacy un-versioned naming `harness_meta_<timestamp>` as v1.
    """
    if not results_root.exists():
        return None
    versioned = list(results_root.glob(f"harness_meta_{version}_*"))
    if version == "v1":
        # Include legacy un-versioned dirs as v1 fallback.
        for p in results_root.glob("harness_meta_*"):
            if not p.name.startswith(f"harness_meta_v"):
                versioned.append(p)
    versioned = [p for p in versioned if p.is_dir() and (p / "summary.json").exists()]
    if not versioned:
        return None
    newest = max(versioned, key=lambda p: (p / "summary.json").stat().st_mtime)
    return _load_summary_json(newest)


def load_latest_ablation(*, results_root: Path = DEFAULT_RESULTS_ROOT) -> dict | None:
    """Latest Stage-6 sentiment ablation summary + per_fold (as 'per_fold')."""
    if not results_root.exists():
        return None
    dirs = [p for p in results_root.glob("sentiment_ablation_*") if p.is_dir() and (p / "summary.json").exists()]
    if not dirs:
        return None
    newest = max(dirs, key=lambda p: (p / "summary.json").stat().st_mtime)
    summary = _load_summary_json(newest)
    if summary is None:
        return None
    pf = newest / "per_fold.csv"
    if pf.exists():
        summary["per_fold"] = pd.read_csv(pf)
    return summary


def load_equity_curves(*, results_root: Path = DEFAULT_RESULTS_ROOT) -> dict[str, pd.Series]:
    """Equity series by label. Sources:
       - results/momentum_topn_*/equity.csv (label='Stage-2 momentum')
       - results/momentum_topn_*/spy_close.csv → normalized to equity (label='SPY')
    """
    out: dict[str, pd.Series] = {}
    if not results_root.exists():
        return out
    momenta = sorted(results_root.glob("momentum_topn_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if momenta:
        latest = momenta[0]
        eq = latest / "equity.csv"
        if eq.exists():
            df = pd.read_csv(eq)
            ts_col = next((c for c in df.columns if c.lower() in {"date", "ts", "timestamp"}), df.columns[0])
            val_col = next((c for c in df.columns if "equity" in c.lower()), df.columns[-1])
            out["Stage-2 momentum"] = pd.Series(
                df[val_col].values,
                index=pd.to_datetime(df[ts_col]),
                name="equity",
            )
        spy = latest / "spy_close.csv"
        if spy.exists():
            sdf = pd.read_csv(spy)
            ts_col = next((c for c in sdf.columns if c.lower() in {"date", "ts", "timestamp"}), sdf.columns[0])
            val_col = next((c for c in sdf.columns if "close" in c.lower() or "spy" in c.lower()), sdf.columns[-1])
            base = float(sdf[val_col].iloc[0])
            series = sdf[val_col].astype(float) / base * 100_000.0
            out["SPY (normalized)"] = pd.Series(
                series.values,
                index=pd.to_datetime(sdf[ts_col]),
                name="equity",
            )
    return out


def load_news_for_ticker(
    ticker: str,
    *,
    window_days: int = 180,
    news_root: Path = _PROJECT_ROOT / "data" / "news",
    scores_root: Path = _PROJECT_ROOT / "nlp" / "cache" / "finbert-v1",
) -> pd.DataFrame:
    """Headlines + sentiment scores joined for one ticker, last `window_days`."""
    ticker_dir = news_root / ticker
    if not ticker_dir.exists():
        return pd.DataFrame(columns=["created_at", "headline", "score"])
    rows = [pd.read_parquet(s) for s in sorted(ticker_dir.glob("*.parquet"))]
    if not rows:
        return pd.DataFrame(columns=["created_at", "headline", "score"])
    df = pd.concat(rows, ignore_index=True)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=window_days)
    df = df[df["created_at"] >= cutoff].copy()
    scores_path = scores_root / f"{ticker}.parquet"
    if scores_path.exists():
        sc = pd.read_parquet(scores_path)
        df = df.merge(sc[["id", "score"]], on="id", how="left")
    else:
        df["score"] = float("nan")
    return df.sort_values("created_at", ascending=False).reset_index(drop=True)


def load_model_coverage(*, cache_dir: Path) -> pd.DataFrame:
    """Map model artifacts in `cache_dir` to (combo, cutoff) rows."""
    if not cache_dir.exists():
        return pd.DataFrame(columns=["combo", "cutoff", "path"])
    rows = []
    for p in sorted(cache_dir.glob("*.joblib")):
        m = _FILENAME_RE.match(p.name)
        if not m:
            continue
        rows.append({
            "combo": m.group("combo"),
            "cutoff": pd.Timestamp(m.group("cutoff")),
            "path": str(p),
        })
    return pd.DataFrame(rows, columns=["combo", "cutoff", "path"])


def load_feature_importances(
    *, cache_dir: Path, combo: str, cutoff: pd.Timestamp,
) -> pd.Series | None:
    """Load a single joblib and return importances as a Series keyed by feature_name."""
    import joblib
    path = cache_dir / f"{combo}_{cutoff.strftime('%Y-%m-%d')}.joblib"
    if not path.exists():
        return None
    payload = joblib.load(path)
    model = payload.get("model")
    names = payload.get("feature_names") or []
    if model is None or not hasattr(model, "feature_importances_"):
        return None
    imp = model.feature_importances_
    if len(imp) != len(names):
        return None
    return pd.Series(imp, index=names, name="importance").sort_values(ascending=False)


def load_journal_entries(
    *, roots: Iterable[Path] = (
        _PROJECT_ROOT / "docs" / "superpowers" / "specs",
        _PROJECT_ROOT / "docs" / "superpowers" / "plans",
    ),
) -> list[Path]:
    """All journal markdown files, newest first by mtime."""
    out: list[Path] = []
    for root in roots:
        if not Path(root).exists():
            continue
        for p in Path(root).glob("*.md"):
            out.append(p)
    return sorted(out, key=lambda p: p.stat().st_mtime, reverse=True)
```

- [ ] **Step 4:** Run tests — 5 passed.
- [ ] **Step 5:** Full pytest: 137 passed.
- [ ] **Step 6:** Commit: `feat(stage7): typed data loaders for the dashboard (T2)`.

---

## Task 3: `dashboard/views/equity.py`

- [ ] Implement and commit. Code:

```python
"""Equity tab: equity curves + rolling Sharpe + drawdown."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


def _rolling_sharpe(returns: pd.Series, window: int = 60) -> pd.Series:
    return (returns.rolling(window).mean() / returns.rolling(window).std()) * np.sqrt(252)


def _drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def render(data_sources) -> None:
    st.header("Equity")
    curves = data_sources.load_equity_curves()
    if not curves:
        st.info("No equity curves yet. Run `uv run python examples/backtest_momentum.py`.")
        return
    df = pd.DataFrame(curves).sort_index()
    st.subheader("Equity curves")
    st.line_chart(df)
    rets = df.pct_change().dropna()
    if not rets.empty:
        st.subheader("Rolling 60-day Sharpe")
        st.line_chart(rets.apply(_rolling_sharpe))
        st.subheader("Drawdown")
        st.line_chart(df.apply(_drawdown))
```

Commit: `feat(stage7): equity view (T3)`.

---

## Task 4: `dashboard/views/harness_verdict.py`

```python
"""Harness Verdict tab: Stage-3 winner + Stage-5 baseline-vs-meta cards."""
from __future__ import annotations

import streamlit as st


def render(data_sources) -> None:
    st.header("Harness Verdicts")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Stage 3 — Walk-forward winner")
        s3 = data_sources.load_latest_harness()
        if s3 is None:
            st.info("No Stage-3 harness yet. Run `uv run python examples/run_harness.py`.")
        else:
            st.json(s3)
    with col2:
        st.subheader("Stage 5 — Baseline vs Meta")
        v1 = data_sources.load_latest_meta_harness("v1")
        v2 = data_sources.load_latest_meta_harness("v2")
        if v1 is None and v2 is None:
            st.info("No Stage-5/6 harness yet. Run `examples/run_harness_meta.py`.")
        else:
            if v1 is not None:
                st.markdown("**v1 (price-only)**")
                bm = v1.get("baseline_median_sharpe")
                mw = v1.get("meta_winner", {}).get("median_window_sharpe")
                if bm is not None and mw is not None:
                    delta = ((mw - bm) / bm) * 100 if bm != 0 else float("nan")
                    st.metric("Baseline Sharpe", f"{bm:.4f}")
                    st.metric("Meta winner Sharpe", f"{mw:.4f}", delta=f"{delta:+.1f}%")
                else:
                    st.json(v1)
            if v2 is not None:
                st.markdown("**v2 (price + sentiment)**")
                bm = v2.get("baseline_median_sharpe")
                mw = v2.get("meta_winner", {}).get("median_window_sharpe")
                if bm is not None and mw is not None:
                    delta = ((mw - bm) / bm) * 100 if bm != 0 else float("nan")
                    st.metric("Baseline Sharpe", f"{bm:.4f}")
                    st.metric("Meta winner Sharpe", f"{mw:.4f}", delta=f"{delta:+.1f}%")
                else:
                    st.json(v2)
```

Commit: `feat(stage7): harness verdict view (T4)`.

---

## Task 5: `dashboard/views/sentiment.py`

```python
"""Sentiment tab: Stage-6 ablation verdict + per-fold ΔIC + per-ticker timeseries."""
from __future__ import annotations

import streamlit as st


def render(data_sources) -> None:
    st.header("Sentiment Ablation (Stage 6)")
    a = data_sources.load_latest_ablation()
    if a is None:
        st.info("No ablation run yet. Run `uv run python examples/ablate_sentiment.py`.")
        return
    cols = st.columns(3)
    cols[0].metric("Median ΔIC", f"{a.get('median_delta_ic', float('nan')):+.4f}")
    cols[1].metric("Permutation p", f"{a.get('p_value', float('nan')):.4f}")
    cols[2].metric("Verdict", str(a.get("verdict", "")).split(" — ")[0])
    st.caption(str(a.get("verdict", "")))
    per_fold = a.get("per_fold")
    if per_fold is not None and len(per_fold) > 0:
        st.subheader("Per-fold ΔIC")
        st.bar_chart(per_fold.set_index("cutoff")["delta_ic"])
        st.subheader("Per-fold IC table")
        st.dataframe(per_fold)
    st.subheader("Per-ticker sentiment (last 180 days)")
    ticker = st.selectbox("Ticker", ["SPY", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
                                     "XLP", "XLU", "XLB", "XLRE", "XLC"])
    news = data_sources.load_news_for_ticker(ticker, window_days=180)
    if news.empty:
        st.info(f"No news for {ticker} in last 180 days.")
    else:
        st.write(f"{len(news)} articles")
        st.dataframe(news[["created_at", "headline", "score"]].head(50))
```

Commit: `feat(stage7): sentiment ablation view (T5)`.

---

## Task 6: `dashboard/views/models.py`

```python
"""Models tab: cache coverage maps + per-combo feature importances."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.data_sources import DEFAULT_CACHE_V1, DEFAULT_CACHE_V2


def render(data_sources) -> None:
    st.header("Model Cache Coverage")
    for label, cache in (("v1 (price-only)", DEFAULT_CACHE_V1), ("v2 (price+sentiment)", DEFAULT_CACHE_V2)):
        st.subheader(label)
        df = data_sources.load_model_coverage(cache_dir=cache)
        if df.empty:
            st.info(f"No models in {cache}.")
            continue
        st.write(f"{len(df)} models across {df['combo'].nunique()} combos × {df['cutoff'].nunique()} cutoffs")
        pivot = df.assign(present=1).pivot_table(
            index="combo", columns="cutoff", values="present", fill_value=0
        )
        st.dataframe(pivot)
        # Feature importances for the latest cutoff of the first combo.
        latest_cutoff = df["cutoff"].max()
        first_combo = sorted(df["combo"].unique())[0]
        imp = data_sources.load_feature_importances(
            cache_dir=cache, combo=first_combo, cutoff=latest_cutoff,
        )
        if imp is not None:
            st.markdown(f"**Feature importances for {first_combo} @ {latest_cutoff.date()}**")
            st.bar_chart(imp)
```

Commit: `feat(stage7): models view — coverage + feature importances (T6)`.

---

## Task 7: `dashboard/views/journal.py`

```python
"""Journal tab: render all spec/plan/summary markdown, newest first."""
from __future__ import annotations

import streamlit as st


def render(data_sources) -> None:
    st.header("Research Journal")
    entries = data_sources.load_journal_entries()
    if not entries:
        st.info("No journal entries found.")
        return
    st.write(f"{len(entries)} entries, newest first")
    options = [p.name for p in entries]
    selected = st.selectbox("Document", options)
    chosen = next(p for p in entries if p.name == selected)
    st.markdown(chosen.read_text())
```

Commit: `feat(stage7): journal view (T7)`.

---

## Task 8: `dashboard/app.py` — wire all tabs

```python
"""Stage-7 Streamlit dashboard entry.

Usage:
    uv run streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from dashboard import data_sources as ds
from dashboard.views import equity, harness_verdict, sentiment, models, journal


# Wrap every loader with Streamlit's cache so the dashboard isn't
# re-reading parquet/json on every interaction. TTL 5 min — refresh by
# hitting "Rerun" in the Streamlit UI when a fresh acceptance run completes.
class _CachedDataSources:
    load_latest_harness = staticmethod(st.cache_data(ttl=300)(ds.load_latest_harness))
    load_latest_meta_harness = staticmethod(st.cache_data(ttl=300)(ds.load_latest_meta_harness))
    load_latest_ablation = staticmethod(st.cache_data(ttl=300)(ds.load_latest_ablation))
    load_equity_curves = staticmethod(st.cache_data(ttl=300)(ds.load_equity_curves))
    load_news_for_ticker = staticmethod(st.cache_data(ttl=300)(ds.load_news_for_ticker))
    load_model_coverage = staticmethod(st.cache_data(ttl=300)(ds.load_model_coverage))
    load_feature_importances = staticmethod(st.cache_data(ttl=300)(ds.load_feature_importances))
    load_journal_entries = staticmethod(st.cache_data(ttl=300)(ds.load_journal_entries))


def main() -> None:
    st.set_page_config(page_title="trading-ml", layout="wide")
    st.title("trading-ml — Research Dashboard")
    sources = _CachedDataSources

    tabs = st.tabs(["Equity", "Harness Verdicts", "Sentiment", "Models", "Journal"])
    with tabs[0]:
        equity.render(sources)
    with tabs[1]:
        harness_verdict.render(sources)
    with tabs[2]:
        sentiment.render(sources)
    with tabs[3]:
        models.render(sources)
    with tabs[4]:
        journal.render(sources)


if __name__ == "__main__":
    main()
```

Verify: `uv run streamlit run dashboard/app.py --server.headless true &` then `curl -sf http://localhost:8501 > /dev/null && echo OK`. Kill the server after.

Commit: `feat(stage7): wire the 5 dashboard tabs (T8)`.

---

## Task 9: README + Stage 7 wrap-up

Update `README.md`:
- Stage 7 entry: `[x] Stage 7 — Streamlit dashboard` with the 5 tabs listed and the run command.
- Add a `### Dashboard (Stage 7)` section in Backtest with `uv run streamlit run dashboard/app.py`.
- Update Layout to include `dashboard/`.

Commit: `docs: README — Stage 7 complete`.

---

## Self-review

- All 4 spec acceptance items mapped to tasks: data_sources (T2), each tab (T3-T7), wiring (T8), README (T9).
- No placeholders; every step has concrete code or commands.
- Type/name consistency: `data_sources` module attribute names match across all view files.
- Test count progression: 132 → 137 (just T2 adds tests; views are visual).
