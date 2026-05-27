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

    Accepts the legacy un-versioned naming `harness_meta_<timestamp>` as v1.
    """
    if not results_root.exists():
        return None
    versioned = list(results_root.glob(f"harness_meta_{version}_*"))
    if version == "v1":
        # Include legacy un-versioned dirs as v1 fallback.
        for p in results_root.glob("harness_meta_*"):
            if not p.name.startswith("harness_meta_v"):
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
