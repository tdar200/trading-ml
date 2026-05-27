"""Tests for dashboard.data_sources — typed loaders for the Streamlit dashboard."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

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
    base_ts = 1_700_000_000
    for offset, sh in [(0, 0.5), (200, 1.2), (100, 0.8)]:
        d = results / f"harness_t{offset}"
        _write_summary(d, sharpe=sh)
        os.utime(d / "summary.json", (base_ts + offset, base_ts + offset))
    out = load_latest_harness(results_root=results)
    assert out is not None
    assert out["sharpe"] == 1.2  # offset=200 is newest


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
    for i, name in enumerate(["old.md", "newer.md", "newest.md"]):
        p = docs / name
        p.write_text(f"# {name}\n")
        # Increasing mtimes => newest.md (i=2) is newest.
        os.utime(p, (1_700_000_000 + i * 1000, 1_700_000_000 + i * 1000))
    entries = load_journal_entries(roots=[docs])
    assert [p.name for p in entries[:3]] == ["newest.md", "newer.md", "old.md"]
