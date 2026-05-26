"""Harness report aggregation + rendering tests."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis.harness_report import (
    render_markdown,
    summarize_walk_forward,
    write_report,
)


def _sample_wf_df() -> pd.DataFrame:
    """4 rows: 2 combos × 2 windows, with combo (3, 126, 21) clearly winning."""
    return pd.DataFrame([
        {"window_idx": 0, "start": pd.Timestamp("2021-01-01"), "end": pd.Timestamp("2022-01-01"),
         "top_n": 3, "lookback_days": 126, "skip_days": 21,
         "cagr": 0.10, "sharpe": 0.9, "max_drawdown": -0.15,
         "total_return": 0.10, "final_equity": 110_000.0, "n_days": 252, "n_trades": 30},
        {"window_idx": 1, "start": pd.Timestamp("2022-01-01"), "end": pd.Timestamp("2023-01-01"),
         "top_n": 3, "lookback_days": 126, "skip_days": 21,
         "cagr": 0.08, "sharpe": 0.8, "max_drawdown": -0.20,
         "total_return": 0.08, "final_equity": 108_000.0, "n_days": 252, "n_trades": 30},
        {"window_idx": 0, "start": pd.Timestamp("2021-01-01"), "end": pd.Timestamp("2022-01-01"),
         "top_n": 5, "lookback_days": 126, "skip_days": 21,
         "cagr": 0.02, "sharpe": 0.1, "max_drawdown": -0.30,
         "total_return": 0.02, "final_equity": 102_000.0, "n_days": 252, "n_trades": 30},
        {"window_idx": 1, "start": pd.Timestamp("2022-01-01"), "end": pd.Timestamp("2023-01-01"),
         "top_n": 5, "lookback_days": 126, "skip_days": 21,
         "cagr": -0.05, "sharpe": -0.4, "max_drawdown": -0.40,
         "total_return": -0.05, "final_equity": 95_000.0, "n_days": 252, "n_trades": 30},
    ])


def test_summarize_walk_forward_picks_winner_by_median_window_sharpe():
    wf_df = _sample_wf_df()
    summary = summarize_walk_forward(wf_df, param_cols=["top_n", "lookback_days", "skip_days"])
    # Winner is (3, 126, 21) — median Sharpe 0.85 vs (5, 126, 21)'s -0.15.
    assert summary["winner"] == {"top_n": 3, "lookback_days": 126, "skip_days": 21}
    assert summary["winner_median_window_sharpe"] == 0.85
    assert summary["winner_window_sharpe_range"] == (0.8, 0.9)
    assert "all_combos_sharpe_p05" in summary
    assert "all_combos_sharpe_p95" in summary
    assert summary["winner_rank"] == (1, 2)


def test_render_markdown_includes_all_five_sections():
    wf_df = _sample_wf_df()
    summary = summarize_walk_forward(wf_df, param_cols=["top_n", "lookback_days", "skip_days"])
    dsr = {"observed_sharpe": 0.85, "n_trials": 2,
           "expected_max_sharpe": 0.5, "deflated_sharpe": 1.2, "p_value": 0.88}
    cost_df = pd.DataFrame([
        {"slippage_prob": 0.0,  "cagr": 0.11, "sharpe": 1.0, "max_drawdown": -0.14, "final_equity": 111_000.0, "n_trades": 30, "n_days": 252, "total_return": 0.11},
        {"slippage_prob": 1.0,  "cagr": 0.05, "sharpe": 0.4, "max_drawdown": -0.18, "final_equity": 105_000.0, "n_trades": 30, "n_days": 252, "total_return": 0.05},
    ])
    md = render_markdown(strategy_name="MomentumTopN", timestamp="2026-05-26T00:00:00Z",
                         wf_summary=summary, dsr=dsr, cost_df=cost_df)
    for header in ("# Honesty Harness", "## 1. Time stability",
                   "## 2. Parameter robustness", "## 3. Cost survival",
                   "## 4. Deflated Sharpe", "## 5. Bottom line"):
        assert header in md, f"missing section: {header}"


def test_write_report_creates_all_artifacts(tmp_path: Path):
    wf_df = _sample_wf_df()
    summary = summarize_walk_forward(wf_df, param_cols=["top_n", "lookback_days", "skip_days"])
    dsr = {"observed_sharpe": 0.85, "n_trials": 2,
           "expected_max_sharpe": 0.5, "deflated_sharpe": 1.2, "p_value": 0.88}
    cost_df = pd.DataFrame([
        {"slippage_prob": 0.0, "cagr": 0.11, "sharpe": 1.0, "max_drawdown": -0.14, "final_equity": 111_000.0, "n_trades": 30, "n_days": 252, "total_return": 0.11},
    ])
    out_dir = tmp_path / "harness_2026"
    write_report(out_dir=out_dir, wf_df=wf_df, cost_df=cost_df, dsr=dsr,
                 strategy_name="MomentumTopN")
    assert (out_dir / "walk_forward.csv").exists()
    assert (out_dir / "cost_stress.csv").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "summary.md").exists()
    summary_json = json.loads((out_dir / "summary.json").read_text())
    assert "winner" in summary_json["walk_forward_summary"]
    assert "deflated_sharpe" in summary_json
