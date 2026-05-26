"""
Aggregate walk-forward + DSR + cost stress into a written verdict.

`summarize_walk_forward` picks a winner (median window Sharpe) and computes
the time-stability + parameter-robustness summary stats.
`render_markdown` formats the five-section Markdown report.
`write_report` persists CSVs + JSON + Markdown to a results directory.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def summarize_walk_forward(wf_df: pd.DataFrame, param_cols: list[str]) -> dict[str, Any]:
    """Pick the winner combo (highest median window Sharpe) and compute summary stats."""
    if wf_df.empty:
        raise ValueError("wf_df must be non-empty")

    grouped = wf_df.groupby(param_cols)
    medians = grouped["sharpe"].median().sort_values(ascending=False)
    winner_tuple = medians.index[0]
    if not isinstance(winner_tuple, tuple):
        winner_tuple = (winner_tuple,)
    winner = dict(zip(param_cols, winner_tuple, strict=True))

    winner_rows = wf_df
    for col, val in winner.items():
        winner_rows = winner_rows[winner_rows[col] == val]
    winner_sharpes = winner_rows["sharpe"]

    all_sharpes = wf_df["sharpe"]
    n_combos = grouped.ngroups

    return {
        "winner": winner,
        "winner_median_window_sharpe": round(float(medians.iloc[0]), 10),
        "winner_window_sharpe_range": (round(float(winner_sharpes.min()), 10), round(float(winner_sharpes.max()), 10)),
        "all_combos_sharpe_p05": float(all_sharpes.quantile(0.05)),
        "all_combos_sharpe_p95": float(all_sharpes.quantile(0.95)),
        "winner_rank": (1, int(n_combos)),
        "n_windows": int(wf_df["window_idx"].nunique()),
    }


def _stable_verdict(low: float, high: float) -> str:
    spread = high - low
    if spread < 0.4:
        return "stable across windows"
    if spread < 1.0:
        return "varies moderately across windows"
    return "wildly unstable across windows"


def _robust_verdict(p05: float, p95: float, winner_sr: float) -> str:
    if winner_sr - p95 > 0.3:
        return "lone island (winner well above the 95th percentile of combos)"
    if winner_sr - p05 < 0.2:
        return "noise (winner barely above the 5th percentile of combos)"
    return "cluster of similarly-good configs"


def _cost_verdict(cost_df: pd.DataFrame) -> str:
    cagr0 = cost_df.iloc[0]["cagr"]
    cagr_max = cost_df.iloc[-1]["cagr"]
    if cagr0 > 0 and cagr_max > cagr0 / 2:
        return "robust to cost increases"
    if cagr0 > 0 and cagr_max > 0:
        return "degrades but remains positive"
    return "negative at realistic costs"


def render_markdown(
    strategy_name: str,
    timestamp: str,
    wf_summary: dict[str, Any],
    dsr: dict[str, Any],
    cost_df: pd.DataFrame,
) -> str:
    """Produce the 5-section Markdown report."""
    w = wf_summary["winner"]
    win_med = wf_summary["winner_median_window_sharpe"]
    win_lo, win_hi = wf_summary["winner_window_sharpe_range"]
    p05 = wf_summary["all_combos_sharpe_p05"]
    p95 = wf_summary["all_combos_sharpe_p95"]
    rank_pos, rank_total = wf_summary["winner_rank"]
    n_windows = wf_summary["n_windows"]

    cost_rows_md = "\n".join(
        f"| {row['slippage_prob']:.2f} | {row['cagr']:.4f} | {row['sharpe']:.4f} | {row['max_drawdown']:.4f} |"
        for _, row in cost_df.iterrows()
    )

    return f"""# Honesty Harness — {strategy_name} — {timestamp}

## 1. Time stability
- Winner combo: `{w}`
- Median window Sharpe (winner): **{win_med:.4f}**
- Range across {n_windows} windows: [{win_lo:.4f}, {win_hi:.4f}]
- Verdict: {_stable_verdict(win_lo, win_hi)}

## 2. Parameter robustness
- Best combo: `{w}`
- Winner rank by median window Sharpe: {rank_pos}/{rank_total}
- 5th–95th percentile of per-(combo,window) Sharpes: [{p05:.4f}, {p95:.4f}]
- Verdict: {_robust_verdict(p05, p95, win_med)}

## 3. Cost survival

| slippage_prob | CAGR | Sharpe | max-DD |
|---|---|---|---|
{cost_rows_md}

- Verdict: {_cost_verdict(cost_df)}

## 4. Deflated Sharpe (multiple-testing penalty)
- Observed (winner) Sharpe: **{dsr['observed_sharpe']:.4f}**
- Expected max-Sharpe under H0 (n_trials={dsr['n_trials']}): {dsr['expected_max_sharpe']:.4f}
- Deflated Sharpe (z-score): **{dsr['deflated_sharpe']:.4f}**
- p-value (P[winner SR > 0 | H0]): **{dsr['p_value']:.4f}**
- Verdict: {"cannot reject 'true Sharpe is zero' at α=0.05" if dsr['p_value'] < 0.95 else "winner's Sharpe is unlikely to be pure selection bias"}

## 5. Bottom line
{strategy_name} ran a {dsr['n_trials']}-combo × {n_windows}-window walk-forward.
The winner (`{w}`) shows a median Sharpe of {win_med:.2f} across windows ({_stable_verdict(win_lo, win_hi)});
parameter-wise it's a {_robust_verdict(p05, p95, win_med)};
costs make it {_cost_verdict(cost_df)};
and the Deflated Sharpe gives p={dsr['p_value']:.2f} for the null hypothesis that the
edge is selection bias. Treat the harness output as a *characterization* of the
strategy, not a verdict on whether to deploy it.
"""


def write_report(
    out_dir: Path,
    wf_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    dsr: dict[str, Any],
    strategy_name: str,
) -> dict[str, Any]:
    """Write walk_forward.csv, cost_stress.csv, summary.json, summary.md."""
    out_dir.mkdir(parents=True, exist_ok=True)
    param_cols = [c for c in wf_df.columns
                  if c not in {"window_idx", "start", "end", "cagr", "sharpe",
                               "max_drawdown", "total_return", "final_equity",
                               "n_days", "n_trades"}]
    wf_summary = summarize_walk_forward(wf_df, param_cols=param_cols)

    from datetime import datetime, timezone
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    md = render_markdown(strategy_name=strategy_name, timestamp=timestamp,
                         wf_summary=wf_summary, dsr=dsr, cost_df=cost_df)

    wf_df.to_csv(out_dir / "walk_forward.csv", index=False)
    cost_df.to_csv(out_dir / "cost_stress.csv", index=False)
    (out_dir / "summary.md").write_text(md)

    summary = {
        "strategy_name": strategy_name,
        "timestamp": timestamp,
        "walk_forward_summary": {
            **wf_summary,
            "winner_rank": list(wf_summary["winner_rank"]),
            "winner_window_sharpe_range": list(wf_summary["winner_window_sharpe_range"]),
        },
        "deflated_sharpe": dsr,
        "cost_stress": cost_df.to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
