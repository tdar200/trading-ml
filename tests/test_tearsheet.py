"""Tearsheet smoke test — quantstats actually produces an HTML report."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.tearsheet import report


def _synthetic_equity_curve(start: float, daily_return: float, n: int, start_date: str) -> pd.Series:
    idx = pd.date_range(start=start_date, periods=n, freq="B")
    vals = [start]
    for _ in range(n - 1):
        vals.append(vals[-1] * (1.0 + daily_return))
    return pd.Series(vals, index=idx, name="equity")


def test_tearsheet_writes_html_file(tmp_path: Path):
    equity = _synthetic_equity_curve(100_000.0, 0.0006, 252, "2024-01-02")  # ~16% annual
    spy = _synthetic_equity_curve(450.0, 0.0004, 252, "2024-01-02")         # ~10% annual
    out_dir = tmp_path / "results"
    report(equity_curve=equity, spy_closes=spy, out_dir=out_dir)
    html = out_dir / "tearsheet.html"
    assert html.exists() and html.stat().st_size > 1000, "tearsheet HTML missing or trivially small"


def test_tearsheet_returns_summary_dict(tmp_path: Path):
    equity = _synthetic_equity_curve(100_000.0, 0.0006, 252, "2024-01-02")
    spy = _synthetic_equity_curve(450.0, 0.0004, 252, "2024-01-02")
    summary = report(equity_curve=equity, spy_closes=spy, out_dir=tmp_path / "results")
    # Summary must include the key metrics named in the spec milestone (§1).
    for key in ("strategy_cagr", "strategy_sharpe", "strategy_max_dd",
                "spy_cagr", "spy_sharpe", "spy_max_dd"):
        assert key in summary, f"missing key: {key}"
        assert summary[key] is not None
