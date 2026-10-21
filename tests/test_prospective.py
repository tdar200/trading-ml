from pathlib import Path

import pandas as pd

from analysis.prospective import prospective_progress


def test_prospective_progress_counts_only_post_freeze_sessions(tmp_path: Path):
    bars = pd.DataFrame({"ts": pd.to_datetime([
        "2026-01-02T15:00:00Z", "2026-01-05T15:00:00Z", "2026-01-05T16:00:00Z",
    ])})
    paper = tmp_path / "paper.csv"
    pd.DataFrame({
        "session_date": ["2026-01-05"], "orders_filled": [2],
    }).to_csv(paper, index=False)
    result = prospective_progress(
        data_end="2026-01-02", benchmark_bars=bars, paper_path=paper,
    )
    assert result["post_freeze_market_sessions"] == 1
    assert result["broker_verified_paper_sessions"] == 1
    assert result["broker_verified_fills"] == 2
    assert not result["approval_ready_by_count_only"]
