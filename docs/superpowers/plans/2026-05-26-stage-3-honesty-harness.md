# Stage 3: Honesty Harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the reusable evaluation harness — walk-forward parameter sweep + Deflated Sharpe + cost stress — that lets us answer "is the edge real, or an artifact of one parameter choice / one time period / one cost assumption?" with a number. Output is a Markdown summary + JSON dump per harness run.

**Architecture:** A `RunSpec` dataclass defines one backtest configuration. `runner.run(spec, bars, instruments)` executes it via NautilusTrader and returns metrics + equity curve. `walk_forward.py` generates the cross-product of a parameter grid × time windows; `costs.py` re-runs the winner at multiple slippage levels; `deflated_sharpe.py` applies the Bailey-LdP penalty for multiple testing; `harness_report.py` glues the three into a written verdict. A runnable in `examples/run_harness.py` ties it all to the Stage-1 catalog (with a `--smoke` mode for ~6-minute dev iteration).

**Tech Stack:**
- Python 3.12 (`uv`-managed; already pinned)
- `nautilus_trader==1.227.0` — engine reused via existing helpers
- `pandas` — long-format result frames
- `scipy` — `scipy.stats.norm` for the Deflated Sharpe Φ/Φ⁻¹ (added in Task 2)
- `pytest`, `pytest-mock` — testing

---

## File Structure

| Path | Responsibility |
|---|---|
| `pyproject.toml` | Add `scipy` to dependencies |
| `analysis/__init__.py` | (existing; unchanged) |
| `analysis/runspec.py` | Pure dataclasses: `RunSpec`, `RunResult` |
| `analysis/deflated_sharpe.py` | Pure Bailey-LdP DSR math |
| `analysis/metrics.py` | Pure metric functions (CAGR, Sharpe, max-DD) — extracted from `tearsheet.py` |
| `analysis/tearsheet.py` | Modified to import metrics from `metrics.py` |
| `analysis/runner.py` | Execute one RunSpec → RunResult; the only Nautilus-engine consumer in `analysis/` |
| `analysis/walk_forward.py` | Grid × windows: build specs, run them, aggregate to long-format DataFrame |
| `analysis/costs.py` | Run a winner spec at multiple slippage levels |
| `analysis/harness_report.py` | `summarize_walk_forward` + `render_markdown` + `write_report` |
| `examples/run_harness.py` | Runnable: full harness against Stage-1 catalog, with `--smoke` mode |
| `tests/test_runspec.py` | Dataclass invariants |
| `tests/test_deflated_sharpe.py` | Pure-math unit tests |
| `tests/test_metrics.py` | Metric-function unit tests (post-refactor) |
| `tests/test_runner.py` | Synthetic-catalog integration |
| `tests/test_walk_forward.py` | Tiny grid × windows on synthetic data |
| `tests/test_costs.py` | Cost-stress on synthetic data |
| `tests/test_harness_report.py` | Markdown/summary rendering |

**Boundaries:**
- `runspec.py`, `deflated_sharpe.py`, `metrics.py`, `harness_report.py` are pure modules — no Nautilus, no IO. Fully unit-testable.
- `runner.py` is the only file in `analysis/` that constructs a `BacktestEngine`. All other harness modules consume its outputs.
- `walk_forward.py` and `costs.py` compose `runner.py` and `runspec.py`.
- `examples/run_harness.py` is the only place that wires the catalog. Same convention as `backtest_momentum.py`.

---

## Task 1: `RunSpec` + `RunResult` dataclasses

**Files:**
- Create: `tests/test_runspec.py`
- Create: `analysis/runspec.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_runspec.py`:
```python
"""Pure dataclass tests for RunSpec + RunResult — no Nautilus, no IO."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from analysis.runspec import RunResult, RunSpec
from strategies.momentum_topn import MomentumTopNConfig, MomentumTopNStrategy


def _basic_config() -> MomentumTopNConfig:
    return MomentumTopNConfig(universe=("XLK", "XLF"), benchmark="SPY")


def test_runspec_constructs_with_required_fields():
    spec = RunSpec(
        label="test",
        strategy_class=MomentumTopNStrategy,
        strategy_config=_basic_config(),
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 6, 1, tzinfo=timezone.utc),
    )
    assert spec.label == "test"
    assert spec.start < spec.end
    assert spec.slippage_prob == 0.25       # default
    assert spec.starting_cash == 100_000     # default
    assert spec.random_seed == 42            # default


def test_runspec_rejects_end_before_start():
    with pytest.raises(ValueError, match="end must be after start"):
        RunSpec(
            label="bad",
            strategy_class=MomentumTopNStrategy,
            strategy_config=_basic_config(),
            start=datetime(2024, 6, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )


def test_runspec_rejects_non_positive_cash():
    with pytest.raises(ValueError, match="starting_cash must be positive"):
        RunSpec(
            label="bad",
            strategy_class=MomentumTopNStrategy,
            strategy_config=_basic_config(),
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 6, 1, tzinfo=timezone.utc),
            starting_cash=0,
        )


def test_runspec_rejects_slippage_out_of_range():
    with pytest.raises(ValueError, match="slippage_prob must be in"):
        RunSpec(
            label="bad",
            strategy_class=MomentumTopNStrategy,
            strategy_config=_basic_config(),
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 6, 1, tzinfo=timezone.utc),
            slippage_prob=1.5,
        )


def test_runresult_carries_spec_metrics_and_equity():
    spec = RunSpec(
        label="ok",
        strategy_class=MomentumTopNStrategy,
        strategy_config=_basic_config(),
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 6, 1, tzinfo=timezone.utc),
    )
    equity = pd.Series(
        [100_000.0, 101_000.0],
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        name="equity",
    )
    metrics = {"cagr": 0.10, "sharpe": 1.2, "max_drawdown": -0.05,
               "total_return": 0.01, "final_equity": 101_000.0,
               "n_days": 2, "n_trades": 0}
    result = RunResult(spec=spec, metrics=metrics, equity_curve=equity)
    assert result.metrics["sharpe"] == 1.2
    assert len(result.equity_curve) == 2
    assert result.spec.label == "ok"
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /home/tahseen-dar/Projects/trading-ml
uv run pytest tests/test_runspec.py -v
```
Expected: collection error (`ModuleNotFoundError: No module named 'analysis.runspec'`).

- [ ] **Step 3: Implement**

Create `analysis/runspec.py`:
```python
"""
Pure dataclasses for the Stage-3 harness — no Nautilus, no IO.

RunSpec describes one backtest configuration; RunResult carries its outputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from nautilus_trader.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy


@dataclass(frozen=True)
class RunSpec:
    """A single backtest configuration."""
    label: str
    strategy_class: type[Strategy]
    strategy_config: StrategyConfig
    start: datetime
    end: datetime
    slippage_prob: float = 0.25
    starting_cash: int = 100_000
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"end must be after start (got start={self.start}, end={self.end})")
        if self.starting_cash <= 0:
            raise ValueError(f"starting_cash must be positive (got {self.starting_cash})")
        if not (0.0 <= self.slippage_prob <= 1.0):
            raise ValueError(f"slippage_prob must be in [0.0, 1.0] (got {self.slippage_prob})")


@dataclass(frozen=True)
class RunResult:
    """Output of a single backtest run."""
    spec: RunSpec
    metrics: dict[str, float]
    equity_curve: pd.Series  # noqa: NPY002 — pandas Series is fine in dataclass

    @property
    def label(self) -> str:
        return self.spec.label
```

- [ ] **Step 4: Run to confirm pass**

```bash
uv run pytest tests/test_runspec.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Full-suite sanity**

```bash
uv run pytest -q 2>&1 | tail -3
```
Expected: 43 passed (38 + 5).

- [ ] **Step 6: Commit**

```bash
git add analysis/runspec.py tests/test_runspec.py
git commit -m "feat(analysis): RunSpec + RunResult dataclasses for the harness

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Deflated Sharpe (pure math, with scipy)

**Files:**
- Modify: `pyproject.toml` (add `scipy`)
- Create: `tests/test_deflated_sharpe.py`
- Create: `analysis/deflated_sharpe.py`

- [ ] **Step 1: Add `scipy` as an explicit dependency**

Edit `pyproject.toml` `[project].dependencies`. Add `"scipy>=1.13"` as the last entry. The new dependencies block:
```toml
dependencies = [
    "nautilus_trader==1.227.0",
    "alpaca-py==0.43.4",
    "pandas>=2.2",
    "polars>=1.0",
    "python-dotenv>=1.0",
    "yfinance>=0.2",
    "quantstats>=0.0.62",
    "scipy>=1.13",
]
```

Then sync:
```bash
uv sync
uv run python -c "from scipy.stats import norm; print('scipy', __import__('scipy').__version__, 'norm.cdf(0)=', norm.cdf(0))"
```
Expected: prints a scipy version and `norm.cdf(0)= 0.5`.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_deflated_sharpe.py`:
```python
"""Pure-math tests for the Bailey-LdP Deflated Sharpe Ratio."""
from __future__ import annotations

import math

import pytest

from analysis.deflated_sharpe import deflated_sharpe


def test_empty_sharpes_raises():
    with pytest.raises(ValueError, match="non-empty"):
        deflated_sharpe([], n_obs=252)


def test_n_obs_must_be_positive():
    with pytest.raises(ValueError, match="n_obs must be"):
        deflated_sharpe([1.0], n_obs=0)


def test_single_trial_returns_all_keys():
    out = deflated_sharpe([1.0], n_obs=252)
    for key in ("observed_sharpe", "n_trials", "expected_max_sharpe",
                "deflated_sharpe", "p_value"):
        assert key in out


def test_single_trial_no_penalty():
    """With only 1 trial there is no selection bias — expected_max_sharpe = 0."""
    out = deflated_sharpe([1.0], n_obs=252)
    assert out["observed_sharpe"] == pytest.approx(1.0)
    assert out["n_trials"] == 1
    assert out["expected_max_sharpe"] == pytest.approx(0.0)
    # With SR=1, n_obs=252: sigma_SR ≈ sqrt(1.5/251) ≈ 0.0773; z ≈ 12.94 → p ≈ 1.0
    assert out["deflated_sharpe"] > 10
    assert out["p_value"] == pytest.approx(1.0, abs=1e-6)


def test_winner_in_many_trials_with_mean_zero_depresses():
    """If 100 trial Sharpes are drawn ~N(0, 0.5), the max is much smaller after deflation."""
    import random
    random.seed(123)
    sharpes = [random.gauss(0.0, 0.5) for _ in range(100)]
    out = deflated_sharpe(sharpes, n_obs=252)
    # The observed (max) Sharpe is positive (selection bias), but DSR should be
    # near zero or negative — the multiple-testing penalty dominates.
    assert out["observed_sharpe"] > 0.5      # selection-biased high value
    assert out["deflated_sharpe"] < out["observed_sharpe"]
    assert out["p_value"] < 0.5              # cannot reject "real Sharpe is zero"


def test_all_equal_sharpes_no_variance():
    """Identical Sharpes → Var=0 → expected_max=0 → no penalty (no information about luck)."""
    out = deflated_sharpe([0.5, 0.5, 0.5, 0.5], n_obs=252)
    assert out["observed_sharpe"] == pytest.approx(0.5)
    assert out["expected_max_sharpe"] == pytest.approx(0.0)


def test_p_value_always_in_unit_interval():
    """p_value must be in [0, 1] regardless of inputs."""
    for sharpes in ([0.0], [-1.0], [5.0], [-2.0, 0.0, 2.0, 4.0]):
        for n_obs in (10, 100, 1000):
            out = deflated_sharpe(sharpes, n_obs=n_obs)
            assert 0.0 <= out["p_value"] <= 1.0


def test_higher_n_obs_tightens_p_value():
    """For a fixed positive SR, more observations → larger z-score → p closer to 1."""
    sharpes = [0.5, 0.3, 0.1]
    low = deflated_sharpe(sharpes, n_obs=50)
    high = deflated_sharpe(sharpes, n_obs=5000)
    assert high["p_value"] >= low["p_value"]
```

- [ ] **Step 3: Run to confirm failure**

```bash
uv run pytest tests/test_deflated_sharpe.py -v
```
Expected: collection error.

- [ ] **Step 4: Implement**

Create `analysis/deflated_sharpe.py`:
```python
"""
Bailey & López de Prado (2014) Deflated Sharpe Ratio.

Given the Sharpe ratios from a set of `n` trial strategies / parameter combos,
return the deflated (multiple-testing-penalized) Sharpe and the implied p-value
that the winner's Sharpe is real (> 0) under the null hypothesis that all true
Sharpes are zero.

Reference: "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
Overfitting and Non-Normality" (Bailey & López de Prado, 2014).

This module uses the no-skew/no-kurt simplification of the σ_SR_obs variance
term (σ² ≈ (1 + 0.5·SR²) / (N − 1)), which the paper notes is acceptable when
return distributions are roughly normal — sufficient for daily ETF returns.
Higher-moment adjustments are deferred to Stage 5+ where ML predictions may
have fatter tails.

Pure module: no IO, no Nautilus.
"""
from __future__ import annotations

import math
from statistics import pstdev

from scipy.stats import norm

# Euler-Mascheroni constant.
_GAMMA = 0.5772156649015329


def _expected_max_sharpe(n_trials: int, sigma_sr: float) -> float:
    """E[max SR over N iid trials] under H0: all true SRs are zero."""
    if n_trials <= 1 or sigma_sr == 0.0:
        return 0.0
    # Approximation from Bailey-LdP.
    return sigma_sr * (
        (1 - _GAMMA) * norm.ppf(1 - 1.0 / n_trials)
        + _GAMMA * norm.ppf(1 - 1.0 / (n_trials * math.e))
    )


def _sigma_sr_obs(sr_obs: float, n_obs: int) -> float:
    """Standard error of the observed Sharpe (no-skew/no-kurt simplification)."""
    denom = max(n_obs - 1, 1)
    return math.sqrt((1.0 + 0.5 * sr_obs * sr_obs) / denom)


def deflated_sharpe(sharpes: list[float], n_obs: int) -> dict:
    """Compute the Deflated Sharpe Ratio.

    Args:
        sharpes: All in-sample Sharpe ratios from the trial set (e.g. all 18
            combos of a parameter grid). The "winner" is max(sharpes).
        n_obs: Number of return observations the winner's Sharpe was computed
            over (e.g. number of trading days in the full backtest).

    Returns:
        dict with:
            observed_sharpe:    max(sharpes)
            n_trials:           len(sharpes)
            expected_max_sharpe: E[max SR over n_trials] under H0
            deflated_sharpe:    z-score after subtracting selection-bias mean
                                and dividing by σ_SR_obs
            p_value:            Φ(deflated_sharpe) — probability the winner's
                                Sharpe is real (> 0)
    """
    if not sharpes:
        raise ValueError("sharpes must be non-empty")
    if n_obs <= 0:
        raise ValueError(f"n_obs must be positive (got {n_obs})")

    n = len(sharpes)
    sr_obs = max(sharpes)
    sigma_sr_trials = pstdev(sharpes) if n > 1 else 0.0
    expected_max = _expected_max_sharpe(n, sigma_sr_trials)
    sigma_sr_obs = _sigma_sr_obs(sr_obs, n_obs)
    deflated_z = (sr_obs - expected_max) / sigma_sr_obs if sigma_sr_obs > 0 else 0.0
    p_value = float(norm.cdf(deflated_z))

    return {
        "observed_sharpe": float(sr_obs),
        "n_trials": n,
        "expected_max_sharpe": float(expected_max),
        "deflated_sharpe": float(deflated_z),
        "p_value": p_value,
    }
```

- [ ] **Step 5: Run to confirm pass**

```bash
uv run pytest tests/test_deflated_sharpe.py -v
```
Expected: 8 passed.

- [ ] **Step 6: Full-suite sanity**

```bash
uv run pytest -q 2>&1 | tail -3
```
Expected: 51 passed (43 + 8).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock analysis/deflated_sharpe.py tests/test_deflated_sharpe.py
git commit -m "feat(analysis): Deflated Sharpe Ratio (Bailey-LdP, no-skew form) + scipy dep

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Extract metric helpers from `tearsheet.py` to `analysis/metrics.py`

The metric helpers `_cagr`, `_sharpe`, `_max_drawdown` currently live as private functions in `analysis/tearsheet.py`. Stage 3's `runner.py` needs the same functions. Promote them to a small public module and update `tearsheet.py` to import them.

**Files:**
- Create: `analysis/metrics.py`
- Modify: `analysis/tearsheet.py` (remove private helpers, import from `metrics`)
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write failing tests for the public functions**

Create `tests/test_metrics.py`:
```python
"""Pure metric-function tests."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from analysis.metrics import cagr, max_drawdown, sharpe


def test_cagr_empty_returns_nan():
    assert math.isnan(cagr(pd.Series([], dtype=float)))


def test_cagr_one_year_doubling():
    """1 year of daily returns that compound to 2× → CAGR = 1.0."""
    # 252 returns, each (2)^(1/252) - 1 → product = 2 → CAGR = 1.0
    daily = pd.Series([(2.0) ** (1.0 / 252) - 1.0] * 252)
    assert cagr(daily) == pytest.approx(1.0, abs=1e-6)


def test_sharpe_zero_std_returns_nan():
    assert math.isnan(sharpe(pd.Series([0.001] * 100)))


def test_sharpe_constant_positive_returns_high():
    """All returns +0.1%/day (no variance fails the std test but tiny noise should give very high Sharpe)."""
    rng = np.random.default_rng(42)
    daily = pd.Series([0.001 + rng.normal(0, 1e-9) for _ in range(252)])
    # ddof=0 will be ~0.001/1e-9 * sqrt(252) — astronomical. Just confirm it's finite & large.
    s = sharpe(daily)
    assert math.isfinite(s)
    assert s > 100  # arbitrarily large


def test_max_drawdown_monotone_growth():
    """Equity that only grows has max-DD = 0."""
    equity = pd.Series([100.0, 101.0, 102.0, 103.0])
    assert max_drawdown(equity) == pytest.approx(0.0)


def test_max_drawdown_50pct_peak_to_trough():
    """Equity peaks at 200, troughs at 100 → -50% drawdown."""
    equity = pd.Series([100.0, 200.0, 150.0, 100.0])
    assert max_drawdown(equity) == pytest.approx(-0.5)
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_metrics.py -v
```
Expected: collection error.

- [ ] **Step 3: Create `analysis/metrics.py` by extracting from `tearsheet.py`**

First, read `analysis/tearsheet.py` to find the `_cagr`, `_sharpe`, `_max_drawdown` definitions. Then create `analysis/metrics.py` with the SAME bodies but renamed (drop the underscore prefix; they are now public):

```python
"""
Pure metric functions: CAGR, Sharpe, max drawdown.

Used by analysis.tearsheet for individual-run summaries and by analysis.runner
for backtest result metrics. No IO; deterministic given inputs.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compound annual growth rate from a series of period returns."""
    if len(returns) == 0:
        return float("nan")
    total = (1.0 + returns).prod()
    years = len(returns) / periods_per_year
    if years <= 0 or total <= 0:
        return float("nan")
    return float(total ** (1.0 / years) - 1.0)


def sharpe(returns: pd.Series, periods_per_year: int = 252, rf: float = 0.0) -> float:
    """Annualized Sharpe ratio (ddof=0 for population std — consistent across the project)."""
    if returns.std(ddof=0) == 0 or len(returns) == 0:
        return float("nan")
    excess = returns - rf / periods_per_year
    return float(excess.mean() / excess.std(ddof=0) * math.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (a negative number, or 0 for monotone growth)."""
    if len(equity) == 0:
        return float("nan")
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return float(drawdown.min())
```

- [ ] **Step 4: Update `analysis/tearsheet.py` to import from `metrics`**

In `analysis/tearsheet.py`:
1. Add at the top of the imports: `from analysis.metrics import cagr, max_drawdown, sharpe`
2. DELETE the three private function definitions `_cagr`, `_sharpe`, `_max_drawdown`.
3. Replace every internal call site `_cagr(` → `cagr(`, `_sharpe(` → `sharpe(`, `_max_drawdown(` → `max_drawdown(`.

Don't change anything else in `tearsheet.py` — same public function signature, same fail-soft behavior, same padding logic.

- [ ] **Step 5: Run all metric tests + tearsheet tests + full suite**

```bash
uv run pytest tests/test_metrics.py tests/test_tearsheet.py -v
uv run pytest -q 2>&1 | tail -3
```
Expected: 6 metric tests + 2 tearsheet tests pass. Full suite at 57 (51 + 6 metrics). The existing tearsheet behavior must NOT change (its tests stay green).

- [ ] **Step 6: Commit**

```bash
git add analysis/metrics.py analysis/tearsheet.py tests/test_metrics.py
git commit -m "refactor(analysis): extract metric helpers to analysis/metrics.py

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `runner.run(spec, bars_by_ticker, instruments) → RunResult`

The core execution unit: take a RunSpec, build a BacktestEngine, run it, return metrics. All other harness modules consume this.

**Files:**
- Create: `analysis/runner.py`
- Create: `tests/test_runner.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_runner.py`:
```python
"""Integration test for runner.run against synthetic bar data."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId, Symbol
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Price, Quantity

from analysis.runner import run
from analysis.runspec import RunResult, RunSpec
from data.instruments import VENUE_MIC
from strategies.momentum_topn import MomentumTopNConfig, MomentumTopNStrategy


def _instrument(ticker: str) -> Equity:
    return Equity(
        instrument_id=InstrumentId.from_str(f"{ticker}.{VENUE_MIC}"),
        raw_symbol=Symbol(ticker),
        currency=USD,
        price_precision=2,
        price_increment=Price.from_str("0.01"),
        lot_size=Quantity.from_int(1),
        ts_event=0,
        ts_init=0,
    )


def _bars(ticker: str, closes: list[float], start_ts_ns: int) -> list[Bar]:
    bar_type = BarType.from_str(f"{ticker}.{VENUE_MIC}-1-DAY-LAST-EXTERNAL")
    one_day_ns = 86_400_000_000_000
    bars: list[Bar] = []
    for i, c in enumerate(closes):
        ts = start_ts_ns + i * one_day_ns
        p = Price.from_str(f"{c:.2f}")
        bars.append(Bar(
            bar_type=bar_type, open=p, high=p, low=p, close=p,
            volume=Quantity.from_int(1_000_000),
            ts_event=ts, ts_init=ts,
        ))
    return bars


def _grow(rate: float, n: int) -> list[float]:
    out = [100.0]
    for _ in range(n - 1):
        out.append(out[-1] * (1.0 + rate))
    return out


def test_runner_returns_RunResult_with_metrics_and_equity():
    n = 380
    start_ts_ns = 1_577_836_800_000_000_000  # 2020-01-01 UTC
    closes_by_ticker = {"A": _grow(0.0020, n), "B": _grow(0.0010, n), "BENCH": _grow(0.0005, n)}
    bars_by_ticker = {t: _bars(t, c, start_ts_ns) for t, c in closes_by_ticker.items()}
    instruments = [_instrument(t) for t in bars_by_ticker]

    spec = RunSpec(
        label="runner-test",
        strategy_class=MomentumTopNStrategy,
        strategy_config=MomentumTopNConfig(
            universe=("A", "B"), benchmark="BENCH",
            top_n=1, lookback_days=63, skip_days=21,
        ),
        start=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end=datetime(2021, 1, 17, tzinfo=timezone.utc),  # ~380 days
        slippage_prob=0.0,  # deterministic for the test
    )

    result = run(spec, bars_by_ticker=bars_by_ticker, instruments=instruments)
    assert isinstance(result, RunResult)
    # Metrics required by the harness:
    for key in ("cagr", "sharpe", "max_drawdown", "total_return",
                "final_equity", "n_days", "n_trades"):
        assert key in result.metrics, f"missing metric: {key}"
    # Sanity: equity curve non-empty, final equity > 0
    assert len(result.equity_curve) > 0
    assert result.metrics["final_equity"] > 0
    # With top_n=1 and ticker A having highest momentum, the strategy held A
    # and grew with it → final_equity should be above 100k.
    assert result.metrics["final_equity"] > 100_000


def test_runner_filters_bars_by_date_window():
    """Bars outside [spec.start, spec.end) must be excluded — the strategy only
    sees bars in the window. Verified by running with two different windows on
    the SAME bars and confirming different n_days."""
    n = 380
    start_ts_ns = 1_577_836_800_000_000_000  # 2020-01-01
    closes_by_ticker = {"A": _grow(0.0020, n), "B": _grow(0.0010, n), "BENCH": _grow(0.0005, n)}
    bars_by_ticker = {t: _bars(t, c, start_ts_ns) for t, c in closes_by_ticker.items()}
    instruments = [_instrument(t) for t in bars_by_ticker]

    config = MomentumTopNConfig(universe=("A", "B"), benchmark="BENCH",
                                top_n=1, lookback_days=63, skip_days=21)
    short_spec = RunSpec(
        label="short", strategy_class=MomentumTopNStrategy, strategy_config=config,
        start=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end=datetime(2020, 7, 1, tzinfo=timezone.utc),
        slippage_prob=0.0,
    )
    long_spec = RunSpec(
        label="long", strategy_class=MomentumTopNStrategy, strategy_config=config,
        start=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end=datetime(2021, 1, 17, tzinfo=timezone.utc),
        slippage_prob=0.0,
    )
    r_short = run(short_spec, bars_by_ticker=bars_by_ticker, instruments=instruments)
    r_long = run(long_spec, bars_by_ticker=bars_by_ticker, instruments=instruments)
    assert r_long.metrics["n_days"] > r_short.metrics["n_days"]
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_runner.py -v
```
Expected: collection error.

- [ ] **Step 3: Implement `analysis/runner.py`**

Create `analysis/runner.py`:
```python
"""
The single execution unit of the Stage-3 harness: run one RunSpec, return one
RunResult. The only file in analysis/ that imports a BacktestEngine.

The strategy's _equity_history (Stage-2 mark-to-market snapshots) is the
source of truth for the equity curve. Trade counts come from cache.orders()
filtered to fills (verified path in docs/nautilus_api_notes.md §9).
"""
from __future__ import annotations

import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.enums import OrderStatus
from nautilus_trader.model.data import Bar
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.instruments import Equity

from analysis.metrics import cagr, max_drawdown, sharpe
from analysis.runspec import RunResult, RunSpec
from strategies.backtest_setup import add_default_backtest_venue


def _bars_in_window(bars: list[Bar], start_ns: int, end_ns: int) -> list[Bar]:
    return [b for b in bars if start_ns <= b.ts_event < end_ns]


def _equity_curve_from_strategy(strategy) -> pd.Series:
    """Mark-to-market daily equity from strategy._equity_history (Stage-2 mechanism)."""
    history = getattr(strategy, "_equity_history", None) or []
    if not history:
        return pd.Series(name="equity", dtype=float)
    rows = [
        (pd.Timestamp(ts_ns, unit="ns", tz="UTC").tz_convert(None).normalize(), eq)
        for ts_ns, eq in history
    ]
    df = pd.DataFrame(rows, columns=["date", "equity"]).groupby("date").last()
    return df["equity"].sort_index()


def _count_fills(engine: BacktestEngine) -> int:
    n = 0
    for order in engine.cache.orders():
        # OrderStatus.FILLED indicates the order was fully filled.
        if order.status == OrderStatus.FILLED:
            n += 1
    return n


def _compute_metrics(equity: pd.Series, n_trades: int) -> dict[str, float]:
    if len(equity) == 0:
        return {
            "cagr": float("nan"), "sharpe": float("nan"),
            "max_drawdown": float("nan"), "total_return": float("nan"),
            "final_equity": 0.0, "n_days": 0, "n_trades": n_trades,
        }
    returns = equity.pct_change().dropna()
    final = float(equity.iloc[-1])
    initial = float(equity.iloc[0])
    total_return = (final / initial - 1.0) if initial > 0 else float("nan")
    return {
        "cagr": cagr(returns),
        "sharpe": sharpe(returns),
        "max_drawdown": max_drawdown(equity),
        "total_return": total_return,
        "final_equity": final,
        "n_days": len(equity),
        "n_trades": n_trades,
    }


def run(
    spec: RunSpec,
    bars_by_ticker: dict[str, list[Bar]],
    instruments: list[Equity],
) -> RunResult:
    """Execute the backtest defined by `spec` against the provided bars + instruments."""
    engine_config = BacktestEngineConfig(
        trader_id=TraderId(f"HARNESS-{spec.label[:32]}"),  # TraderId requires a hyphen
        logging=LoggingConfig(log_level="WARNING"),
    )
    engine = BacktestEngine(config=engine_config)
    venue = add_default_backtest_venue(
        engine,
        starting_cash=spec.starting_cash,
        slippage_prob=spec.slippage_prob,
        random_seed=spec.random_seed,
    )

    for instrument in instruments:
        engine.add_instrument(instrument)

    start_ns = int(spec.start.timestamp() * 1e9)
    end_ns = int(spec.end.timestamp() * 1e9)
    for ticker, bars in bars_by_ticker.items():
        windowed = _bars_in_window(bars, start_ns, end_ns)
        if windowed:
            engine.add_data(windowed)

    strategy = spec.strategy_class(config=spec.strategy_config)
    engine.add_strategy(strategy)
    engine.run()

    equity = _equity_curve_from_strategy(strategy)
    n_trades = _count_fills(engine)
    metrics = _compute_metrics(equity, n_trades)
    engine.dispose()
    return RunResult(spec=spec, metrics=metrics, equity_curve=equity)
```

If `OrderStatus.FILLED` doesn't exist at that import path in v1.227, find the verified equivalent via `inspect` and check `docs/nautilus_api_notes.md` §9. An alternative is checking `order.is_filled` (a boolean property) if available — adjust `_count_fills` accordingly.

- [ ] **Step 4: Run to confirm pass**

```bash
uv run pytest tests/test_runner.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Full-suite sanity**

```bash
uv run pytest -q 2>&1 | tail -3
```
Expected: 59 passed (57 + 2).

- [ ] **Step 6: Commit**

```bash
git add analysis/runner.py tests/test_runner.py
git commit -m "feat(analysis): runner.run — execute one RunSpec → RunResult

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Walk-forward grid × windows runner

**Files:**
- Create: `analysis/walk_forward.py`
- Create: `tests/test_walk_forward.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_walk_forward.py`:
```python
"""Walk-forward harness tests on synthetic data."""
from __future__ import annotations

from datetime import datetime, timezone

from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId, Symbol
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Price, Quantity

from analysis.walk_forward import expand_param_grid, walk_forward
from data.instruments import VENUE_MIC
from strategies.momentum_topn import MomentumTopNConfig, MomentumTopNStrategy


def test_expand_param_grid_cross_product():
    grid = {"top_n": [2, 3], "lookback_days": [63, 126], "skip_days": [21]}
    combos = expand_param_grid(grid)
    assert len(combos) == 4
    assert {"top_n": 2, "lookback_days": 63, "skip_days": 21} in combos
    assert {"top_n": 3, "lookback_days": 126, "skip_days": 21} in combos


def test_expand_param_grid_empty_grid_returns_one_empty_combo():
    assert expand_param_grid({}) == [{}]


# ---- synthetic-data helpers (same pattern as test_runner.py) ----

def _instrument(ticker: str) -> Equity:
    return Equity(
        instrument_id=InstrumentId.from_str(f"{ticker}.{VENUE_MIC}"),
        raw_symbol=Symbol(ticker), currency=USD, price_precision=2,
        price_increment=Price.from_str("0.01"),
        lot_size=Quantity.from_int(1),
        ts_event=0, ts_init=0,
    )


def _bars(ticker: str, closes: list[float], start_ts_ns: int) -> list[Bar]:
    bar_type = BarType.from_str(f"{ticker}.{VENUE_MIC}-1-DAY-LAST-EXTERNAL")
    one_day_ns = 86_400_000_000_000
    out: list[Bar] = []
    for i, c in enumerate(closes):
        ts = start_ts_ns + i * one_day_ns
        p = Price.from_str(f"{c:.2f}")
        out.append(Bar(bar_type=bar_type, open=p, high=p, low=p, close=p,
                       volume=Quantity.from_int(1_000_000),
                       ts_event=ts, ts_init=ts))
    return out


def _grow(rate: float, n: int) -> list[float]:
    out = [100.0]
    for _ in range(n - 1):
        out.append(out[-1] * (1.0 + rate))
    return out


def test_walk_forward_produces_one_row_per_combo_per_window():
    n = 380
    start_ts_ns = 1_577_836_800_000_000_000  # 2020-01-01
    closes_by_ticker = {"A": _grow(0.002, n), "B": _grow(0.001, n), "BENCH": _grow(0.0005, n)}
    bars_by_ticker = {t: _bars(t, c, start_ts_ns) for t, c in closes_by_ticker.items()}
    instruments = [_instrument(t) for t in bars_by_ticker]

    grid = {"top_n": [1, 2], "lookback_days": [63], "skip_days": [21]}
    windows = [
        (datetime(2020, 1, 1, tzinfo=timezone.utc),
         datetime(2020, 7, 1, tzinfo=timezone.utc)),
        (datetime(2020, 7, 1, tzinfo=timezone.utc),
         datetime(2021, 1, 17, tzinfo=timezone.utc)),
    ]
    base = dict(universe=("A", "B"), benchmark="BENCH")

    df = walk_forward(
        strategy_class=MomentumTopNStrategy,
        config_class=MomentumTopNConfig,
        base_config_kwargs=base,
        param_grid=grid,
        windows=windows,
        bars_by_ticker=bars_by_ticker,
        instruments=instruments,
        slippage_prob=0.0,
    )

    # 2 combos × 2 windows = 4 rows
    assert len(df) == 4
    # Required columns
    for col in ("window_idx", "start", "end", "top_n", "lookback_days",
                "skip_days", "cagr", "sharpe", "max_drawdown",
                "n_days", "n_trades", "final_equity"):
        assert col in df.columns
    # window_idx ranges over 0..1
    assert set(df["window_idx"]) == {0, 1}
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_walk_forward.py -v
```
Expected: collection error.

- [ ] **Step 3: Implement**

Create `analysis/walk_forward.py`:
```python
"""
Walk-forward parameter sweep: cross-product of param grid × time windows.

`expand_param_grid` is a pure helper. `walk_forward` composes RunSpecs and
delegates execution to `analysis.runner`. Returns a long-format DataFrame:
one row per (combo, window) with all metrics and the parameter values used.
"""
from __future__ import annotations

from datetime import datetime
from itertools import product
from typing import Any

import pandas as pd

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar
from nautilus_trader.model.instruments import Equity
from nautilus_trader.trading.strategy import Strategy

from analysis.runner import run
from analysis.runspec import RunSpec


def expand_param_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Enumerate the cross-product of a parameter grid.

    Example: {"top_n": [2, 3], "lookback_days": [63, 126]} →
        [{"top_n": 2, "lookback_days": 63}, {"top_n": 2, "lookback_days": 126},
         {"top_n": 3, "lookback_days": 63}, {"top_n": 3, "lookback_days": 126}].
    Empty grid → [{}] so the caller still runs the base config once.
    """
    if not grid:
        return [{}]
    keys = list(grid.keys())
    return [dict(zip(keys, values, strict=True)) for values in product(*grid.values())]


def walk_forward(
    *,
    strategy_class: type[Strategy],
    config_class: type[StrategyConfig],
    base_config_kwargs: dict[str, Any],
    param_grid: dict[str, list[Any]],
    windows: list[tuple[datetime, datetime]],
    bars_by_ticker: dict[str, list[Bar]],
    instruments: list[Equity],
    slippage_prob: float = 0.25,
    starting_cash: int = 100_000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Run every (param combo) × (window) and return a long-format DataFrame.

    Each row carries: window_idx, start, end, the parameter values, plus the
    metrics from analysis.runner (cagr, sharpe, max_drawdown, total_return,
    n_days, n_trades, final_equity).
    """
    combos = expand_param_grid(param_grid)
    rows: list[dict[str, Any]] = []
    for window_idx, (start, end) in enumerate(windows):
        for combo in combos:
            config_kwargs = {**base_config_kwargs, **combo}
            config = config_class(**config_kwargs)
            label_bits = "_".join(f"{k}{v}" for k, v in combo.items()) or "base"
            spec = RunSpec(
                label=f"w{window_idx}-{label_bits}",
                strategy_class=strategy_class,
                strategy_config=config,
                start=start,
                end=end,
                slippage_prob=slippage_prob,
                starting_cash=starting_cash,
                random_seed=random_seed,
            )
            result = run(spec, bars_by_ticker=bars_by_ticker, instruments=instruments)
            row: dict[str, Any] = {
                "window_idx": window_idx,
                "start": start,
                "end": end,
                **combo,
                **result.metrics,
            }
            rows.append(row)
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run to confirm pass**

```bash
uv run pytest tests/test_walk_forward.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Full-suite sanity**

```bash
uv run pytest -q 2>&1 | tail -3
```
Expected: 62 passed (59 + 3).

- [ ] **Step 6: Commit**

```bash
git add analysis/walk_forward.py tests/test_walk_forward.py
git commit -m "feat(analysis): walk_forward — param-grid × time-window cross-product

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Cost stress

**Files:**
- Create: `analysis/costs.py`
- Create: `tests/test_costs.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_costs.py`:
```python
"""Cost stress tests on synthetic data."""
from __future__ import annotations

from datetime import datetime, timezone

from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId, Symbol
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Price, Quantity

from analysis.costs import cost_stress
from analysis.runspec import RunSpec
from data.instruments import VENUE_MIC
from strategies.momentum_topn import MomentumTopNConfig, MomentumTopNStrategy


def _instrument(ticker: str) -> Equity:
    return Equity(
        instrument_id=InstrumentId.from_str(f"{ticker}.{VENUE_MIC}"),
        raw_symbol=Symbol(ticker), currency=USD, price_precision=2,
        price_increment=Price.from_str("0.01"),
        lot_size=Quantity.from_int(1),
        ts_event=0, ts_init=0,
    )


def _bars(ticker: str, closes: list[float], start_ts_ns: int) -> list[Bar]:
    bar_type = BarType.from_str(f"{ticker}.{VENUE_MIC}-1-DAY-LAST-EXTERNAL")
    one_day_ns = 86_400_000_000_000
    out: list[Bar] = []
    for i, c in enumerate(closes):
        ts = start_ts_ns + i * one_day_ns
        p = Price.from_str(f"{c:.2f}")
        out.append(Bar(bar_type=bar_type, open=p, high=p, low=p, close=p,
                       volume=Quantity.from_int(1_000_000),
                       ts_event=ts, ts_init=ts))
    return out


def _grow(rate: float, n: int) -> list[float]:
    out = [100.0]
    for _ in range(n - 1):
        out.append(out[-1] * (1.0 + rate))
    return out


def test_cost_stress_returns_one_row_per_slippage_level():
    n = 380
    start_ts_ns = 1_577_836_800_000_000_000  # 2020-01-01
    closes_by_ticker = {"A": _grow(0.002, n), "B": _grow(0.001, n), "BENCH": _grow(0.0005, n)}
    bars_by_ticker = {t: _bars(t, c, start_ts_ns) for t, c in closes_by_ticker.items()}
    instruments = [_instrument(t) for t in bars_by_ticker]

    base_spec = RunSpec(
        label="winner",
        strategy_class=MomentumTopNStrategy,
        strategy_config=MomentumTopNConfig(
            universe=("A", "B"), benchmark="BENCH",
            top_n=1, lookback_days=63, skip_days=21,
        ),
        start=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end=datetime(2021, 1, 17, tzinfo=timezone.utc),
    )

    df = cost_stress(
        base_spec=base_spec,
        slippage_probs=[0.0, 0.25, 0.5, 1.0],
        bars_by_ticker=bars_by_ticker,
        instruments=instruments,
    )
    assert len(df) == 4
    assert list(df["slippage_prob"]) == [0.0, 0.25, 0.5, 1.0]
    for col in ("slippage_prob", "cagr", "sharpe", "max_drawdown",
                "total_return", "final_equity", "n_days", "n_trades"):
        assert col in df.columns
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_costs.py -v
```
Expected: collection error.

- [ ] **Step 3: Implement**

Create `analysis/costs.py`:
```python
"""
Cost-stress: re-run a chosen RunSpec at multiple slippage levels and report
how the headline metrics degrade. Used after walk-forward selects a "winner".
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import pandas as pd

from nautilus_trader.model.data import Bar
from nautilus_trader.model.instruments import Equity

from analysis.runner import run
from analysis.runspec import RunSpec


def cost_stress(
    *,
    base_spec: RunSpec,
    slippage_probs: list[float],
    bars_by_ticker: dict[str, list[Bar]],
    instruments: list[Equity],
) -> pd.DataFrame:
    """Re-run `base_spec` at each slippage level; return one row per level."""
    rows: list[dict[str, Any]] = []
    for prob in slippage_probs:
        spec = replace(base_spec, slippage_prob=prob, label=f"{base_spec.label}-slip{prob:.2f}")
        result = run(spec, bars_by_ticker=bars_by_ticker, instruments=instruments)
        rows.append({"slippage_prob": prob, **result.metrics})
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run to confirm pass**

```bash
uv run pytest tests/test_costs.py -v
```
Expected: 1 passed.

- [ ] **Step 5: Full-suite sanity**

```bash
uv run pytest -q 2>&1 | tail -3
```
Expected: 63 passed (62 + 1).

- [ ] **Step 6: Commit**

```bash
git add analysis/costs.py tests/test_costs.py
git commit -m "feat(analysis): cost_stress — re-run a winner at multiple slippage levels

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Harness report (`summarize_walk_forward`, `render_markdown`, `write_report`)

**Files:**
- Create: `analysis/harness_report.py`
- Create: `tests/test_harness_report.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_harness_report.py`:
```python
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
    # Median window Sharpe of winner
    assert summary["winner_median_window_sharpe"] == 0.85
    # Range across windows for the winner
    assert summary["winner_window_sharpe_range"] == (0.8, 0.9)
    # All-combos Sharpe percentile range (5th–95th)
    assert "all_combos_sharpe_p05" in summary
    assert "all_combos_sharpe_p95" in summary
    # Winner rank: 1 of 2 unique combos
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
    assert "winner" in summary_json
    assert "deflated_sharpe" in summary_json
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_harness_report.py -v
```
Expected: collection error.

- [ ] **Step 3: Implement**

Create `analysis/harness_report.py`:
```python
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

    # Median window Sharpe per combo.
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

    # Winner's rank among unique combos.
    n_combos = grouped.ngroups
    winner_rank = 1  # by construction (we picked the max)

    return {
        "winner": winner,
        "winner_median_window_sharpe": float(medians.iloc[0]),
        "winner_window_sharpe_range": (float(winner_sharpes.min()), float(winner_sharpes.max())),
        "all_combos_sharpe_p05": float(all_sharpes.quantile(0.05)),
        "all_combos_sharpe_p95": float(all_sharpes.quantile(0.95)),
        "winner_rank": (winner_rank, int(n_combos)),
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
    """Write walk_forward.csv, cost_stress.csv, summary.json, summary.md.

    Returns the structured summary dict (also persisted as summary.json).
    """
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
        "walk_forward_summary": wf_summary,
        "deflated_sharpe": dsr,
        "cost_stress": cost_df.to_dict(orient="records"),
    }
    # winner_rank is a tuple — make it JSON-friendly.
    summary["walk_forward_summary"] = {
        **wf_summary,
        "winner_rank": list(wf_summary["winner_rank"]),
        "winner_window_sharpe_range": list(wf_summary["winner_window_sharpe_range"]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
```

- [ ] **Step 4: Run to confirm pass**

```bash
uv run pytest tests/test_harness_report.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Full-suite sanity**

```bash
uv run pytest -q 2>&1 | tail -3
```
Expected: 66 passed (63 + 3).

- [ ] **Step 6: Commit**

```bash
git add analysis/harness_report.py tests/test_harness_report.py
git commit -m "feat(analysis): harness_report — summarize WF + DSR + costs → JSON + Markdown

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Runnable harness + `--smoke` verification

**Files:**
- Create: `examples/run_harness.py`

- [ ] **Step 1: Implement the runnable**

Create `examples/run_harness.py`:
```python
"""
Stage-3 honesty harness: walk-forward × parameter grid + DSR + cost stress
on the Stage-2 momentum strategy, against the Stage-1 catalog.

Usage:
    uv run python examples/run_harness.py            # full battery, ~45–50 min
    uv run python examples/run_harness.py --smoke    # tiny grid, ~6 min

Writes results/harness_<timestamp>/ with walk_forward.csv, cost_stress.csv,
summary.json, summary.md. Prints summary.md to stdout.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

from analysis.costs import cost_stress
from analysis.deflated_sharpe import deflated_sharpe
from analysis.harness_report import write_report
from analysis.runspec import RunSpec
from analysis.walk_forward import walk_forward
from data.instruments import VENUE_MIC, build_universe_instruments
from data.tickers import ALL_TICKERS, BENCHMARK, UNIVERSE
from strategies.momentum_topn import MomentumTopNConfig, MomentumTopNStrategy


CATALOG_PATH = _PROJECT_ROOT / "data" / "catalog"
RESULTS_ROOT = _PROJECT_ROOT / "results"


# Stage-3 default param grid (spec §3): 3 × 3 × 2 = 18 combos.
DEFAULT_PARAM_GRID = {
    "top_n": [2, 3, 5],
    "lookback_days": [63, 126, 252],
    "skip_days": [0, 21],
}
SMOKE_PARAM_GRID = {"top_n": [3], "lookback_days": [126], "skip_days": [21]}

DEFAULT_SLIPPAGE_PROBS = [0.0, 0.25, 0.5, 1.0]
SMOKE_SLIPPAGE_PROBS = [0.0, 0.25]


def _load_catalog_data(catalog: ParquetDataCatalog) -> tuple[dict, list]:
    bars_by_ticker: dict = {}
    for ticker in ALL_TICKERS:
        bars = catalog.bars(instrument_ids=[f"{ticker}.{VENUE_MIC}"])
        bars_by_ticker[ticker] = bars
        print(f"  {ticker}: {len(bars)} bars")
    instruments = build_universe_instruments()
    return bars_by_ticker, instruments


def _build_windows(bars_by_ticker: dict, n_windows: int) -> list[tuple[datetime, datetime]]:
    """Build n_windows expanding test windows over the universe-bar period.

    Each window has the SAME start (the earliest universe-bar timestamp, so
    every window has the same warm-up history), and a different end split
    evenly across the bar range. Windows are evaluated by the per-window
    metrics — overlap in their start segment doesn't matter because each
    metric is computed only on the bars actually fed in.
    """
    # Earliest start across all UNIVERSE tickers (excluding the benchmark, which
    # may have a different starting date).
    universe_starts = [
        bars_by_ticker[t][0].ts_event
        for t in UNIVERSE if bars_by_ticker.get(t)
    ]
    if not universe_starts:
        raise RuntimeError("no universe bars to slice")
    start_ns = max(universe_starts)
    # Latest end across all tickers.
    all_ends = [bars_by_ticker[t][-1].ts_event for t in ALL_TICKERS if bars_by_ticker.get(t)]
    end_ns = max(all_ends)

    start = datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc)
    end = datetime.fromtimestamp(end_ns / 1e9, tz=timezone.utc)
    span = end - start
    out: list[tuple[datetime, datetime]] = []
    for i in range(1, n_windows + 1):
        window_end = start + (span * i) / n_windows
        out.append((start, window_end))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Tiny grid + few windows for dev iteration (~6 min)")
    args = parser.parse_args()

    grid = SMOKE_PARAM_GRID if args.smoke else DEFAULT_PARAM_GRID
    slippage_probs = SMOKE_SLIPPAGE_PROBS if args.smoke else DEFAULT_SLIPPAGE_PROBS
    n_windows = 2 if args.smoke else 5

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = RESULTS_ROOT / f"harness_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out_dir}")
    print(f"mode: {'SMOKE' if args.smoke else 'FULL'}")

    catalog = ParquetDataCatalog(path=str(CATALOG_PATH.resolve()))
    print("loading catalog...")
    bars_by_ticker, instruments = _load_catalog_data(catalog)
    total_bars = sum(len(v) for v in bars_by_ticker.values())
    print(f"total bars: {total_bars}")
    if total_bars == 0:
        print("ERROR: catalog is empty. Run data/ingest_alpaca.py first.")
        return 1

    windows = _build_windows(bars_by_ticker, n_windows=n_windows)
    print(f"windows ({len(windows)}):")
    for i, (s, e) in enumerate(windows):
        print(f"  [{i}] {s.date()} → {e.date()}")

    base_kwargs = dict(universe=UNIVERSE, benchmark=BENCHMARK)
    print("running walk-forward...")
    wf_df = walk_forward(
        strategy_class=MomentumTopNStrategy,
        config_class=MomentumTopNConfig,
        base_config_kwargs=base_kwargs,
        param_grid=grid,
        windows=windows,
        bars_by_ticker=bars_by_ticker,
        instruments=instruments,
        slippage_prob=0.25,
    )
    print(f"walk-forward done: {len(wf_df)} runs")

    # Pick winner: highest median window Sharpe.
    param_cols = list(grid.keys())
    medians = wf_df.groupby(param_cols)["sharpe"].median().sort_values(ascending=False)
    winner_tuple = medians.index[0]
    if not isinstance(winner_tuple, tuple):
        winner_tuple = (winner_tuple,)
    winner_kwargs = {**base_kwargs, **dict(zip(param_cols, winner_tuple, strict=True))}
    print(f"winner: {dict(zip(param_cols, winner_tuple, strict=True))}")

    # Full-catalog winner spec (no windowing) for cost stress + DSR n_obs.
    earliest_start = min(b.ts_event for b in bars_by_ticker[UNIVERSE[0]])
    latest_end = max(b.ts_event for b in bars_by_ticker[BENCHMARK])
    full_start = datetime.fromtimestamp(earliest_start / 1e9, tz=timezone.utc)
    full_end = datetime.fromtimestamp(latest_end / 1e9, tz=timezone.utc)
    winner_spec = RunSpec(
        label="winner-fullcatalog",
        strategy_class=MomentumTopNStrategy,
        strategy_config=MomentumTopNConfig(**winner_kwargs),
        start=full_start,
        end=full_end,
        slippage_prob=0.25,
    )

    # Cost stress.
    print("running cost stress...")
    cost_df = cost_stress(
        base_spec=winner_spec,
        slippage_probs=slippage_probs,
        bars_by_ticker=bars_by_ticker,
        instruments=instruments,
    )
    print(f"cost stress done: {len(cost_df)} runs")

    # DSR: use all per-(combo, window) Sharpes as the trial set; n_obs ≈ winner's n_days at baseline slippage.
    sharpes = wf_df["sharpe"].dropna().tolist()
    n_obs = int(cost_df[cost_df["slippage_prob"] == 0.25].iloc[0]["n_days"]) \
        if any(cost_df["slippage_prob"] == 0.25) else int(cost_df.iloc[0]["n_days"])
    dsr = deflated_sharpe(sharpes=sharpes, n_obs=n_obs)
    print(f"DSR: p-value={dsr['p_value']:.4f}, deflated_z={dsr['deflated_sharpe']:.4f}")

    # Report.
    write_report(out_dir=out_dir, wf_df=wf_df, cost_df=cost_df, dsr=dsr,
                 strategy_name="MomentumTopN")
    md = (out_dir / "summary.md").read_text()
    print("\n" + md)
    print(f"\nartifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify imports + `--help`**

```bash
cd /home/tahseen-dar/Projects/trading-ml
uv run python examples/run_harness.py --help
```
Expected: argparse help text, exit 0.

- [ ] **Step 3: Run the SMOKE mode against the real catalog**

```bash
uv run python examples/run_harness.py --smoke
```
Expected: ~6 minutes (2 windows × 1 combo + 2 cost levels = 4 backtests). Output ends with the Markdown summary printed to stdout. Confirm:
- An output directory `results/harness_<timestamp>/` was created
- It contains `walk_forward.csv`, `cost_stress.csv`, `summary.json`, `summary.md`
- The Markdown summary has all 5 sections

If it errors, debug. The most likely failure modes:
- An import path mismatch — fix and re-run.
- `OrderStatus.FILLED` import (in runner.py) — verify against `docs/nautilus_api_notes.md` §9 and adjust.
- `_build_windows` math producing degenerate ranges if the catalog is unusually shaped — print the windows and adjust.

- [ ] **Step 4: Full-suite sanity (no test changes)**

```bash
uv run pytest -q 2>&1 | tail -3
```
Expected: still 66 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/run_harness.py
git commit -m "feat(examples): Stage-3 runnable harness with --smoke dev mode

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Full acceptance run

**Files:** none new — just runs `examples/run_harness.py` without `--smoke`.

- [ ] **Step 1: Pre-flight check**

Verify the smoke run from Task 8 actually produced a sane Markdown summary. Look at its `summary.md` — does the verdict text read coherently? Are the cost-stress and DSR values present and finite? If anything looks degenerate, fix the runnable before committing 50 minutes to the full run.

- [ ] **Step 2: Run the full harness**

```bash
cd /home/tahseen-dar/Projects/trading-ml
time uv run python examples/run_harness.py
```
Expected: ~45–60 minutes. Output: a new `results/harness_<timestamp>/` with the four artifacts AND the Markdown summary printed at the end.

Capture from the summary:
- Winner combo (which `top_n` / `lookback` / `skip` won)
- Winner median window Sharpe + window range
- 5th–95th percentile of all combo-window Sharpes
- CAGRs across the 4 slippage levels
- Deflated Sharpe + p-value

Sanity checks:
- Total runs in `walk_forward.csv` should equal `18 × 5 = 90`.
- Total runs in `cost_stress.csv` should equal `4`.
- All metric values should be finite (no NaNs except for degenerate edge cases).
- The Markdown verdict should be readable English with concrete numbers (not template placeholders).

If any of those fail, investigate before declaring the task done. Do NOT alter the harness to make a failing run look better.

- [ ] **Step 3: Commit nothing (this task produces output, not code)**

The results directory is gitignored. There's no commit for this task — it's the acceptance evidence for Stage 3.

- [ ] **Step 4: Save the headline numbers for Task 10's README update**

Note the actual numbers from the run (winner config, p-value, CAGR at each slippage) so the README update in Task 10 can cite them honestly. You may keep these in a scratch buffer / note for the next task to use.

---

## Task 10: README + Stage 3 wrap-up

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the Status section**

Read the current `README.md`. Update the Status section to mark Stage 3 complete, citing the actual numbers from Task 9. The block should read like:
```markdown
## Status

- [x] Stage 0 — Project skeleton + Nautilus smoke backtest
- [x] Stage 1 — Alpaca → Nautilus ParquetDataCatalog ingestion pipeline
  - [x] Data pipeline built and unit-tested
  - [x] Live catalog populated (12 tickers; SPY ~2018-11, sector ETFs ~2020-07 → today on free-tier IEX). Cross-check passes at 1.5% tolerance (see design spec §7).
- [x] Stage 2 — Momentum top-N strategy + tearsheet
  - [x] Pure ranker + Nautilus Strategy (monthly top-3 by 6m-skip-1 momentum) + quantstats tearsheet
  - [x] Acceptance backtest: `uv run python examples/backtest_momentum.py`
  - On the ~2020-07→2026 window the simple momentum strategy underperformed SPY buy-and-hold (≈9% vs ≈21% CAGR) — the expected honest baseline per the design spec; the milestone is a credible, reproducible tearsheet, not beating SPY.
- [x] Stage 3 — Honesty harness (walk-forward × param grid + Deflated Sharpe + cost stress)
  - [x] `analysis/{runspec,deflated_sharpe,metrics,runner,walk_forward,costs,harness_report}.py`
  - [x] Runnable: `uv run python examples/run_harness.py` (full) or `--smoke` (~6 min dev mode)
  - [x] Acceptance run: 90 walk-forward + 4 cost-stress backtests over the full catalog. Winner: `<INSERT actual winner combo from Task 9>`. Deflated Sharpe p-value: `<INSERT actual p>` (verdict: `<INSERT "cannot reject H0" or similar>`). Strategy <improves / degrades / inverts> as costs rise.
- [ ] Stage 4 — Sim-live loop
- [ ] Stages 5–8 — see design spec
```

REPLACE the `<INSERT …>` placeholders with the ACTUAL values from Task 9's full run summary.md. Do NOT leave any literal `<INSERT …>` in the committed file.

- [ ] **Step 2: Add a new entry to the "## Backtest" section (or create one)**

If the README has a `## Backtest` section, APPEND below the existing `examples/backtest_momentum.py` block:
```markdown
### Honesty harness

```bash
# Stage 3: walk-forward × parameter grid + Deflated Sharpe + cost stress
uv run python examples/run_harness.py            # full battery (~45-60 min)
uv run python examples/run_harness.py --smoke    # tiny grid (~6 min) for dev iteration
```
Writes `walk_forward.csv`, `cost_stress.csv`, `summary.json`, `summary.md` to `results/harness_<timestamp>/`.
Prints the 5-section Markdown verdict (time stability / parameter robustness / cost survival / Deflated Sharpe / bottom line).
```

- [ ] **Step 3: Run the full pytest as the acceptance gate**

```bash
uv run pytest -q 2>&1 | tail -3
```
Expected: 66 passed (38 from Stages 0–2 + 28 from Stage 3). Zero failures.

- [ ] **Step 4: Final commit**

```bash
git add README.md
git commit -m "docs: README — Stage 3 complete, honesty-harness documented

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: Stage 3 done**

The spec milestones are met:
- `analysis/walk_forward.py`, `analysis/deflated_sharpe.py`, `analysis/costs.py` all exist and are unit-tested.
- The runnable `examples/run_harness.py` produces a credible Markdown verdict + JSON dump from the real catalog.
- Stage 3's bottom-line question — "is the edge real or overfit?" — is answered with concrete numbers in the Markdown summary.

Stage 4 (sim-live loop) is the next plan.

---

## Self-review

**1. Spec coverage**

| Spec section / requirement | Implementing task(s) |
|---|---|
| §1 Goal: walk-forward + DSR + cost stress producing Markdown + JSON | Tasks 5, 2, 6, 7, 8, 9 |
| §1 Milestone: 5-section verdict (time stability / param robustness / cost survival / DSR / bottom line) | Task 7 (rendering) + Tasks 8/9 (real run) |
| §2 Architecture: pure modules + one runner | Tasks 1–7 each implement one boundary |
| §2 Architectural commitments (runner is sole engine consumer, RunSpec is unit, pure modules pure) | Tasks 1, 3, 4 (only Task 4 imports BacktestEngine); §1–§7 file-by-file responsibility |
| §3 Param grid: 3 × 3 × 2 = 18 combos | Task 8 `DEFAULT_PARAM_GRID` |
| §3 5 expanding walk-forward windows | Task 8 `_build_windows` + `n_windows=5` default |
| §3 Per-run metrics (cagr/sharpe/max_dd/total_return/final_equity/n_days/n_trades + equity_curve) | Task 4 `_compute_metrics`; Task 1 RunResult schema |
| §3 Deflated Sharpe (no-skew/no-kurt simplification, public function signature) | Task 2 |
| §3 Cost stress at {0, 0.25, 0.5, 1.0} | Task 6 + Task 8 `DEFAULT_SLIPPAGE_PROBS` |
| §3 5-section Markdown report + JSON dump | Task 7 `render_markdown` + `write_report` |
| §4 Tests (DSR, RunSpec, runner, walk_forward, costs, harness_report, metrics refactor) | Tasks 1–7 (each TDD) |
| §5 Runnable entry with `--smoke` flag | Task 8 |
| §6 Risks: ~5.5 yr data window | Acknowledged in spec; nothing to do in plan |
| §6 Risks: portfolio.equity warns | Acknowledged; no change |
| §6 Risks: DSR simplification | Documented in `analysis/deflated_sharpe.py` docstring (Task 2) |
| §6 Risks: 18 combos is a deliberate compromise | Documented in spec; nothing to do |
| §6 Risks: winner picked by median window Sharpe | Implemented in Task 7 `summarize_walk_forward` |
| §6 Risks: 45–50 min runtime + `--smoke` mode | Implemented in Task 8 |
| §7 Out-of-scope (CPCV, Bayesian opt, train/test split, parallel exec, HTML report, multi-strategy compare) | None implemented (verified by file inventory: no parallelism, no CPCV, no Bayesian opt, no train/test split, summary is Markdown not HTML, runner is per-strategy) |

No spec section is uncovered.

**2. Placeholder scan**

- The README in Task 10 deliberately contains `<INSERT …>` placeholders that Task 9's run-output replaces. These are NOT placeholders in the plan-failure sense — they are explicit instructions to the implementer to fill in the actual run numbers before committing, and Task 10 Step 1 says "Do NOT leave any literal `<INSERT …>` in the committed file."
- No `TBD`, no `implement later`, no `similar to Task N`, no "add appropriate error handling" generalities.

**3. Type consistency**

- `RunSpec` defined in Task 1; consumed identically in Tasks 4, 5, 6, 8.
- `RunResult.metrics` keys (`cagr`, `sharpe`, `max_drawdown`, `total_return`, `final_equity`, `n_days`, `n_trades`) are set in Task 4 `_compute_metrics` and asserted on in test_runner (Task 4), test_walk_forward (Task 5), test_costs (Task 6), test_harness_report (Task 7).
- `walk_forward(...)` keyword-only signature in Task 5 is called identically in Task 8.
- `cost_stress(...)` keyword-only signature in Task 6 is called identically in Task 8.
- `deflated_sharpe(sharpes, n_obs) -> dict` signature in Task 2 matches the call in Task 8.
- `summarize_walk_forward(wf_df, param_cols) -> dict` in Task 7 matches Task 8's usage indirectly (Task 8 calls `write_report(...)` which calls `summarize_walk_forward` internally).
- Metric helpers renamed from `_cagr/_sharpe/_max_drawdown` (Task 3) → `cagr/sharpe/max_drawdown` consistently throughout tearsheet.py + runner.py.
