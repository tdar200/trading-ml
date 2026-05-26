# Stage 3 — Honesty Harness: Design Spec

**Date:** 2026-05-26
**Status:** Approved through brainstorming; pending user spec-review before implementation planning.
**Parent spec:** `2026-05-26-trading-ml-design.md` (architecture, philosophy, full roadmap)
**Stage:** 3 of 8 — the harness that lets us answer "is the edge real?" with a number.

## 1. Goal & milestone

Build reusable infrastructure that evaluates any Nautilus strategy across multiple time windows and a parameter grid, applies the Deflated Sharpe Ratio (Bailey & López de Prado) to penalize multiple-testing, and stress-tests the chosen configuration under elevated trading costs. The harness is strategy-agnostic — Stage 5's ML strategy will plug into it unchanged.

**Milestone (from parent §5):** A `analysis/` package + a runnable that, given the Stage-2 strategy, produces a written verdict — Markdown summary + JSON dump — answering three blunt questions:
1. **Time stability:** is the Sharpe consistent across windows, or concentrated in one period?
2. **Parameter robustness:** is the "winner" an island in a sea of bad combos (red flag) or part of a cluster of similarly-good configs (credible)?
3. **Cost survival:** does the winner survive 2× and 4× the baseline slippage?

The milestone is having a *reproducible, honest answer to those three questions* — not finding alpha. The Stage-2 strategy underperforms SPY at baseline; the harness will most likely confirm that the underperformance is stable rather than overturn it. That is a valid Stage-3 outcome.

## 2. Architecture

```
analysis/
  runspec.py           # Pure dataclass: one backtest configuration
  runner.py            # Execute one RunSpec → metrics + equity curve
  walk_forward.py      # Run grid × windows; aggregate per-combo / per-window
  deflated_sharpe.py   # Pure Bailey-LdP DSR math
  costs.py             # Cost-stress runs of the winner config
  harness_report.py    # WF + DSR + cost stress → JSON + Markdown summary
  tearsheet.py         # (existing — unchanged)

examples/
  run_harness.py       # Runnable: full Stage-3 battery on Stage-2 strategy

results/
  harness_<timestamp>/
    walk_forward.csv   # combo × window × metric long-format table
    cost_stress.csv    # cost-multiplier × metric
    summary.json       # machine-readable everything
    summary.md         # human-readable 1-page verdict

tests/
  test_deflated_sharpe.py   # pure-math TDD
  test_runspec.py
  test_runner.py            # synthetic-catalog integration
  test_walk_forward.py      # tiny grid × windows on synthetic data
```

### Architectural commitments

1. **`runner.py` is the only file that constructs a `BacktestEngine`.** All other harness modules consume the runner's outputs. This keeps Nautilus-specific concerns isolated and the rest of the harness testable without engine setup.
2. **`RunSpec` is the unit of execution.** A pure dataclass — strategy class, strategy-config kwargs, date window, slippage probability, starting cash. Nothing else.
3. **Strategy-agnostic.** Wiring a future strategy in means passing a different `strategy_class` + kwargs to RunSpec; the rest of the harness doesn't change.
4. **Pure math lives in pure modules.** `deflated_sharpe.py` has no IO; `walk_forward.py` aggregates already-computed metrics; `costs.py` is a thin wrapper over `runner.py`. Each is independently testable.
5. **No new tests for `examples/run_harness.py`.** It's a runnable demo (same convention as `examples/smoke.py` and `examples/backtest_momentum.py`); exercised manually for the acceptance run.

## 3. Key behaviors

### Parameter grid (default, overridable via CLI flags)

| Parameter | Values | Rationale |
|---|---|---|
| `top_n` | {2, 3, 5} | The "concentration" knob: how many names to hold |
| `lookback_days` | {63, 126, 252} | 3-month / 6-month / 12-month momentum windows |
| `skip_days` | {0, 21} | Raw vs. classic skip-recent-month pattern |

**Total: 3 × 3 × 2 = 18 combos.** Small enough that the DSR penalty is meaningful but not crushing; large enough to surface obvious overfitting risk.

### Walk-forward windows

**5 expanding test windows** spanning the post-warm-up era of the catalog (~2021-04 → 2026-05). Each window:
- Starts at the catalog's beginning so the strategy has ≥147 bars (lookback 126 + skip 21) of warm-up before its first rebalance.
- Has a contiguous, non-overlapping **test** segment of approximately 12-15 months.
- Metrics are computed only on the test segment (warm-up returns are zero by design and excluded).

Expanding (not rolling) windows are chosen because the strategy has no learned parameters — there's nothing to "train" — so the goal is purely to evaluate Sharpe stability across distinct out-of-sample periods. Expanding gives every window the same warm-up history, simplifying interpretation.

### Per-run metrics (computed in `runner.py`)

- `cagr` (annualized)
- `sharpe` (annualized, ddof=0 for consistency with `analysis/tearsheet.py`)
- `max_drawdown`
- `total_return`
- `final_equity` (mark-to-market via the same `_equity_history` mechanism from Stage 2)
- `n_days`
- `n_trades` (count of `OrderFilled` events)
- `equity_curve: pd.Series` (the full daily equity series — for plotting / downstream analysis)

### Deflated Sharpe (`analysis/deflated_sharpe.py`)

Pure function, no IO:

```python
def deflated_sharpe(
    sharpes: list[float],   # all in-sample SRs from the trial set (the 18 combos)
    n_obs: int,             # number of return observations the winner's SR was computed over
) -> dict:
    """Bailey & López de Prado (2014) Deflated Sharpe Ratio.

    Returns: {
        'observed_sharpe': float,   # max(sharpes) — the winner
        'n_trials': int,            # len(sharpes)
        'expected_max_sharpe': float,  # E[max SR] under H0: all SRs are iid N(0, var)
        'deflated_sharpe': float,   # the haircut-applied SR (z-score)
        'p_value': float,           # P(SR_obs > 0) given H0
    }
    """
```

The math (from the paper):
- `E[max_n_trials] ≈ sqrt(Var(SR_trials)) × ((1−γ)×Φ⁻¹(1 − 1/N) + γ×Φ⁻¹(1 − 1/(N·e)))` where γ is the Euler-Mascheroni constant ≈ 0.5772.
- `σ_SR_obs ≈ sqrt((1 − γ₃·SR_obs + (γ₄−1)/4 · SR_obs²) / (n_obs − 1))` where γ₃, γ₄ are the skew/kurtosis of the winner's returns. For the harness, we use the simplification `σ ≈ sqrt((1 + 0.5·SR² )/(n_obs − 1))` (the "no skew/kurt adjustment" form), which the paper notes is acceptable when the return distribution is roughly normal — fine for daily ETF returns.
- `deflated_z = (SR_obs − E[max]) / σ_SR_obs`
- `p_value = Φ(deflated_z)` — probability the observed SR is "real" (> 0) under H0.

Unit tests cover: known canonical examples (e.g. 1 trial with SR=1 → DSR ≈ 1, p ≈ Φ(1) ≈ 0.84); 100 trials with mean-zero gaussian SRs → strong depression; edge cases (empty list, single trial, all-equal SRs).

### Cost stress (`analysis/costs.py`)

After the walk-forward selects a "winner" combo (highest median Sharpe across windows), re-run the winner's strategy across the **full catalog** at each of:
- `prob_slippage = 0.00` (no slippage — sanity baseline)
- `prob_slippage = 0.25` (Stage-2 default, ≈ 0.5 bps avg)
- `prob_slippage = 0.50` (~ 1 bps avg)
- `prob_slippage = 1.00` (deterministic 1 tick, ~ 2 bps)

Report CAGR / Sharpe / max-DD for each.

### Report (`analysis/harness_report.py` + `summary.md`)

The Markdown summary, written in plain English, contains:

```
# Honesty Harness — <strategy_name> — <timestamp>

## 1. Time stability
- Median window Sharpe (winner): X
- Range across 5 windows: [min, max]
- Verdict: stable / concentrated in one period / wildly unstable

## 2. Parameter robustness
- Best combo: <top_n=X, lookback=Y, skip=Z>
- Sharpe rank of best: 1/18 (median across windows)
- 5th–95th percentile of combo Sharpes: [a, b]
- Verdict: cluster of good configs / lone island

## 3. Cost survival
- CAGR at 0×: A
- CAGR at 1× (default): B
- CAGR at 2×: C
- CAGR at 4×: D
- Verdict: edge robust to costs / degrades sharply / negative at realistic costs

## 4. Deflated Sharpe (multiple-testing penalty)
- Observed Sharpe (winner, full catalog): S
- Expected max-Sharpe under H0 (18 trials): M
- Deflated Sharpe: D
- p-value: P
- Verdict: P > 0.05 → cannot reject "Sharpe is zero" hypothesis

## 5. Bottom line
One-paragraph honest summary.
```

The same data is dumped to `summary.json` for machine consumption (future Streamlit dashboard in Stage 7 can read this).

## 4. Testing

| Test | Approach | Verifies |
|---|---|---|
| `test_deflated_sharpe.py` (5-8 tests) | TDD with hand-computed examples | Pure math correctness; edge cases (empty, single trial, large N) |
| `test_runspec.py` (2-3 tests) | Plain assertions | Dataclass invariants (date order, positive cash, non-empty universe) |
| `test_runner.py` (2-3 tests) | Synthetic 3-ticker catalog (similar to Stage-2 fixture) | RunSpec → real metrics returned, sane values |
| `test_walk_forward.py` (1-2 tests) | Tiny grid (2 combos × 2 windows) on synthetic data | Output dataframe shape; per-combo aggregation |

Total: ~10-15 new tests on top of the 38 from Stages 0-2. None hit the real Stage-1 catalog — synthetic data only. The runnable `examples/run_harness.py` exercises the real catalog manually during the Stage-3 acceptance run.

## 5. Runnable entry point (`examples/run_harness.py`)

```bash
uv run python examples/run_harness.py
```

- Loads the Stage-1 catalog.
- Generates RunSpecs for the 18-combo × 5-window grid (the cross-product).
- Runs them sequentially via `runner.run(spec)`, capturing metrics.
- Selects the winner combo (highest median window Sharpe).
- Cost-stresses the winner at 4 slippage levels (additional 4 backtests).
- Writes `walk_forward.csv`, `cost_stress.csv`, `summary.json`, `summary.md` to `results/harness_<timestamp>/`.
- Prints the Markdown summary to stdout.

Total runtime estimate: 18 × 5 + 4 ≈ **94 backtests × ~30s = 45-50 minutes** on commodity hardware. Acceptable for an offline analysis tool.

## 6. Risks & open items

1. **Free-tier IEX data window is only ~5.5 years.** With 5 walk-forward windows we get ~12-15 months each — short, but enough to surface gross instability. Acknowledged limitation; richer history requires Stage-1's paid-SIP upgrade lever.
2. **`portfolio.equity(venue)` "warns once" for unpriced instruments.** During the early bars of each window before bars stream in, this may emit log spam. Mitigation: same as Stage-2 — log level WARNING and accept the noise. If it becomes overwhelming, can introduce a per-RunSpec log filter.
3. **The 0.5×0.5×n_obs simplification of the DSR variance term** ignores higher-moment effects. For daily ETF returns this is fine; for fat-tailed or skewed distributions it understates the variance and overstates significance. The harness documents this explicitly in the report.
4. **18 combos is a small grid.** A larger grid would catch more overfitting but also makes the DSR penalty crushing. 18 is a deliberate compromise between coverage and statistical honesty. Documented in the spec.
5. **The walk-forward selects the winner by median in-window Sharpe**, not by best in-sample then test out-of-sample. This is appropriate for a parameter-free strategy where there's no "fitting" step; it characterizes which params are robust across regimes. For a future ML strategy with a fit step, the harness would need a true train/test split — that extension is Stage 5+ territory.
6. **Sequential runtime ~45-50 minutes.** A single CLI flag could opt into a smaller "smoke" grid (`--smoke`: 2 combos × 2 windows + 2 cost levels ≈ 6 min) for development iteration without sacrificing the full mode. Including as a Task in the plan.

## 7. Out of scope (deliberately, do not add)

- Combinatorial purged cross-validation, embargo periods, the rest of the López de Prado AFML stack. (Stage 5+ once we have an ML model with predictions worth purging.)
- Hyperparameter optimization via Bayesian search / Optuna. (Premature; needs Stage 5's ML model first.)
- Per-window train/test split. (Strategy has no fittable parameters; would be ceremony.)
- Parallel execution. (Adds Nautilus-engine-state complexity not worth the runtime savings.)
- An HTML harness report. (The existing per-run tearsheet HTML stays; the harness output is JSON + Markdown.)
- Multi-strategy comparison. (One strategy per harness run; comparing strategies is a Stage 7 dashboard concern.)
- Anything intraday or higher-frequency than daily.
- Bayesian DSR / PBO / Probability of Backtest Overfitting beyond the basic DSR formula.

---

*This spec is the Stage-3 contract. Implementation plan to be written next via `superpowers:writing-plans`.*
