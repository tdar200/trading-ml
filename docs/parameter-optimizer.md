# Parameter Optimizer

Parameter Optimizer is the project's cTrader-style optimization tool. It uses
the same NautilusTrader event backtester as single runs and the honesty
harness, so optimization cannot quietly switch to a more flattering fill
model.

This optimizes settings for an already implemented strategy. It does not invent
new indicators, entry/exit rules, or strategy structures; that broader capability
is implemented separately by the Strategy Finder layered above this optimizer.

## Search engines

- **Exhaustive grid** evaluates every value from every finite parameter range.
- **Adaptive TPE** uses a seeded Bayesian sampler to focus a fixed budget on
  promising regions. It is the preferred mode when the full Cartesian product
  is too large.
- **Hybrid tensor + Nautilus** (the default) evaluates the complete grid as
  batched portfolio tensors, selects strong candidates from separated
  parameter regions, and validates only those finalists with the canonical
  event engine.

Both modes evaluate each unique configuration across contiguous,
non-overlapping time windows. A candidate is ineligible if any window has an
invalid metric or too few trades.

The ranking score is deliberately transparent:

```text
median window Sharpe
  - 0.25 × interquartile range of window Sharpes
  - 0.10 × absolute worst-window drawdown
```

The output also reports worst-window Sharpe, median CAGR, total trades, and a
Pareto frontier. The score orders the table; it is not a deployment verdict.

## Run it

```bash
# Fast end-to-end check (2 configurations × 2 windows)
uv run python examples/optimize_parameters.py --smoke

# Default hybrid: full tensor pre-screen, 20 diverse Nautilus finalists
uv run python examples/optimize_parameters.py --algorithm hybrid --finalists 20 --windows 4

# Adaptive search
uv run python examples/optimize_parameters.py --algorithm tpe --trials 100 --windows 4

# Every default combination
uv run python examples/optimize_parameters.py --algorithm grid --windows 4
```

Each run writes `rankings.csv`, `window_evaluations.csv`, and `summary.json`;
hybrid runs also write `prescreen.csv`. Artifacts live under
`results/parameter_optimizer_<timestamp>/`. The same feature is available in
Dashboard → Backtest Lab → Parameter Optimizer.

## GPU plan and current boundary

The project lock intentionally pins the portable CPU PyTorch wheel. On a CUDA
machine, activate the CUDA wheel in the existing environment and invoke Python
directly so `uv run` does not restore the CPU lock:

```bash
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 torch --reinstall
.venv/bin/python examples/optimize_parameters.py --algorithm hybrid --device cuda \
  --finalists 20 --windows 4
```

This was acceptance-tested on the local NVIDIA RTX 3050 Laptop GPU: 165 tensor
combinations on CUDA followed by 80 canonical Nautilus runs (20 finalists × 4
windows). NautilusTrader's event-driven simulation itself remains CPU-bound and
cannot accelerate Python orchestration around that engine in a meaningful way.

The implemented two-stage pipeline:

1. Computes signals and approximate portfolio paths for all parameter sets as
   batched PyTorch tensors on CUDA when available, with an identical CPU fallback.
2. Selects the strongest and most diverse regions, then rerun every finalist
   through the canonical Nautilus engine with costs and independent windows.
3. Has a synthetic winner-parity gate against Nautilus plus batching, cost,
   warm-up, device-failure, alignment, and diversity tests.

The CUDA wheel is deliberately a local activation rather than a lockfile default,
so CPU-only machines retain a working install. Optional future CUDA model training
for ML strategies is separate from this pre-screen.

This hybrid gives useful GPU speed without replacing realistic order/fill
simulation with an unverified approximation. `accelerator_status()` exposes
the active capability in both the CLI and dashboard.

## Backtest safety checks

The canonical runner now rejects:

- naive/ambiguous datetimes;
- missing data for a configured universe or benchmark;
- duplicate or out-of-order timestamps;
- bars stored under the wrong ticker key;
- bars without a corresponding instrument definition.

Optimizer windows are independent rather than the older harness's overlapping
expanding windows. Full-suite tests cover deterministic synthetic execution,
date boundaries, live-simulation parity, cost stress, metrics, search-space
enumeration, adaptive-search repeatability, eligibility rules, robust ranking,
and Pareto selection.
