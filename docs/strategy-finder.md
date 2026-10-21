# Strategy Finder

Strategy Finder discovers rule structures as well as tuning their parameters.
It is a separate layer above the Parameter Optimizer.

## Implemented grammar

Version 1 searches registered, serializable building blocks:

- signal family: cross-sectional momentum or mean reversion;
- optional absolute-trend filter: disabled, 126 days, or 200 days;
- optional positive-signal gate;
- top-N holdings, lookback, and skip-period parameter grids;
- monthly rebalancing, long-only exposure, and a 2% cash buffer.

The default grammar contains 12 distinct structures. Crossing it with the
default 180-point parameter grid produces 2,160 candidates.

Candidates are plain data such as:

```json
{
  "signal_family": "momentum",
  "trend_filter_days": 200,
  "require_positive": true,
  "top_n": 3,
  "lookback_days": 126,
  "skip_days": 21
}
```

The finder never generates or executes arbitrary Python.

## Validation pipeline

1. Enumerate every permitted structure and parameter combination.
2. Run the complete population through the tensor portfolio screen on CUDA or CPU.
3. Preserve the best representatives from distinct structures.
4. Run every finalist across independent windows with the canonical Nautilus engine.
5. Apply stability, drawdown, and explicit complexity penalties.
6. Export all rankings, per-window evidence, and pre-screen results.

The tensor stage is an approximation used only for triage. Nautilus results are
the source of truth.

## Run

```bash
# Small real-catalog CUDA acceptance
.venv/bin/python examples/find_strategies.py --smoke --device cuda

# Default: 2,160 structural/parameter candidates and 24 finalists
.venv/bin/python examples/find_strategies.py \
  --device cuda --finalists 24 --windows 4
```

Artifacts are written under `results/strategy_finder_<timestamp>/`. The same
workflow is available at Dashboard → Backtest Lab → Strategy Finder.

## Current boundary

This first grammar intentionally covers one portfolio archetype:
cross-sectional monthly rotation. A later grammar can register breakout,
moving-average crossover, intraday, exit-rule, stop, and position-sizing
families. Each addition needs both a tensor implementation and a canonical
Nautilus implementation plus parity tests before it is searchable.

Finder rankings are research evidence, not deployment approval. Candidates
still require untouched holdout evaluation and paper trading.
