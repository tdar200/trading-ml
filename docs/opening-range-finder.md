# Opening Range Finder

This CUDA-only finder tests selective long/short opening-range breakout
portfolios across the 30-stock scalping basket.

## Design

- exact 15, 30, and 60-minute ranges anchored at 09:30 US/Eastern;
- sessions with missing opening bars are ineligible;
- long, short, and bidirectional structures;
- 0/0.25/0.5 ATR breakout distance;
- 1.0/1.5/2.0 relative-volume gates based only on prior sessions at the same
  time of day;
- optional same-direction SPY confirmation;
- causal 24-bar ATR targets and stops;
- optional fresh-cross confirmation to reject stale post-breakout entries;
- 5/10/15/30/60/120-minute maximum holding periods;
- at most one entry per stock per session;
- next-bar fills, mandatory end-of-day flattening, selection at 2.5 bp per side,
  and 5 bp per-side stress.
- every discovery window must be profitable with strictly positive worst-window Sharpe;
- at least 15 trades are required in every discovery window;
- the selected winner writes an execution-level ledger with exit reasons, MFE/MAE, cost drag, and available quote state;
- target and stop fills use each bar's high/low path, with conservative stop-first handling when both are touched;
- selection is nested: an orientation segment, a five-session purge, four chronological validation windows, then a sealed 20% holdout;
- PBO, Deflated Sharpe, and a moving-block bootstrap explicitly penalize multiple testing and serial dependence;
- an optional causal prior-session volatility/trend regime gate, pooled context confirmation, event exclusion, and trailing-shadow-P&L circuit breaker can suppress entries;
- SIP quote coverage, point-in-time data, event coverage, sufficient history and breadth remain mandatory deployment gates.

CUDA creates the signed signals, screens 324 entry structures, and validates
2,592 fully specified finalists after retaining 48 family-diverse entries. Discovery ranking uses four chronological,
shared-capital, equal-weight portfolio windows. The final 20% is opened only after one candidate survives every discovery window. A run writes `frozen_candidate.json`, `evaluation_cost_stress.csv`, a trade ledger, and a gate-by-gate historical qualification. Because previously inspected dates are exposed, such a run remains research evidence and cannot self-certify for deployment.

Run:

```bash
.venv/bin/python examples/find_opening_range_strategies.py --tickers SOXS --device cuda

# Optional causal context, regime, and scheduled-event controls
.venv/bin/python examples/find_opening_range_strategies.py --tickers SOXS \
  --context-tickers SOXX,SMH --context-relation inverse \
  --min-regime-strength 0.5 --events data/events.csv
```

The same evidence is shown under **Backtest Lab → Opening Range Finder**.

## Verdict semantics

`research_verdict=PASS` means the sealed backtest also passed PBO, Deflated Sharpe, bootstrap, excess-return, drawdown, and cost-stress rules. It is not permission to trade. `verdict=HISTORICAL_PASS_PAPER_REQUIRED` additionally requires every historical data-quality gate. True deployment approval is produced only by `examples/qualify_profitability_strategy.py` after the unchanged candidate accumulates at least 126 broker-reconciled forward paper sessions and 100 fills. Any parameter change creates a new fingerprint and restarts qualification.

## September 12, 2026 acceptance result

The run used 1,538,750 five-minute bars from 30 stocks. The discovery winner
was a short 15-minute opening-range breakout 0.5 ATR beyond the range, with
relative volume of at least 1× and same-direction SPY confirmation. It used a
1.5 ATR target, 0.5 ATR stop, 30-minute maximum hold, and one trade per day.

| Untouched portfolio metric | Result |
|---|---:|
| Return at 1 bp/side | -2.92% |
| Sharpe | -4.780 |
| Maximum drawdown | -2.94% |
| Trades | 1,563 |
| SPY return | +11.96% |
| Excess return | -14.88% |
| Return at 5 bp/side | -6.87% |

**Verdict: REJECT.**

The strategy was materially more selective and controlled drawdown better than
the generic scalper, but it did not survive the untouched holdout. Evidence is
stored in `results/opening_range_finder_20260912T135021Z/`.

This remains bar-based research. IEX is a single-exchange feed and lacks
consolidated bid/ask state. A future candidate must pass SIP quote-aware
backtesting and paper trading before deployment.
