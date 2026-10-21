# Cross-Sectional Intraday Finder

This finder searches for stock-selection edge instead of market direction. At
each eligible completed five-minute bar it ranks each stock's return after
subtracting its causal rolling beta exposure to SPY. It buys the strongest and
shorts the weakest names for momentum candidates; reversal candidates invert
that ranking. Long and short gross exposure are each 50%.

Each candidate makes at most one portfolio entry per day at a fixed scheduled
time. Entries fill at the next bar open and exits fill at a later bar open in
the same regular session. Selected weights are projected to zero net dollars
and zero rolling SPY beta, then normalized to 100% gross exposure. The
simulation charges the configured cost on both entry and exit and rejects
signals lacking sufficient relative volume, cross-sectional dispersion, or a
calm-enough market regime.

Run the complete CUDA search:

```bash
.venv/bin/python examples/find_cross_sectional_strategies.py
```

Use `--smoke` for a six-stock, two-candidate end-to-end check. The complete
grammar has 972 candidates, four non-overlapping discovery windows, and a final
20% holdout that is not read during ranking. The selected candidate is repeated
at 1, 2.5, and 5 basis points per side.

`PASS` requires positive performance in at least 75% of discovery windows, a
positive holdout with Sharpe above 1 and drawdown below 10%, at least 200 stock
round trips, profitability at 5 bps per side, mean absolute beta exposure no
greater than 0.02, no stock exceeding 25% of absolute P&L contribution,
combinatorial probability of backtest overfitting no greater than 20%, and at
least 95% Deflated Sharpe confidence.

Bars do not contain NBBO quotes. Therefore even `PASS` means research-qualified,
not live-qualified. Quote-aware replay, paper trading, reconciliation, and live
risk controls remain mandatory before capital deployment.
