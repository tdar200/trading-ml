# Support/Resistance Confluence Finder

This CUDA finder combines beta-residual cross-sectional rankings with six
mechanical level families: previous regular-session high/low, previous-week
high/low, available IEX premarket proxy, 30-minute opening range, causal
24-bar swing range, and expanding intraday VWAP bands.

It searches breakout-and-retest, rejection, and failed-breakout behaviors over
the last three completed bars, both per level and across an `any_level`
ensemble.
Signals are determined from completed bars, enter at the next bar open, and
hold a dollar- and beta-neutral long/short book. Stops are 0.5 causal ATR and
targets are 2R. If both occur inside one five-minute bar, the stop is assumed
first. Costs are charged on entry and exit.

```bash
.venv/bin/python examples/find_support_resistance_strategies.py --smoke
.venv/bin/python examples/find_support_resistance_strategies.py
```

The complete grammar contains 1,680 candidates. Four chronological discovery
windows determine the winner. The final 20% is reported as a reused
post-selection evaluation period—not an untouched holdout—because prior
strategy experiments already exposed those dates. The program always emits a
deployment `REJECT`; only genuinely new future data can clear that limitation.

The cached IEX extended-hours data usually begins around 08:00 ET and is sparse.
`premarket_proxy` therefore does not claim to represent the complete 04:00–09:30
premarket range. SIP/NBBO quotes, spread-aware fills, paper trading, borrow
availability, and operational risk controls remain required for deployment.
