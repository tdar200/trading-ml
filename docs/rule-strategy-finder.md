# Rule Strategy Finder

A complete reference workflow for searching **instrument × timeframe × entry
trigger × trend filter × volume confirmation × holding period**. Each candidate
trades one instrument, long-only, with a single fractional-share position and a
configurable capital allocation, stop and target. This is separate from the
existing signal-consensus and portfolio finders.

## Run

Open the **Rule Strategy Finder** tab in the main dashboard, or run the dedicated
application:

```bash
.venv/bin/python -m streamlit run dashboard/finder_app.py
```

Select local market data or the explicitly labelled synthetic demo. The demo is
for software acceptance and never constitutes market evidence. Intraday data is
loaded from the selected directory using the existing loader; daily bars come
from `data/catalog`. Nothing is downloaded automatically.

```bash
# Small deterministic acceptance run
.venv/bin/python examples/discover_rules.py --demo --periods 7 --trends 0 --holds 4

# Search instruments and timeframes together, if local coverage exists
.venv/bin/python examples/discover_rules.py --instruments AAPL MSFT \
  --timeframes 15m 1h --cost-bps 3

# Verify the exact saved candidate, data, costs, trades and equity
.venv/bin/python examples/discover_rules.py --replay results/rule_discovery/RUN_ID
```

## Rules and execution

- RSI uses rolling average gains/losses and triggers on a cross above 30.
- Moving-average entry triggers when close crosses above its rolling mean.
- Breakout triggers when close exceeds the preceding rolling high.
- Optional trend filter requires close above its moving average.
- Optional relative volume compares completed-bar volume with the preceding
  rolling average, excluding the current bar from that average.
- Signals use completed bars. Entries occur at the next observed open, including
  across missing bars or overnight. There is no forward-fill or invented bar.
- A holding-period exit occurs at the open after the configured number of bars.
- Stops that gap through their level fill at the open. Other stops fill at the
  stop; targets fill at their limit. If a bar touches both, stop wins.
- Positions are liquidated at the last evaluation close. Exit timestamps label
  the bar, not an inferred exact intrabar execution time.
- Per-side costs apply to entry and exit notionals. Entry sizing reserves its
  fees, never borrows cash, and applies the configured allocation.

Search and replay use the same simulator. There is no accelerated approximation
in this version. The grid is capped at 2,000 candidates.

## Chronological validation

At least 80 common session dates are required. Date boundaries are shared by
all instruments and timeframes, with the final 20% reserved for evaluation.
The development interval has three expanding training/outer-test splits.
Each split selects the instrument, timeframe and rule exclusively on earlier
data and then evaluates that selection on the next interval. Positions restart
flat at each boundary; earlier prices remain available solely for warm-up.

Training ranks candidates by daily Sharpe minus absolute maximum drawdown,
subject to minimum completed-trade activity. The final candidate is selected
using all development data. The final period opens only if development return
is positive and all three outer windows have positive net returns and the
required activity. Otherwise the report contains a development replay and the
final period stays sealed. Outer results describe the selection procedure;
they do not imply every fold used the final candidate.

The reserved period is isolated **within this run**. Historical data may already
have been examined in prior research. Repeated searches do not make that history
new again. All outputs keep `deployment_approved=false`; a positive historical
result is only a candidate for separate forward research.

## Saved runs and failures

Each run has a unique directory, atomic progress state, a report containing the
whole candidate grid and folds, and Parquet snapshots with SHA-256 fingerprints.
Replay uses those snapshots instead of current market files and rejects altered
reports, altered data, or a changed simulator source file. These hashes detect
accidental drift; they are not signatures against a malicious editor.

The UI binds every panel to the selected run. Failed/cancelled jobs cannot show
an older winner. Searches run in a background thread, can be cancelled, and
survive browser reloads. A server restart interrupts unfinished work; it does
not silently resume it. Completed runs remain available. One search runs per
server at a time. `TRADING_DISCOVERY_RESULTS` can override the result directory.

## Tests

```bash
uv sync --group dev
.venv/bin/python -m playwright install chromium
.venv/bin/python -m pytest tests/test_rule_discovery.py -q
.venv/bin/python -m pytest tests/e2e/test_rule_discovery_browser.py -q
```

The browser suite launches an isolated Streamlit server and exercises search,
custom costs, trades, exact replay, JSON/CSV downloads, reload, invalid inputs,
missing data and cancellation. Screenshots and Playwright traces are written
under `results/rule_discovery_e2e/`. Backend tests verify hand-calculated cash
accounting, stop ambiguity, gaps, future-data isolation, chronology, no-trade
rejection, and snapshot integrity.

## Boundaries

This version supports single-instrument long-only research on 15-minute, hourly
and daily US-equity bars. It does not yet implement shared-capital baskets,
short borrow, broker fills, tick-level execution, GPU acceleration, resumable
search checkpoints or forward paper trading. Fixed bps costs are a model, not
quote-derived execution evidence. Corporate-action quality follows the supplied
bars; this finder does not independently verify adjustment consistency.
