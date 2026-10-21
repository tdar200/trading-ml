# Scalping Finder

The Scalping Finder is the intraday branch of Strategy Finder. It searches
rule families and their parameters together, then rejects or accepts the
discovery winner on data that was not used during selection.

## Run

First populate matching 5-minute history for the 50 exchange-traded securities,
the SPY benchmark, and the separate 10-crypto sleeve:

```bash
.venv/bin/python data/ingest_alpaca_intraday.py \
  --asset-class stock --timeframe 5m --since 2024-01-01 --until 2026-09-12
.venv/bin/python data/ingest_alpaca_intraday.py \
  --asset-class crypto --timeframe 5m --since 2024-01-01 --until 2026-09-12
```

Then run the complete search:

```bash
.venv/bin/python examples/find_scalping_strategies.py --device cuda
```

The same workflow is available at **Backtest Lab → Scalping Finder**.

## Search and validation design

The constrained, serializable grammar covers three long-only entry structures:

- mean reversion: close below its rolling mean by a z-score threshold;
- momentum: positive return over a rolling lookback;
- breakout: close above the prior rolling high.

It crosses 3/6/12/24-bar lookbacks, family-specific thresholds, 1/3/6-bar
maximum holding periods, and 5/10/20 bp profit targets and stops. The default
grammar contains 108 entry rules and 972 theoretical entry/exit combinations.
CUDA generates signals, screens every entry rule independently, and runs the
stateful execution validation across 60 instruments: 30 individual
stocks, 10 commodity ETF/proxies, 10 crypto pairs, and 10 other high-volatility
contenders. The median
cross-instrument screen keeps 18 family-diverse entry finalists, which are
crossed with nine exit combinations to produce 162 candidates. Each candidate
is then evaluated in 240 instrument-window cells (60 instruments × 4 windows).

All signals use completed bars and fill at the next bar's open. Entries are
limited to 09:45–15:30 US/Eastern and every position is flat by the close.
The simulator charges a configurable cost on each side because IEX OHLCV bars
do not contain the bid/ask spread, queue position, or fees.

All 60 instruments and SPY are restricted to common US trading-session dates.
Crypto is deliberately sampled only during the same weekday 09:30–16:00
US/Eastern window, so it is comparable with the exchange-traded securities.
The first 80% forms four non-overlapping discovery windows per instrument; the
last 20% of every instrument remains untouched until one generic rule has been
selected. Discovery ranking penalizes unstable Sharpe and drawdown and requires
trades in every instrument-window cell. The winner is tested on all 60 holdouts
at 1, 2.5, and 5 bp per side, always against passive SPY.

A PASS requires all of the following:

- median instrument holdout Sharpe above SPY;
- positive median instrument excess return;
- at least 20 aggregate holdout trades per traded instrument;
- at least 60% of instruments profitable and at least 60% beating SPY;
- positive median instrument holdout return at 5 bp per side.

## September 12, 2026 acceptance result

The acceptance dataset contains 2,125,227 traded regular-session bars across
the 60-instrument basket, aligned with SPY over 490 common sessions from August
26, 2024 through September 11, 2026. Alpaca's free-tier IEX feed supplied the
stock and ETF bars; Alpaca's US crypto feed supplied the crypto bars.

The best discovery candidate was a 6-bar breakout with a 10 bp threshold,
20 bp target, 20 bp stop, and a maximum six-bar (30-minute) hold. It was the
least-bad generic candidate, not a profitable winner:

| Metric | Candidate | Passive SPY |
|---|---:|---:|
| Average instrument holdout return at 1 bp/side | -10.80% | +8.17% |
| Median instrument holdout Sharpe | -2.209 | 1.738 |
| Aggregate holdout trades | 28,425 | — |
| Positive / SPY-beating instruments | 1 / 0 of 60 | — |
| Median instrument return at 5 bp/side | -34.76% | — |

Commodity proxies were the least-negative sleeve at -6.46% average holdout
return, followed by individual stocks at -10.49%, crypto at -12.21%, and other
contenders at -14.66%. AAPL was the only profitable instrument (+4.64%), but it
still trailed SPY by 3.52 percentage points. This is not a deployable edge.

**Verdict: REJECT. Do not deploy this candidate.**

The reproducible evidence is in
`results/scalping_finder_20260912T124116Z/`: aggregate and per-instrument entry
screens, discovery instrument-window evaluations, per-instrument holdouts,
per-sleeve holdouts, aggregate cost stress, and JSON summary.

The complete CUDA-first run took 170.24 seconds, including parquet loading and
report generation. Its 162 discovery rankings, 180 per-instrument cost-stress
rows, aggregate holdout, and sleeve holdout all match the CPU reference within
FP64 numerical tolerance. CPU remains responsible only for storage I/O and
small final metric/table aggregation.

## Scope and limitations

This is a five-minute, same-day strategy test—not tick-level or sub-second
scalping validation. IEX bar aggregates cannot model spread changes, order-book
depth, latency, partial fills, or market impact. The implementation deliberately
uses next-open fills and explicit cost stress, but live eligibility would still
require quote/tick data, broker paper execution, and forward observation.
The commodity sleeve uses exchange-traded proxies because Alpaca does not offer
historical commodity futures through this API. Crypto bars are restricted to
US equity hours for cross-asset comparability; this test does not assess the
remaining 17.5 hours of each crypto trading day.
