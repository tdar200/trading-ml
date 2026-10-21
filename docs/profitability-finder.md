# Profitability Finder

## V3 approval candidate

`examples/find_profitable_intraday_v2.py` is now the default dashboard runner.
It searches 40 configurations without SIP quotes and 48 when genuine quote
coverage is available. The grammar is fixed before evaluation and contains
structure momentum, structure reversal, pooled CUDA ridge, a causal online
expert ensemble, a causal regime router, and an optional quote-confirmed
family. Expert selection uses only trailing, already-realized cross-sectional
information coefficients and abstains when no expert has positive trailing
skill. Every attempted configuration is charged to Deflated Sharpe, including
inactive trials.

The causal feature set contains session VWAP deviation, prior-session high/low
position, opening-range breaks, support/resistance rejection, relative volume,
realized volatility, beta residuals, event exclusions, and quote imbalance.
Entries occur at the following five-minute bar and exits at the regular-session
close. Replay cost is the greater of the stress assumption and half-spread, plus
a latency buffer and relative-volume impact surcharge.

Run each compatible sleeve independently:

```bash
.venv/bin/python examples/find_profitable_intraday_v2.py --sleeve individual_stock
.venv/bin/python examples/find_profitable_intraday_v2.py --sleeve commodity_proxy
.venv/bin/python examples/find_profitable_intraday_v2.py --sleeve other_contender
```

Crypto is intentionally rejected by this runner because its continuous session,
fees and fills require a separate simulator. Inspect all current inputs with
`.venv/bin/python examples/preflight_profitability_data.py`.

### Free, keyless research mode

Yahoo bars are always stored separately and explicitly marked research-only:

```bash
.venv/bin/python data/ingest_yfinance_intraday.py --asset-class stock --timeframe 5m --period 60d
.venv/bin/python examples/find_profitable_intraday_v2.py \
  --sleeve individual_stock --intraday-root data/free_intraday
```

This mode does not set SIP/NBBO, point-in-time, broker-fill, or deployment gates
to true. Never merge `data/free_intraday` into the licensed/provenanced stores.
Each asset class and timeframe has a separate manifest, such as
`manifest_stock_5m.json`, `manifest_stock_1h.json`, or `manifest_crypto_5m.json`.

The keyless 1-hour V3 acceptance run used 723 common sessions, reserving the
last 252 for evaluation. Discovery still selected pooled ridge; evaluation
returned -4.39% at 2.5 bps/side with Sharpe -1.439 and 29 portfolio events.
The ensemble therefore remains **REJECTED**. Its 100% DSR is explicitly scoped
to discovery and does not override the losing evaluation result.

V4 research additions include a GPU asymmetric model trained directly on long
and short returns after conservative round-trip costs, forecast-uncertainty
abstention, a trailing-IC circuit breaker, transaction-level failure
attribution, purged walk-forward split utilities, and a hash-chained shadow
journal. Shadow observations are explicitly research-only and never satisfy
the broker-evidence gate.

```bash
.venv/bin/python examples/diagnose_profitability_failure.py RUN_DIR
.venv/bin/python examples/record_profitability_shadow.py RUN_DIR \
  --observed-at 2026-09-14T15:30:00Z --ticker AAPL --side long --quantity 1 \
  --predicted-edge-bps 12 --intended-price 100 --observed-bid 99.99 \
  --observed-ask 100.01 --reference-exit-price 100.20 \
  --data-source manual_delayed_research
.venv/bin/python examples/prospective_profitability_status.py RUN_DIR \
  --intraday-root data/free_intraday
```

The latest attribution showed that the rejected holdout loss was primarily
wrong-way gross long selection in high-information sessions, not merely fees.
Those exposed findings can inform a newly frozen prospective candidate but
cannot retroactively improve the existing qualification result.

## Sixty-signal combinatorial finder

`examples/find_signal_combinations.py` implements the indicator-level strategy
finder. It provides 60 causal MA/EMA, ROC, RSI, stochastic, Williams %R, CCI,
MACD, Bollinger, ATR, Donchian support/resistance, volume, OBV, VWAP, candle and
intraday-range primitives. It orients standalone signals using discovery data,
builds all 1–4 signal combinations among the top 12 at each decision time,
GPU-screens 1,586 research candidates, and sends no more than 48 finalists to
the canonical beta-neutral cost replay and qualification pipeline.

The first complete run selected `EMA(5) distance + stochastic(5) + ROC(3)` but
lost 28.99% on evaluation, with Sharpe -2.826, 30.09% drawdown, PBO 66.7%, and
DSR confidence 0.04%. It is rejected. Because every canonical discovery
finalist also had a non-positive robust score or a negative worst window,
subsequent runs fail before opening a holdout unless a genuinely positive,
stable discovery candidate exists.

V5 corrects that search design. Indicator values are converted to natural
multi-bar states such as RSI recovery from oversold, stochastic crosses,
Bollinger re-entry, VWAP reclaim, and Donchian breakout. Correlation filtering
removes primitives with at least 0.85 absolute dependence, and combinations may
contain at most one trend, momentum, volatility, volume, or structure signal.
Orientation uses 280 sessions, followed by a five-session purge and 126-session
nested validation. Early setups use a one-bar exit; later setups use the session
close. The V5 run screened 426 combinations and correctly returned
`DISCOVERY_REJECT` without opening the 252-session holdout: the leading
Donchian/stochastic/Bollinger/MACD candidate had median Sharpe 1.118 but a
worst-window Sharpe of -4.712 and negative robust score.

This stage implements four fixed strategy families:

1. first-half-hour beta-residual momentum, entered at a later fixed decision time and held to the close;
2. the same signal combined with causal same-horizon intraday seasonality;
3. a pooled, regularized walk-forward ridge forecast.
4. a pooled CUDA elastic-net forecast with sparse feature selection.

When sufficiently complete SIP/NBBO quotes are installed, a fourth execution-confirmation variant is enabled. It requires momentum and causal quote imbalance to agree. The runner never substitutes IEX quotes for SIP/NBBO.

Every signal fills at the next five-minute open, exits at the final regular-session close, and uses weights projected to zero cash and rolling SPY-beta exposure. Missing bars remove only that stock for that session; at least 15 eligible names are required, eliminating the lossy global inner join. Signals are also demeaned by sector. Expected returns, regularized models, model-error haircuts, and seasonality are calculated only from earlier sessions. The optimizer varies decision time, breadth, minimum expected edge, cost multiple, forecast-error haircut, information regime, and seasonality horizon. A candidate may make no trade when estimated edge does not clear estimated round-trip costs.

## Run

```bash
.venv/bin/python examples/find_profitable_intraday_strategies.py --smoke
.venv/bin/python examples/find_profitable_intraday_strategies.py
```

CUDA is mandatory. The pooled models and forecast calibration run in FP64 on CUDA; session-level portfolio construction is reported honestly as `hybrid-cuda`.

Optional bias-control inputs:

```bash
.venv/bin/python examples/find_profitable_intraday_strategies.py \
  --membership data/reference/membership.csv \
  --events data/reference/events.csv
```

Membership columns are `ticker,start,end` with an open-ended blank `end` permitted. Event columns are `date,ticker,event`; use ticker `*` for market-wide macro events. Missing explicitly requested files fail closed. Without these inputs, the summary records the corresponding deployment blockers.

Historical SIP quotes require an entitled Alpaca subscription:

```bash
.venv/bin/python data/ingest_alpaca_quotes.py \
  --tickers AAPL,MSFT,NVDA --since 2026-01-01 --until 2026-02-01
```

Quotes are validated and stored under `data/quotes/<ticker>/<YYYY-MM>.parquet`. Quote strategies are absent from the grammar until all stocks have data and each decision dataset reaches at least 80% causal coverage.

## Acceptance result

The expanded acceptance run is `results/profitability_finder_20260912T164940Z`. It searched 4,536 candidates over 30 stocks and 667 dynamic-universe sessions: 60-session unscored warm-up, 485 discovery sessions in four windows, and 122 evaluation sessions. The best discovery rule used sector-adjusted first-half-hour residual momentum plus 40-session seasonality, entered 120 minutes after the open, long/short three, and a 3x modeled-cost gate.

The rule returned **+7.94%** at 1 bp per side on evaluation, with Sharpe **1.620**, maximum drawdown **-6.51%**, and 60 portfolio events. At 2.5 bps it returned **-0.33%** with only three events; at 5 bps the cost gate made no trades. Contribution concentration was 22.3%, PBO was **50.0%**, and Deflated Sharpe confidence was effectively zero. Verdict: **REJECT**.

This is not deployable. The evaluation dates were already seen by earlier project searches; even a future historical research pass can only advance to new forward paper trading. SIP/NBBO replay, point-in-time membership, an earnings/macro calendar, and live fill calibration remain mandatory deployment gates.

Each run writes `frozen_candidate.json` with a canonical SHA-256 fingerprint. Record broker-reconciled forward sessions without changing that fingerprint:

```bash
.venv/bin/python examples/record_profitability_paper_session.py RUN_DIR \
  --date 2026-09-14 --predicted-gross-bps 20 --realized-gross-bps 15 \
  --modeled-cost-bps 2 --realized-cost-bps 3 --net-return 0.001 \
  --orders-submitted 6 --orders-filled 6 \
  --broker-evidence broker-exports/orders-2026-09-14.json
.venv/bin/python examples/qualify_profitability_strategy.py RUN_DIR \
  --paper-log RUN_DIR/paper_observations.csv
```

The paper journal records actual submitted and filled order counts, binds every
session to a SHA-256 of its broker export, and forms a row-hash chain so edits
fail closed. The qualifier binds the frozen candidate, cost-stress table, and
paper journal into `deployment_qualification.json`; changing any artifact
invalidates runtime approval.

The qualifier requires 252 historical evaluation sessions, 200 historical events, Sharpe 1.25, positive 2.5-bps stress, PBO ≤20%, DSR confidence ≥95%, all bias-control datasets, an unexposed interval, then at least 126 forward paper sessions and 100 fills. `livesim/profitability_guard.py` separately rejects stale data, inadequate quote coverage, wide spreads, reconciliation failures, duplicates, excessive name/sector/gross exposure, out-of-distribution inputs, daily loss breaches, or an active kill switch. A future real-money profitability execution adapter must call `require_approved_profitability_order` immediately before every batch; the repository still has no enabled real-money path.
