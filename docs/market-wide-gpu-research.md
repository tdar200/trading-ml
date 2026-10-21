# Market-wide CUDA research

Implemented and first run on 30 September 2026. This expands the research candidate pool; it does not certify complete worldwide coverage or recommend a trade.

## Discovery and honest denominators

The archived catalogue contains **214,723 distinct source/symbol identities**:

| Catalogue | Distinct listings |
|---|---:|
| International equities via Yahoo exchange screeners | 166,171 |
| Alpaca equities/funds, including inactive broker assets | 33,307 |
| Nasdaq-traded issues: ETFs | 5,747 |
| Nasdaq-traded issues: other securities | 7,503 |
| Kraken spot pairs | 1,455 |
| Alpaca crypto pairs | 73 |
| Generated FX reference crosses | 435 |
| Continuous futures proxies | 32 |

These are listings/pairs, **not distinct companies**. Cross-listed securities and multiple currency pairs can represent the same economic exposure. Nasdaq/Alpaca aliases share one history request where identified. There are **201,944 distinct provider/symbol history requests** after that reconciliation. No top-N market-cap filter is applied. Data failures are stored rather than silently dropping candidates from the denominator.

The exchange screener's paginated responses contain overlaps/count mismatches. Raw rows, unique symbols, duplicate counts, advertised totals and partial/error status are recorded per venue. A successful HTTP response is not proof of complete venue coverage. The catalogue is a dated current snapshot, supplemented by inactive broker assets; it is not a survivorship-free historical universe. Refresh into a new results root to preserve a previous experiment's catalogue.

Unknown/uncovered: comprehensive global options and futures contracts, bonds, OTC instruments, CFDs, spread bets, prediction markets, every crypto venue/token, complete delisting/corporate-action histories, UK account eligibility, and broker execution permissions. FX crosses are reference pairs, not confirmed executable broker products. Futures proxies do not include contract multipliers, expiry, rolling costs or margin and must not be interpreted as executable futures-account returns.

## Actual GPU work

`analysis/market_gpu.py` uses PyTorch CUDA float64 tensors on the NVIDIA GeForce RTX 3050 Laptop GPU (4 GiB), batching independent asset/strategy cases. CUDA is required by the default CLI; it does not silently fall back to CPU. Downloads, SQLite and canonical replays remain CPU work. The GPU is idle while a data batch downloads; sustained 100% utilization is not expected.

Each usable history receives **90 configurations**: six families (Donchian, moving average, RSI, Bollinger, volume breakout, gap), five periods (10, 15, 20, 30, 50), three maximum holding periods (5, 10, 20 calendar days). Minimum history: 300 completed daily bars. Histories begin in 2020 where available; Kraken's endpoint has a much shorter rolling window. No missing candles are manufactured.

For each history, first 60% provides development/warmup, next 20% is validation, final 20% is reserved. The reserved bars are removed before tensorization. These are per-instrument date splits, **not a globally aligned cross-asset out-of-sample portfolio**. There is no headline out-of-sample winner yet.

Execution: completed-bar signals, following-bar open entries, adverse spread/slippage, fees on both sides, ATR stops, stop-first OHLC collisions, gap-aware stop fills and marked equity. Risk is capped at 0.5% per trade, 25% position exposure, one cash long position, with the existing loss/drawdown controls replicated. Three best-per-history configurations per processed GPU batch are replayed through the existing paper/RiskEngine path; final-equity disagreement above 0.000001 halts the worker. This sampling is not a claim that every configuration has received canonical replay.

Cost assumptions per side (basis points):

| Class | Fee | Spread | Slippage |
|---|---:|---:|---:|
| Equities and generic proxies | 15 | 2 | 3 |
| Crypto | 80 | 10 | 5 |
| FX | 2 | 10 | 3 |

These are screening assumptions, not individually verified venue tariffs. Capital is **100 normalized quote-currency units**, not a historically FX-converted GBP account. Funding, borrow, contract size, venue minimums, currency subunits and margin need instrument-specific qualification before a result is eligible for account-level research. No broad-universe result is approved for deployment.

## Fundamentals and selection bias

Current catalogue fields are not backdated fundamental features. Broad point-in-time financials, analyst estimates and vintage macro coverage are missing; the sweep does **not** invent fundamental scores. Existing fundamental infrastructure remains separate. A combined recommendation requires independently timestamped coverage and a new matched-universe, matched-cost validation experiment.

Millions of configurations create severe selection bias. Maximum validation ROI is not evidence of predictive edge. After data qualification, freeze the selection rule, account for the number of trials, use aligned chronological walk-forward evaluation, run parameter/cost sensitivity and only then open reserved data once. Evaluate finalist portfolio returns, realistic GBP cash constraints and Monte Carlo separately. No 10× target influences sizing.

## Files and monitor

- `data/market_universe.py`: archived discovery, provenance and coverage.
- `analysis/market_gpu.py`: batched CUDA signals and financial replay.
- `analysis/market_sweep.py`: resumable SQLite work queue, downloaded history hashes, costs, canonical comparisons and provider throttling stop.
- `examples/run_market_sweep.py`: CLI.
- `dashboard/market_sweep_app.py`: auto-refreshing live counts, GPU telemetry, partial coverage, failures and explicitly provisional recent results.
- `tests/test_market_gpu.py`, `tests/test_market_sweep.py`: financial parity, future-data isolation, catalogue identity, invalid inputs, batch isolation and throttling regression tests.

Open <http://127.0.0.1:8521>. Monitor values come from `results/market_sweep/status.json`, `coverage.json`, `gpu_latest.json` and `sweep.sqlite`. The ledger distinguishes PENDING, READY, GPU_TESTED, EXCLUDED and FETCH_ERROR. `canonical` contains replay comparisons; `screening` retains every tested configuration. Bar hashes, source responses, initial run configuration and timestamped resume manifests support reconstruction. A different GPU model hash requires a new results root.

Run from repository root:

```bash
.venv/bin/python examples/run_market_sweep.py --device cuda --workers 6
.venv/bin/python -m streamlit run dashboard/market_sweep_app.py --server.address=127.0.0.1 --server.port=8521 --server.headless=true
```

For a new archived discovery use `--root results/market_sweep_new --discover`. A worker lock rejects concurrent workers on one root. Create `results/market_sweep/STOP` to stop after the current bounded batch. Inspect the reason, remove that marker and rerun the same command to resume pending/ready work. Provider throttling creates the marker automatically; failed requests remain recorded and are not blindly retried. After an external kill, inspect the saved PID/log: the most recent status timestamp alone does not establish worker liveness.

## Verification and defects repaired

Ten dedicated tests pass, including CUDA-vs-CPU comparisons and canonical accounting checks, all six signal families, variable calendar lengths, future-data canaries, bad OHLC, duplicate bars, literal ticker `NA`, duplicate provider identity, invalid-symbol batch recovery and provider rate limits. Actual CUDA histories and canonical replays are recorded in the ledger; this is not a synthetic GPU demonstration. The full suite passed: **607 tests**, five warnings, 113.80 seconds. Full-suite output is retained separately in `results/market_sweep/full_suite.txt`.

Fixed: float32 cost expansion causing small equity discrepancies; Nasdaq's literal `NA` ticker being parsed as missing; Alpaca `asset_class` schema handling; a rejected symbol poisoning its entire download batch. Previously affected batch members were returned to the pending queue. Pagination overlaps are flagged rather than represented as full coverage.

## Source documentation

- [Nasdaq directory field definitions](https://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs).
- [Alpaca historical stock bars](https://docs.alpaca.markets/reference/stockbars).
- [Kraken OHLC endpoint and rolling history limit](https://docs.kraken.com/api/docs/rest-api/get-ohlc-data/).
- [Frankfurter reference currency API](https://frankfurter.dev/).
- [yfinance screener wrapper](https://ranaroussi.github.io/yfinance/reference/api/yfinance.screen.html).
- [PyTorch CUDA installation/runtime documentation](https://pytorch.org/get-started/locally/).

The full queue is a long-running network-bound experiment. Its current counts must be read from the monitor; discovery completion does not mean backtest completion.


## Monitoring repairs, 30 September 2026

Minute-by-minute monitoring found two defects after the initial run:

1. The rate-limit detector matched `429` inside ticker `073429.TW`. The provider actually reported missing prices. Detection now uses structured HTTP status, the provider rate-limit exception or explicit HTTP/rate-limit wording. A permanent regression test covers numeric ticker names.
2. Canonical validation stopped on ABR.PRD: a nearly flat Bollinger window produced different strict comparisons with Torch and pandas floating-point reductions (ending-equity discrepancy 0.02057754). Signal and ATR inputs now come from the exact canonical CPU feature implementation; the financial paths, fills, sizing and metrics still run in GPU batches. This prioritizes reproducibility over indicator throughput. A dated real-price fixture reproduces the defect. Holding time now uses elapsed UTC seconds rather than truncated dates, matching canonical behavior across DST. Crypto annualization includes Alpaca crypto as well as Kraken.

All previous GPU results were archived in `results/market_sweep/archive_before_signal_repair/` with their original SQLite database and configuration. The active screening and canonical tables were cleared and all previously tested histories requeued from cached prices for recomputation. The final reserved period remains unopened. Old results must not be combined with the repaired run. Fifteen targeted tests passed before restarting; the full regression suite is recorded in `full_suite_after_repairs.txt`.
