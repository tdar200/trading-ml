# trading-ml — Design Spec

**Date:** 2026-05-26
**Status:** Approved through brainstorming; pending user spec-review before implementation planning.
**Owner:** tahseen-dar

## 1. Context & goal

An ongoing, paper-trading-first ML + trading project on US stocks/ETFs. Goal is to build a research-and-execution *system* whose primary ROI is durable skill and reusable infrastructure; any net-positive edge is a bonus, not the plan. The project is structured so the same code path runs in backtest and (eventually) live execution.

### Honest framing (informs every decision below)
- ~79% of professional active funds underperform the S&P 500; <1% of retail day traders are reliably profitable net of fees.
- Backtest Sharpe predicts live Sharpe with R² < 0.025; ~30–50% edge degradation in live trading is the realistic baseline.
- ML does not *create* an edge; it *amplifies* a signal that already has a structural reason to work. Naive next-day price prediction (LSTMs etc.) is the #1 retail time-sink and is excluded by design.
- Validation method matters more than model choice. Walk-forward / purged CV and Deflated Sharpe are non-negotiable before any signal is trusted.

## 2. Architecture

Single Python project. NautilusTrader owns everything that moves orders or time (engine, backtest, sim-live, message bus, cache). The project owns the data ingestion, strategy logic, ML signal generation, and analysis.

```
                ┌───────────────────────────────────────────────┐
                │                trading-ml/                    │
                │                                               │
   Alpaca   ───▶ data/  (Nautilus ParquetDataCatalog, local)    │
                │   │                                           │
                │   ▼                                           │
                │ strategies/                                   │
                │   momentum_topn.py  ──┐                       │
                │   (later) momentum_meta.py                    │
                │                       ▼                       │
                │              NautilusTrader engine            │
                │      ┌────────────────────────────────┐       │
                │      │ BacktestEngine / SimVenue /    │       │
                │      │ Message Bus / Cache / LiveNode │       │
                │      └────────────────────────────────┘       │
                │                       │                       │
                │                       ▼                       │
                │      results/  (orders, fills, equity)        │
                │                       │                       │
                │                       ▼                       │
                │      analysis/  (tearsheets, walk-forward,    │
                │      Deflated Sharpe, cost stress)            │
                │                       │                       │
                │                       ▼                       │
                │      dashboard/  (Streamlit, Stage 7+)        │
                └───────────────────────────────────────────────┘
```

### Architectural commitments
1. **One strategy file = one self-contained Nautilus `Strategy`.** Same code path in backtest and sim-live. No bifurcation.
2. **The ParquetDataCatalog is the source of truth** for historical data. Alpaca is fetched once and cached locally; no re-fetching on every run.
3. **ML runs outside the strategy hot path** — predictions are generated in a separate process and persisted; the strategy consumes them as a feature/signal. Keeps the engine fast and the ML reproducible.
4. **Universe stays small and survivorship-bias-free** to start: the 11 SPDR sector ETFs + SPY as benchmark. Tiny data footprint, clean cross-sectional structure for the eventual ranker.

## 3. Stack

| Concern | Choice | Notes |
|---|---|---|
| Engine | `nautilus_trader==1.227.0` | Rust-native, event-driven, deterministic; backtest + sim-live parity |
| Python | 3.12 (uv-managed) | Nautilus requires 3.12–3.14 |
| Env mgmt | `uv` | Modern, fast, lockfile-based |
| Market data (Stage 1+) | `alpaca-py==0.43.4` | Free tier; daily bars since 2016; Benzinga news bundled |
| Analysis | `pandas`, `polars`, `quantstats` | Tearsheets vs SPY |
| ML (Stage 5+) | `lightgbm`, `scikit-learn` | Gradient-boosted trees on tabular features; lambdarank or top-quantile classifier |
| NLP (Stage 6+) | `transformers` + ProsusAI/finbert | Sentiment as a *feature*, ablated vs price-only |
| Dashboard (Stage 7+) | `streamlit` | Plays to existing user strength |
| Out of scope (now) | IBKR adapter, Redis state, Databento, tick/L2 data | Reserved as later upgrade levers |

### Why Alpaca instead of Nautilus's native Databento adapter
- $0 forever on free tier (vs Databento's free credit then per-symbol billing) — matches the stated paper-only constraint.
- Starter universe (12 daily-bar tickers since 2016) fits comfortably in free-tier limits; IEX-only real-time and 15-min SIP withhold are non-issues for daily bars.
- Alpaca credentials will be needed anyway for eventual real paper trading.
- Cost: one-time ~100-line `ingest_alpaca.py` script that writes to Nautilus's ParquetDataCatalog. Bonus: forces fluency with Nautilus's data model in week one.
- Databento remains the Stage 3+ upgrade lever if tick or order-book data is ever needed; flipping it on is an adapter swap, not an architecture change.

## 4. Universe & data scope

- **Tickers:** XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XLB, XLRE, XLC (11 SPDR sector ETFs) + SPY (benchmark).
- **Resolution:** daily bars only at this stage. No intraday, no tick.
- **History:** Free-tier `DataFeed.IEX` provides ~5.5 years of history in practice (~2020-07 onward); the catalog effectively starts ~2020-07. The `2016-01-01` start date is left in the ingest CLI as an aspirational lower bound — Alpaca's free tier simply returns what it has, so there is no harm in asking for more. Full history to 2016+ requires Alpaca's paid Algo Trader Plus subscription (`DataFeed.SIP`), which is out of scope for the $0 paper-only commitment.
- **Adjustment:** `adjustment="all"` (splits + dividends) on the Alpaca side; bars persisted into Nautilus catalog with a pinned, documented adjustment convention.

Survivorship bias is sidestepped because all 12 tickers exist for the full window; this is a deliberate choice to keep the early stages clean and is revisited if/when the universe expands beyond ETFs.

## 5. Staged roadmap

Each stage has a single milestone that gates the next. No advancing on vibes.

| Stage | Deliverable | Milestone |
|---|---|---|
| 0 | Repo skeleton, uv env, Nautilus installed, smoke backtest runs | `examples/smoke.py` produces a tearsheet |
| 1 | `data/ingest_alpaca.py` writes daily bars (12 tickers, 2016→today) into Nautilus ParquetDataCatalog; idempotent | Catalog loads into `BacktestEngine`, replays cleanly, cross-checks vs yfinance within 0.5% |
| 2 | `strategies/momentum_topn.py` — Nautilus `Strategy`: rank 11 ETFs by 6-month momentum (skip last month), hold top-3 equal-weight, monthly rebalance, backtest 2016–today with 0.5 bps slippage | Tearsheet vs SPY: CAGR, Sharpe, max-DD on one chart |
| 3 | `analysis/walk_forward.py`, `analysis/deflated_sharpe.py`, `analysis/costs.py` | Can answer "is the edge real or overfit?" with a number; expected baseline: probably underperforms SPY net of costs |
| 4 | Strategy wired to Nautilus `LiveNode` against simulated venue; nightly scheduled rebalance | 2+ weeks of sim-live fills logged, matching backtest within tolerance |
| 5 | `models/lightgbm_ranker.py` + `strategies/momentum_meta.py` — ML as a *filter/sizer* (meta-labeling) on Stage-2 trades; walk-forward retraining | ML version beats Stage-2 baseline OOS *and* on Deflated Sharpe |
| 6 | Pull Alpaca Benzinga news feed; FinBERT-score per ticker; add as feature; ablation vs price-only | Measurable incremental IC or a clean rejection — both are real results |
| 7 | Streamlit dashboard: rolling Sharpe vs backtest, positions, signal health, feature importances, research-journal markdown | Can read "is the strategy still working or just noisy?" off the screen |
| 8+ (optional) | Custom Nautilus execution adapter for Alpaca paper trading, then real $ | Live paper Sharpe ≥ 50% of backtest Sharpe over ≥3 months before any real capital |

### Why this order
- Stage 3 (honesty harness) comes **before** any ML. Building ML on top of an unvalidated edge teaches nothing trustworthy.
- The dashboard is **Stage 7, not Stage 1**. A dashboard on a strategy you haven't validated is a machine for making pretty lies. Build the truth first.
- The "ongoing" part of the ongoing project lives in Stages 5–7. Stages 0–4 are bounded setup; ~1–2 weeks of focused evenings.

## 6. First implementation chunk (Stage 0 + 1)

### Repo layout
```
trading-ml/
├── pyproject.toml          # uv-managed, Python 3.12
├── README.md
├── .gitignore
├── .env.example            # ALPACA_API_KEY, ALPACA_API_SECRET (paper)
├── data/
│   ├── ingest_alpaca.py    # Stage 1: pulls bars → catalog
│   └── catalog/            # Nautilus ParquetDataCatalog (gitignored)
├── strategies/             # empty in Stage 1; populated in Stage 2
├── examples/
│   └── smoke.py            # Stage 0: minimal Nautilus backtest
├── tests/
│   ├── test_ingest.py
│   └── test_catalog.py
└── docs/
    └── superpowers/specs/  # this doc
```

### Stage 0 — done when
- `uv sync` succeeds on a fresh clone; `nautilus_trader==1.227.0` and `alpaca-py==0.43.4` resolve.
- `uv run python examples/smoke.py` runs Nautilus's canonical backtest example end-to-end and prints an account balance change.
- README has a runnable 3-line "how to run" section.

### Stage 1 — done when
- `.env` populated with paper-account keys (free Alpaca signup at this step).
- `uv run python data/ingest_alpaca.py` fetches daily bars (2016-01-01 → today) for the 12 tickers and writes a Nautilus ParquetDataCatalog to `data/catalog/`.
- Re-running ingest is **idempotent**: no duplicate bars; only new dates fetched.
- `tests/test_catalog.py` loads the catalog into a `BacktestEngine`, replays 2024 bars, and asserts:
  - per-ticker bar count > 240 (trading days/year)
  - no nulls in OHLCV
  - timestamps are monotonic and timezone-aware
- A cross-check compares one ticker's daily closes vs yfinance and asserts max relative drift < 0.5% (catches adjustment bugs).

### Non-goals for these stages
- No strategy logic. No ML. No Streamlit. No live execution. No tick or minute data.

## 7. Risks & open items

1. **Splits/dividends adjustment convention** — Alpaca `adjustment="all"` vs Nautilus's expected bar schema needs one-time alignment in the ingest script. Will be pinned and documented.
2. **Nautilus data model has a learning curve** — `BarType`, `InstrumentId`, `Venue`, `ParquetDataCatalog`. Real cost in Stage 1 even with no strategy yet. The payoff is Stage 4 sim-live with zero rewrite.
3. **Alpaca free-tier 15-min SIP withhold** — irrelevant for daily-bar backtests; will matter near market hours in Stage 4 sim-live. Stage 4 design should snapshot data *after* market close to sidestep it.
4. **Nautilus breaking changes between releases** — README warns of this. Pin the version, write integration tests, upgrade deliberately.
5. **The realistic outcome of Stage 2** is "doesn't beat SPY net of costs." That is the expected baseline and validates the honesty harness — not a project failure.
6. **Free-tier IEX vs SIP venue spread** — Alpaca's free tier serves bars from IEX (a single exchange); yfinance returns SIP-consolidated closes. This creates legitimate venue noise of ~1% worst-case on isolated days (e.g. SPDR ex-dividend days); median drift is ~0.011%; p99 is ~0.061%; only ~0.27% of days exceed 0.1%. The yfinance cross-check tolerance is calibrated to 1.5% to catch adjustment bugs while accepting this venue noise. Migrating to a paid SIP feed (Alpaca Algo Trader Plus, or Tiingo) would eliminate this spread and extend history to 2016+; it is a Stage-3+ upgrade lever.

## 8. Out of scope (durably, not just for now)

- High-frequency / latency-sensitive strategies. Structurally unwinnable on home hardware.
- Naive deep-learning price prediction (LSTM-predicts-tomorrow's-close).
- Deep reinforcement learning as anything other than a learning sandbox.
- Strategies whose justification is "the backtest looks good" with no structural *why* (no answer to "who's losing money on the other side, and why do they keep doing it?").

---

*This spec is the design contract. Implementation plan to be written next via `superpowers:writing-plans`.*
