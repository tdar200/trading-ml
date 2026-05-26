# Stage 2 — Momentum Top-N Strategy: Design Spec

**Date:** 2026-05-26
**Status:** Approved through brainstorming; pending user spec-review before implementation planning.
**Parent spec:** `2026-05-26-trading-ml-design.md` (architecture, philosophy, full roadmap)
**Stage:** 2 of 8 — first real strategy, no ML yet (ML arrives in Stage 5).

## 1. Goal & milestone

Build the project's first real Nautilus `Strategy`: a cross-sectional momentum top-N rotator on the 11 SPDR sector ETFs, with monthly rebalancing, backtested against SPY buy-and-hold via the catalog produced in Stage 1.

**Milestone (from parent §5):** Tearsheet vs SPY showing CAGR, Sharpe, and max drawdown on one chart. Passing or failing the benchmark is *not* the milestone — having a credible, reproducible answer is.

**This stage does NOT include:** ML, walk-forward / Deflated Sharpe, sim-live, or any cost-stress beyond the baseline 0.5 bps slippage. Those are Stages 3, 4, 5.

## 2. Architecture

Stage 2 leaves the data pipeline (Stage 1) untouched. It adds three modules and one runnable:

```
strategies/
  __init__.py
  ranking.py            # PURE: 6m-skip-1 momentum ranker. No Nautilus deps.
  momentum_topn.py      # Nautilus Strategy — composes ranking + engine.

analysis/
  __init__.py
  tearsheet.py          # quantstats wrapper: CAGR / Sharpe / max-DD vs SPY.

examples/
  backtest_momentum.py  # Runnable: load catalog → run engine → save → tearsheet.

results/                # Backtest outputs (gitignored).

tests/
  test_ranking.py
  test_momentum_strategy.py
```

### Why this split

- **`ranking.py` is pure.** No Nautilus, no IO. The heart of the strategy (and the part most likely to grow features later, e.g. risk weighting in Stage 4-5) lives where it can be exhaustively unit-tested in isolation.
- **`momentum_topn.py` is a thin Nautilus coordinator** that wires the ranker into the engine's bar stream and order placement. Mirrors the Stage-1 pattern (pure `bar_converter` + composing `ingest_alpaca`).
- **`analysis/` is separate** so the strategy never imports plotting/reporting libraries. The Strategy class stays lean enough to run in the fast path; reporting only loads at tearsheet time.

### Architectural commitments

1. **No catalog queries from inside the Strategy.** The strategy gets bars via Nautilus's subscription mechanism, maintains its own in-memory price history, and never reaches back to the catalog mid-run. Keeps the engine fast and the strategy's behavior deterministic.
2. **Pure ranker is the single source of truth for "what would I hold today?"** Strategy code just executes the ranker's verdict and reconciles with current positions.
3. **Same code path will run in Stage 4 sim-live.** Don't bake in any backtest-only assumptions (e.g. don't look ahead, don't peek at SPY's future bars for the benchmark — SPY's close-to-close path is built passively from the bars the engine streams).

## 3. Key behaviors

### Universe & benchmark

- **Tradeable universe** (long-only): the 11 `UNIVERSE` ETFs from `data/tickers.py` — XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XLB, XLRE, XLC.
- **Benchmark:** SPY. Added to the engine as an instrument so its bars stream into the Strategy alongside the others, but the Strategy NEVER places an order on SPY. Used by `tearsheet.py` for the comparison curve.

### Rebalance trigger

Inside `on_bar`: when the incoming bar's calendar month differs from `self._last_rebalance_month`, this is the first bar of a new month → fire a rebalance. Simple, deterministic, no timers.

Edge case: the very first bar received initializes `_last_rebalance_month` without rebalancing — we don't rebalance until we've seen at least one full month roll over after warm-up. (See "Warm-up" below.)

### Ranking

Pure function in `strategies/ranking.py`:

```python
def rank_by_momentum(
    closes_by_ticker: dict[str, list[float]],
    lookback_days: int = 126,   # ≈ 6 trading months
    skip_days: int = 21,        # ≈ 1 trading month
) -> list[str]:
    """Return tickers sorted descending by lookback return ending skip_days ago.
    Tickers with insufficient history (< lookback_days + skip_days closes) are excluded."""
```

The "6-month return skipping the last month" is the classic momentum-skip-recent pattern: it captures the medium-horizon trend while avoiding the well-known short-term mean-reversion effect that contaminates raw 6m returns.

### Position sizing

On rebalance:
1. Compute the ranker's verdict (ordered list of tradeable tickers).
2. Target USD per name = `equity / top_n` for each ticker in the top-N; `0` for everyone else.
3. Target shares = `int(target_usd / current_price)` (floor to whole shares — the simulated venue doesn't model fractional execution).
4. Delta = `target_shares - current_shares` for each ticker in the universe (held or not).
5. For every ticker where `delta != 0`, submit a market order for `abs(delta)` shares in the appropriate direction (buy if positive, sell if negative).
6. Default `top_n = 3` (per parent spec §5). Configurable.

No no-trade band — always reconcile to the floored integer target. Tiny rebalances are accepted as a known cost in exchange for simpler, more deterministic behavior.

### Cost model

- **Slippage:** 0.5 bps per fill (per parent §5). Wired via Nautilus's simulated venue fill model. Exact symbol confirmed against `docs/nautilus_api_notes.md` at plan time — likely `FillModel` with a basis-point or fractional-spread parameter.
- **Commissions:** $0 (Alpaca paper baseline; Stage 3's honesty harness will stress this).
- **Borrow fees, dividends, taxes:** not modeled. Long-only, paper-only. Acknowledged limitation; documented in the tearsheet output.

### Warm-up

No trades placed until every tradeable-universe ticker's price history is at least `lookback_days + skip_days` ≈ **147 daily bars** (~7 trading months). On a ~5.5-year catalog this consumes the first ~7 months and leaves ~60 months for actual trading. Documented in the tearsheet.

### Configuration surface (`MomentumTopNConfig`)

A Nautilus `StrategyConfig`:

```
universe: tuple[str, ...]     # default ALL 11 UNIVERSE tickers
benchmark: str                # default "SPY" (subscribed but not traded)
top_n: int = 3
lookback_days: int = 126
skip_days: int = 21
venue: str = "ARCX"           # imports VENUE_MIC at plan time
starting_cash: int = 100_000  # used by backtest entry, not the strategy itself
```

## 4. Backtest entry point

`examples/backtest_momentum.py` — a runnable script that:

1. Loads instruments + daily bars for the 12 tickers from the Stage-1 catalog.
2. Builds a `BacktestEngine` (verified config from `docs/nautilus_api_notes.md`), adds an ARCX venue with $100k cash + 0.5 bps slippage, adds all 12 instruments, adds the bars, adds `MomentumTopNStrategy` with default config.
3. Runs the engine.
4. Dumps fills, orders, and the equity curve to `results/momentum_topn_<ISO-timestamp>/` (gitignored).
5. Calls `analysis.tearsheet.report(...)` on the results to produce the comparison vs SPY.

Same "examples are runnable demos, not unit-tested" convention as `examples/smoke.py`.

## 5. Tearsheet (`analysis/tearsheet.py`)

Single public function:

```python
def report(
    equity_curve: pd.Series,      # strategy equity over time, indexed by date
    spy_closes: pd.Series,        # SPY close series over the same range
    out_dir: Path,
) -> None:
    """Generate a quantstats HTML tearsheet comparing strategy to SPY buy-and-hold.
    Writes <out_dir>/tearsheet.html and prints a CAGR/Sharpe/max-DD summary table."""
```

Uses `quantstats.reports.html(...)` for the full report; also prints a small console summary (CAGR, Sharpe, max-DD, total return, alpha-vs-SPY) so a CI run can capture key numbers without opening the HTML.

`quantstats` is added to `pyproject.toml` in Task 1 of the implementation plan.

## 6. Testing

### `test_ranking.py` — pure-function tests (TDD)

Exhaustive coverage of the pure ranker:

- **Known-ranking case.** Build a synthetic dict of close series with known returns; verify the output order matches the expected ranking.
- **Skip-days actually skips.** Tickers with strong returns concentrated in the last `skip_days` should rank LOWER than tickers with the same total return spread over the full window — proves the skip is wired correctly.
- **Insufficient history excludes.** A ticker with fewer than `lookback_days + skip_days` closes is excluded from the ranking (not silently ranked at the bottom).
- **Empty input.** No tickers → empty list (no exception).
- **Tied returns.** Stable ordering on exact ties (don't care about which one wins, just don't crash).

### `test_momentum_strategy.py` — integration test on synthetic data

Set up a tiny `BacktestEngine` with 3 fake instruments and ~18 months of hand-crafted bars (no catalog dependency). Run `MomentumTopNStrategy` and assert:

- **No orders during warm-up.** No `submit_order` calls before the buffer fills.
- **First trade fires on the first bar of the first post-warm-up month** — not earlier, not later.
- **Top-N selection is correct** for a known-rank synthetic dataset (build closes so ticker A has the highest 6m-skip-1 return, B second, C third; assert positions reflect that order).
- **Holdings rotate when ranks change.** Flip the synthetic relative returns mid-backtest; verify the next monthly rebalance closes the dropped name and opens the new one.

### Runnable backtest (`examples/backtest_momentum.py`)

Exercised end-to-end manually (and in Stage 2's final acceptance run). NOT covered by automated tests — same convention as `smoke.py`. The Stage-2 milestone check (tearsheet vs SPY) is satisfied by manually running the entry point on the real catalog and inspecting the output.

## 7. Open items / risks

1. **First-bar-of-month detection** assumes daily bars are received in chronological order. Stage 1's `bar_converter.py` was patched to guarantee this (commit `6fd9b07`); restating the assumption here so future Stage-1 changes don't silently break Stage 2.
2. **0.5 bps slippage is generous** for sector ETFs. The Stage-3 honesty harness should stress this at 2×/5×. Don't anchor on the Stage-2 numbers as gospel.
3. **No transaction-cost realism beyond slippage.** No commissions (Alpaca paper accurate), no taxes, no half-spread modeling. Tearsheet output will note these omissions.
4. **The `Strategy`'s in-memory ring buffer assumes ≤ 12 tickers × ≤ 200 bars** — trivial memory. Future stages with broader universes will need to revisit.
5. **`quantstats` HTML reports are heavy** (~5 MB per run, browser-rendered). Not committed; `results/` is gitignored.
6. **Free-tier IEX data gap on 2024-12-23 (the SPDR ex-div day flagged in Stage 1)** is in the historical bars. Spec inheritance: the strategy treats whatever the catalog says is the close as ground truth. The roughly ~1% one-day anomaly on that date may produce a small artifact in the backtest equity curve; the tearsheet will be honest about it.

## 8. Out of scope (deliberately, do not add)

- ML / meta-labeling / sentiment features. (Stages 5–6.)
- Walk-forward, purged CV, Deflated Sharpe, cost stress. (Stage 3.)
- Live or sim-live execution loop. (Stage 4.)
- Streamlit dashboard. (Stage 7.)
- Multi-strategy comparison or per-sector tilts beyond top-N. (Future.)
- Anything intraday or higher-frequency than daily. (Out of scope for the entire project unless explicitly revisited.)

---

*This spec is the Stage-2 contract. Implementation plan to be written next via `superpowers:writing-plans`.*
