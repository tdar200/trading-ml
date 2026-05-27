# Stage 8 — Real-Money Deploy Readiness Checklist

**Audience**: the human operator, before pointing the live adapter at a real-money Alpaca account.

**Parent gate** (design spec §5): *"Live paper Sharpe ≥ 50% of backtest Sharpe over ≥3 months before any real capital."*

This document enumerates what to observe during the paper phase, what would halt the experiment, and what must be true before flipping to real money. It is *not* a substitute for legal / tax / risk consultation appropriate to your jurisdiction.

---

## A. Pre-deployment (one-time, before kicking off paper)

- [ ] Alpaca paper account has at least $100k starting balance (matches backtest assumptions).
- [ ] `.env` has valid paper-account keys. (If rotated since Stage 1, re-paste.)
- [ ] `uv run python examples/preflight_alpaca.py` returns 0 with the expected balance.
- [ ] Universe (`data/tickers.py::UNIVERSE`) matches the strategy's expectation (12 sector ETFs + SPY).
- [ ] Stage-3 winner config is reflected in `examples/run_trading_node.py` (currently `top_n=2, lookback_days=63, skip_days=0`).
- [ ] You have read this checklist end-to-end.

## B. Run cadence (daily, during the ≥3-month observation)

- [ ] Once daily, run `uv run python data/ingest_alpaca.py` to refresh the catalog with overnight bars.
- [ ] Once daily, kick the live node: `uv run python examples/run_trading_node.py --exec-mode alpaca-paper --poll-seconds 60`. Leave running through market close; Ctrl-C after the daily snapshot fires.
- [ ] After the session, check `dashboard/app.py`'s Equity + Harness Verdicts tabs.

## C. Per-session metrics to log (suggest: a research-journal markdown file you append to)

- [ ] Date of session.
- [ ] Account equity at start (USD).
- [ ] Account equity at end.
- [ ] Number of rebalances triggered.
- [ ] List of fills (ticker, side, qty, fill price).
- [ ] Any REJECTED orders (with reason).
- [ ] Any uncaught exceptions in the node log.
- [ ] Comparison: actual fill prices vs the bar close the strategy used for sizing — log the slippage delta.

## D. Weekly metrics to compute (suggest: a Streamlit cell or notebook)

- [ ] Rolling 4-week Sharpe of paper equity curve.
- [ ] Rolling 4-week Sharpe of the Stage-3 winner backtest over the same window (for comparison).
- [ ] Total cumulative return paper vs backtest.
- [ ] Max drawdown paper vs backtest.

## E. Halt conditions (stop the paper experiment and investigate before continuing)

- [ ] **Paper Sharpe < 25% of backtest Sharpe** sustained over any 4-week window. ⟶ likely strategy decay or implementation drift; do not roll to real money.
- [ ] **Cumulative slippage > 0.5%** of equity per month. ⟶ the cost model is wrong; revisit Stage 3 cost-stress assumptions.
- [ ] **Any uncaught exception** in the adapter or any `REJECTED` count > 5% of total submits per week. ⟶ adapter bug; fix and restart the 3-month clock.
- [ ] **Account equity falls below $80k** (20% drawdown). ⟶ pause regardless of Sharpe — the strategy or its execution may have broken.

## F. Gates to real money (ALL must be true for ≥3 months)

- [ ] **Live paper Sharpe ≥ 50% of backtest Sharpe** over the full observation window (parent spec gate).
- [ ] **Zero uncaught exceptions** for the most recent 4-week stretch.
- [ ] **Slippage within 2× the Stage-3 cost-stress range** (`slippage_prob ∈ [0, 1]` produced CAGR 0.17 → 0.17; if live slippage shrinks CAGR by >5 percentage points, the cost model needs revisiting before real money).
- [ ] **No regulatory / tax surprise** in your jurisdiction (not a code concern — your call).
- [ ] **You are emotionally prepared to lose the position size** you'll deploy. (Not a code concern, but worth saying.)

## G. Real-money rollout (when all of F is true)

- [ ] **Do not** swap the entire balance at once. Start with 10% of the paper notional in real money for ≥1 more month.
- [ ] Code change required: in `livesim/alpaca_exec_client.py`, the `TradingClient(...)` constructor takes `paper: bool`. Real money = `paper=False` and pointing at the real-account API keys. This is a one-line config change — gated entirely by this checklist, not by code complexity.
- [ ] After 1 month at 10%, if still meeting all F criteria, scale up — slowly.

---

## H. What this checklist does NOT cover

- **Tax loss harvesting** and other portfolio-level concerns.
- **Cross-strategy capital allocation** if you ever run more than one.
- **Failover** if your home machine dies during a session (the node is a local Python process; you'd miss that day's rebalance).
- **Adversarial market regimes** (Stage 3 cost-stress is the closest defense; real life is harder).

If any of these are concerns, treat them as Stage 8.5+ research.
