# Profitability Deployment Runbook

This runbook is fail closed. A historical backtest is never permission to trade.

## 1. Data and research

1. Run `examples/preflight_profitability_data.py` with licensed point-in-time
   membership and event-calendar files.
2. Ingest entitled SIP/NBBO quotes. IEX-only data does not satisfy the gate.
3. Run `examples/find_profitable_intraday_v2.py` independently for each
   US-equity-session sleeve. Never combine crypto with the equity replay.
4. Reject any candidate that fails one strict historical gate. Do not lower a
   threshold or add trials after viewing evaluation results.
5. Preregister a date after the repository's exposed cutoff and freeze exactly
   one candidate. Accumulate at least 252 untouched sessions and 200 events.

## 2. Paper observation

1. Run `examples/preflight_profitability_deployment.py RUN_DIR --mode paper`.
2. Run `examples/preflight_alpaca.py` to authenticate and verify the paper account.
3. Keep the node supervised through the full session. Do not run if quotes are
   stale, SIP coverage is below 80%, reconciliation fails, or the kill switch is set.
4. Export broker orders/fills after every session and record them with
   `examples/record_profitability_paper_session.py`. The export is copied and hashed.
5. Investigate any rejection, partial fill, exception, restart, data gap, or risk
   breach. A risk breach fails approval; changing the strategy restarts observation.
6. After at least 126 sessions and 100 fills, run
   `examples/qualify_profitability_strategy.py`.

## 3. Incident response

1. Set `PROFITABILITY_KILL_SWITCH=1` and cancel open orders.
2. Reconcile broker orders, fills, positions, and cash against local state.
3. Flatten only under the operator's broker-approved procedure; do not improvise
   automated liquidation when market data is stale.
4. Preserve logs and broker exports. Diagnose before clearing the switch.

## 4. Real capital

Real-money routing for the profitability strategy is intentionally absent. Even
an approved report only makes the candidate eligible for a separately reviewed
adapter. That adapter must call `require_approved_profitability_order` immediately
before every order batch. Begin with limited capital only after legal, tax, broker,
and human risk review; the existing Alpaca adapter still rejects `paper=False`.
