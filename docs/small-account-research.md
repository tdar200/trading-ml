> Audit update (2026-09-30): execution, monitoring and probability-boundary defects were repaired after this report. Preserve this as the original research record; use [the QA report](qa-financial-audit.md) and `results/qa_audit_20260930/corrected_research` for current-source replay. The selected diagnostic remains economically unchanged; no candidate qualifies.

# £100 trading research: findings, implementation and runbook

Research date: 2026-09-30. **Decision: NO_TRADE for real capital.** A working research and local paper framework is implemented; no strategy has enough independent evidence to justify risking this £100. The experimental 10× target did not affect candidate selection, sizing or risk limits.

## 1. Executive findings

- Compared **144 fixed candidates**: six rule families × three nearby periods × eight instruments. **0 qualified** under the stated development gate.
- Highest validation score: **XLE_bollinger_20**. Its later historical diagnostic ended at **£100.0188**, 0.02% total across 366 observations, with only **8 trades**. Doubling modeled fees/slippage made it lose money.
- This later period was chronologically separated in this run, but prior repository searches already exposed historical data. It is **not a pristine out-of-sample sample**. No claim of a statistically validated positive edge is made.
- Public live BTC/GBP data was fetched successfully; the default paper account stayed at £100 with no trade. That is one connectivity/behavior check, not 30 days of forward evidence.
- A full-account 10× needs **900% return**, **7.98% per calendar day**, **11.59% per trading day over 21 sessions**, or **71.13% compounded every seven days over 30 days**. Four equal weeks instead requires 77.83% per week.

## 2. Research coverage and what was actually tested

[Market/broker and literature review](small-account-market-evidence.md) covers UK/US shares, ETFs, FX, spot crypto, futures, options, CFDs, spread betting, prediction markets, and ten named broker/exchange routes. Markets rejected for access, capital, cost or missing execution data did not receive fabricated backtests.

Fresh tests used SPY, XLK, XLE, XLV, XLF, XLP daily ETF bars: **1,464 common sessions, 2020-07-27 to 2026-05-22**. BTC/GBP and ETH/GBP used **720 daily observations, 2024-10-10 to 2026-09-29**, fetched from Kraken's public endpoint. GBP crypto prices are directly denominated in GBP. ETF results are a constant-FX GBP-capital proxy, with 15 bp FX charges per side; they are not a reconstructed GBPUSD portfolio.

The data includes the 2022 equity bear market, later equity advances and volatile 2025 periods; it does **not** include the COVID crash, 2008, or a full crypto-cycle sample. The initial SPY file contained a 634-day gap; common-session alignment excluded its isolated early bar. No missing prices were invented. Input Parquet snapshots and SHA-256 hashes are saved. IEX venue coverage, corporate-action verification, dividends and distribution adjustment remain limitations. Fixed current ETFs reduce but do not eliminate universe-selection bias.

## 3. Test design and strategy comparison

Fixed families: prior-high Donchian breakout, moving-average upward cross, RSI crossing back above 30, Bollinger lower-band reversion, price/volume continuation, and positive-gap continuation. Periods 18/20/22; 10-calendar-day maximum hold; 2×14-bar ATR stop; 2R target. Gap continuation is a price proxy, not an earnings-event strategy.

First 50%: descriptive training. Next 25%: validation and three chronological reset-flat subwindows. Final 25%: later diagnostics. Ranking uses validation Sharpe minus maximum drawdown. Development qualification requires positive return in all three windows, at least 12 completed validation trades, and a positive 99% circular-block bootstrap mean-return lower bound. The 12-trade floor is a minimal rejection filter, not proof of sufficiency. Pointwise intervals are not a correction for 144 searches. All candidates remain deployment-disabled regardless.

Signals use completed observations. Orders execute on the subsequent observed open, with spread, slippage and fees. Stops gap at the open when necessary; stop wins if stop and target both touch. Fractional quantity is rounded down and fees reserved. Every entry passes the central risk engine. Positions start flat at evaluation boundaries and are liquidated at the final close; that forced endpoint is explicitly an evaluation convention. Indicator history before each boundary is warm-up only. Positions and P&L never cross training/validation boundaries.

Family representatives below were chosen by **validation**, not by the later result:

| Validation-selected representative | Validation Sharpe | Later total return | Later Sharpe | Trades | Max drawdown |
| --- | --- | --- | --- | --- | --- |
| XLF_donchian_20 | 1.707 | -0.15% | -0.127 | 10 | 1.11% |
| XLK_ma_18 | 1.622 | -2.08% | -1.087 | 18 | 2.62% |
| XLF_rsi_20 | 1.605 | 0.89% | 1.376 | 3 | 0.30% |
| XLE_bollinger_20 | 1.890 | 0.02% | 0.018 | 8 | 1.49% |
| SPY_volume_22 | 0.852 | 0.30% | 0.586 | 2 | 0.21% |
| XLV_gap_18 | 1.056 | -0.58% | -0.803 | 3 | 0.77% |

The full train/validation/diagnostic metrics for every candidate are in `results/small_account/strategy_comparison.csv` and `research.json`: CAGR, monthly geometric return, daily mean, Sharpe, Sortino, Calmar, profit factor, expectancy, drawdown/recovery, win/loss sizes, reward/risk, trade count, exposure, turnover and fees. Undefined ratios are null, not fabricated finite scores.

Selected diagnostic: CAGR 0.01%; monthly geometric 0.00%; Sharpe 0.0178; Sortino 0.0245; profit factor 1.0126; win rate 62.50%; average win £0.3023; average loss £-0.4975; realized payoff ratio 0.608; average trade expectancy £0.0024. A nominal 2R target did not produce a realized 2:1 payoff because time exits and stops determine actual outcomes.

## 4. Cost, parameter and exit robustness

| Fee/slippage multiplier | Signal-to-entry bars | Later return |
| --- | --- | --- |
| 1 | 1 | 0.02% |
| 2 | 1 | -0.24% |
| 4 | 1 | -0.68% |
| 1 | 2 | 0.22% |

The stress multiplier increases fee and slippage assumptions; spread is held fixed. One additional bar of execution delay is separately replayed. Stop variants fixed 2%, 2×ATR, prior-low structure and trailing ATR were crossed with 1R/1.5R/2R/3R targets **on development data only**; all 16 comparisons are saved. None was promoted using final-period performance. Neighboring periods 18/20/22 are retained for inspection rather than pruning poor neighbors away.

Previously stored intraday ORB discovery at `results/opening_range_discovery_reject_20260914T091855Z/summary.json` says DISCOVERY_REJECT, holdout unopened, with missing quote/event readiness. Other previously reported intraday search failures were not rebranded as successful strategies. No ML overlay, regime switch, order-book alpha, funding carry or pairs strategy was deployed without suitable evidence/data.

## 5. £100 → £1,000 mathematics

For win probability p, gross winning payoff bR, losing payoff −1R and cost cR:

`E[R] = p*b − (1−p) − c`

`Var(R) = p*(1−p)*(b+1)^2`

`g(f) = p*log(1+f*(b−c)) + (1−p)*log(1−f*(1+c))`

Geometric growth uses `exp(g)-1`, not arithmetic expectancy. A 1% price stop requires notional/equity approximately `f / 0.01`: risking 10% of equity implies about 10× exposure, and 20% implies about 20×, before reserving costs. This is incompatible with the new cash-only 25% notional cap. The aggressive simulations are analytical comparisons, not executable portfolio plans. At 10× exposure, a 38 bp round trip would consume about 3.8% of equity; the base 0.1R cost assumption at 10% risk is only 1%, showing how optimistic the assumed-edge scenario can be.

The approximate number of trades to multiply wealth by 10 at the log mean is `ceil(log(10)/g)` only when g>0; it is not a promised hitting time. At fixed costs, variance drag can make g negative despite positive arithmetic expectancy.

| Win rate | Gross reward:risk | Gross expectancy | Net at 0.1R costs | Full Kelly, analytical only |
| --- | --- | --- | --- | --- |
| 40.00% | 2 | 0.200R | 0.100R | 4.78% |
| 45.00% | 2 | 0.350R | 0.250R | 11.96% |
| 50.00% | 1.5 | 0.250R | 0.150R | 9.74% |
| 55.00% | 1.5 | 0.375R | 0.275R | 17.86% |
| 60.00% | 1 | 0.200R | 0.100R | 10.10% |
| 65.00% | 1 | 0.300R | 0.200R | 20.20% |

With a fixed fractional risk f, exact terminal tail probabilities use a binomial count of wins and product `(1+f*(b-c))^W * (1-f*(1+c))^(N-W)`. This also detects impossible targets even if a Monte Carlo run sees zero hits. For example, 21 consecutive 2R wins at 1% risk and 0.1R costs cannot produce 10×. The script records required wins and exact terminal probability for all 48 combinations.

## 6. Conditional Monte Carlo — 100,000 paths per case

Seed 20260930. Six win/payoff combinations × eight risk fractions (0.25%, 0.5%, 1%, 2%, 3%, 5%, 10%, 20%) × 100,000 paths; recorded after 21, 63, 126 and 252 trades, interpreted as one opportunity per equity session over approximately 30/90 days/6/12 months. Additional 60-trade cases include 50% risk solely to illustrate failure, never as executable configuration. Five cost scenarios span 0 to 1.6R. Overall there are 61 binary scenarios (6.1 million simulated paths), plus two 100,000-path empirical bootstraps.

These scenarios **assume** the edge; none of the listed win rates was validated for future trading. Base cost 0.1R is optimistic relative to many short-stop retail strategies. No shortage of setups, signal dependence, margin limit or participation constraint is modeled in the IID case; real small-account feasibility is worse.

At assumed 45% wins and 2R winners, 21 opportunities:

| Scenario | Risk/trade | Median end | 5th–95th | Hit £1,000 | Ever ≤£75 | Ever ≤£50 | Trades at log mean to 10× |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A Conservative | 0.50% | £101.91 | £97.43–£108.20 | 0.00% | 0.00% | 0.00% | 1886 |
| B Moderate | 1.00% | £103.73 | £94.84–£116.91 | 0.00% | 0.00% | 0.00% | 965 |
| C Aggressive | 3.00% | £110.10 | £84.30–£157.18 | 0.00% | 3.37% | 0.00% | 355 |
| D Extreme | 10.00% | £118.20 | £49.45–£377.77 | 0.15% | 48.09% | 15.65% | 163 |

For 60 opportunities in 30 days, the same assumed edge:

| Risk | Model | Median end | Ever £1,000 | End ≥£1,000 | Ever ≤£50 | Ever ≤£25 | Exact zero | Ever <£1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5.00% | IID | £179.2383 | 0.19% | 0.14% | 3.72% | 0.04% | 0.00% | 0.00% |
| 5.00% | stress | £110.6977 | 0.03% | 0.02% | 18.50% | 1.40% | 0.00% | 0.00% |
| 10.00% | IID | £234.2277 | 15.01% | 12.03% | 27.06% | 6.09% | 0.00% | 0.00% |
| 10.00% | stress | £89.3049 | 4.43% | 2.24% | 54.01% | 25.28% | 0.00% | 0.02% |
| 20.00% | IID | £164.3831 | 34.12% | 17.99% | 65.33% | 41.53% | 0.00% | 1.92% |
| 20.00% | stress | £15.3165 | 15.51% | 5.84% | 85.30% | 71.84% | 16.53% | 25.55% |
| 50.00% | IID | £0.0244 | 24.27% | 2.58% | 98.74% | 97.17% | 0.00% | 84.79% |
| 50.00% | stress | £0.0000 | 15.83% | 0.57% | 99.71% | 99.34% | 16.54% | 95.96% |

Stress model: transition to adverse win-rate regime with 5% probability per trade, 90% persistence; win probability falls by 15 percentage points; cost becomes lognormal with log-sigma 0.5; a 0.5% per-trade independent gap event makes a losing trade lose 5R. Losses are clipped at 100% of equity. These are **illustrative sensitivity assumptions, not fitted market frequencies**. They quantify dependence on tail assumptions rather than estimate this trader's real ruin probability.

“Hit” means ever reached the target; trading continues afterward. “End” means terminal wealth. The same path may hit £1,000 and later lose most of it. Loss probabilities use starting £100; drawdown probabilities use each path's running peak. Ruin at exactly £0 is mathematically impossible for finite bounded fractional losses smaller than 100%; its zero estimate is not reassurance. Operational failure is separately measured at <£1. Model counterparty/default risks are unquantified.

Zero hits in 100,000 simulations gives an approximate one-sided 95% bound of **0.003% within that model**. It is not a confidence bound on unknown real-world probabilities.

## 7. Empirical-candidate probability analysis

Circular 5-observation block bootstrap of the frozen candidate's 366 later-period net daily returns. 100,000 paths; 21/63/126/252 observation horizons. It preserves short local dependence but assumes the observed process is representative; eight trades are insufficient for reliable tail inference. The stressed bootstrap adds persistent adverse regimes, 1.5× volatility and random active-day execution drag. It is not a reconstructed order-book simulation or a replay of all risk halts.

| Horizon | Median end | 5th–95th | Hit £125 | £150 | £200 | £250 | £500 | £1,000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30_days | £100.00 | £99.45–£100.39 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 90_days | £100.05 | £99.12–£100.69 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 6_months | £100.08 | £98.83–£101.01 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 12_months | £100.07 | £98.37–£101.47 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

All 25%/50%/75%/100% capital-loss and 20%/25%/50%/75% drawdown probabilities, mean/95th maximum drawdowns, and 5th/25th/50th/75th/95th ending-equity percentiles are saved in `research.json`. Zero large-loss observations arise from the very sparse, tightly sized source returns; they do not establish zero market risk. No defensible precise real-world 30-day return or ruin probability can be inferred from this sample. The observed bootstrap median is about £100, and the correct trading decision is to preserve cash rather than present that as a guaranteed forecast.

## 8. Exact selected rules and risk model

Research candidate only: XLE daily close below its 20-day simple mean minus 2 sample standard deviations. Next observed open entry; initial stop 2×14-day simple-average true range below execution price; target twice that distance above entry; maximum hold 10 calendar days. No shorting, averaging down or martingale. After exit, the same signal ID cannot reenter. Confidence is null because predictive calibration has not been established.

Real-capital risk justified by current evidence: **0%**. Research-shadow budget: **0.5% per trade**, total/correlated notional at most 25%, one position, cash only, no leverage. Fees plus modeled adverse stop-exit costs are included in sizing. Quantity is rounded down to the venue increment; insufficient economically/venue-valid size produces NO_TRADE. Stops are not guarantees against gaps.

Conservative engineering defaults, not optimized empirical constants: 2% daily loss disables new entries for that day; 4% weekly loss disables new entries for that week; at half the weekly limit or after three consecutive nonpositive trades, size halves; 10% peak drawdown latches a halt requiring review. Maximum modeled spread/slippage each 20 bp; data age at most 90 seconds, clock skew at most 5 seconds. Current implementation uses one position, so correlated exposure cannot be diversified away.

Fixed fractional, ATR, volatility targeting, inverse-volatility single-asset allocation and fractional/full Kelly sizing arithmetic is available in `sizing_comparison`. Actual orders use only the capped fee-aware fixed-risk method. Full Kelly is analytical only and may be enormous under a falsely assumed edge; this sample does not justify an estimated Kelly allocation. No risk adjustment depends on progress toward £1,000.

## 9. Implementation, monitoring and verification

[Architecture and full audit](small-account-architecture.md). Historical data → causal signal → independent risk → fractional quantity → simulated order/fill → position → exit → cash/P&L → SQLite journal → dashboard is exercised with hand-calculated scenarios and actual historical replay. Runtime handling includes duplicate messages, stale/invalid quotes, clock skew, disconnected feed, inconsistent balance, unknown order state, rejection, partial IOC execution, database failure/rollback and concurrent writers.

Rolling monitoring exposes win rate, expectancy, profit factor, Sharpe and drawdown, and explicitly marks insufficient samples. An optional frozen matching validation baseline with at least 30 trades can enable approximate 99% one-sided mean-P&L degradation detection after 30 paper closes; hard loss limits do not depend on it. This is an engineering alarm, not a universally calibrated sequential test. A halted state is persisted; no automatic restart/upsizing.

Journal fields include timestamp, instrument, signal time, reason, setup, regime, entry/stop/target, requested/filled quantity, spread, slippage, fees, expected risk/reward, exit reason, realized P&L and rule adherence. Unvalidated confidence is null. Source timestamp limitations are labeled. No external notification endpoint or messaging credential is configured.

Baseline: 323 offline tests passed. The final expanded suite and its exact result are recorded in `results/small_account/verification.json`. Five pre-existing Quantstats/NumPy warnings occurred. Network-dependent Alpaca integration and unrelated existing browser suites are excluded from this count. A dedicated local Chrome render and Streamlit AppTest verify the new monitor. A live public-data paper cycle recorded healthy quotes, no signal and £100 equity. No 30-day paper history is claimed.

## 10. Runbook

From the project directory (existing `.venv`):

```bash
# Re-run complete research and 100,000 paths per scenario.
.venv/bin/python examples/research_small_account.py

# Verify frozen input/code hashes and exact equity replay.
.venv/bin/python examples/replay_small_account.py

# Journal historical replay into a NEW database (not forward evidence).
.venv/bin/python examples/replay_small_account.py --database results/small_account/replay-new.sqlite

# Public live data; default unqualified strategy remains NO_TRADE.
TRADING_MODE=PAPER .venv/bin/python examples/paper_small_account.py --once

# Continuous local paper monitor, 60-second polling; Ctrl-C stops it.
TRADING_MODE=PAPER .venv/bin/python examples/paper_small_account.py

# Explicit hypothetical shadow experiment; separate journal, never real orders.
TRADING_MODE=PAPER .venv/bin/python examples/paper_small_account.py --shadow --database results/small_account/shadow.sqlite

# Dashboard (default monitors paper.sqlite; results path is editable).
.venv/bin/python -m streamlit run dashboard/small_account_app.py

# Main app also includes the £100 Research tab.
.venv/bin/python -m streamlit run dashboard/app.py

# Unit/integration tests with no private API calls.
.venv/bin/python -m pytest tests --ignore=tests/test_catalog_integration.py --ignore=tests/e2e -q
```

The public-feed paper example defaults to BTC/GBP Donchian 20 as a connectivity/research template, not the selected XLE hypothesis or a qualified alternative. The selected XLE strategy is integrated in the shared engine and snapshot replay; live XLE/UK-equivalent execution is deliberately unavailable until data, product access and evidence qualify it.

`STOP_TRADING=true` is read immediately before each proposed entry. Set it in the process environment before launch/restart; modifying an unrelated shell's environment cannot change an already running process. Ctrl-C stops the local process immediately. For a running process, create `results/small_account/STOP_TRADING` (or the path in `STOP_TRADING_FILE`); every proposed entry checks this file, so no restart is needed. The switch blocks new entries and does not promise automatic external flattening; there are no external positions created by this path. A persisted hard halt also blocks entries. No live activation recipe exists: `TRADING_MODE=LIVE`, even with `LIVE_TRADING_CONFIRMED=true`, fails closed. Do not remove guards to pursue the target.

## 11. Remaining risks and ranked next experiments

1. Collect genuinely new forward observations with frozen rules and costs. A month is a calendar milestone, not enough statistical evidence for a low-frequency rule. Preserve every rejected setup and quotation.
2. Verify a UK-accessible fractional cash ETF, broker permissions, current API order support and venue minima; gather bid/ask quotes and actual GBP FX series. Re-test that exact instrument, not a renamed US proxy.
3. Repair/extend historical coverage, verify corporate actions/distributions, include pre-2020 crises and wider point-in-time universes. Check market-calendar gaps rather than only OHLC validity.
4. Test lower-turnover diversified trend/relative strength against buy-and-hold and cash, with UK execution costs and much longer horizons. Better supported general literature does not guarantee a single-account edge.
5. Pre-register a larger independent test and simultaneous/multiple-testing correction. Bootstrap uncertainty in estimated edge, not only the order of observed returns. Do not repeatedly reuse this final quarter as new OOS evidence.
6. Only after evidence and capital justify it: authenticated broker paper reconciliation, broker-native stops, restart/partial-fill recovery, exchange status and order timeouts; then independent operational review before real money.
7. Only if simple rules already survive: nested OOS tests of regime filters or a logistic/LightGBM overlay. Reject improvements smaller than cost/estimation uncertainty.

Unresolved items are explicit: no professional market-data entitlement, no historical NBBO/queue data, no validated funding/borrow/event sample, no authenticated new broker execution, no crash/default probability estimate, no qualified ML confidence, and no demonstrated profitable strategy. Brokerage minimums, laws and fees can change after this research date.

## 12. Answers to the final recommendation questions

1. Best **validation-ranked** candidate was XLE Bollinger reversion; no deployable winner.
2. Tested six US ETFs and BTC/GBP/ETH/GBP; other markets were researched and excluded, not backtested without data.
3. Reversion may compensate liquidity provision/temporary pressure; that mechanism does not establish net tradable alpha here.
4. Relevant academic evidence and positive development windows support investigation.
5. Eight later trades, near-zero return, cost sensitivity, selection bias and lack of true new OOS contradict deployment.
6. Modeled round trips: ~38 bp equity proxy and ~180 bp crypto; actual quotes/venue/account matter.
7. A reliable future monthly return cannot be estimated. The candidate's historical geometric monthly rate is ~0.0011%, not a forecast.
8. Conditional 30-day bootstrap median: £100.00; cash NO_TRADE also remains £100 before inflation/interest.
9. Zero £1,000 hits in 100,000 empirical bootstrap paths; not a precise real-world probability. Extreme assumed-edge probabilities are tabulated separately.
10. Zero large-capital-loss observations in that sparse bootstrap; actual loss probabilities unknown. Aggressive hypothetical cases quantify much larger downside and operational failure.
11. Statistically justified real-money risk is 0%; 0.5% is a conservative research budget, not an empirically optimal fraction.
12. 900% return / ~11.59% per equity session, or many unusually favorable leveraged/risky outcomes, would be required.
13. Yes: high risk raises severe-loss probability dramatically; at 20% risk under the stated 60-trade stress model, about 85% hit ≤£50 and about 16.5% reach exact zero. These depend on the assumed gap model.
14. Best growth/survival trade-off supported now: retain the £100, collect evidence in paper, and investigate low-cost low-turnover cash strategies over a longer horizon. Trying to force 10× now is gambling-level risk, not a demonstrated investment edge.

Artifacts: `research.json`, `monte_carlo.json`, `strategy_comparison.csv`, `scenario_probabilities.csv`, `candidate_probabilities.csv`, `equity.csv`, `journal.csv`, hashed input Parquet, `paper.sqlite`, `paper.status.json`, `secret_audit.json`, `research_summary.png` and `dashboard.png` under `results/small_account/`. Generated outputs are intentionally gitignored; this report and code are reviewable working-tree changes.
