"""Build reviewable research tables/plots from saved experiment outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT=Path('results/small_account')
r=json.loads((ROOT/'research.json').read_text()); mc=json.loads((ROOT/'monte_carlo.json').read_text())
rows=[]
for x in r['candidates']:
    row={'id':x['id'],'symbol':x['symbol'],**x['setup'],'eligible':x['eligible']}
    for split in ('train','validation','diagnostic'):
        row.update({split+'_'+k:v for k,v in x[split].items()})
    rows.append(row)
pd.DataFrame(rows).to_csv(ROOT/'strategy_comparison.csv',index=False)
mcrows=[]
for x in mc['scenarios']:
    for horizon,v in x['horizons'].items():
        mcrows.append({'win_probability':x['p'],'reward_risk':x['reward_risk'],'risk':x['risk'],
                       'cost_r':x['cost_r'],'horizon':horizon,'expectancy_r':x['expectancy_r'],
                       'expected_log_growth':x['expected_log_growth'],'kelly_analytic':x['kelly_analytic'],**v})
pd.DataFrame(mcrows).to_csv(ROOT/'scenario_probabilities.csv',index=False)
pd.DataFrame([{'horizon':h,**v} for h,v in r['monte_carlo']['base']['horizons'].items()]).to_csv(ROOT/'candidate_probabilities.csv',index=False)

fig,axs=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
e=pd.read_csv(ROOT/'equity.csv'); axs[0].plot(pd.to_datetime(e.ts),e.equity,color='#216e90')
axs[0].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3,maxticks=6)); axs[0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[0].xaxis.get_major_locator()))
axs[0].axhline(100,color='gray',ls='--'); axs[0].set(title='Rejected research candidate: later historical period',ylabel='Equity (£)',xlabel='Date')
xs=mc['extreme_60trades']; labels=[f"{x['risk']:.0%} risk\n{'stress' if x['stress'] else 'IID'}" for x in xs]
h=[x['horizons']['30_days_60_trades'] for x in xs]; pos=np.arange(len(xs))
axs[1].bar(pos-.18,[x['hit_1000']*100 for x in h],.36,label='Ever reaches £1,000',color='#216e90')
axs[1].bar(pos+.18,[x['loss_50']*100 for x in h],.36,label='Ever falls to £50',color='#bb5544')
axs[1].set(xticks=pos,xticklabels=labels,ylabel='Conditional probability (%)',title='Assumed 45% wins, 2R reward, 60 trades — not an estimated edge')
axs[1].legend(fontsize=8); fig.savefig(ROOT/'research_summary.png',dpi=160); plt.close(fig)

def pct(x): return f'{x*100:.2f}%'
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)
sel=r['selected']; d=sel['diagnostic']; req=mc['requirements']
familyrows=[]
for family in ('donchian','ma','rsi','bollinger','volume','gap'):
    x=max((x for x in r['candidates'] if x['setup']['family']==family),key=lambda x:x['score']); v=x['diagnostic']
    familyrows.append([x['id'],f"{x['validation']['sharpe']:.3f}",pct(v['total_return']),f"{v['sharpe']:.3f}" if v['sharpe'] is not None else 'N/A',v['trades'],pct(v['max_drawdown'])])
scenario_rows=[]
for risk,name in ((.005,'A Conservative'),(.01,'B Moderate'),(.03,'C Aggressive'),(.1,'D Extreme')):
    x=next(x for x in mc['scenarios'] if x['p']==.45 and x['risk']==risk); h=x['horizons']['30_days']
    scenario_rows.append([name,pct(risk),f"£{h['p50']:.2f}",f"£{h['p5']:.2f}–£{h['p95']:.2f}",pct(h['hit_1000']),pct(h['loss_25']),pct(h['loss_50']),x['trades_to_10x_at_log_mean']])
extreme_rows=[]
for x in mc['extreme_60trades']:
    h=x['horizons']['30_days_60_trades']; extreme_rows.append([pct(x['risk']),'stress' if x['stress'] else 'IID',f"£{h['p50']:.4f}",pct(h['hit_1000']),pct(h['end_above_1000']),pct(h['loss_50']),pct(h['loss_75']),pct(h['loss_100']),pct(h['operational_failure_below_1'])])
prob_rows=[]
for horizon,h in r['monte_carlo']['base']['horizons'].items():
    prob_rows.append([horizon,f"£{h['p50']:.2f}",f"£{h['p5']:.2f}–£{h['p95']:.2f}",*[pct(h[f'hit_{t}']) for t in (125,150,200,250,500,1000)]])
expectancy=[]
for p,b in ((.4,2),(.45,2),(.5,1.5),(.55,1.5),(.6,1),(.65,1)):
    expectancy.append([pct(p),b,f'{p*b-(1-p):.3f}R',f'{p*b-(1-p)-.1:.3f}R',pct((p*b-(1-p)-.1)/((b-.1)*1.1))])
text=f'''# £100 trading research: findings, implementation and runbook

Research date: 2026-09-30. **Decision: NO_TRADE for real capital.** A working research and local paper framework is implemented; no strategy has enough independent evidence to justify risking this £100. The experimental 10× target did not affect candidate selection, sizing or risk limits.

## 1. Executive findings

- Compared **144 fixed candidates**: six rule families × three nearby periods × eight instruments. **{r['eligible_count']} qualified** under the stated development gate.
- Highest validation score: **{sel['id']}**. Its later historical diagnostic ended at **£{d['final_equity']:.4f}**, {pct(d['total_return'])} total across {d['observations']} observations, with only **{d['trades']} trades**. Doubling modeled fees/slippage made it lose money.
- This later period was chronologically separated in this run, but prior repository searches already exposed historical data. It is **not a pristine out-of-sample sample**. No claim of a statistically validated positive edge is made.
- Public live BTC/GBP data was fetched successfully; the default paper account stayed at £100 with no trade. That is one connectivity/behavior check, not 30 days of forward evidence.
- A full-account 10× needs **900% return**, **{pct(req['calendar_daily'])} per calendar day**, **{pct(req['trading_daily_21'])} per trading day over 21 sessions**, or **{pct(req['weekly_30_days'])} compounded every seven days over 30 days**. Four equal weeks instead requires {pct(req['weekly_four_weeks'])} per week.

## 2. Research coverage and what was actually tested

[Market/broker and literature review](small-account-market-evidence.md) covers UK/US shares, ETFs, FX, spot crypto, futures, options, CFDs, spread betting, prediction markets, and ten named broker/exchange routes. Markets rejected for access, capital, cost or missing execution data did not receive fabricated backtests.

Fresh tests used SPY, XLK, XLE, XLV, XLF, XLP daily ETF bars: **1,464 common sessions, 2020-07-27 to 2026-05-22**. BTC/GBP and ETH/GBP used **720 daily observations, 2024-10-10 to 2026-09-29**, fetched from Kraken's public endpoint. GBP crypto prices are directly denominated in GBP. ETF results are a constant-FX GBP-capital proxy, with 15 bp FX charges per side; they are not a reconstructed GBPUSD portfolio.

The data includes the 2022 equity bear market, later equity advances and volatile 2025 periods; it does **not** include the COVID crash, 2008, or a full crypto-cycle sample. The initial SPY file contained a 634-day gap; common-session alignment excluded its isolated early bar. No missing prices were invented. Input Parquet snapshots and SHA-256 hashes are saved. IEX venue coverage, corporate-action verification, dividends and distribution adjustment remain limitations. Fixed current ETFs reduce but do not eliminate universe-selection bias.

## 3. Test design and strategy comparison

Fixed families: prior-high Donchian breakout, moving-average upward cross, RSI crossing back above 30, Bollinger lower-band reversion, price/volume continuation, and positive-gap continuation. Periods 18/20/22; 10-calendar-day maximum hold; 2×14-bar ATR stop; 2R target. Gap continuation is a price proxy, not an earnings-event strategy.

First 50%: descriptive training. Next 25%: validation and three chronological reset-flat subwindows. Final 25%: later diagnostics. Ranking uses validation Sharpe minus maximum drawdown. Development qualification requires positive return in all three windows, at least 12 completed validation trades, and a positive 99% circular-block bootstrap mean-return lower bound. The 12-trade floor is a minimal rejection filter, not proof of sufficiency. Pointwise intervals are not a correction for 144 searches. All candidates remain deployment-disabled regardless.

Signals use completed observations. Orders execute on the subsequent observed open, with spread, slippage and fees. Stops gap at the open when necessary; stop wins if stop and target both touch. Fractional quantity is rounded down and fees reserved. Every entry passes the central risk engine. Positions start flat at evaluation boundaries and are liquidated at the final close; that forced endpoint is explicitly an evaluation convention. Indicator history before each boundary is warm-up only. Positions and P&L never cross training/validation boundaries.

Family representatives below were chosen by **validation**, not by the later result:

{table(['Validation-selected representative','Validation Sharpe','Later total return','Later Sharpe','Trades','Max drawdown'],familyrows)}

The full train/validation/diagnostic metrics for every candidate are in `results/small_account/strategy_comparison.csv` and `research.json`: CAGR, monthly geometric return, daily mean, Sharpe, Sortino, Calmar, profit factor, expectancy, drawdown/recovery, win/loss sizes, reward/risk, trade count, exposure, turnover and fees. Undefined ratios are null, not fabricated finite scores.

Selected diagnostic: CAGR {pct(d['cagr'])}; monthly geometric {pct(d['monthly_geometric'])}; Sharpe {d['sharpe']:.4f}; Sortino {d['sortino']:.4f}; profit factor {d['profit_factor']:.4f}; win rate {pct(d['win_rate'])}; average win £{d['average_win']:.4f}; average loss £{d['average_loss']:.4f}; realized payoff ratio {d['reward_risk_realized']:.3f}; average trade expectancy £{d['expectancy_gbp']:.4f}. A nominal 2R target did not produce a realized 2:1 payoff because time exits and stops determine actual outcomes.

## 4. Cost, parameter and exit robustness

{table(['Fee/slippage multiplier','Signal-to-entry bars','Later return','Later Sharpe'],[[x['cost_multiple'],x['delay_bars'],pct(x['metrics']['total_return']),f"{{x['metrics']['sharpe']:.3f}}".format(x=x)] for x in []]) if False else table(['Fee/slippage multiplier','Signal-to-entry bars','Later return'],[[x['cost_multiple'],x['delay_bars'],pct(x['metrics']['total_return'])] for x in r['cost_stress']])}

The stress multiplier increases fee and slippage assumptions; spread is held fixed. One additional bar of execution delay is separately replayed. Stop variants fixed 2%, 2×ATR, prior-low structure and trailing ATR were crossed with 1R/1.5R/2R/3R targets **on development data only**; all 16 comparisons are saved. None was promoted using final-period performance. Neighboring periods 18/20/22 are retained for inspection rather than pruning poor neighbors away.

Previously stored intraday ORB discovery at `results/opening_range_discovery_reject_20260914T091855Z/summary.json` says DISCOVERY_REJECT, holdout unopened, with missing quote/event readiness. Other previously reported intraday search failures were not rebranded as successful strategies. No ML overlay, regime switch, order-book alpha, funding carry or pairs strategy was deployed without suitable evidence/data.

## 5. £100 → £1,000 mathematics

For win probability p, gross winning payoff bR, losing payoff −1R and cost cR:

`E[R] = p*b − (1−p) − c`

`Var(R) = p*(1−p)*(b+1)^2`

`g(f) = p*log(1+f*(b−c)) + (1−p)*log(1−f*(1+c))`

Geometric growth uses `exp(g)-1`, not arithmetic expectancy. A 1% price stop requires notional/equity approximately `f / 0.01`: risking 10% of equity implies about 10× exposure, and 20% implies about 20×, before reserving costs. This is incompatible with the new cash-only 25% notional cap. The aggressive simulations are analytical comparisons, not executable portfolio plans. At 10× exposure, a 38 bp round trip would consume about 3.8% of equity; the base 0.1R cost assumption at 10% risk is only 1%, showing how optimistic the assumed-edge scenario can be.

The approximate number of trades to multiply wealth by 10 at the log mean is `ceil(log(10)/g)` only when g>0; it is not a promised hitting time. At fixed costs, variance drag can make g negative despite positive arithmetic expectancy.

{table(['Win rate','Gross reward:risk','Gross expectancy','Net at 0.1R costs','Full Kelly, analytical only'],expectancy)}

With a fixed fractional risk f, exact terminal tail probabilities use a binomial count of wins and product `(1+f*(b-c))^W * (1-f*(1+c))^(N-W)`. This also detects impossible targets even if a Monte Carlo run sees zero hits. For example, 21 consecutive 2R wins at 1% risk and 0.1R costs cannot produce 10×. The script records required wins and exact terminal probability for all 48 combinations.

## 6. Conditional Monte Carlo — 100,000 paths per case

Seed 20260930. Six win/payoff combinations × eight risk fractions (0.25%, 0.5%, 1%, 2%, 3%, 5%, 10%, 20%) × 100,000 paths; recorded after 21, 63, 126 and 252 trades, interpreted as one opportunity per equity session over approximately 30/90 days/6/12 months. Additional 60-trade cases include 50% risk solely to illustrate failure, never as executable configuration. Five cost scenarios span 0 to 1.6R. Overall there are 61 binary scenarios (6.1 million simulated paths), plus two 100,000-path empirical bootstraps.

These scenarios **assume** the edge; none of the listed win rates was validated for future trading. Base cost 0.1R is optimistic relative to many short-stop retail strategies. No shortage of setups, signal dependence, margin limit or participation constraint is modeled in the IID case; real small-account feasibility is worse.

At assumed 45% wins and 2R winners, 21 opportunities:

{table(['Scenario','Risk/trade','Median end','5th–95th','Hit £1,000','Ever ≤£75','Ever ≤£50','Trades at log mean to 10×'],scenario_rows)}

For 60 opportunities in 30 days, the same assumed edge:

{table(['Risk','Model','Median end','Ever £1,000','End ≥£1,000','Ever ≤£50','Ever ≤£25','Exact zero','Ever <£1'],extreme_rows)}

Stress model: transition to adverse win-rate regime with 5% probability per trade, 90% persistence; win probability falls by 15 percentage points; cost becomes lognormal with log-sigma 0.5; a 0.5% per-trade independent gap event makes a losing trade lose 5R. Losses are clipped at 100% of equity. These are **illustrative sensitivity assumptions, not fitted market frequencies**. They quantify dependence on tail assumptions rather than estimate this trader's real ruin probability.

“Hit” means ever reached the target; trading continues afterward. “End” means terminal wealth. The same path may hit £1,000 and later lose most of it. Loss probabilities use starting £100; drawdown probabilities use each path's running peak. Ruin at exactly £0 is mathematically impossible for finite bounded fractional losses smaller than 100%; its zero estimate is not reassurance. Operational failure is separately measured at <£1. Model counterparty/default risks are unquantified.

Zero hits in 100,000 simulations gives an approximate one-sided 95% bound of **0.003% within that model**. It is not a confidence bound on unknown real-world probabilities.

## 7. Empirical-candidate probability analysis

Circular 5-observation block bootstrap of the frozen candidate's 366 later-period net daily returns. 100,000 paths; 21/63/126/252 observation horizons. It preserves short local dependence but assumes the observed process is representative; eight trades are insufficient for reliable tail inference. The stressed bootstrap adds persistent adverse regimes, 1.5× volatility and random active-day execution drag. It is not a reconstructed order-book simulation or a replay of all risk halts.

{table(['Horizon','Median end','5th–95th','Hit £125','£150','£200','£250','£500','£1,000'],prob_rows)}

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
'''
Path('docs/small-account-research.md').write_text(text)
print('Generated docs/small-account-research.md, tables and research_summary.png')
