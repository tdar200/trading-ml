"""Buy-and-hold versus the strategy, per instrument, on identical windows, capital and costs.

Strategy per instrument = the candidate with the best validation score (eligible ones first), chosen
BEFORE the diagnostic window was opened, from the 18 pre-registered candidates in
results/small_account/research.json. Buy-and-hold puts the same GBP 100 in at the open of the first bar
(the engine's own entry convention), pays the same per-side costs on entry and on the final liquidation,
and holds. Windows are the research windows: train = bars 60..50%, validation = ..75%, diagnostic = last 25%.

    .venv/bin/python examples/compare_buy_and_hold.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.small_account_research import replay  # noqa: E402
from strategies.small_account import Setup  # noqa: E402

# Current-source research after the 30 Sep audit repairs (results/small_account is the pre-audit record).
SOURCE = Path('results/qa_audit_20260930/corrected_research')
OUT = Path('results/buy_and_hold_comparison')
SYMBOLS = ('SPY', 'XLE', 'XLF', 'XLK', 'XLP', 'XLV', 'XBTGBP', 'ETHGBP')


def costs(symbol):
    crypto = symbol.endswith('GBP')
    return (80. if crypto else 15.) / 10000, (10. if crypto else 2.), (5. if crypto else 3.), (365 if crypto else 252)


def hold(frame, begin, end, symbol, capital=100.):
    """Buy at the open of bar `begin`, hold, liquidate at the close of bar end-1. Returns (equity curve per bar, summary)."""
    fee, spread, slip, annual = costs(symbol)
    o = frame.open.to_numpy(float)
    c = frame.close.to_numpy(float)
    entry = o[begin] * (1 + (spread / 2 + slip) / 10000)
    qty = capital / (entry * (1 + fee))                         # fully invested, fee reserved
    marks = qty * c[begin:end]
    exit_price = c[end - 1] * (1 - (spread / 2 + slip) / 10000)
    marks = marks.copy()
    marks[-1] = exit_price * qty * (1 - fee)                    # the final bar is the liquidation, as in the engine
    eq = np.r_[capital, marks]
    r = eq[1:] / eq[:-1] - 1
    peak = np.maximum.accumulate(eq)
    std = r.std(ddof=1) if len(r) > 1 else 0.
    gross = c[end - 1] / o[begin] - 1
    return eq, {'final_gbp': eq[-1], 'return': eq[-1] / capital - 1, 'gross_return': gross,
                'max_drawdown': float((1 - eq / peak).max()), 'sharpe': float(r.mean() / std * np.sqrt(annual)) if std > 0 else None,
                'observations': len(r)}


def main():
    research = json.loads((SOURCE / 'research.json').read_text())
    by_symbol = {}
    for cand in research['candidates']:
        by_symbol.setdefault(cand['symbol'], []).append(cand)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, chains = [], {}
    for symbol in SYMBOLS:
        frame = pd.read_parquet(SOURCE / f'{symbol}.parquet').sort_values('ts').reset_index(drop=True)
        n = len(frame)
        a, b = int(n * .5), int(n * .75)
        windows = {'train': (60, a), 'validation': (a, b), 'diagnostic': (b, n)}
        cands = sorted(by_symbol[symbol], key=lambda x: (not x['eligible'], -x['score']))
        pick = cands[0]
        setup = Setup(**pick['setup'])
        fee, spread, slip, annual = costs(symbol)
        window_curves = {}
        for name, (begin, end) in windows.items():
            engine = replay(frame, setup, symbol=symbol, begin=begin, end=end, fee_bps=fee * 10000, spread_bps=spread, slip_bps=slip)
            curve = np.r_[100., [x['equity'] for x in engine.curve]]
            window_curves[name] = curve
            bh_curve, bh = hold(frame, begin, end, symbol)
            stored = pick[name]
            strategy_return = curve[-1] / 100 - 1
            assert abs(strategy_return - stored['total_return']) < 1e-9, (symbol, name, strategy_return, stored['total_return'])
            diag_all = [c[name]['total_return'] for c in by_symbol[symbol]] if name in pick else []
            rows.append({
                'instrument': symbol, 'window': name, 'start': str(frame.ts[begin].date()), 'end': str(frame.ts[end - 1].date()), 'bars': end - begin,
                'strategy': f"{pick['setup']['family']}_{pick['setup']['period']}",
                'strategy_return': strategy_return, 'buy_hold_return': bh['return'], 'buy_hold_gross_return': bh['gross_return'],
                'strategy_minus_hold': strategy_return - bh['return'],
                'strategy_final_gbp': curve[-1], 'buy_hold_final_gbp': bh['final_gbp'],
                'strategy_max_drawdown': stored['max_drawdown'], 'buy_hold_max_drawdown': bh['max_drawdown'],
                'strategy_sharpe': stored['sharpe'], 'buy_hold_sharpe': bh['sharpe'],
                'strategy_trades': stored['trades'], 'strategy_exposure': stored['exposure'],
                'all_18_candidates_median_return': float(np.median(diag_all)), 'all_18_candidates_beating_hold': int(sum(r > bh['return'] for r in diag_all)),
            })
        # the whole span: the strategy windows chained (flat at each boundary), the hold a single round trip
        chained = np.concatenate([window_curves['train'], window_curves['validation'][1:] * window_curves['train'][-1] / 100,
                                  window_curves['diagnostic'][1:] * window_curves['train'][-1] / 100 * window_curves['validation'][-1] / 100])
        bh_full, bhf = hold(frame, 60, n, symbol)
        rows.append({
            'instrument': symbol, 'window': 'full span (chained)', 'start': str(frame.ts[60].date()), 'end': str(frame.ts[n - 1].date()), 'bars': n - 60,
            'strategy': f"{pick['setup']['family']}_{pick['setup']['period']}", 'strategy_return': chained[-1] / 100 - 1, 'buy_hold_return': bhf['return'],
            'buy_hold_gross_return': bhf['gross_return'], 'strategy_minus_hold': chained[-1] / 100 - 1 - bhf['return'],
            'strategy_final_gbp': chained[-1], 'buy_hold_final_gbp': bhf['final_gbp'],
            'strategy_max_drawdown': float((1 - chained / np.maximum.accumulate(chained)).max()), 'buy_hold_max_drawdown': bhf['max_drawdown'],
            'strategy_sharpe': None, 'buy_hold_sharpe': bhf['sharpe'], 'strategy_trades': sum(r['strategy_trades'] for r in rows[-3:]),
            'strategy_exposure': None, 'all_18_candidates_median_return': None, 'all_18_candidates_beating_hold': None})
        chains[symbol] = {'strategy': chained, 'hold': bh_full, 'diag_strategy': window_curves['diagnostic'],
                          'diag_hold': hold(frame, b, n, symbol)[0], 'dates': frame.ts}
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'comparison.csv', index=False)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for key, title, name in (('full', 'Full span (strategy windows chained)', 'equity_full_span.png'), ('diag', 'Final diagnostic window', 'equity_diagnostic.png')):
            fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharey=False)
            for ax, symbol in zip(axes.ravel(), SYMBOLS):
                d = chains[symbol]
                s, h = (d['strategy'], d['hold']) if key == 'full' else (d['diag_strategy'], d['diag_hold'])
                ax.plot(h, label='buy and hold', color='#1f77b4')
                ax.plot(s, label='strategy', color='#d62728')
                ax.set_title(f"{symbol}: hold {h[-1] - 100:+.1f}, strategy {s[-1] - 100:+.1f}", fontsize=10)
                ax.axhline(100, color='grey', lw=.5)
            axes[0, 0].legend(fontsize=8)
            fig.suptitle(f'GBP 100 invested: {title}', fontsize=13)
            fig.tight_layout()
            fig.savefig(OUT / name, dpi=110)
            plt.close(fig)
    except ImportError:
        pass
    print(table.to_string(index=False, float_format=lambda x: f'{x:.4f}'))


if __name__ == '__main__':
    main()
