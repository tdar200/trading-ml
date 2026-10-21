"""
The single execution unit of the Stage-3 harness: run one RunSpec, return one
RunResult. The only file in analysis/ that imports a BacktestEngine.

The strategy's _equity_history (Stage-2 mark-to-market snapshots) is the
source of truth for the equity curve. Trade counts come from cache.orders()
filtered to fills (verified path in docs/nautilus_api_notes.md §9).
"""
from __future__ import annotations

import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderStatus
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.instruments import Equity

from analysis.metrics import cagr, max_drawdown, sharpe
from analysis.runspec import RunResult, RunSpec
from strategies.backtest_setup import add_default_backtest_venue


def _required_tickers(spec: RunSpec) -> set[str]:
    """Return the instruments explicitly required by a strategy config."""
    config = spec.strategy_config
    universe = set(getattr(config, "universe", ()) or ())
    benchmark = getattr(config, "benchmark", None)
    if benchmark:
        universe.add(str(benchmark))
    return universe


def _validate_inputs(
    spec: RunSpec,
    bars_by_ticker: dict[str, list[Bar]],
    instruments: list[Equity],
) -> None:
    """Fail closed on data defects which can invalidate a backtest."""
    required = _required_tickers(spec)
    missing = sorted(t for t in required if not bars_by_ticker.get(t))
    if missing:
        raise ValueError(f"missing bars for required tickers: {missing}")

    instrument_ids = {str(instrument.id) for instrument in instruments}
    for ticker in sorted(required or bars_by_ticker.keys()):
        bars = bars_by_ticker.get(ticker, [])
        previous_ts: int | None = None
        for bar in bars:
            if previous_ts is not None and bar.ts_event <= previous_ts:
                kind = "duplicate" if bar.ts_event == previous_ts else "unsorted"
                raise ValueError(f"{kind} bar timestamp for {ticker}: {bar.ts_event}")
            previous_ts = bar.ts_event
            actual_ticker = str(bar.bar_type).split(".", 1)[0]
            if actual_ticker != ticker:
                raise ValueError(
                    f"bar mapping mismatch: key {ticker!r} contains {actual_ticker!r} bar"
                )
            if str(bar.bar_type.instrument_id) not in instrument_ids:
                raise ValueError(f"no instrument supplied for {bar.bar_type.instrument_id}")


def _bars_in_window(bars: list[Bar], start_ns: int, end_ns: int) -> list[Bar]:
    return [b for b in bars if start_ns <= b.ts_event < end_ns]


def _equity_curve_from_strategy(strategy) -> pd.Series:
    """Mark-to-market daily equity from strategy._equity_history (Stage-2 mechanism)."""
    history = getattr(strategy, "_equity_history", None) or []
    if not history:
        return pd.Series(name="equity", dtype=float)
    rows = [
        (pd.Timestamp(ts_ns, unit="ns", tz="UTC").tz_convert(None).normalize(), eq)
        for ts_ns, eq in history
    ]
    df = pd.DataFrame(rows, columns=["date", "equity"]).groupby("date").last()
    return df["equity"].sort_index()


def _count_fills(engine: BacktestEngine) -> int:
    n = 0
    for order in engine.cache.orders():
        if order.status == OrderStatus.FILLED:
            n += 1
    return n


def _compute_metrics(equity: pd.Series, n_trades: int) -> dict[str, float]:
    if len(equity) == 0:
        return {
            "cagr": float("nan"), "sharpe": float("nan"),
            "max_drawdown": float("nan"), "total_return": float("nan"),
            "final_equity": 0.0, "n_days": 0, "n_trades": n_trades,
        }
    returns = equity.pct_change().dropna()
    final = float(equity.iloc[-1])
    initial = float(equity.iloc[0])
    total_return = (final / initial - 1.0) if initial > 0 else float("nan")
    return {
        "cagr": cagr(returns),
        "sharpe": sharpe(returns),
        "max_drawdown": max_drawdown(equity),
        "total_return": total_return,
        "final_equity": final,
        "n_days": len(equity),
        "n_trades": n_trades,
    }


def run(
    spec: RunSpec,
    bars_by_ticker: dict[str, list[Bar]],
    instruments: list[Equity],
) -> RunResult:
    """Execute the backtest defined by `spec` against the provided bars + instruments."""
    if spec.validate_data:
        _validate_inputs(spec, bars_by_ticker, instruments)
    engine_config = BacktestEngineConfig(
        trader_id=TraderId(f"HARNESS-{spec.label[:32]}"),  # TraderId requires a hyphen
        logging=LoggingConfig(log_level=spec.log_level),
    )
    engine = BacktestEngine(config=engine_config)
    venue = add_default_backtest_venue(
        engine,
        starting_cash=spec.starting_cash,
        slippage_prob=spec.slippage_prob,
        random_seed=spec.random_seed,
    )

    for instrument in instruments:
        engine.add_instrument(instrument)

    start_ns = int(spec.start.timestamp() * 1e9)
    end_ns = int(spec.end.timestamp() * 1e9)
    for ticker, bars in bars_by_ticker.items():
        windowed = _bars_in_window(bars, start_ns, end_ns)
        if windowed:
            engine.add_data(windowed)

    strategy = spec.strategy_class(config=spec.strategy_config)
    engine.add_strategy(strategy)
    engine.run()

    equity = _equity_curve_from_strategy(strategy)
    n_trades = _count_fills(engine)
    metrics = _compute_metrics(equity, n_trades)
    # NOTE: deliberately NOT calling engine.dispose() — it tears down the
    # process-global Rust logging system, and a subsequent BacktestEngine in
    # the same process will panic on re-init. The harness runs MANY engines
    # per process (e.g. ~90 for walk-forward); GC handles cleanup. Matches
    # the pattern used in tests/test_momentum_strategy.py.
    return RunResult(spec=spec, metrics=metrics, equity_curve=equity)
