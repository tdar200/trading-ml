"""
Backtest Lab — Stage 9 self-service strategy tester.

Tabs (sub-modes): Single | Grid sweep | Parameter Optimizer | Strategy Finder.

Strategy discovery via ``pkgutil`` walks ``strategies/`` for
``(Strategy, StrategyConfig)`` class pairs. The parameter form is
auto-rendered from the Pydantic ``StrategyConfig`` schema, so adding a
new strategy = drop a file in ``strategies/`` and the form appears.

All modes ultimately share ``analysis.runner.run`` as the canonical event
engine. Parameter Optimizer adds exhaustive/TPE search, non-overlapping windows,
robust scoring, and Pareto results instead of a single-period Sharpe winner.
"""
from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from data.tickers import ALL_TICKERS, BENCHMARK, SCALPING_STOCKS, UNIVERSE


_PROJECT_ROOT = Path(__file__).parent.parent.parent

# ---- pure helpers (tested in tests/test_backtest_lab.py) -------------------

def discover_strategies(strategies_pkg: str = "strategies") -> dict[str, tuple[type, type]]:
    """Walk ``strategies/`` for ``(Strategy, StrategyConfig)`` class pairs.

    Returns a dict {module_name: (StrategyClass, ConfigClass)}. Modules that
    don't expose at least one of each are skipped.
    """
    from nautilus_trader.config import StrategyConfig
    from nautilus_trader.trading.strategy import Strategy

    out: dict[str, tuple[type, type]] = {}
    pkg = importlib.import_module(strategies_pkg)
    for _, name, _ in pkgutil.iter_modules(pkg.__path__):
        try:
            mod = importlib.import_module(f"{strategies_pkg}.{name}")
        except Exception:
            continue
        strategies = [
            obj for _, obj in inspect.getmembers(mod, inspect.isclass)
            if obj is not Strategy and issubclass(obj, Strategy)
            and obj.__module__ == mod.__name__
        ]
        configs = [
            obj for _, obj in inspect.getmembers(mod, inspect.isclass)
            if obj is not StrategyConfig and issubclass(obj, StrategyConfig)
            and obj.__module__ == mod.__name__
        ]
        if strategies and configs:
            out[name] = (strategies[0], configs[0])
    return out


def make_param_grid(ranges: dict[str, list]) -> list[dict]:
    """Cross-product of parameter ranges → list of single-config dicts."""
    if not ranges:
        return [{}]
    keys = list(ranges.keys())
    return [dict(zip(keys, vals, strict=True))
            for vals in product(*(ranges[k] for k in keys))]


def valid_numeric_space(space: dict[str, dict]) -> bool:
    """Return whether every numeric optimizer range can be enumerated safely."""
    return bool(space) and all(
        spec.get("low") is not None
        and spec.get("high") is not None
        and spec.get("step") is not None
        and spec["low"] <= spec["high"]
        and spec["step"] > 0
        for spec in space.values()
    )


def config_fields(config_cls: type) -> dict[str, tuple[Any, Any]]:
    """Return {field_name: (annotation, default)} for SUBCLASS-defined fields
    of a msgspec-Struct StrategyConfig.

    Inherited fields from StrategyConfig / NautilusConfig / Struct base
    classes are intentionally excluded — we only want the strategy's own
    parameters to appear in the form.

    Note: msgspec's ``__struct_defaults__`` is RIGHT-ALIGNED with
    ``__struct_fields__`` — required fields take the leftmost slots without
    a default entry, then defaulted fields fill the rest. So the index
    into defaults is ``field_idx - n_required``.
    """
    import msgspec
    all_fields = config_cls.__struct_fields__
    all_defaults = config_cls.__struct_defaults__
    n_required = len(all_fields) - len(all_defaults)
    subclass_anns = config_cls.__annotations__  # only the subclass's own fields
    out: dict[str, tuple[Any, Any]] = {}
    for name, ann in subclass_anns.items():
        if name not in all_fields:
            continue
        idx = all_fields.index(name)
        if idx < n_required:
            default = None  # required field
        else:
            default = all_defaults[idx - n_required]
            if default is msgspec.NODEFAULT:
                default = None
        out[name] = (ann, default)
    return out


def field_default(name_or_field, config_cls: type | None = None) -> Any:
    """Backwards-compat helper. Two call shapes:

    - ``field_default(name, config_cls)`` → preferred; returns subclass default for `name`.
    - ``field_default(field_info)`` → legacy Pydantic-shaped FieldInfo with .default / .annotation.
    """
    if config_cls is not None:
        fields = config_fields(config_cls)
        return fields.get(str(name_or_field), (None, None))[1]
    # Legacy single-arg call (no longer used in the view but kept for test back-compat)
    field_info = name_or_field
    default = getattr(field_info, "default", None)
    if default is not None and default is not Ellipsis:
        return default
    return None


# ---- view -----------------------------------------------------------------

def _ann_kind(ann) -> str:
    """Classify an annotation (which may be a string under PEP 563 lazy
    evaluation OR an actual type) into one of: 'int', 'float', 'bool',
    'str', 'tuple-str', 'unknown'.

    Both shapes are common: ``from __future__ import annotations`` in the
    strategy file leaves ``__annotations__`` as raw strings; without it the
    annotations are resolved types.
    """
    if ann in (int, "int"):
        return "int"
    if ann in (float, "float"):
        return "float"
    if ann in (bool, "bool"):
        return "bool"
    if ann in (str, "str"):
        return "str"
    ann_str = str(ann).lower()
    if "tuple" in ann_str and "str" in ann_str:
        return "tuple-str"
    return "unknown"


def _render_param_form(config_cls: type, key_prefix: str) -> dict:
    """Auto-render a parameter form per the msgspec StrategyConfig schema.

    Iterates the subclass's own annotations (so the 16+ inherited base
    fields don't clutter the form). Returns a dict {field_name: value}
    ready to splat into the config constructor.

    Handles both PEP-563 stringified annotations and resolved types via
    ``_ann_kind`` — without that, ``int`` annotations evaluated as the
    string ``"int"`` would fall through to ``st.text_input``, and the user
    would submit a string into an int-typed config, producing a
    ``"can only concatenate str (not 'int') to str"`` TypeError downstream.
    """
    values: dict = {}
    for name, (ann, default) in config_fields(config_cls).items():
        widget_key = f"{key_prefix}__{name}"
        kind = _ann_kind(ann)
        if kind == "str":
            values[name] = st.text_input(name, value=str(default or ""), key=widget_key)
        elif kind == "int":
            values[name] = int(
                st.number_input(name, value=int(default if default is not None else 0),
                                step=1, key=widget_key)
            )
        elif kind == "float":
            values[name] = float(
                st.number_input(name, value=float(default if default is not None else 0.0),
                                format="%.4f", key=widget_key)
            )
        elif kind == "bool":
            values[name] = st.checkbox(name, value=bool(default), key=widget_key)
        elif kind == "tuple-str":
            values[name] = tuple(
                st.multiselect(name, options=list(ALL_TICKERS),
                               default=list(default) if default else list(UNIVERSE),
                               key=widget_key)
            )
        else:
            values[name] = st.text_input(name, value=str(default), key=widget_key)
    return values


def _run_single(strategy_cls, config_cls, params: dict, start, end) -> dict:
    """Run one backtest via analysis.runner.run, return metrics dict."""
    from analysis.runner import run as backtest_run
    from analysis.runspec import RunSpec
    from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
    from data.instruments import VENUE_MIC, build_universe_instruments

    catalog = ParquetDataCatalog(path=str((_PROJECT_ROOT / "data" / "catalog").resolve()))
    bars_by_ticker = {
        t: catalog.bars(instrument_ids=[f"{t}.{VENUE_MIC}"])
        for t in ALL_TICKERS
    }
    cfg = config_cls(**params)
    spec = RunSpec(
        label=f"lab-{strategy_cls.__name__}",
        strategy_class=strategy_cls,
        strategy_config=cfg,
        start=start, end=end,
        slippage_prob=0.0,
    )
    result = backtest_run(spec, bars_by_ticker=bars_by_ticker,
                          instruments=build_universe_instruments())
    return result.metrics


def _run_grid(strategy_cls, config_cls, base_params: dict,
              param_ranges: dict[str, list], windows) -> pd.DataFrame:
    """Walk-forward grid sweep via analysis.walk_forward.walk_forward."""
    from analysis.walk_forward import walk_forward
    from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
    from data.instruments import VENUE_MIC, build_universe_instruments

    catalog = ParquetDataCatalog(path=str((_PROJECT_ROOT / "data" / "catalog").resolve()))
    bars_by_ticker = {
        t: catalog.bars(instrument_ids=[f"{t}.{VENUE_MIC}"])
        for t in ALL_TICKERS
    }
    return walk_forward(
        strategy_class=strategy_cls,
        config_class=config_cls,
        base_config_kwargs=base_params,
        param_grid=param_ranges,
        windows=windows,
        bars_by_ticker=bars_by_ticker,
        instruments=build_universe_instruments(),
        slippage_prob=0.0,
    )


def _run_parameter_optimizer(
    strategy_cls, config_cls, base_params: dict, param_space: dict,
    start, end, *, algorithm: str, n_trials: int, n_windows: int,
    n_finalists: int = 20, device: str = "auto",
):
    """Run robust multi-window search through the canonical backtester."""
    from analysis.optimizer import (
        SearchSettings, build_non_overlapping_windows, evaluate_candidates,
        exhaustive_combinations, search,
    )
    from analysis.runner import run as backtest_run
    from analysis.runspec import RunSpec
    from analysis.vectorized_screen import bars_to_close_frame, screen_momentum, select_diverse_finalists
    from data.instruments import VENUE_MIC, build_universe_instruments
    from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

    catalog = ParquetDataCatalog(path=str((_PROJECT_ROOT / "data" / "catalog").resolve()))
    bars_by_ticker = {
        ticker: catalog.bars(instrument_ids=[f"{ticker}.{VENUE_MIC}"])
        for ticker in ALL_TICKERS
    }
    instruments = build_universe_instruments()
    windows = build_non_overlapping_windows(start, end, n_windows)

    def evaluator(params, window):
        result = backtest_run(
            RunSpec(
                label=f"finder-{window[0].date()}-{window[1].date()}",
                strategy_class=strategy_cls,
                strategy_config=config_cls(**params),
                start=window[0], end=window[1], slippage_prob=0.25,
                log_level="ERROR",
            ),
            bars_by_ticker=bars_by_ticker,
            instruments=instruments,
        )
        return result.metrics

    settings = SearchSettings(algorithm="grid" if algorithm == "hybrid" else algorithm, n_trials=n_trials)
    if algorithm == "hybrid":
        candidates = exhaustive_combinations(param_space)
        screen_candidates = [{**base_params, **candidate} for candidate in candidates]
        closes = bars_to_close_frame(bars_by_ticker, base_params["universe"])
        screened = screen_momentum(closes, screen_candidates, device=device)
        finalists = select_diverse_finalists(screened, list(param_space), min(n_finalists, len(candidates)))
        ranked, evaluations = evaluate_candidates(
            base_params=base_params,
            candidates=finalists[list(param_space)].to_dict(orient="records"),
            windows=windows, evaluator=evaluator, settings=settings,
            param_names=list(param_space),
        )
        return ranked, evaluations, screened
    ranked, evaluations = search(
        base_params=base_params, param_space=param_space, windows=windows,
        evaluator=evaluator, settings=settings,
    )
    return ranked, evaluations, None



def _run_strategy_finder(
    structures, parameter_space: dict, start, end, *,
    n_windows: int, n_finalists: int, complexity_penalty: float, device: str,
):
    """Discover structures, then validate finalists with the canonical engine."""
    from analysis.optimizer import SearchSettings, build_non_overlapping_windows
    from analysis.runner import run as backtest_run
    from analysis.runspec import RunSpec
    from analysis.strategy_finder import FinderSettings, find_strategies
    from analysis.vectorized_screen import bars_to_close_frame
    from data.instruments import VENUE_MIC, build_universe_instruments
    from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
    from strategies.rule_rotation import RuleRotationConfig, RuleRotationStrategy

    catalog = ParquetDataCatalog(path=str((_PROJECT_ROOT / "data" / "catalog").resolve()))
    bars_by_ticker = {
        ticker: catalog.bars(instrument_ids=[f"{ticker}.{VENUE_MIC}"])
        for ticker in ALL_TICKERS
    }
    instruments = build_universe_instruments()
    windows = build_non_overlapping_windows(start, end, n_windows)
    config_fields = {
        "signal_family", "trend_filter_days", "require_positive",
        "top_n", "lookback_days", "skip_days",
    }

    def evaluator(params, window):
        config_params = {name: params[name] for name in config_fields}
        result = backtest_run(
            RunSpec(
                label=f"strategy-{params['structure_id']}-{window[0].date()}",
                strategy_class=RuleRotationStrategy,
                strategy_config=RuleRotationConfig(
                    universe=UNIVERSE, benchmark=BENCHMARK, **config_params,
                ),
                start=window[0], end=window[1], slippage_prob=0.25,
                log_level="ERROR",
            ),
            bars_by_ticker=bars_by_ticker,
            instruments=instruments,
        )
        return result.metrics

    return find_strategies(
        closes=bars_to_close_frame(bars_by_ticker, UNIVERSE),
        structures=structures,
        parameter_space=parameter_space,
        windows=windows,
        evaluator=evaluator,
        settings=FinderSettings(
            finalists=n_finalists,
            complexity_penalty=complexity_penalty,
            device=device,
            search=SearchSettings(algorithm="grid"),
        ),
    )


def _latest_scalping_result() -> tuple[Path, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """Load the newest completed scalping-finder artifact, if one exists."""
    runs = sorted(
        (_PROJECT_ROOT / "results").glob("scalping_finder_*/summary.json"),
        key=lambda path: path.parent.name,
        reverse=True,
    )
    if not runs:
        return None
    summary_path = runs[0]
    run_dir = summary_path.parent
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rankings = pd.read_csv(run_dir / "discovery_rankings.csv")
    holdout = pd.read_csv(run_dir / "winner_holdout_cost_stress.csv")
    instrument_path = run_dir / "winner_holdout_by_instrument.csv"
    by_instrument = (
        pd.read_csv(instrument_path)
        if instrument_path.exists() else pd.DataFrame()
    )
    group_path = run_dir / "winner_holdout_by_group.csv"
    by_group = pd.read_csv(group_path) if group_path.exists() else pd.DataFrame()
    return run_dir, summary, rankings, holdout, by_instrument, by_group


def _run_scalping_finder(*, device: str, cost_bps: float) -> subprocess.CompletedProcess[str]:
    """Run the reproducible 5-minute discovery/holdout pipeline."""
    return subprocess.run(
        [
            sys.executable,
            str(_PROJECT_ROOT / "examples" / "find_scalping_strategies.py"),
            "--device", device,
            "--cost-bps", str(cost_bps),
        ],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )


def _latest_opening_range_result() -> tuple[Path, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """Load the newest completed opening-range artifact."""
    runs = sorted(
        (_PROJECT_ROOT / "results").glob("opening_range_finder_*/summary.json"),
        key=lambda path: path.parent.name,
        reverse=True,
    )
    if not runs:
        return None
    summary_path = runs[0]
    run_dir = summary_path.parent
    return (
        run_dir,
        json.loads(summary_path.read_text(encoding="utf-8")),
        pd.read_csv(run_dir / "discovery_rankings.csv"),
        pd.read_csv(run_dir / "winner_portfolio_cost_stress.csv"),
        pd.read_csv(run_dir / "winner_holdout_by_instrument.csv"),
    )


def _run_opening_range_finder(*, cost_bps: float) -> subprocess.CompletedProcess[str]:
    """Run the CUDA-only selective opening-range portfolio finder."""
    return subprocess.run(
        [
            sys.executable,
            str(_PROJECT_ROOT / "examples" / "find_opening_range_strategies.py"),
            "--device", "cuda",
            "--cost-bps", str(cost_bps),
        ],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )


def _latest_cross_sectional_result() -> tuple[Path, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """Load the newest completed cross-sectional finder artifact."""
    runs = sorted(
        (_PROJECT_ROOT / "results").glob("cross_sectional_finder_*/summary.json"),
        key=lambda path: path.parent.name,
        reverse=True,
    )
    if not runs:
        return None
    summary_path = runs[0]
    run_dir = summary_path.parent
    return (
        run_dir,
        json.loads(summary_path.read_text(encoding="utf-8")),
        pd.read_csv(run_dir / "discovery_rankings.csv"),
        pd.read_csv(run_dir / "winner_holdout_cost_stress.csv"),
        pd.read_csv(run_dir / "winner_holdout_contributions.csv"),
    )


def _run_cross_sectional_finder(*, cost_bps: float) -> subprocess.CompletedProcess[str]:
    """Run the CUDA-only market-neutral cross-sectional finder."""
    return subprocess.run(
        [
            sys.executable,
            str(_PROJECT_ROOT / "examples" / "find_cross_sectional_strategies.py"),
            "--device", "cuda", "--cost-bps", str(cost_bps),
        ],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )


def _latest_support_resistance_result() -> tuple[Path, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """Load the newest completed support/resistance finder artifact."""
    runs = sorted(
        (_PROJECT_ROOT / "results").glob("support_resistance_finder_*/summary.json"),
        key=lambda path: path.parent.name, reverse=True,
    )
    if not runs:
        return None
    summary_path = runs[0]
    run_dir = summary_path.parent
    return (
        run_dir, json.loads(summary_path.read_text(encoding="utf-8")),
        pd.read_csv(run_dir / "discovery_rankings.csv"),
        pd.read_csv(run_dir / "winner_evaluation_cost_stress.csv"),
        pd.read_csv(run_dir / "winner_evaluation_contributions.csv"),
    )


def _run_support_resistance_finder(*, cost_bps: float) -> subprocess.CompletedProcess[str]:
    """Run the CUDA mechanical support/resistance confluence finder."""
    return subprocess.run(
        [
            sys.executable,
            str(_PROJECT_ROOT / "examples" / "find_support_resistance_strategies.py"),
            "--device", "cuda", "--cost-bps", str(cost_bps),
        ],
        cwd=_PROJECT_ROOT, capture_output=True, text=True,
        timeout=1800, check=False,
    )


def _latest_profitability_result() -> tuple[Path, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """Load the newest completed profitability finder artifact."""
    runs = sorted(
        list((_PROJECT_ROOT / "results").glob("profitability_v[23]_*/summary.json"))
        + list((_PROJECT_ROOT / "results").glob("signal_strategy_finder_*/summary.json"))
        + list((_PROJECT_ROOT / "results").glob("profitability_finder_*/summary.json")),
        key=lambda path: path.stat().st_mtime, reverse=True,
    )
    if not runs:
        return None
    summary_path = runs[0]
    run_dir = summary_path.parent
    return (
        run_dir, json.loads(summary_path.read_text(encoding="utf-8")),
        pd.read_csv(run_dir / "discovery_rankings.csv"),
        pd.read_csv(run_dir / "evaluation_cost_stress.csv"),
        pd.read_csv(run_dir / "evaluation_contributions.csv"),
    )


def _latest_gpu_screen_path() -> Path | None:
    """Return the newest complete Strategy Finder GPU ledger, including rejects."""
    screens = list((_PROJECT_ROOT / "results").glob(
        "signal_strategy_finder_*/gpu_research_screen.csv"
    )) + list((_PROJECT_ROOT / "results").glob(
        "signal_strategy_discovery_reject_*/gpu_research_screen.csv"
    ))
    return max(screens, key=lambda path: path.stat().st_mtime) if screens else None


@st.cache_data(show_spinner=False)
def _gpu_screen_row_count(path: str, modified_ns: int) -> int:
    """Count result rows without loading a potentially million-row CSV."""
    del modified_ns
    with Path(path).open("rb") as handle:
        return max(0, sum(1 for _ in handle) - 1)


@st.cache_data(show_spinner=False)
def _gpu_screen_page(
    path: str, modified_ns: int, offset: int, page_size: int,
) -> pd.DataFrame:
    """Read one virtual page from the complete GPU result ledger."""
    del modified_ns
    return pd.read_csv(
        path, skiprows=range(1, offset + 1), nrows=page_size,
    )


def _queue_strategy_replay(row: dict[str, Any]) -> None:
    """Persist a row selection before Streamlit starts its button rerun."""
    st.session_state["strategy_finder_replay_row"] = row
    st.session_state["strategy_finder_replay_pending"] = True


def _execution_price_chart(spec: dict[str, Any]) -> alt.Chart:
    """OHLC chart with explicit entry and exit markers for one trade leg."""
    bars = pd.DataFrame(spec["price_bars"])
    bars["ts"] = pd.to_datetime(bars["ts"])
    bars["direction"] = bars["close"] >= bars["open"]
    candles = alt.Chart(bars).encode(x=alt.X("ts:T", title="Time (New York)"))
    wicks = candles.mark_rule().encode(
        y=alt.Y("low:Q", title="Price"), y2="high:Q",
    )
    bodies = candles.mark_bar(size=12).encode(
        y="open:Q", y2="close:Q",
        color=alt.condition("datum.direction", alt.value("#16a34a"), alt.value("#dc2626")),
    )
    markers = pd.DataFrame([
        {"ts": spec["entry_time"], "price": spec["entry_price"], "event": f"ENTRY {spec['side'].upper()}"},
        {"ts": spec["exit_time"], "price": spec["exit_price"], "event": "EXIT"},
    ])
    markers["ts"] = pd.to_datetime(markers["ts"])
    points = alt.Chart(markers).mark_point(filled=True, size=140).encode(
        x="ts:T", y="price:Q", color=alt.Color("event:N"), shape=alt.Shape("event:N"),
        tooltip=["event:N", "ts:T", alt.Tooltip("price:Q", format=".4f")],
    )
    return (wicks + bodies + points).properties(
        title=f"{spec['ticker']} · {spec['session_date']} · {spec['side']}", height=320,
    ).interactive()


def _latest_signal_replay() -> tuple[Path, dict, pd.DataFrame, pd.DataFrame, list[dict]] | None:
    summaries = list((_PROJECT_ROOT / "results" / "signal_replays").glob("*/summary.json"))
    if not summaries:
        return None
    summary_path = max(summaries, key=lambda path: path.stat().st_mtime)
    run_dir = summary_path.parent
    chart_path = run_dir / "execution_charts.json"
    charts = json.loads(chart_path.read_text()) if chart_path.exists() else []
    return (
        run_dir, json.loads(summary_path.read_text()),
        pd.read_csv(run_dir / "equity.csv"), pd.read_csv(run_dir / "trades.csv"), charts,
    )


def _render_latest_signal_replay(container) -> None:
    loaded = _latest_signal_replay()
    if loaded is None:
        return
    run_dir, summary, equity, trades, charts = loaded
    container.subheader("Latest replay")
    metrics = container.columns(4)
    metrics[0].metric("Return", f"{summary['total_return']:.2%}")
    metrics[1].metric("Sharpe", f"{summary['sharpe']:.3f}")
    metrics[2].metric("Max drawdown", f"{summary['max_drawdown']:.2%}")
    metrics[3].metric("Trades", f"{int(summary['n_trades']):,}")
    equity["session_date"] = pd.to_datetime(equity["session_date"])
    container.markdown("**Portfolio equity**")
    container.line_chart(
        equity.set_index("session_date")[["equity"]],
        x_label="Session", y_label="Equity (start = 1.0)",
    )
    if charts:
        labels = [
            f"{index + 1}: {item['session_date']} · {item['ticker']} · {item['side']}"
            for index, item in enumerate(charts)
        ]
        selected_label = container.selectbox(
            "Trade chart", labels, key=f"replay_trade_chart_{run_dir.name}",
        )
        selected_chart = charts[labels.index(selected_label)]
        container.altair_chart(_execution_price_chart(selected_chart), width="stretch")
    container.markdown("**Every executed trade leg**")
    container.dataframe(trades, width="stretch", hide_index=True)


def _render_complete_gpu_results() -> None:
    """Render every GPU candidate through paging and provide the raw ledger."""
    screen_path = _latest_gpu_screen_path()
    if screen_path is None:
        return
    stat = screen_path.stat()
    row_count = _gpu_screen_row_count(str(screen_path), stat.st_mtime_ns)
    st.subheader("All GPU results")
    st.caption(
        f"{row_count:,} evaluated combinations in `{screen_path.parent.name}`. "
        "Paging changes only the view; the CSV contains every result."
    )
    st.caption("Instruments: " + ", ".join(SCALPING_STOCKS))
    with st.expander("Replay settings"):
        replay_columns = st.columns(3)
        with replay_columns[0]:
            replay_period = st.selectbox(
                "Replay period", ("discovery", "holdout", "full"),
                help="Discovery reproduces the screen interval; holdout and full are diagnostic research replays.",
            )
        with replay_columns[1]:
            replay_positions = int(st.number_input(
                "Positions per side", min_value=1, max_value=5, value=3, step=1,
            ))
        with replay_columns[2]:
            replay_cost = float(st.number_input(
                "Replay cost per side (bps)", min_value=0.0, value=2.5, step=0.5,
            ))
        replay_rvol = float(st.number_input(
            "Minimum relative volume", min_value=0.0, value=0.75, step=0.05,
        ))
    replay_output = st.container()
    selected = st.session_state.get("strategy_finder_replay_row")
    run_requested = bool(st.session_state.pop("strategy_finder_replay_pending", False))
    if not run_requested:
        _render_latest_signal_replay(replay_output)

    page_col, size_col = st.columns(2)
    with size_col:
        page_size = int(st.selectbox(
            "Rows per page", (10, 25, 50, 100), index=1,
            key="strategy_finder_result_page_size",
        ))
    page_count = max(1, (row_count + page_size - 1) // page_size)
    with page_col:
        page = int(st.number_input(
            "Result page", min_value=1, max_value=page_count, value=1, step=1,
            key="strategy_finder_result_page",
        ))
    offset = (page - 1) * page_size
    frame = _gpu_screen_page(
        str(screen_path), stat.st_mtime_ns, offset, page_size,
    )
    if not frame.empty:
        frame.insert(0, "rank", range(offset + 1, offset + len(frame) + 1))
        header = st.columns([0.45, 0.55, 1.0, 4.2, 1.1, 1.1, 1.1, 1.3])
        for column, label in zip(
            header,
            ("Play", "Rank", "Time", "Signals", "Instruments", "Sharpe", "Return", "Events"),
            strict=True,
        ):
            column.markdown(f"**{label}**")
        for _, row in frame.iterrows():
            columns = st.columns([0.45, 0.55, 1.0, 4.2, 1.1, 1.1, 1.1, 1.3])
            rank = int(row["rank"])
            columns[0].button(
                "▶", key=f"play_gpu_result_{stat.st_mtime_ns}_{rank}",
                help=f"Replay result {rank}",
                on_click=_queue_strategy_replay, args=(row.to_dict(),),
            )
            columns[1].write(f"{rank:,}")
            columns[2].write(f"Bar {int(row['decision_bar'])}")
            columns[3].code(str(row["signals"]), language=None)
            columns[4].write("30 stocks")
            columns[5].write(f"{float(row['screen_sharpe']):.3f}")
            columns[6].write(f"{float(row['screen_total_return']):.2%}")
            columns[7].write(f"{int(row['screen_events']):,}")
    st.caption(f"Showing rows {offset + 1:,}–{min(offset + page_size, row_count):,} of {row_count:,}.")
    with screen_path.open("rb") as handle:
        st.download_button(
            "Download all GPU results (CSV)", data=handle,
            file_name=screen_path.name, mime="text/csv",
            key=f"download_gpu_screen_{stat.st_mtime_ns}",
        )

    if not selected:
        st.info("Press ▶ beside any result to stream its canonical backtest.")
        return
    if run_requested:
        replay_output.subheader(f"Streaming replay — result {int(selected['rank']):,}")
        replay_output.code(str(selected["signals"]), language=None)
        progress = replay_output.progress(0, text="Starting canonical replay...")
        metric_slots = [column.empty() for column in replay_output.columns(4)]
        replay_output.markdown("**Portfolio equity**")
        chart_slot = replay_output.empty()
        replay_output.markdown("**Latest executed trade — entry and exit**")
        execution_chart_slot = replay_output.empty()
        replay_output.markdown("**Every executed trade leg**")
        trade_slot = replay_output.empty()
        log_slot = replay_output.empty()
        equity_rows: list[dict[str, Any]] = []
        trade_rows: list[dict[str, Any]] = []
        logs: list[str] = []
        final_summary: dict[str, Any] = {}

        def show_replay(line: str) -> None:
            if line.startswith("PROGRESS|"):
                _, percent, message = line.split("|", 2)
                progress.progress(int(percent), text=message)
                return
            if line.startswith("REPLAY|"):
                event = json.loads(line.split("|", 1)[1])
                equity_rows.append({
                    "session_date": event["session_date"],
                    "equity": event["equity"],
                    "daily_return": event["daily_return"],
                })
                trade_rows.extend(event["trades"])
                progress.progress(
                    int(event["progress"]),
                    text=f"Replaying {event['session_index'] + 1:,}/{event['session_count']:,} sessions",
                )
                curve = pd.DataFrame(equity_rows)
                curve["session_date"] = pd.to_datetime(curve["session_date"])
                chart_slot.line_chart(
                    curve.set_index("session_date")[["equity"]],
                    x_label="Session", y_label="Equity (start = 1.0)",
                )
                daily = curve["daily_return"]
                volatility = daily.std(ddof=1)
                live_sharpe = (
                    daily.mean() / volatility * (252 ** 0.5)
                    if len(daily) > 1 and volatility > 0 else 0.0
                )
                running_peak = curve["equity"].cummax()
                live_drawdown = float((curve["equity"] / running_peak - 1).min())
                metric_slots[0].metric("Return", f"{event['equity'] - 1:.2%}")
                metric_slots[1].metric("Live Sharpe", f"{live_sharpe:.3f}")
                metric_slots[2].metric("Max drawdown", f"{live_drawdown:.2%}")
                metric_slots[3].metric("Trades", f"{event['cumulative_trades']:,}")
                if trade_rows:
                    trade_slot.dataframe(
                        pd.DataFrame(trade_rows), width="stretch", hide_index=True,
                    )
                if event.get("trade_charts"):
                    execution_chart_slot.altair_chart(
                        _execution_price_chart(event["trade_charts"][-1]),
                        width="stretch",
                    )
                return
            if line.startswith("SUMMARY|"):
                final_summary.update(json.loads(line.split("|", 1)[1]))
                return
            if line:
                logs.append(line)
                log_slot.code("\n".join(logs[-10:]))

        completed = _run_signal_replay(
            signals=str(selected["signals"]),
            decision_bar=int(selected["decision_bar"]),
            period=replay_period, cost_bps=replay_cost,
            positions_per_side=replay_positions,
            min_relative_volume=replay_rvol,
            on_output=show_replay,
        )
        if completed.returncode != 0:
            replay_output.error("Replay failed.")
            replay_output.code(completed.stdout)
        else:
            progress.progress(100, text="Replay complete")
            replay_output.success(
                f"Canonical replay complete: {final_summary.get('total_return', 0):.2%} return, "
                f"Sharpe {final_summary.get('sharpe', 0):.3f}."
            )


def _run_profitability_finder(
    *, cost_bps: float, indicators: tuple[str, ...], value_min: int,
    value_max: int, value_step: int, top_signals: int, max_combination_size: int,
    on_output: Callable[[str], None] | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
            sys.executable,
            str(_PROJECT_ROOT / "examples" / "find_signal_combinations.py"),
            "--device", "cuda", "--cost-bps", str(cost_bps),
            "--indicators", *indicators,
            "--value-min", str(value_min), "--value-max", str(value_max),
            "--value-step", str(value_step), "--top-signals", str(top_signals),
            "--max-combination-size", str(max_combination_size),
        ]
    process = subprocess.Popen(
        command, cwd=_PROJECT_ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line)
        if on_output is not None:
            on_output(line.rstrip())
    return subprocess.CompletedProcess(
        command, process.wait(), stdout="".join(lines), stderr="",
    )


def _run_signal_replay(
    *, signals: str, decision_bar: int, period: str, cost_bps: float,
    positions_per_side: int, min_relative_volume: float,
    on_output: Callable[[str], None] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one selected GPU row through the canonical streaming replay."""
    command = [
        sys.executable, str(_PROJECT_ROOT / "examples" / "replay_signal_combination.py"),
        "--device", "cuda", "--signals", signals,
        "--decision-bar", str(decision_bar), "--period", period,
        "--cost-bps", str(cost_bps), "--positions-per-side", str(positions_per_side),
        "--min-relative-volume", str(min_relative_volume),
    ]
    process = subprocess.Popen(
        command, cwd=_PROJECT_ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line)
        if on_output is not None:
            on_output(line.rstrip())
    return subprocess.CompletedProcess(
        command, process.wait(), stdout="".join(lines), stderr="",
    )


def render(data_sources) -> None:
    st.header("Backtest Lab")
    st.caption(
        "Self-service backtest, Parameter Optimizer, and structural Strategy Finder. "
        "Auto-discovers strategies from `strategies/`, auto-renders param "
        "forms from each StrategyConfig."
    )

    strategies = discover_strategies()
    if not strategies:
        st.error("No strategies discovered under `strategies/`.")
        return

    sel_col, mode_col = st.columns([2, 3])
    with sel_col:
        strategy_name = st.selectbox("Strategy", sorted(strategies))
        strategy_cls, config_cls = strategies[strategy_name]
        st.caption(f"`{strategy_cls.__module__}.{strategy_cls.__name__}`")
    with mode_col:
        mode = st.segmented_control(
            "Mode",
            options=(
                "Single", "Grid", "Parameter Optimizer", "Rotation Rule Finder",
                "Scalping Finder", "Opening Range Finder", "Cross-Sectional Finder",
                "Support/Resistance Finder", "Strategy Finder",
            ),
            default="Single",
        )
        if mode is None:
            mode = "Single"

    if mode == "Rotation Rule Finder":
        base_params = {}
        st.info(
            "Legacy Rotation Rule Finder searches momentum and mean-reversion ETF "
            "rotation structures. For the 60-signal combinatorial search, use Strategy Finder."
        )
    elif mode == "Scalping Finder":
        base_params = {}
        st.info(
            "Scalping Finder is the intraday Strategy Finder: it discovers entry-rule "
            "families and optimizes exits across 60 instruments on 5-minute bars: "
            "30 stocks, 10 commodity proxies, 10 cryptocurrencies, and 10 other "
            "high-volatility contenders. The fixed-strategy "
            "selector and daily window controls are not used."
        )
    elif mode == "Opening Range Finder":
        base_params = {}
        st.info(
            "CUDA Opening Range Finder searches a shared-capital portfolio of "
            "30 stocks using long/short breakouts, relative volume, ATR-scaled "
            "risk, optional SPY alignment, and at most one trade per stock/day."
        )
    elif mode == "Cross-Sectional Finder":
        base_params = {}
        st.info(
            "CUDA Cross-Sectional Finder ranks beta-residual stock moves, holds "
            "a dollar- and beta-neutral long/short portfolio, and makes at most "
            "one scheduled portfolio entry per day."
        )
    elif mode == "Support/Resistance Finder":
        base_params = {}
        st.info(
            "CUDA Support/Resistance Finder combines six causal level families "
            "with beta-residual rankings, neutral portfolio weights, ATR stops, "
            "2R targets, and conservative cost stress."
        )
    elif mode == "Strategy Finder":
        base_params = {}
        st.info(
            "CUDA Strategy Finder chooses an indicator and value for each signal "
            "slot, then GPU-screens unrestricted 1–4 signal combinations before it "
            "sends at most 48 finalists through canonical cost and holdout validation."
        )
    else:
        st.subheader("Strategy parameters")
        base_params = _render_param_form(config_cls, key_prefix=f"params_{strategy_name}_{mode}")

    if mode not in {
        "Scalping Finder", "Opening Range Finder", "Cross-Sectional Finder",
        "Support/Resistance Finder", "Strategy Finder",
    }:
        st.subheader("Window")
        c1, c2 = st.columns(2)
        with c1:
            start_date = st.date_input("Start", value=datetime(2021, 1, 1).date())
        with c2:
            end_date = st.date_input("End", value=datetime(2024, 12, 31).date())
        start_ts = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
        end_ts = datetime.combine(end_date, datetime.min.time(), tzinfo=timezone.utc)

    if mode == "Single":
        if st.button("Run single backtest", type="primary"):
            with st.spinner("Running backtest..."):
                try:
                    metrics = _run_single(strategy_cls, config_cls, base_params,
                                          start_ts, end_ts)
                except Exception as exc:
                    st.error(f"Backtest failed: {exc}")
                    return
            st.success("Done")
            cols = st.columns(4)
            cols[0].metric("Sharpe", f"{metrics.get('sharpe', float('nan')):.4f}")
            cols[1].metric("CAGR", f"{metrics.get('cagr', float('nan')):.4f}")
            cols[2].metric("Max DD", f"{metrics.get('max_drawdown', float('nan')):.4f}")
            cols[3].metric("Trades", int(metrics.get("n_trades", 0)))
            st.json(metrics)

    elif mode == "Grid":
        st.markdown(
            "Define ranges for one or more numeric parameters. **Total cells "
            "must be ≤ 200** (1-minute compute budget). For larger searches, "
            "use Parameter Optimizer mode."
        )
        numeric_fields = [k for k, (ann, _) in config_fields(config_cls).items()
                          if ann in ("int", "float", int, float)]
        sweep_param = st.selectbox("Sweep parameter", options=numeric_fields)
        c1, c2, c3 = st.columns(3)
        with c1:
            sweep_lo = st.number_input("low", value=1)
        with c2:
            sweep_hi = st.number_input("high", value=10)
        with c3:
            sweep_step = st.number_input("step", value=1, min_value=1)
        if st.button("Run grid sweep", type="primary"):
            param_ranges = {sweep_param: list(range(int(sweep_lo), int(sweep_hi) + 1, int(sweep_step)))}
            cells = len(make_param_grid(param_ranges))
            if cells > 200:
                st.error(f"{cells} cells > 200 limit; tighten the range or use Parameter Optimizer mode.")
                return
            windows = [(start_ts, end_ts)]
            with st.spinner(f"Running grid sweep ({cells} cells)..."):
                try:
                    df = _run_grid(strategy_cls, config_cls, base_params, param_ranges, windows)
                except Exception as exc:
                    st.error(f"Grid sweep failed: {exc}")
                    return
            st.success(f"Done — {len(df)} runs")
            st.dataframe(df.sort_values("sharpe", ascending=False))
            if len(df) > 1 and sweep_param in df.columns:
                st.bar_chart(df.set_index(sweep_param)["sharpe"])

    elif mode == "Parameter Optimizer":
        from analysis.optimizer import accelerator_status, exhaustive_combinations

        st.markdown(
            "Exhaustive mode tests every combination; adaptive TPE concentrates "
            "a fixed budget in promising regions. Candidates are scored across "
            "non-overlapping windows and all robust/Pareto-optimal results remain visible."
        )
        accelerator = accelerator_status()
        if accelerator["cuda_available"]:
            st.success(f"GPU ready: {accelerator['device']} for compatible model/vector stages. Nautilus simulation remains CPU.")
        else:
            st.info("CUDA is not enabled here. Nautilus event simulation is CPU-bound; GPU is reserved for compatible model/vector stages.")
        algorithm_label = st.radio("Search engine", ("Hybrid GPU/CPU", "Adaptive TPE", "Exhaustive grid"), horizontal=True)
        algorithm = {"Hybrid GPU/CPU": "hybrid", "Adaptive TPE": "tpe", "Exhaustive grid": "grid"}[algorithm_label]
        n_windows = int(st.number_input("Independent evaluation windows", min_value=2, max_value=10, value=4))
        n_trials = int(st.number_input("Trial budget", min_value=5, value=50, step=5, disabled=algorithm != "tpe"))
        n_finalists = int(st.number_input("Hybrid finalists", min_value=1, value=20, disabled=algorithm != "hybrid"))
        device = st.selectbox("Tensor device", ("auto", "cpu", "cuda"), disabled=algorithm != "hybrid")
        numeric_fields = [k for k, (ann, _) in config_fields(config_cls).items()
                          if ann in ("int", "float", int, float)]
        opt_params = st.multiselect("Parameters to optimize", options=numeric_fields)
        param_space: dict = {}
        for p in opt_params:
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                lo = st.number_input(f"{p} low", value=1, key=f"opt_lo_{p}")
            with cc2:
                hi = st.number_input(f"{p} high", value=10, key=f"opt_hi_{p}")
            with cc3:
                step = st.number_input(f"{p} step", min_value=0.0001, value=1.0, key=f"opt_step_{p}")
            ann = config_fields(config_cls)[p][0]
            param_space[p] = {
                "type": "int" if ann in ("int", int) else "float",
                "low": lo, "high": hi,
                "step": int(step) if ann in ("int", int) else float(step),
            }
        space_valid = valid_numeric_space(param_space)
        if param_space and not space_valid:
            st.warning("Each parameter range must have low ≤ high and step > 0.")
        elif param_space and algorithm in {"grid", "hybrid"}:
            cells = len(exhaustive_combinations(param_space))
            canonical_runs = cells * n_windows if algorithm == "grid" else min(cells, n_finalists) * n_windows
            st.caption(f"{cells:,} tensor combinations; {canonical_runs:,} canonical backtests")
        if st.button("Run Parameter Optimizer", type="primary") and param_space:
            if not space_valid:
                st.error("Fix the invalid parameter ranges before starting the search.")
                return
            with st.spinner("Searching and validating candidates across windows..."):
                try:
                    ranked, evaluations, screened = _run_parameter_optimizer(
                        strategy_cls, config_cls, base_params, param_space,
                        start_ts, end_ts, algorithm=algorithm,
                        n_trials=n_trials, n_windows=n_windows,
                        n_finalists=n_finalists, device=device,
                    )
                except Exception as exc:
                    st.error(f"Parameter Optimizer failed: {exc}")
                    return
            eligible = ranked[ranked["eligible"]]
            st.success(f"Done — {len(ranked)} unique candidates; {len(eligible)} eligible")
            st.dataframe(ranked, width="stretch")
            st.download_button("Download rankings CSV", ranked.to_csv(index=False), "parameter_optimizer_rankings.csv")
            st.download_button("Download per-window CSV", evaluations.to_csv(index=False), "parameter_optimizer_windows.csv")
            if screened is not None:
                st.download_button("Download tensor pre-screen CSV", screened.to_csv(index=False), "parameter_optimizer_prescreen.csv")


    elif mode == "Rotation Rule Finder":
        from analysis.optimizer import accelerator_status
        from analysis.strategy_finder import (
            StrategyGrammar, describe_strategy, generate_candidates, generate_structures,
        )

        st.markdown(
            "Searches registered rule structures—not just parameter values. Every "
            "candidate is GPU/CPU tensor-screened, and diverse finalists are "
            "validated with the canonical Nautilus backtester."
        )
        accelerator = accelerator_status()
        if accelerator["cuda_available"]:
            st.success(f"GPU ready: {accelerator['device']}; Nautilus validation remains CPU.")
        else:
            st.info("CUDA unavailable; structural screening will use the CPU.")

        families = st.multiselect(
            "Rotation rule families", ("momentum", "mean_reversion"),
            default=("momentum", "mean_reversion"),
        )
        trend_days = st.multiselect(
            "Trend-filter structures (days; 0 disables)", (0, 126, 200),
            default=(0, 126, 200),
        )
        gate_labels = st.multiselect(
            "Signal gates", ("Ungated", "Positive signal only"),
            default=("Ungated", "Positive signal only"),
        )
        max_complexity = int(st.slider("Maximum rule complexity", 1, 3, 3))
        top_values = st.multiselect("Top-N values", (1, 2, 3, 4, 5), default=(1, 2, 3, 4, 5))
        lookback_values = st.multiselect(
            "Lookback values", tuple(range(21, 253, 21)), default=tuple(range(21, 253, 21)),
        )
        skip_values = st.multiselect("Skip values", (0, 21, 42), default=(0, 21, 42))
        n_windows = int(st.number_input("Strategy evaluation windows", min_value=2, max_value=10, value=4))
        n_finalists = int(st.number_input("Structural finalists", min_value=1, value=24))
        complexity_penalty = float(st.number_input("Complexity penalty", min_value=0.0, value=0.03, step=0.01))
        device = st.selectbox("Strategy tensor device", ("auto", "cpu", "cuda"))

        selections_valid = bool(
            families and trend_days and gate_labels
            and top_values and lookback_values and skip_values
        )
        if not selections_valid:
            st.warning("Select at least one value in every Rotation Rule Finder dimension.")
        else:
            gates = tuple(label == "Positive signal only" for label in gate_labels)
            grammar = StrategyGrammar(
                signal_families=tuple(families),
                trend_filter_days=tuple(int(value) for value in trend_days),
                positive_gates=gates,
                max_complexity=max_complexity,
            )
            structures = generate_structures(grammar)
            parameter_space = {
                "top_n": {"type": "categorical", "choices": list(top_values)},
                "lookback_days": {"type": "categorical", "choices": list(lookback_values)},
                "skip_days": {"type": "categorical", "choices": list(skip_values)},
            }
            candidate_count = len(generate_candidates(structures, parameter_space))
            canonical_runs = min(candidate_count, n_finalists) * n_windows
            st.caption(
                f"{len(structures):,} structures · {candidate_count:,} tensor candidates · "
                f"{canonical_runs:,} canonical backtests"
            )

        if st.button("Run Rotation Rule Finder", type="primary") and selections_valid:
            with st.spinner("Discovering, tuning, and validating rule structures..."):
                try:
                    ranked, evaluations, screened = _run_strategy_finder(
                        structures, parameter_space, start_ts, end_ts,
                        n_windows=n_windows, n_finalists=n_finalists,
                        complexity_penalty=complexity_penalty, device=device,
                    )
                except Exception as exc:
                    st.error(f"Rotation Rule Finder failed: {exc}")
                    return
            ranked.insert(1, "description", [
                describe_strategy(row) for _, row in ranked.iterrows()
            ])
            eligible = ranked[ranked["eligible"]]
            st.success(f"Done — {len(ranked)} structural finalists; {len(eligible)} eligible")
            st.dataframe(ranked, width="stretch")
            st.download_button("Download strategy rankings CSV", ranked.to_csv(index=False), "strategy_finder_rankings.csv")
            st.download_button("Download strategy windows CSV", evaluations.to_csv(index=False), "strategy_finder_windows.csv")
            st.download_button("Download structural pre-screen CSV", screened.to_csv(index=False), "strategy_finder_prescreen.csv")

    elif mode == "Strategy Finder":
        from math import comb

        from analysis.signal_library import PARAMETRIC_INDICATORS

        st.warning(
            "Historical evaluation dates are already exposed. This screen can only reject "
            "or qualify research for a new forward paper period; it cannot approve deployment."
        )
        st.markdown(
            "Each signal slot is represented as `indicator` + `value`; there are "
            "no signal families and periods are not hard-coded strategy types."
        )
        selected_indicators = tuple(st.multiselect(
            "Indicators", PARAMETRIC_INDICATORS, default=PARAMETRIC_INDICATORS,
        ))
        value_columns = st.columns(3)
        with value_columns[0]:
            value_min = int(st.number_input("Minimum value", min_value=3, value=3))
        with value_columns[1]:
            value_max = int(st.number_input("Maximum value", min_value=3, value=50))
        with value_columns[2]:
            value_step = int(st.number_input("Value step", min_value=1, value=1))
        variant_count = (
            len(selected_indicators) * (((value_max - value_min) // value_step) + 1)
            if selected_indicators and value_max >= value_min else 0
        )
        search_columns = st.columns(2)
        with search_columns[0]:
            top_signals = int(st.number_input(
                "Indicator/value variants retained for combinations",
                min_value=4, max_value=60, value=min(60, max(4, variant_count)),
            ))
        with search_columns[1]:
            max_combination_size = int(st.slider(
                "Signals per strategy", min_value=1, max_value=4, value=4,
            ))
        search_valid = bool(
            selected_indicators and value_max >= value_min
            and variant_count >= top_signals
        )
        combination_count = (
            sum(comb(top_signals, size) for size in range(1, max_combination_size + 1))
            if search_valid else 0
        )
        st.caption(
            f"{len(selected_indicators)} indicators × value grid = {variant_count:,} "
            f"indicator/value variants; {combination_count:,} unrestricted sets × "
            f"2 decision times = {2 * combination_count:,} GPU candidates."
        )
        profit_cost = float(st.number_input(
            "Strategy search base cost per side (bps)", min_value=0.0, value=2.5, step=0.5,
        ))
        if not search_valid:
            st.warning("Choose indicators and a valid value range large enough for the retained variants.")
        if st.button("Run Strategy Finder", type="primary", disabled=not search_valid):
            progress = st.progress(0, text="Starting Strategy Finder...")
            log_output = st.empty()
            st.subheader("Live GPU leaders")
            live_results = st.empty()
            log_lines: list[str] = []

            def show_progress(line: str) -> None:
                if line.startswith("PROGRESS|"):
                    _, percent, message = line.split("|", 2)
                    progress.progress(int(percent), text=message)
                elif line.startswith("RESULTS|"):
                    rows = json.loads(line.split("|", 1)[1])
                    result_frame = pd.DataFrame(rows)
                    if not result_frame.empty:
                        result_frame["provisional_return"] = result_frame[
                            "provisional_return"
                        ].map(lambda value: f"{value:.2%}")
                        result_frame["provisional_sharpe"] = result_frame[
                            "provisional_sharpe"
                        ].map(lambda value: f"{value:.3f}")
                        live_results.dataframe(
                            result_frame, width="stretch", hide_index=True,
                        )
                elif line:
                    log_lines.append(line)
                    log_output.code("\n".join(log_lines[-12:]))

            with st.spinner("Running full CUDA signal-combination validation..."):
                completed = _run_profitability_finder(
                    cost_bps=profit_cost, indicators=selected_indicators,
                    value_min=value_min, value_max=value_max, value_step=value_step,
                    top_signals=top_signals,
                    max_combination_size=max_combination_size,
                    on_output=show_progress,
                )
            discovery_reject = (
                completed.returncode == 2
                and "DISCOVERY_REJECT" in completed.stdout
            )
            if completed.returncode != 0 and not discovery_reject:
                st.error("Strategy Finder did not produce a holdout-qualified candidate.")
                st.code(completed.stderr or completed.stdout)
                return
            progress.progress(100, text="Strategy Finder complete")
            if discovery_reject:
                st.warning("Strategy Finder completed: discovery rejected; holdout stayed sealed.")
            else:
                st.success("Strategy Finder completed.")
        _render_complete_gpu_results()
        loaded = _latest_profitability_result()
        if loaded is None:
            st.info("No completed profitability search yet.")
            return
        run_dir, summary, rankings, costs, contributions = loaded
        st.error(f"Deployment verdict: {summary['verdict']}")
        base, harsh = summary["evaluation_base_cost"], summary["evaluation_harsh_cost"]
        st.subheader("Best discovery candidate")
        st.write(summary["winner"]["description"])
        metrics = st.columns(6)
        metrics[0].metric("Evaluation return", f"{base['total_return']:.2%}")
        metrics[1].metric("Sharpe", f"{base['sharpe']:.3f}")
        metrics[2].metric("Max drawdown", f"{base['max_drawdown']:.2%}")
        metrics[3].metric("Return @ 5 bps", f"{harsh['total_return']:.2%}")
        metrics[4].metric("PBO", f"{summary['probability_backtest_overfit']['probability_overfit']:.1%}")
        metrics[5].metric("SIP ready", "Yes" if summary["sip_quote_ready"] else "No")
        st.caption(
            f"{summary['candidate_count']:,} candidates · {summary['common_session_count']} "
            f"common sessions · {summary['simulation_device']} · artifact {run_dir.name}"
        )
        if "research_trial_count" in summary:
            st.caption(
                f"{summary['signal_primitive_count']} signal primitives · "
                f"{summary['research_trial_count']:,} GPU research combinations"
            )
        qualification = summary.get("historical_deployment_qualification", {})
        if qualification:
            gate_frame = pd.DataFrame([
                {"gate": name, "passed": passed}
                for name, passed in qualification["gates"].items()
            ])
            st.subheader("Strict deployment gates")
            st.dataframe(gate_frame, width="stretch", hide_index=True)
            st.caption(f"Frozen candidate: `{summary.get('candidate_fingerprint', 'not frozen')}`")
        st.dataframe(rankings, width="stretch")
        st.subheader("Evaluation cost stress")
        st.dataframe(costs, width="stretch")
        st.subheader("Per-stock contribution")
        st.dataframe(contributions, width="stretch")
        st.download_button("Download profitability rankings CSV", rankings.to_csv(index=False), "profitability_rankings.csv")
        st.download_button("Download profitability summary JSON", json.dumps(summary, indent=2), "profitability_summary.json")

    elif mode == "Support/Resistance Finder":
        st.markdown(
            "Searches previous-day/week, available premarket, opening-range, "
            "swing, VWAP-band, and any-level confluence using breakout/retest, "
            "rejection, and failed-breakout rules over the last three bars."
        )
        st.warning(
            "These evaluation dates were exposed by earlier strategy searches. "
            "This mode always rejects live deployment until genuinely new future "
            "data and SIP/NBBO quote replay are available."
        )
        sr_cost = float(st.number_input(
            "Support/resistance base cost per side (bps)",
            min_value=0.0, value=1.0, step=0.5,
        ))
        if st.button("Run Support/Resistance Finder", type="primary"):
            with st.spinner("Running CUDA level discovery and cost validation..."):
                completed = _run_support_resistance_finder(cost_bps=sr_cost)
            if completed.returncode != 0:
                st.error("Support/Resistance Finder failed.")
                st.code(completed.stderr or completed.stdout)
                return
            st.success("Support/Resistance Finder completed.")
        loaded = _latest_support_resistance_result()
        if loaded is None:
            st.info("No completed support/resistance search yet.")
            return
        run_dir, summary, rankings, costs, contributions = loaded
        st.error("Deployment verdict: REJECT")
        if summary["research_verdict"] == "PASS":
            st.success("Historical research gates: PASS")
        else:
            st.error("Historical research gates: REJECT")
        base, harsh = summary["evaluation_base_cost"], summary["evaluation_harsh_cost"]
        st.subheader("Best discovery candidate")
        st.write(summary["winner"]["description"])
        metrics = st.columns(5)
        metrics[0].metric("Evaluation return", f"{base['total_return']:.2%}")
        metrics[1].metric("Sharpe", f"{base['sharpe']:.3f}")
        metrics[2].metric("Maximum drawdown", f"{base['max_drawdown']:.2%}")
        metrics[3].metric("Return @ 5 bps", f"{harsh['total_return']:.2%}")
        metrics[4].metric("Trades", int(base["n_trades"]))
        st.caption(
            f"{summary['candidate_count']:,} candidates · "
            f"{summary['instrument_count']} stocks / "
            f"{summary['aligned_bar_count']:,} aligned bars · "
            f"{summary['evaluation_status']} · artifact {run_dir.name}"
        )
        st.dataframe(rankings, width="stretch")
        st.subheader("Cost stress")
        st.dataframe(costs, width="stretch")
        st.subheader("Per-stock contribution")
        st.dataframe(contributions, width="stretch")
        st.download_button(
            "Download support/resistance rankings CSV", rankings.to_csv(index=False),
            "support_resistance_rankings.csv",
        )
        st.download_button(
            "Download support/resistance summary JSON", json.dumps(summary, indent=2),
            "support_resistance_summary.json",
        )

    elif mode == "Cross-Sectional Finder":
        st.markdown(
            "Searches momentum and reversal rankings of 30 stocks after removing "
            "rolling SPY beta. Entries use next-bar opens; candidates vary decision "
            "time, lookback, holding period, breadth, dispersion, and relative volume."
        )
        st.warning(
            "A PASS is only research qualification. Live deployment still requires "
            "SIP/NBBO quote replay and a forward paper-trading observation period."
        )
        cross_cost = float(st.number_input(
            "Cross-sectional base cost per side (bps)",
            min_value=0.0, value=1.0, step=0.5,
        ))
        if st.button("Run Cross-Sectional Finder", type="primary"):
            with st.spinner("Running CUDA portfolio discovery and sealed holdout..."):
                completed = _run_cross_sectional_finder(cost_bps=cross_cost)
            if completed.returncode != 0:
                st.error("Cross-Sectional Finder failed.")
                st.code(completed.stderr or completed.stdout)
                return
            st.success("Cross-Sectional Finder completed.")

        loaded = _latest_cross_sectional_result()
        if loaded is None:
            st.info("No completed cross-sectional search yet.")
            return
        run_dir, summary, rankings, cost_stress, contributions = loaded
        if summary["verdict"] == "PASS":
            st.success("Holdout verdict: PASS")
        else:
            st.error("Holdout verdict: REJECT — do not deploy this candidate")
        base = summary["holdout_base_cost"]
        harsh = summary["holdout_harsh_cost"]
        st.subheader("Best discovery candidate")
        st.write(summary["winner"]["description"])
        metrics = st.columns(6)
        metrics[0].metric("Holdout return", f"{base['total_return']:.2%}")
        metrics[1].metric("Sharpe", f"{base['sharpe']:.3f}")
        metrics[2].metric("Maximum drawdown", f"{base['max_drawdown']:.2%}")
        metrics[3].metric("Return @ 5 bps", f"{harsh['total_return']:.2%}")
        metrics[4].metric("Mean |beta|", f"{base['mean_abs_beta_exposure']:.3f}")
        metrics[5].metric(
            "PBO", f"{summary['probability_backtest_overfit']['probability_overfit']:.1%}",
        )
        st.caption(
            f"{summary['candidate_count']:,} candidates · "
            f"{summary['portfolio_window_count']} discovery windows · "
            f"{summary['instrument_count']} stocks / "
            f"{summary['aligned_bar_count']:,} aligned bars · "
            f"Deflated Sharpe confidence "
            f"{summary['deflated_sharpe']['p_value']:.1%} · artifact {run_dir.name}"
        )
        st.dataframe(rankings, width="stretch")
        st.subheader("Untouched holdout cost stress")
        st.dataframe(cost_stress, width="stretch")
        st.subheader("Per-stock holdout contribution")
        st.dataframe(contributions, width="stretch")
        st.download_button(
            "Download cross-sectional rankings CSV", rankings.to_csv(index=False),
            "cross_sectional_rankings.csv",
        )
        st.download_button(
            "Download cross-sectional holdout CSV", cost_stress.to_csv(index=False),
            "cross_sectional_holdout.csv",
        )
        st.download_button(
            "Download cross-sectional summary JSON", json.dumps(summary, indent=2),
            "cross_sectional_summary.json",
        )

    elif mode == "Opening Range Finder":
        st.markdown(
            "Searches exact-clock **15/30/60-minute opening ranges** with "
            "long, short, and bidirectional entries. Signals require configurable "
            "relative volume and breakout distance; exits and stops scale with "
            "causal ATR. Candidate evaluation uses a shared-capital, equal-weight "
            "30-stock portfolio and an untouched final 20% holdout."
        )
        st.warning(
            "IEX bars can omit intervals and do not contain consolidated bid/ask "
            "quotes. Incomplete opening ranges are rejected, and explicit cost "
            "stress is applied, but paper trading and SIP quote validation remain required."
        )
        orb_cost = float(st.number_input(
            "Opening-range base cost per side (bps)",
            min_value=0.0, value=1.0, step=0.5,
        ))
        if st.button("Run Opening Range Finder", type="primary"):
            with st.spinner(
                "Running CUDA entry discovery, portfolio validation, and holdout..."
            ):
                completed = _run_opening_range_finder(cost_bps=orb_cost)
            if completed.returncode != 0:
                st.error("Opening Range Finder failed.")
                st.code(completed.stderr or completed.stdout)
                return
            st.success("Opening Range Finder completed.")

        loaded = _latest_opening_range_result()
        if loaded is None:
            st.info("No completed opening-range search yet.")
            return
        run_dir, summary, rankings, portfolio_holdout, by_instrument = loaded
        if summary["verdict"] == "PASS":
            st.success("Holdout verdict: PASS")
        else:
            st.error("Holdout verdict: REJECT — do not deploy this candidate")
        base = summary["holdout_base_cost"]
        harsh = summary["holdout_harsh_cost"]
        st.subheader("Best discovery candidate")
        st.write(summary["winner"]["description"])
        metrics = st.columns(5)
        metrics[0].metric("Portfolio return", f"{base['total_return']:.2%}")
        metrics[1].metric("SPY return", f"{base['benchmark_total_return']:.2%}")
        metrics[2].metric("Portfolio Sharpe", f"{base['sharpe']:.3f}")
        metrics[3].metric("Maximum drawdown", f"{base['max_drawdown']:.2%}")
        metrics[4].metric("Return @ 5 bps/side", f"{harsh['total_return']:.2%}")
        st.caption(
            f"{summary['entry_rule_count']:,} entry rules → "
            f"{summary['entry_finalist_count']:,} finalists → "
            f"{summary['candidate_count']:,} candidates · "
            f"{summary['instrument_count']} stocks / "
            f"{summary['bar_count']:,} bars · CUDA screen and validation · "
            f"artifact {run_dir.name}"
        )
        st.dataframe(rankings, width="stretch")
        st.subheader("Portfolio cost stress")
        st.dataframe(portfolio_holdout, width="stretch")
        st.subheader("Per-stock untouched holdout")
        base_by_instrument = by_instrument[
            by_instrument["cost_scenario_bps_per_side"] == orb_cost
        ]
        st.dataframe(
            base_by_instrument.sort_values("total_return", ascending=False),
            width="stretch",
        )
        st.download_button(
            "Download opening-range rankings CSV",
            rankings.to_csv(index=False),
            "opening_range_rankings.csv",
        )
        st.download_button(
            "Download opening-range portfolio holdout CSV",
            portfolio_holdout.to_csv(index=False),
            "opening_range_portfolio_holdout.csv",
        )
        st.download_button(
            "Download opening-range per-stock holdout CSV",
            by_instrument.to_csv(index=False),
            "opening_range_holdout_by_stock.csv",
        )
        st.download_button(
            "Download opening-range summary JSON",
            json.dumps(summary, indent=2),
            "opening_range_summary.json",
        )

    elif mode == "Scalping Finder":
        from analysis.optimizer import accelerator_status

        st.markdown(
            "Searches **mean-reversion, momentum, and breakout** entry structures, "
            "then optimizes holding period, profit target, and stop loss. Entry "
            "screening and execution-aware validation use CUDA when available, with "
            "next-bar fills, mandatory end-of-day flattening, explicit costs, four "
            "chronological discovery windows, and a final untouched 20% holdout."
        )
        st.warning(
            "Five-minute IEX bars do not contain bid/ask quotes or order-book state. "
            "This validates bar-based intraday ideas, not sub-second execution."
        )
        accelerator = accelerator_status()
        if accelerator["cuda_available"]:
            st.success(f"GPU ready: {accelerator['device']} for screening and validation.")
        else:
            st.info("CUDA unavailable; entry-rule screening will use the CPU.")
        c1, c2 = st.columns(2)
        with c1:
            scalp_device = st.selectbox("Scalping tensor device", ("auto", "cpu", "cuda"))
        with c2:
            scalp_cost = float(st.number_input(
                "Base cost per side (bps)", min_value=0.0, value=1.0, step=0.5,
            ))

        if st.button("Run Scalping Finder", type="primary"):
            with st.spinner("Screening entries, optimizing exits, and testing the untouched holdout..."):
                completed = _run_scalping_finder(device=scalp_device, cost_bps=scalp_cost)
            if completed.returncode != 0:
                st.error("Scalping Finder failed.")
                st.code(completed.stderr or completed.stdout)
                return
            st.success("Scalping Finder completed. The result below is the new run.")

        loaded = _latest_scalping_result()
        if loaded is None:
            st.info(
                "No completed scalping search yet. Populate SPY 5-minute data with "
                "`data/ingest_alpaca_intraday.py --timeframe 5m --since 2024-01-01`, then run the finder."
            )
            return

        run_dir, summary, rankings, holdout, by_instrument, by_group = loaded
        verdict = summary["verdict"]
        if verdict == "PASS":
            st.success("Holdout verdict: PASS")
        else:
            st.error("Holdout verdict: REJECT — do not deploy this candidate")
        winner = summary["winner"]
        base = summary["holdout_base_cost"]
        harsh = summary["holdout_harsh_cost"]
        st.subheader("Best discovery candidate")
        st.write(winner["description"])
        metrics = st.columns(5)
        metrics[0].metric("Average instrument return", f"{base['total_return']:.2%}")
        metrics[1].metric("SPY return", f"{base['benchmark_total_return']:.2%}")
        metrics[2].metric("Median instrument Sharpe", f"{base['sharpe']:.3f}")
        metrics[3].metric("Trades", int(base["n_trades"]))
        metrics[4].metric("Median @ 5 bps/side", f"{harsh['median_total_return']:.2%}")
        st.caption(
            f"{summary['entry_rule_count']:,} entry rules → "
            f"{summary['entry_finalist_count']:,} entry finalists → "
            f"{summary['candidate_count']:,} candidates across "
            f"{summary.get('instrument_window_cells', 4):,} instrument-window cells · "
            f"{summary['bar_count']:,} bars / {summary['session_count']:,} sessions · "
            f"artifact `{run_dir.name}`"
        )
        st.dataframe(rankings, width="stretch")
        if not by_group.empty:
            st.subheader("Untouched holdout by asset group")
            base_by_group = by_group[by_group["cost_scenario_bps_per_side"] == scalp_cost]
            st.dataframe(base_by_group.sort_values("total_return", ascending=False), width="stretch")
        if not by_instrument.empty:
            st.subheader("Per-instrument untouched holdout")
            base_by_instrument = by_instrument[by_instrument["cost_scenario_bps_per_side"] == scalp_cost]
            st.dataframe(base_by_instrument.sort_values("total_return", ascending=False), width="stretch")
        st.dataframe(holdout, width="stretch")
        st.download_button(
            "Download scalping rankings CSV", rankings.to_csv(index=False),
            "scalping_finder_rankings.csv",
        )
        st.download_button(
            "Download scalping holdout CSV", holdout.to_csv(index=False),
            "scalping_finder_holdout.csv",
        )
        if not by_instrument.empty:
            st.download_button(
                "Download per-instrument holdout CSV", by_instrument.to_csv(index=False),
                "scalping_finder_holdout_by_instrument.csv",
            )
        if not by_group.empty:
            st.download_button(
                "Download asset-group holdout CSV", by_group.to_csv(index=False),
                "scalping_finder_holdout_by_group.csv",
            )
        st.download_button(
            "Download scalping summary JSON", json.dumps(summary, indent=2),
            "scalping_finder_summary.json",
        )
