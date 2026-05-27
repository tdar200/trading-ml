"""
Backtest Lab — Stage 9 self-service strategy tester.

Tabs (sub-modes): Single | Grid sweep | Optuna optimization.

Strategy discovery via ``pkgutil`` walks ``strategies/`` for
``(Strategy, StrategyConfig)`` class pairs. The parameter form is
auto-rendered from the Pydantic ``StrategyConfig`` schema, so adding a
new strategy = drop a file in ``strategies/`` and the form appears.

All three modes share ``analysis.runner.run`` / ``analysis.walk_forward.walk_forward``
as the engine, so in-dashboard backtest numbers agree exactly with the
canonical Stage-3 harness.

The Optuna mode hands off the rich visualization (parallel coords,
hyperparam importance, contour plots) to ``optuna-dashboard``, launched
as a separate browser tab — no point reimplementing what's already free.
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil
import subprocess
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from data.tickers import ALL_TICKERS, BENCHMARK, UNIVERSE


_PROJECT_ROOT = Path(__file__).parent.parent.parent
_OPTUNA_DB = _PROJECT_ROOT / "results" / "optuna.db"


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


def _run_optuna(strategy_cls, config_cls, base_params: dict,
                param_space: dict, n_trials: int, start, end) -> Any:
    """Run an Optuna study; objective wraps a single-backtest run."""
    import optuna
    from analysis.runner import run as backtest_run
    from analysis.runspec import RunSpec
    from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
    from data.instruments import VENUE_MIC, build_universe_instruments

    catalog = ParquetDataCatalog(path=str((_PROJECT_ROOT / "data" / "catalog").resolve()))
    bars_by_ticker = {
        t: catalog.bars(instrument_ids=[f"{t}.{VENUE_MIC}"])
        for t in ALL_TICKERS
    }
    instruments = build_universe_instruments()

    def objective(trial):
        params = dict(base_params)
        for name, spec in param_space.items():
            if spec["type"] == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
            elif spec["type"] == "float":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"])
        cfg = config_cls(**params)
        spec_obj = RunSpec(
            label=f"optuna-{trial.number}",
            strategy_class=strategy_cls,
            strategy_config=cfg,
            start=start, end=end,
            slippage_prob=0.0,
        )
        result = backtest_run(spec_obj, bars_by_ticker=bars_by_ticker, instruments=instruments)
        return result.metrics.get("sharpe", 0.0)

    _OPTUNA_DB.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        storage=f"sqlite:///{_OPTUNA_DB}",
        study_name=f"lab-{strategy_cls.__name__}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}",
        direction="maximize",
    )
    study.optimize(objective, n_trials=n_trials)
    return study


def render(data_sources) -> None:
    st.header("Backtest Lab")
    st.caption(
        "Self-service backtest + grid sweep + Optuna optimization. "
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
        mode = st.segmented_control("Mode", options=("Single", "Grid", "Optuna"), default="Single")
        if mode is None:
            mode = "Single"

    st.subheader("Strategy parameters")
    base_params = _render_param_form(config_cls, key_prefix=f"params_{strategy_name}_{mode}")

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
            "use Optuna mode."
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
                st.error(f"{cells} cells > 200 limit; tighten the range or use Optuna mode.")
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

    elif mode == "Optuna":
        st.markdown(
            "Suggest int / float ranges for one or more parameters. Optuna "
            "runs `n_trials` backtests, maximizing Sharpe. After it finishes, "
            "launch the rich hyperparam dashboard separately:"
        )
        st.code(f"uv run optuna-dashboard sqlite:///{_OPTUNA_DB}", language="bash")
        n_trials = st.number_input("n_trials", min_value=5, value=30, step=5)
        numeric_fields = [k for k, (ann, _) in config_fields(config_cls).items()
                          if ann in ("int", "float", int, float)]
        opt_params = st.multiselect("Parameters to optimize", options=numeric_fields)
        param_space: dict = {}
        for p in opt_params:
            cc1, cc2 = st.columns(2)
            with cc1:
                lo = st.number_input(f"{p} low", value=1, key=f"opt_lo_{p}")
            with cc2:
                hi = st.number_input(f"{p} high", value=10, key=f"opt_hi_{p}")
            ann = config_fields(config_cls)[p][0]
            param_space[p] = {
                "type": "int" if ann in ("int", int) else "float",
                "low": lo, "high": hi,
            }
        if st.button("Run Optuna optimization", type="primary") and param_space:
            with st.spinner(f"Running Optuna ({n_trials} trials)..."):
                try:
                    study = _run_optuna(strategy_cls, config_cls, base_params,
                                        param_space, int(n_trials), start_ts, end_ts)
                except Exception as exc:
                    st.error(f"Optuna run failed: {exc}")
                    return
            st.success("Done")
            cols = st.columns(2)
            cols[0].metric("Best Sharpe", f"{study.best_value:.4f}")
            cols[1].metric("Best trial", study.best_trial.number)
            st.json(study.best_params)
            st.info(
                f"For the full hyperparameter view (parallel coords, importance, "
                f"contour plots), run:\n\n`uv run optuna-dashboard sqlite:///{_OPTUNA_DB}`"
            )
