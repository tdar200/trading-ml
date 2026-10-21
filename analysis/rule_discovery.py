"""Reproducible single-instrument rule discovery with chronological selection.

Bars are labelled by their opening timestamp. Signals use completed bars and
execute at the next observed open. Daily labels are session dates. One long
position, fractional shares, no leverage; intrabar stop/target ambiguity is
resolved stop-first. This research simulator does not model order-book fills.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

VERSION = 1
TIMEFRAMES = ("15m", "1h", "1d")
TRIGGERS = ("rsi_cross", "ma_cross", "breakout")


@dataclass(frozen=True)
class Rule:
    instrument: str
    timeframe: str
    trigger: str = "rsi_cross"
    period: int = 7
    trend_period: int = 0
    relative_volume: float = 0.0
    hold_bars: int = 4
    stop_pct: float = 0.02
    target_pct: float = 0.04
    allocation: float = 0.95

    def __post_init__(self):
        if not self.instrument or self.timeframe not in TIMEFRAMES or self.trigger not in TRIGGERS:
            raise ValueError("Invalid instrument, timeframe or entry trigger")
        if self.period < 2 or self.trend_period < 0 or self.hold_bars < 1:
            raise ValueError("Periods and holding duration are invalid")
        numbers = (self.relative_volume, self.stop_pct, self.target_pct, self.allocation)
        if not all(np.isfinite(v) for v in numbers) or self.relative_volume < 0:
            raise ValueError("Rule values must be finite and volume nonnegative")
        if not 0 <= self.stop_pct < 1 or not 0 <= self.target_pct < 1 or not 0 < self.allocation <= 1:
            raise ValueError("Invalid stop, target or allocation")

    @property
    def key(self):
        return sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]

    @property
    def description(self):
        entry = {"rsi_cross": "RSI crosses above 30", "ma_cross": "price crosses above its moving average",
                 "breakout": "close breaks above the prior rolling high"}[self.trigger]
        return (f"{self.instrument} · {self.timeframe} · {entry} ({self.period} bars); "
                f"trend MA {self.trend_period or 'off'}, relative volume ≥ {self.relative_volume:g}; "
                f"hold {self.hold_bars} bars, stop {self.stop_pct:.1%}, target {self.target_pct:.1%}")


def validate_bars(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["ts", "open", "high", "low", "close", "volume"]
    if not set(columns).issubset(frame):
        raise ValueError("Bars require ts, open, high, low, close and volume")
    result = frame[columns].copy()
    result["ts"] = pd.to_datetime(result.ts, utc=True, errors="raise")
    if result.empty or result.ts.isna().any() or result.ts.duplicated().any():
        raise ValueError("Bars must be nonempty with unique valid timestamps")
    result = result.sort_values("ts").reset_index(drop=True)
    values = result[columns[1:]].to_numpy(float)
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
        raise ValueError("OHLC must be finite and positive; volume must be nonnegative")
    if ((result.high < result[["open", "close", "low"]].max(axis=1)) |
        (result.low > result[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("Inconsistent OHLC bounds")
    return result


def signals(frame: pd.DataFrame, rule: Rule) -> pd.Series:
    close = frame.close
    if rule.trigger == "rsi_cross":
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(rule.period).mean()
        loss = (-delta.clip(upper=0)).rolling(rule.period).mean()
        rsi = 100 * gain / (gain + loss).replace(0, np.nan)
        entry = (rsi.shift(1) <= 30) & (rsi > 30)
    elif rule.trigger == "ma_cross":
        average = close.rolling(rule.period).mean()
        entry = (close.shift(1) <= average.shift(1)) & (close > average)
    else:
        prior_high = frame.high.shift(1).rolling(rule.period).max()
        entry = close > prior_high
    if rule.trend_period:
        entry &= close > close.rolling(rule.trend_period).mean()
    if rule.relative_volume:
        prior_volume = frame.volume.shift(1).rolling(rule.period).mean().replace(0, np.nan)
        entry &= frame.volume / prior_volume >= rule.relative_volume
    return entry.fillna(False)


def simulate(frame: pd.DataFrame, rule: Rule, start, end, cost_bps: float = 2.5,
             initial_cash: float = 10_000, *, checked: bool = False) -> dict:
    """Trade only in [start,end); history before start is indicator warm-up.

    Time exits occur at the open after hold_bars, stop gaps fill at the open,
    targets fill at their limit (conservative even on favourable gaps).
    The final bar closes any remaining position for a comparable flat endpoint.
    """
    if not np.isfinite(cost_bps) or not 0 <= cost_bps < 1000 or initial_cash <= 0:
        raise ValueError("Invalid execution cost or starting cash")
    frame = frame if checked else validate_bars(frame)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise ValueError("Evaluation boundaries must be ordered and timezone-aware")
    # Never inspect the held-back suffix, even for feature calculation.
    history = frame.loc[frame.ts < end].reset_index(drop=True)
    entry = signals(history, rule).to_numpy()
    indices = np.flatnonzero((history.ts >= start).to_numpy())
    if len(indices) < 2:
        raise ValueError("Evaluation interval needs at least two bars")
    cash, qty, position = float(initial_cash), 0.0, None
    trades, curve = [], []
    rate = cost_bps / 10_000
    for i in indices:
        bar = history.iloc[i]
        # A position must exist before the open to be eligible for a time exit.
        timed_exit = position is not None and i - position["entry_index"] >= rule.hold_bars
        if position is None and i > 0 and entry[i - 1] and i != indices[-1]:
            budget = cash * rule.allocation
            qty = budget / (float(bar.open) * (1 + rate))
            notional = qty * float(bar.open)
            cash -= notional * (1 + rate)
            position = {"signal_time": history.ts.iloc[i - 1].isoformat(),
                        "entry_time": bar.ts.isoformat(), "entry_price": float(bar.open),
                        "quantity": qty, "entry_fee": notional * rate, "entry_index": int(i)}
        if position is not None:
            stop = position["entry_price"] * (1 - rule.stop_pct)
            target = position["entry_price"] * (1 + rule.target_pct)
            price, reason = None, None
            if rule.stop_pct and bar.open <= stop:
                price, reason = float(bar.open), "stop_gap"
            elif timed_exit:
                price, reason = float(bar.open), "holding_period"
            elif rule.stop_pct and bar.low <= stop:
                price, reason = stop, "stop"
            elif rule.target_pct and bar.high >= target:
                price, reason = target, "target"
            elif i == indices[-1]:
                price, reason = float(bar.close), "evaluation_end"
            if price is not None:
                exit_fee = qty * price * rate
                gross = qty * (price - position["entry_price"])
                cash += qty * price - exit_fee
                trades.append({**position, "exit_bar_time": bar.ts.isoformat(), "exit_price": price,
                               "exit_reason": reason, "exit_fee": exit_fee,
                               "gross_pnl": gross, "net_pnl": gross - position["entry_fee"] - exit_fee})
                qty, position = 0.0, None
        curve.append({"ts": bar.ts.isoformat(), "cash": cash, "quantity": qty,
                      "equity": cash + qty * float(bar.close)})
    equity = pd.DataFrame(curve)
    dates = pd.to_datetime(equity.ts, utc=True).dt.date
    daily = equity.groupby(dates).equity.last().to_numpy()
    returns = np.diff(np.r_[initial_cash, daily]) / np.r_[initial_cash, daily][:-1]
    all_equity = np.r_[initial_cash, equity.equity.to_numpy()]
    std = returns.std(ddof=1) if len(returns) > 1 else 0.0
    metrics = {"total_return": float(daily[-1] / initial_cash - 1),
               "sharpe": float(returns.mean() / std * np.sqrt(252)) if std > 1e-12 else 0.0,
               "max_drawdown": float(np.min(all_equity / np.maximum.accumulate(all_equity) - 1)),
               "trades": len(trades), "sessions": len(daily),
               "gross_pnl": float(sum(t["gross_pnl"] for t in trades)),
               "costs": float(sum(t["entry_fee"] + t["exit_fee"] for t in trades))}
    return {"metrics": metrics, "trades": trades, "equity": curve}


def candidate_grid(instruments, timeframes, triggers, periods=(7, 14), trends=(0, 50),
                   volumes=(0.0, 1.0), holds=(4, 8), **risk) -> list[Rule]:
    dimensions = (instruments, timeframes, triggers, periods, trends, volumes, holds)
    if any(not len(d) for d in dimensions):
        raise ValueError("Select at least one value in every search dimension")
    count = int(np.prod([len(d) for d in dimensions]))
    if count > 2000:
        raise ValueError("Reference search is limited to 2,000 candidates; narrow the grid")
    rules = [Rule(*values, **risk) for values in product(*dimensions)]
    return list({rule.key: rule for rule in rules}.values())


class Cancelled(Exception):
    pass


def discover(data: dict[tuple[str, str], pd.DataFrame], rules: list[Rule], *, cost_bps=2.5,
             min_trades=3, progress: Callable | None = None, cancelled: Callable | None = None) -> dict:
    """Three expanding-window selections followed by a sealed final 20%.

    Each outer fold searches instrument/timeframe/rule using only prior bars.
    The final candidate is selected on all development data. Outer evidence
    describes the selection procedure, not repeated tests of that final rule.
    """
    if not rules or min_trades < 1:
        raise ValueError("Candidates and a positive minimum trade count are required")
    required = {(r.instrument, r.timeframe) for r in rules}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Missing data: {sorted(missing)}")
    data = {key: validate_bars(data[key]) for key in sorted(required)}
    date_sets = [set(frame.ts.dt.date) for frame in data.values()]
    common = sorted(set.intersection(*date_sets))
    if len(common) < 80:
        raise ValueError("At least 80 common sessions are required across selected instruments/timeframes")
    boundary = lambda i: pd.Timestamp(common[i], tz="UTC")
    cut = int(len(common) * .8)
    first = int(cut * .5)
    edges = np.linspace(first, cut, 4, dtype=int)
    start = boundary(0)
    final_end = pd.Timestamp(common[-1], tz="UTC") + pd.Timedelta(days=1)
    done, total = 0, len(rules) * 4 + 3

    def checkpoint(message):
        nonlocal done
        if cancelled and cancelled():
            raise Cancelled("Search cancelled")
        done += 1
        if progress:
            progress(min(95, int(done / total * 95)), message)

    def select(end):
        rows = []
        for rule in rules:
            checkpoint(f"Evaluating {rule.instrument} {rule.timeframe} {rule.trigger}")
            result = simulate(data[(rule.instrument, rule.timeframe)], rule, start, end, cost_bps, checked=True)
            m = result["metrics"]
            eligible = m["trades"] >= min_trades
            score = m["sharpe"] - abs(m["max_drawdown"]) if eligible else -1e100
            rows.append({"candidate_id": rule.key, "description": rule.description, **asdict(rule),
                         **m, "eligible": eligible, "score": score})
        return sorted(rows, key=lambda r: (-r["score"], r["candidate_id"]))

    folds = []
    by_id = {r.key: r for r in rules}
    for left, right in zip(edges[:-1], edges[1:]):
        train_end, test_end = boundary(int(left)), boundary(int(right))
        leaders = select(train_end)
        leader = leaders[0]
        checkpoint("Evaluating frozen outer-fold candidate")
        if leader["eligible"]:
            rule = by_id[leader["candidate_id"]]
            result = simulate(data[(rule.instrument, rule.timeframe)], rule, train_end, test_end, cost_bps, checked=True)
            metrics = result["metrics"]
        else:
            metrics = {"total_return": 0.0, "sharpe": 0.0, "trades": 0}
        folds.append({"candidate_id": leader["candidate_id"], "train_end": train_end.isoformat(),
                      "test_start": train_end.isoformat(), "test_end": test_end.isoformat(),
                      "training_eligible": leader["eligible"], **metrics})
    rankings = select(boundary(cut))
    best = rankings[0]
    rule = by_id[best["candidate_id"]]
    passed = bool(best["eligible"] and best["total_return"] > 0 and
                  all(f["training_eligible"] and f["trades"] >= min_trades and f["total_return"] > 0 for f in folds))
    if cancelled and cancelled():
        raise Cancelled("Search cancelled")
    evaluation = None
    if passed:
        evaluation = simulate(data[(rule.instrument, rule.timeframe)], rule, boundary(cut), final_end, cost_bps, checked=True)
    # Always provide inspectable discovery evidence even when holdout is sealed.
    preview = simulate(data[(rule.instrument, rule.timeframe)], rule, start, boundary(cut), cost_bps, checked=True)
    report = {"version": VERSION, "cost_bps": cost_bps, "min_trades": min_trades,
              "candidate_count": len(rules), "common_sessions": len(common),
              "development_start": start.isoformat(), "holdout_start": boundary(cut).isoformat(),
              "evaluation_end": final_end.isoformat(), "holdout_opened": passed,
              "deployment_approved": False, "rankings": rankings, "folds": folds,
              "candidate": asdict(rule), "candidate_id": rule.key,
              "preview": preview, "evaluation": evaluation,
              "verdict": ("FORWARD_RESEARCH_CANDIDATE" if passed and evaluation["metrics"]["total_return"] > 0
                          and evaluation["metrics"]["trades"] >= min_trades else "REJECT"),
              "reason": ("Outer selection windows passed; final evaluation opened" if passed else
                         "Insufficient activity or nonpositive return in development/outer validation; final period sealed")}
    if progress:
        progress(100, "Search complete")
    return report


def save_run(directory: Path, report: dict, data: dict, rules: list[Rule]) -> None:
    """Write immutable-input snapshot and manifest; caller owns a unique run directory."""
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for index, key in enumerate(sorted({(r.instrument, r.timeframe) for r in rules})):
        path = directory / f"bars_{index}.parquet"
        validate_bars(data[key]).to_parquet(path, index=False)
        files[path.name] = {"instrument": key[0], "timeframe": key[1], "sha256": sha256(path.read_bytes()).hexdigest()}
    report = {**report, "rules": [asdict(r) for r in rules], "data_files": files,
              "simulator_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
    payload = json.dumps(report, indent=2, allow_nan=False)
    (directory / "report.json").write_text(payload)
    (directory / "report.sha256").write_text(sha256(payload.encode()).hexdigest())


def load_run(directory: Path) -> dict:
    payload = (directory / "report.json").read_bytes()
    if sha256(payload).hexdigest() != (directory / "report.sha256").read_text().strip():
        raise ValueError("Saved report fingerprint mismatch")
    report = json.loads(payload)
    if report["version"] != VERSION:
        raise ValueError("Unsupported simulator version")
    return report


def replay_run(directory: Path) -> dict:
    report = load_run(directory)
    if report["simulator_sha256"] != sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("Simulator changed since this run; use its original code version")
    rule = Rule(**report["candidate"])
    if rule.key != report["candidate_id"]:
        raise ValueError("Candidate fingerprint mismatch")
    item = next((name, meta) for name, meta in report["data_files"].items()
                if (meta["instrument"], meta["timeframe"]) == (rule.instrument, rule.timeframe))
    name, metadata = item
    if Path(name).name != name:
        raise ValueError("Invalid snapshot path")
    path = directory / name
    if sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("Saved data fingerprint mismatch")
    start = report["holdout_start"] if report["holdout_opened"] else report["development_start"]
    end = report["evaluation_end"] if report["holdout_opened"] else report["holdout_start"]
    result = simulate(pd.read_parquet(path), rule, start, end, report["cost_bps"])
    expected = report["evaluation"] if report["holdout_opened"] else report["preview"]
    if result != expected:
        raise ValueError("Replay differs from saved evidence")
    return result
