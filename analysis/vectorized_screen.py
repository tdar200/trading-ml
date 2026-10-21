"""Batched tensor pre-screen for momentum parameter combinations.

This is a fast approximation, not an execution simulator. It mirrors monthly
signal timing and portfolio drift, then sends a small diverse finalist set to
the canonical Nautilus backtester. PyTorch automatically uses CUDA when a
CUDA-enabled build is installed; otherwise the exact same code runs on CPU.
"""
from __future__ import annotations

from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd


def bars_to_close_frame(
    bars_by_ticker: dict[str, list],
    tickers: Iterable[str],
) -> pd.DataFrame:
    """Align Nautilus bars on normalized UTC trading dates using an inner join."""
    series: list[pd.Series] = []
    for ticker in tickers:
        bars = bars_by_ticker.get(ticker, [])
        if not bars:
            raise ValueError(f"no bars for {ticker}")
        index = pd.to_datetime([bar.ts_event for bar in bars], unit="ns", utc=True).tz_convert(None).normalize()
        item = pd.Series([float(bar.close) for bar in bars], index=index, name=ticker)
        if item.index.has_duplicates:
            raise ValueError(f"duplicate trading date for {ticker}")
        series.append(item)
    frame = pd.concat(series, axis=1, join="inner").sort_index()
    if frame.empty or frame.isna().any().any():
        raise ValueError("aligned close matrix is empty or incomplete")
    if (frame <= 0).any().any():
        raise ValueError("close prices must be positive")
    return frame


def resolve_device(requested: Literal["auto", "cpu", "cuda"] = "auto") -> str:
    import torch

    cuda = bool(torch.cuda.is_available())
    if requested == "cuda" and not cuda:
        raise RuntimeError("CUDA requested but this PyTorch build/device has no CUDA support")
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    return "cuda" if requested in {"auto", "cuda"} and cuda else "cpu"


def _monthly_rebalance_indices(index: pd.DatetimeIndex) -> set[int]:
    periods = index.to_period("M")
    return {i for i in range(len(index)) if i == 0 or periods[i] != periods[i - 1]}


def screen_momentum(
    closes: pd.DataFrame,
    candidates: list[dict[str, Any]],
    *,
    device: Literal["auto", "cpu", "cuda"] = "auto",
    cash_buffer: float = 0.02,
    turnover_cost_bps: float = 0.5,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Evaluate all momentum configurations concurrently as tensor portfolios.

    ``closes`` must contain only tradeable assets in deterministic strategy
    universe order. Candidate dictionaries require ``top_n``,
    ``lookback_days``, and ``skip_days``.
    """
    import torch

    if not candidates:
        raise ValueError("candidates must be non-empty")
    required_params = {"top_n", "lookback_days", "skip_days"}
    if any(not required_params.issubset(candidate) for candidate in candidates):
        raise ValueError(f"every candidate requires {sorted(required_params)}")
    if len(closes) < 2 or closes.shape[1] < 1:
        raise ValueError("closes needs at least two dates and one asset")
    if not 0 <= cash_buffer < 1:
        raise ValueError("cash_buffer must be in [0, 1)")
    if turnover_cost_bps < 0:
        raise ValueError("turnover_cost_bps must be non-negative")

    used_device = resolve_device(device)
    prices = torch.as_tensor(closes.to_numpy(dtype=np.float64), dtype=torch.float64, device=used_device)
    count, asset_count = len(candidates), prices.shape[1]
    top_n = torch.tensor([int(c["top_n"]) for c in candidates], device=used_device)
    lookback = torch.tensor([int(c["lookback_days"]) for c in candidates], device=used_device)
    skip = torch.tensor([int(c["skip_days"]) for c in candidates], device=used_device)
    if bool(((top_n < 1) | (top_n > asset_count) | (lookback < 1) | (skip < 0)).any()):
        raise ValueError("invalid top_n/lookback_days/skip_days candidate")

    asset_values = torch.zeros((count, asset_count), dtype=torch.float64, device=used_device)
    cash = torch.ones(count, dtype=torch.float64, device=used_device)
    equity_history = torch.ones((len(closes), count), dtype=torch.float64, device=used_device)
    trade_counts = torch.zeros(count, dtype=torch.int64, device=used_device)
    active = torch.zeros(count, dtype=torch.bool, device=used_device)
    rebalance_indices = _monthly_rebalance_indices(closes.index)
    invest_fraction = 1.0 - cash_buffer

    for day in range(1, len(closes)):
        asset_values *= prices[day] / prices[day - 1]
        equity = cash + asset_values.sum(dim=1)

        if day in rebalance_indices:
            ready = day >= lookback + skip
            if bool(ready.any()):
                candidate_idx = torch.nonzero(ready, as_tuple=False).flatten()
                end_idx = day - skip[candidate_idx]
                start_idx = end_idx - lookback[candidate_idx]
                momentum = prices[end_idx] / prices[start_idx] - 1.0
                # Stable ordering preserves universe order for exact ties, as
                # the canonical Python ranker does.
                order = torch.argsort(momentum, dim=1, descending=True, stable=True)
                target = torch.zeros_like(momentum)
                slots = torch.arange(asset_count, device=used_device)[None, :]
                selected = slots < top_n[candidate_idx, None]
                allocations = selected.to(dtype=prices.dtype) * (
                    invest_fraction / top_n[candidate_idx].to(dtype=prices.dtype)
                )[:, None]
                target.scatter_(1, order, allocations)

                current_weights = asset_values[candidate_idx] / equity[candidate_idx, None]
                turnover = torch.abs(target - current_weights).sum(dim=1)
                cost = equity[candidate_idx] * turnover * turnover_cost_bps / 10_000.0
                post_cost_equity = torch.clamp(equity[candidate_idx] - cost, min=0.0)
                changed = torch.abs(target - current_weights) > 1e-12
                trade_counts[candidate_idx] += changed.sum(dim=1)
                asset_values[candidate_idx] = target * post_cost_equity[:, None]
                cash[candidate_idx] = (1.0 - target.sum(dim=1)) * post_cost_equity
                active[candidate_idx] = True
                equity = cash + asset_values.sum(dim=1)

        equity_history[day] = equity

    equity_np = equity_history.detach().cpu().numpy()
    returns = equity_np[1:] / equity_np[:-1] - 1.0
    mean = returns.mean(axis=0)
    std = returns.std(axis=0, ddof=0)
    sharpe = np.divide(mean, std, out=np.full_like(mean, np.nan), where=std > 1e-14) * np.sqrt(periods_per_year)
    years = max((len(closes) - 1) / periods_per_year, 1.0 / periods_per_year)
    final = equity_np[-1]
    cagr = np.where(final > 0, final ** (1.0 / years) - 1.0, np.nan)
    running_max = np.maximum.accumulate(equity_np, axis=0)
    max_drawdown = np.min(equity_np / running_max - 1.0, axis=0)

    rows: list[dict[str, Any]] = []
    active_np = active.detach().cpu().numpy()
    trades_np = trade_counts.detach().cpu().numpy()
    for i, params in enumerate(candidates):
        rows.append({
            **params,
            "screen_sharpe": float(sharpe[i]) if active_np[i] else float("nan"),
            "screen_cagr": float(cagr[i]) if active_np[i] else float("nan"),
            "screen_max_drawdown": float(max_drawdown[i]) if active_np[i] else float("nan"),
            "screen_final_equity": float(final[i]),
            "screen_trades": int(trades_np[i]),
            "screen_device": used_device,
        })
    result = pd.DataFrame(rows)
    result["screen_score"] = (
        result["screen_sharpe"] - 0.10 * result["screen_max_drawdown"].abs()
    )
    return result.sort_values("screen_score", ascending=False, na_position="last").reset_index(drop=True)


def select_diverse_finalists(
    screened: pd.DataFrame,
    param_names: list[str],
    n_finalists: int,
    *,
    pool_multiplier: int = 5,
) -> pd.DataFrame:
    """Greedily retain strong candidates from separated parameter regions."""
    if n_finalists <= 0:
        raise ValueError("n_finalists must be positive")
    required = {"screen_score", *param_names}
    missing = required - set(screened.columns)
    if missing:
        raise ValueError(f"screened frame missing columns: {sorted(missing)}")
    valid = screened.replace([np.inf, -np.inf], np.nan).dropna(subset=["screen_score"]).copy()
    if valid.empty:
        return valid
    pool = valid.nlargest(min(len(valid), n_finalists * pool_multiplier), "screen_score").reset_index(drop=True)
    if len(pool) <= n_finalists:
        return pool
    numeric = pool[param_names].astype(float)
    ranges = numeric.max() - numeric.min()
    normalized = (numeric - numeric.min()) / ranges.replace(0, 1)
    selected = [0]
    while len(selected) < n_finalists:
        remaining = [i for i in range(len(pool)) if i not in selected]
        distances = []
        for idx in remaining:
            minimum = min(float(np.linalg.norm(normalized.iloc[idx] - normalized.iloc[chosen])) for chosen in selected)
            # Favor distance while retaining a modest quality tie-break.
            quality = 1.0 - idx / max(len(pool) - 1, 1)
            distances.append((minimum + 0.05 * quality, idx))
        selected.append(max(distances)[1])
    return pool.iloc[selected].sort_values("screen_score", ascending=False).reset_index(drop=True)
