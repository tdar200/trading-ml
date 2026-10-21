"""GPU/CPU tensor pre-screen for serializable rule-strategy candidates."""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from analysis.vectorized_screen import _monthly_rebalance_indices, resolve_device


SUPPORTED_FAMILIES = {"momentum": 0, "mean_reversion": 1}


def screen_rule_candidates(
    closes: pd.DataFrame,
    candidates: list[dict[str, Any]],
    *,
    device: Literal["auto", "cpu", "cuda"] = "auto",
    cash_buffer: float = 0.02,
    turnover_cost_bps: float = 0.5,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Screen structural candidates as batched tensor portfolios."""
    import torch

    if not candidates:
        raise ValueError("candidates must be non-empty")
    required = {
        "signal_family", "trend_filter_days", "require_positive",
        "top_n", "lookback_days", "skip_days",
    }
    if any(not required.issubset(candidate) for candidate in candidates):
        raise ValueError(f"every candidate requires {sorted(required)}")
    unknown = {str(c["signal_family"]) for c in candidates} - set(SUPPORTED_FAMILIES)
    if unknown:
        raise ValueError(f"unsupported signal families: {sorted(unknown)}")
    if len(closes) < 2 or closes.shape[1] < 1 or closes.isna().any().any():
        raise ValueError("closes must be a complete date-by-asset matrix")
    if not 0 <= cash_buffer < 1 or turnover_cost_bps < 0:
        raise ValueError("invalid cash buffer or turnover cost")

    used_device = resolve_device(device)
    prices = torch.as_tensor(
        closes.to_numpy(dtype=np.float64), dtype=torch.float64, device=used_device,
    )
    count, asset_count = len(candidates), prices.shape[1]
    top_n = torch.tensor([int(c["top_n"]) for c in candidates], device=used_device)
    lookback = torch.tensor([int(c["lookback_days"]) for c in candidates], device=used_device)
    skip = torch.tensor([int(c["skip_days"]) for c in candidates], device=used_device)
    trend_days = torch.tensor([int(c["trend_filter_days"]) for c in candidates], device=used_device)
    require_positive = torch.tensor(
        [bool(c["require_positive"]) for c in candidates], dtype=torch.bool, device=used_device,
    )
    family = torch.tensor(
        [SUPPORTED_FAMILIES[str(c["signal_family"])] for c in candidates], device=used_device,
    )
    invalid = (
        (top_n < 1) | (top_n > asset_count) | (lookback < 1)
        | (skip < 0) | (trend_days < 0)
    )
    if bool(invalid.any()):
        raise ValueError("invalid structural candidate parameters")

    prefix = torch.cat([
        torch.zeros((1, asset_count), dtype=prices.dtype, device=used_device),
        torch.cumsum(prices, dim=0),
    ])
    asset_values = torch.zeros((count, asset_count), dtype=prices.dtype, device=used_device)
    cash = torch.ones(count, dtype=prices.dtype, device=used_device)
    equity_history = torch.ones((len(closes), count), dtype=prices.dtype, device=used_device)
    trade_counts = torch.zeros(count, dtype=torch.int64, device=used_device)
    active = torch.zeros(count, dtype=torch.bool, device=used_device)
    rebalance_indices = _monthly_rebalance_indices(closes.index)
    invest_fraction = 1.0 - cash_buffer

    for day in range(1, len(closes)):
        asset_values *= prices[day] / prices[day - 1]
        equity = cash + asset_values.sum(dim=1)

        if day in rebalance_indices:
            structural_warmup = torch.maximum(
                lookback + skip,
                torch.clamp(trend_days + skip - 1, min=0),
            )
            ready = day >= structural_warmup
            if bool(ready.any()):
                idx = torch.nonzero(ready, as_tuple=False).flatten()
                end_idx = day - skip[idx]
                start_idx = end_idx - lookback[idx]
                raw = prices[end_idx] / prices[start_idx] - 1.0
                scores = torch.where(family[idx, None] == 0, raw, -raw)

                td = trend_days[idx]
                safe_td = torch.clamp(td, min=1)
                trend_start = end_idx - safe_td + 1
                trend_mean = (
                    prefix[end_idx + 1] - prefix[trend_start]
                ) / safe_td[:, None].to(dtype=prices.dtype)
                trend_ok = (td[:, None] == 0) | (prices[end_idx] > trend_mean)
                signal_ok = (~require_positive[idx, None]) | (scores > 0)
                scores = torch.where(
                    trend_ok & signal_ok,
                    scores,
                    torch.full_like(scores, float("-inf")),
                )

                order = torch.argsort(scores, dim=1, descending=True, stable=True)
                ordered_scores = torch.gather(scores, 1, order)
                slots = torch.arange(asset_count, device=used_device)[None, :]
                selected = (slots < top_n[idx, None]) & torch.isfinite(ordered_scores)
                selected_count = selected.sum(dim=1).clamp(min=1)
                allocations = selected.to(dtype=prices.dtype) * (
                    invest_fraction / selected_count.to(dtype=prices.dtype)
                )[:, None]
                target = torch.zeros_like(scores)
                target.scatter_(1, order, allocations)

                current_weights = asset_values[idx] / equity[idx, None]
                turnover = torch.abs(target - current_weights).sum(dim=1)
                cost = equity[idx] * turnover * turnover_cost_bps / 10_000.0
                post_cost = torch.clamp(equity[idx] - cost, min=0.0)
                trade_counts[idx] += (torch.abs(target - current_weights) > 1e-12).sum(dim=1)
                asset_values[idx] = target * post_cost[:, None]
                cash[idx] = (1.0 - target.sum(dim=1)) * post_cost
                active[idx] = True
                equity = cash + asset_values.sum(dim=1)

        equity_history[day] = equity

    equity_np = equity_history.detach().cpu().numpy()
    returns = equity_np[1:] / equity_np[:-1] - 1.0
    mean = returns.mean(axis=0)
    std = returns.std(axis=0, ddof=0)
    sharpe = np.divide(
        mean, std, out=np.full_like(mean, np.nan), where=std > 1e-14,
    ) * np.sqrt(periods_per_year)
    years = max((len(closes) - 1) / periods_per_year, 1.0 / periods_per_year)
    final = equity_np[-1]
    cagr = np.where(final > 0, final ** (1.0 / years) - 1.0, np.nan)
    running_max = np.maximum.accumulate(equity_np, axis=0)
    drawdown = np.min(equity_np / running_max - 1.0, axis=0)
    active_np = active.detach().cpu().numpy()
    trades_np = trade_counts.detach().cpu().numpy()

    rows = [{
        **candidate,
        "screen_sharpe": float(sharpe[i]) if active_np[i] else float("nan"),
        "screen_cagr": float(cagr[i]) if active_np[i] else float("nan"),
        "screen_max_drawdown": float(drawdown[i]) if active_np[i] else float("nan"),
        "screen_final_equity": float(final[i]),
        "screen_trades": int(trades_np[i]),
        "screen_device": used_device,
    } for i, candidate in enumerate(candidates)]
    result = pd.DataFrame(rows)
    result["screen_score"] = (
        result["screen_sharpe"] - 0.10 * result["screen_max_drawdown"].abs()
    )
    return result.sort_values("screen_score", ascending=False, na_position="last").reset_index(drop=True)
