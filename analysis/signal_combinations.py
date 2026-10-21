"""GPU-first discovery of oriented 1–4 primitive-signal combinations."""
from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from analysis.signal_library import (
    PARAMETRIC_INDICATORS, SIGNAL_NAMES, IndicatorVariant, compute_parameterized_semantic_signal_library,
    compute_parameterized_signal_library, compute_semantic_signal_library,
    compute_signal_library,
)
from analysis.vectorized_screen import resolve_device


def sample_signal_tensor(
    bars_by_ticker: dict[str, pd.DataFrame], session_dates: np.ndarray,
    decision_bar: int, *, bar_minutes: int,
    semantic: bool = True,
    variants: tuple[IndicatorVariant, ...] | None = None,
    device: str = "cpu",
) -> np.ndarray:
    """Sample every primitive at a completed decision bar for each asset/session."""
    used_device = resolve_device(device)
    signal_count = len(variants) if variants is not None else len(SIGNAL_NAMES)
    tensor = np.full((len(session_dates), len(bars_by_ticker), signal_count), np.nan)
    date_index = {value: index for index, value in enumerate(session_dates)}
    for column, (_, frame) in enumerate(bars_by_ticker.items()):
        ordered = frame.sort_values("ts").reset_index(drop=True)
        if variants is None:
            signals = (
                compute_semantic_signal_library(ordered)
                if semantic else compute_signal_library(ordered)
            )
        else:
            raw = compute_parameterized_signal_library(ordered, variants)
            if semantic and str(used_device).startswith("cuda"):
                signals = _semanticize_parameterized_cuda(raw, ordered, variants, used_device)
            else:
                signals = (
                    compute_parameterized_semantic_signal_library(ordered, variants)
                    if semantic else raw
                )
        eastern = pd.to_datetime(ordered["ts"], utc=True).dt.tz_convert("America/New_York")
        bars = ((eastern.dt.hour * 60 + eastern.dt.minute - 570) // bar_minutes).astype(int)
        for row in np.flatnonzero(bars.to_numpy() == decision_bar):
            session = eastern.iloc[row].date()
            if session in date_index:
                tensor[date_index[session], column] = signals.iloc[row].to_numpy(float)
    return tensor


def _semanticize_parameterized_cuda(
    raw: pd.DataFrame, frame: pd.DataFrame,
    variants: tuple[IndicatorVariant, ...], device: str,
) -> pd.DataFrame:
    """Apply the parameterized threshold/crossover rules in batched CUDA tensors."""
    import torch

    values = torch.as_tensor(raw.to_numpy(float), dtype=torch.float64, device=device)
    previous = torch.roll(values, 1, dims=0)
    result = torch.zeros_like(values)
    eastern = pd.to_datetime(frame["ts"], utc=True).dt.tz_convert("America/New_York")
    sessions = eastern.dt.date.to_numpy()
    same_session_np = np.zeros(len(sessions), dtype=bool)
    same_session_np[1:] = sessions[1:] == sessions[:-1]
    same_session = torch.as_tensor(same_session_np, device=device)[:, None]

    def columns(indicators: set[str]) -> torch.Tensor:
        return torch.as_tensor(
            [i for i, variant in enumerate(variants) if variant.indicator in indicators],
            dtype=torch.long, device=device,
        )

    def crossing(cols: torch.Tensor, low: float = 0.0, high: float = 0.0) -> None:
        if cols.numel() == 0:
            return
        value, prior = values[:, cols], previous[:, cols]
        result[:, cols] = torch.where(
            same_session & (prior <= low) & (value > low), torch.ones_like(value),
            torch.where(
                same_session & (prior >= high) & (value < high),
                -torch.ones_like(value), torch.zeros_like(value),
            ),
        )

    crossing(columns({"rsi"}), -0.4, 0.4)
    oscillator = columns({"stochastic", "williams_r"})
    if oscillator.numel():
        value, prior = values[:, oscillator], previous[:, oscillator]
        result[:, oscillator] = torch.where(
            same_session & (prior <= -0.3) & (value > prior), torch.ones_like(value),
            torch.where(
                same_session & (prior >= 0.3) & (value < prior),
                -torch.ones_like(value), torch.zeros_like(value),
            ),
        )
    crossing(columns({"stochastic_signal"}))
    cci = columns({"cci"})
    if cci.numel():
        value = values[:, cci]
        result[:, cci] = torch.where(
            value <= -1, torch.ones_like(value),
            torch.where(value >= 1, -torch.ones_like(value), torch.zeros_like(value)),
        )
    crossing(columns({"bollinger"}), -2.0, 2.0)
    donchian = columns({"donchian"})
    if donchian.numel():
        value = values[:, donchian]
        result[:, donchian] = torch.where(
            value >= 0.45, torch.ones_like(value),
            torch.where(value <= -0.45, -torch.ones_like(value), torch.zeros_like(value)),
        )
    volume = columns({"directional_volume", "obv"})
    if volume.numel():
        value = values[:, volume]
        result[:, volume] = torch.where(
            value >= 0.5, torch.ones_like(value),
            torch.where(value <= -0.5, -torch.ones_like(value), torch.zeros_like(value)),
        )
    handled = {"rsi", "stochastic", "williams_r", "stochastic_signal", "cci", "bollinger", "donchian", "directional_volume", "obv"}
    crossing(columns(set(PARAMETRIC_INDICATORS) - handled))
    return pd.DataFrame(result.cpu().numpy(), index=raw.index, columns=raw.columns)


def orient_and_select_signals(
    tensor: np.ndarray, target: np.ndarray, indices: np.ndarray, *, top_k: int = 12,
    redundancy_threshold: float | None = None,
    device: str = "cpu",
) -> list[tuple[int, int, float]]:
    """Rank and orient indicator variants using only the orientation period.

    ``redundancy_threshold`` is optional.  The unrestricted Strategy Finder
    leaves it unset so correlated variants (for example RSI(7) and RSI(14))
    remain distinct searchable inputs rather than being collapsed into a
    coarse signal family.
    """
    if redundancy_threshold is None and str(resolve_device(device)).startswith("cuda"):
        return _orient_and_select_signals_cuda(tensor, target, indices, top_k, device)
    rows = []
    for signal in range(tensor.shape[2]):
        correlations = []
        for index in indices:
            x, y = tensor[index, :, signal], target[index]
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() >= 4 and np.std(x[valid]) > 1e-12 and np.std(y[valid]) > 1e-12:
                correlations.append(float(np.corrcoef(x[valid], y[valid])[0, 1]))
        ic = float(np.nanmedian(correlations)) if correlations else 0.0
        rows.append((signal, 1 if ic >= 0 else -1, abs(ic)))
    ranked = sorted(rows, key=lambda row: row[2], reverse=True)
    if redundancy_threshold is None:
        return ranked[:top_k]
    # Greedy redundancy removal on oriented, flattened discovery values.
    selected = []
    for item in ranked:
        values = tensor[indices, :, item[0]].reshape(-1)
        redundant = False
        for prior in selected:
            other = tensor[indices, :, prior[0]].reshape(-1)
            valid = np.isfinite(values) & np.isfinite(other)
            if valid.sum() >= 20 and np.std(values[valid]) > 1e-12 and np.std(other[valid]) > 1e-12:
                if abs(float(np.corrcoef(values[valid], other[valid])[0, 1])) >= redundancy_threshold:
                    redundant = True
                    break
        if not redundant:
            selected.append(item)
        if len(selected) == top_k:
            break
    return selected


def _orient_and_select_signals_cuda(
    tensor: np.ndarray, target: np.ndarray, indices: np.ndarray,
    top_k: int, device: str,
) -> list[tuple[int, int, float]]:
    """Vectorized per-session information coefficients and ranking on CUDA."""
    import torch

    used_device = resolve_device(device)
    x = torch.as_tensor(tensor[indices], dtype=torch.float64, device=used_device)
    y = torch.as_tensor(target[indices], dtype=torch.float64, device=used_device)[:, :, None]
    valid = torch.isfinite(x) & torch.isfinite(y)
    n = valid.sum(dim=1)
    clean_x, clean_y = torch.nan_to_num(x), torch.nan_to_num(y)
    mean_x = torch.where(valid, clean_x, 0.0).sum(dim=1) / n.clamp_min(1)
    mean_y = torch.where(valid, clean_y, 0.0).sum(dim=1) / n.clamp_min(1)
    dx = torch.where(valid, clean_x - mean_x[:, None, :], 0.0)
    dy = torch.where(valid, clean_y - mean_y[:, None, :], 0.0)
    covariance = (dx * dy).sum(dim=1)
    denominator = torch.sqrt(dx.square().sum(dim=1) * dy.square().sum(dim=1))
    correlations = torch.where((n >= 4) & (denominator > 1e-12), covariance / denominator, torch.nan)
    ic = torch.nan_to_num(torch.nanmedian(correlations, dim=0).values)
    strength, selected = torch.topk(ic.abs(), min(top_k, ic.numel()))
    directions = torch.where(ic[selected] >= 0, 1, -1)
    return [
        (int(index), int(direction), float(score))
        for index, direction, score in zip(
            selected.cpu().tolist(), directions.cpu().tolist(), strength.cpu().tolist(), strict=True,
        )
    ]


def generate_combinations(
    oriented: list[tuple[int, int, float]], *, max_size: int = 4,
    signal_names: tuple[str, ...] = SIGNAL_NAMES,
) -> list[dict[str, Any]]:
    """Enumerate unrestricted indicator combinations from size one to four.

    RSI plus stochastic, two RSI periods, or any other indicator/value
    combination is valid.
    """
    if not 1 <= max_size <= 4:
        raise ValueError("max_size must be between one and four")
    candidates = []
    for size in range(1, min(max_size, len(oriented)) + 1):
        for items in combinations(oriented, size):
            candidates.append({
                "signal_indices": tuple(item[0] for item in items),
                "directions": tuple(item[1] for item in items),
                "signals": tuple(signal_names[item[0]] for item in items),
            })
    return candidates


def combined_score(tensor: np.ndarray, candidate: dict[str, Any]) -> np.ndarray:
    """Cross-sectionally standardize primitives before equal-weight consensus."""
    selected = tensor[:, :, list(candidate["signal_indices"])].copy()
    direction = np.asarray(candidate["directions"], dtype=float)
    selected *= direction[None, None, :]
    finite = np.isfinite(selected)
    asset_count = finite.sum(axis=1, keepdims=True)
    mean = np.divide(
        np.nansum(selected, axis=1, keepdims=True), asset_count,
        out=np.full((selected.shape[0], 1, selected.shape[2]), np.nan),
        where=asset_count > 0,
    )
    variance = np.divide(
        np.nansum(np.where(finite, (selected - mean) ** 2, 0.0), axis=1, keepdims=True),
        asset_count,
        out=np.full((selected.shape[0], 1, selected.shape[2]), np.nan),
        where=asset_count > 0,
    )
    std = np.sqrt(variance)
    normalized = (selected - mean) / np.where(std > 1e-12, std, np.nan)
    count = np.isfinite(normalized).sum(axis=2)
    score = np.divide(
        np.nansum(normalized, axis=2), count,
        out=np.full(count.shape, np.nan), where=count == selected.shape[2],
    )
    return score


def gpu_screen_combinations(
    tensor: np.ndarray, target: np.ndarray, active: np.ndarray,
    candidates: list[dict[str, Any]], indices: np.ndarray, *,
    top_n: int = 3, cost_per_side_bps: float = 2.5, device: str = "cuda",
    batch_size: int = 1024,
    batch_callback: Callable[[list[dict[str, Any]], int, int], None] | None = None,
) -> pd.DataFrame:
    """Batch-screen unrestricted combinations; canonical replay is still required."""
    import torch

    used_device = resolve_device(device)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    x = torch.as_tensor(tensor[indices], dtype=torch.float64, device=used_device)
    y = torch.as_tensor(target[indices], dtype=torch.float64, device=used_device)
    active_mask = torch.as_tensor(active[indices], dtype=torch.bool, device=used_device)
    finite_x = torch.isfinite(x)
    asset_count = finite_x.sum(dim=1, keepdim=True)
    clean_x = torch.nan_to_num(x)
    mean = clean_x.sum(dim=1, keepdim=True) / asset_count.clamp_min(1)
    variance = torch.where(finite_x, (x - mean) ** 2, 0.0).sum(dim=1, keepdim=True)
    variance = variance / asset_count.clamp_min(1)
    std = torch.sqrt(variance)
    usable_x = finite_x & (std > 1e-12)
    usable_numeric = usable_x.to(torch.float64)
    normalized = torch.where(usable_x, (x - mean) / std, 0.0)
    finite_target = torch.isfinite(y)
    rows = []
    feature_count = tensor.shape[2]
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]
        width = len(batch)
        weights_np = np.zeros((feature_count, width), dtype=np.float64)
        membership_np = np.zeros_like(weights_np)
        sizes_np = np.empty(width, dtype=np.float64)
        for column, candidate in enumerate(batch):
            size = len(candidate["signal_indices"])
            sizes_np[column] = size
            for signal_index, direction in zip(
                candidate["signal_indices"], candidate["directions"], strict=True,
            ):
                weights_np[signal_index, column] = float(direction) / size
                membership_np[signal_index, column] = 1.0
        weights = torch.as_tensor(weights_np, device=used_device)
        membership = torch.as_tensor(membership_np, device=used_device)
        sizes = torch.as_tensor(sizes_np, device=used_device)
        score = torch.einsum("saf,fb->sab", normalized, weights)
        available = torch.einsum("saf,fb->sab", usable_numeric, membership)
        valid = (
            active_mask[:, :, None] & finite_target[:, :, None]
            & (available >= sizes[None, None, :])
        )
        high_ranked = torch.where(valid, score, -torch.inf).permute(0, 2, 1)
        low_ranked = torch.where(valid, -score, -torch.inf).permute(0, 2, 1)
        high = torch.topk(high_ranked, top_n, dim=2).indices
        low = torch.topk(low_ranked, top_n, dim=2).indices
        returns = y[:, None, :].expand(-1, width, -1)
        long_return = torch.gather(returns, 2, high).mean(2)
        short_return = torch.gather(returns, 2, low).mean(2)
        enough = valid.sum(dim=1) >= 2 * top_n
        daily = torch.where(
            enough,
            (long_return - short_return) / 2 - 2 * cost_per_side_bps / 10_000,
            torch.zeros_like(long_return),
        )
        means = daily.mean(dim=0)
        stds = daily.std(dim=0, unbiased=True) if daily.shape[0] > 1 else torch.zeros_like(means)
        sharpes = torch.where(stds > 0, means / stds * np.sqrt(252), 0.0)
        totals = torch.prod(1 + daily, dim=0) - 1
        events = torch.count_nonzero(daily, dim=0)
        sharpes_np, totals_np, events_np = (
            value.detach().cpu().numpy() for value in (sharpes, totals, events)
        )
        batch_rows = [{
            "candidate_index": start + offset,
            "signals": "+".join(candidate["signals"]),
            "signal_count": len(candidate["signals"]),
            "screen_sharpe": float(sharpes_np[offset]),
            "screen_total_return": float(totals_np[offset]),
            "screen_events": int(events_np[offset]),
            "simulation_device": f"gpu-{used_device}",
        } for offset, candidate in enumerate(batch)]
        rows.extend(batch_rows)
        if batch_callback is not None:
            batch_callback(batch_rows, start + width, len(candidates))
    return pd.DataFrame(rows).sort_values(
        ["screen_sharpe", "screen_total_return"], ascending=False,
    ).reset_index(drop=True)
