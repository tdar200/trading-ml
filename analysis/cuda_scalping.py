"""CUDA/Triton execution kernel for exact intraday candidate validation."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from analysis.metrics import cagr, max_drawdown, sharpe


def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def simulate(
        open_ptr, high_ptr, low_ptr, close_ptr, session_ptr, minute_ptr, last_ptr, signal_ptr,
        risk_ptr, max_hold_ptr, take_ptr, stop_ptr, trade_cap_ptr,
        daily_ptr, trades_ptr, wins_ptr, gross_win_ptr, gross_loss_ptr,
        cost_ptr, n_bars, n_candidates: tl.constexpr, use_atr: tl.constexpr,
    ):
        candidate = tl.program_id(0)
        zero = tl.load(open_ptr) * 0.0
        equity = zero + 1.0
        position_side = 0
        entry_price = zero + 1.0
        held = 0
        trades = 0
        wins = 0
        gross_win = zero
        gross_loss = zero
        daily_index = 0
        entries_this_session = 0
        max_hold = tl.load(max_hold_ptr + candidate)
        take_value = tl.load(take_ptr + candidate)
        stop_value = tl.load(stop_ptr + candidate)
        trade_cap = tl.load(trade_cap_ptr + candidate)
        entry_take = take_value
        entry_stop = stop_value
        cost = tl.load(cost_ptr)

        for i in tl.range(1, n_bars):
            open_price = tl.load(open_ptr + i)
            previous_close = tl.load(close_ptr + i - 1)
            same_session = (
                tl.load(session_ptr + i) == tl.load(session_ptr + i - 1)
            )
            entries_this_session = tl.where(
                same_session, entries_this_session, 0,
            )
            position = position_side != 0
            mark_return = tl.where(
                position_side > 0,
                previous_close / entry_price - 1.0,
                entry_price / previous_close - 1.0,
            )
            exit_now = position & (
                (mark_return >= entry_take)
                | (mark_return <= -entry_stop)
                | (held >= max_hold)
                | (~same_session)
            )
            trade_return = tl.where(
                position_side > 0,
                open_price * (1.0 - cost) / entry_price - 1.0,
                entry_price / (open_price * (1.0 + cost)) - 1.0,
            )
            equity = tl.where(exit_now, equity * (1.0 + trade_return), equity)
            trades += tl.where(exit_now, 1, 0)
            wins += tl.where(exit_now & (trade_return > 0.0), 1, 0)
            gross_win += tl.where(exit_now, tl.maximum(trade_return, 0.0), 0.0)
            gross_loss += tl.where(exit_now, tl.maximum(-trade_return, 0.0), 0.0)
            position_side = tl.where(exit_now, 0, position_side)
            held = tl.where(exit_now, 0, held)

            minute = tl.load(minute_ptr + i)
            is_last = tl.load(last_ptr + i) != 0
            entry_signal = tl.load(
                signal_ptr + (i - 1) * n_candidates + candidate
            )
            enter = (
                (position_side == 0)
                & (entry_signal != 0)
                & same_session
                & (minute >= 9 * 60 + 45)
                & (minute <= 15 * 60 + 30)
                & (~is_last)
                & (entries_this_session < trade_cap)
            )
            fill_price = tl.where(
                entry_signal > 0,
                open_price * (1.0 + cost),
                open_price * (1.0 - cost),
            )
            entry_price = tl.where(enter, fill_price, entry_price)
            entry_risk = tl.load(risk_ptr + i - 1)
            candidate_take = (
                entry_risk * take_value if use_atr else take_value
            )
            candidate_stop = (
                entry_risk * stop_value if use_atr else stop_value
            )
            entry_take = tl.where(enter, candidate_take, entry_take)
            entry_stop = tl.where(enter, candidate_stop, entry_stop)
            position_side = tl.where(enter, entry_signal, position_side)
            entries_this_session += tl.where(enter, 1, 0)
            held = tl.where(position_side != 0, held + 1, 0)

            high_price = tl.load(high_ptr + i)
            low_price = tl.load(low_ptr + i)
            long_target_price = entry_price * (1.0 + entry_take) / (1.0 - cost)
            long_stop_price = entry_price * (1.0 - entry_stop) / (1.0 - cost)
            short_target_price = entry_price / ((1.0 + entry_take) * (1.0 + cost))
            short_stop_price = entry_price / ((1.0 - entry_stop) * (1.0 + cost))
            target_hit = ((position_side > 0) & (high_price >= long_target_price)) | ((position_side < 0) & (low_price <= short_target_price))
            stop_hit = ((position_side > 0) & (low_price <= long_stop_price)) | ((position_side < 0) & (high_price >= short_stop_price))
            intrabar_exit = (position_side != 0) & (target_hit | stop_hit)
            intrabar_price = tl.where(
                position_side > 0,
                tl.where(stop_hit, long_stop_price, long_target_price),
                tl.where(stop_hit, short_stop_price, short_target_price),
            )
            intrabar_return = tl.where(
                position_side > 0,
                intrabar_price * (1.0 - cost) / entry_price - 1.0,
                entry_price / (intrabar_price * (1.0 + cost)) - 1.0,
            )
            equity = tl.where(intrabar_exit, equity * (1.0 + intrabar_return), equity)
            trades += tl.where(intrabar_exit, 1, 0)
            wins += tl.where(intrabar_exit & (intrabar_return > 0.0), 1, 0)
            gross_win += tl.where(intrabar_exit, tl.maximum(intrabar_return, 0.0), 0.0)
            gross_loss += tl.where(intrabar_exit, tl.maximum(-intrabar_return, 0.0), 0.0)
            position_side = tl.where(intrabar_exit, 0, position_side)
            held = tl.where(intrabar_exit, 0, held)

            close_price = tl.load(close_ptr + i)
            eod_return = tl.where(
                position_side > 0,
                close_price * (1.0 - cost) / entry_price - 1.0,
                entry_price / (close_price * (1.0 + cost)) - 1.0,
            )
            close_eod = (position_side != 0) & is_last
            equity = tl.where(close_eod, equity * (1.0 + eod_return), equity)
            trades += tl.where(close_eod, 1, 0)
            wins += tl.where(close_eod & (eod_return > 0.0), 1, 0)
            gross_win += tl.where(close_eod, tl.maximum(eod_return, 0.0), 0.0)
            gross_loss += tl.where(close_eod, tl.maximum(-eod_return, 0.0), 0.0)
            position_side = tl.where(close_eod, 0, position_side)
            held = tl.where(close_eod, 0, held)
            tl.store(
                daily_ptr + daily_index * n_candidates + candidate,
                equity,
                mask=is_last,
            )
            daily_index += tl.where(is_last, 1, 0)

        tl.store(trades_ptr + candidate, trades)
        tl.store(wins_ptr + candidate, wins)
        tl.store(gross_win_ptr + candidate, gross_win)
        tl.store(gross_loss_ptr + candidate, gross_loss)

    return simulate


_SIMULATION_KERNEL = None


def _signal_matrix(close, sessions, candidates: list[dict[str, Any]]):
    """Create completed-bar entry signals directly on the CUDA device."""
    import torch

    cache = {}
    for candidate in candidates:
        key = (
            str(candidate["family"]),
            int(candidate["lookback_bars"]),
            float(candidate["entry_threshold"]),
        )
        if key in cache:
            continue
        family, lookback, threshold = key
        signal = torch.zeros(len(close), dtype=torch.bool, device=close.device)
        if len(close) > lookback:
            history = close.unfold(0, lookback, 1)[:-1]
            current = close[lookback:]
            same_session = sessions[lookback:] == sessions[:-lookback]
            if family == "mean_reversion":
                std = history.std(dim=1, correction=0)
                condition = (std > 1e-12) & (
                    (current - history.mean(dim=1)) / std <= -threshold
                )
            elif family == "momentum":
                condition = (
                    current / close[:-lookback] - 1.0
                    >= threshold / 10_000.0
                )
            else:
                condition = (
                    current / history.max(dim=1).values - 1.0
                    >= threshold / 10_000.0
                )
            signal[lookback:] = condition & same_session
        cache[key] = signal
    return torch.stack([
        cache[(
            str(candidate["family"]),
            int(candidate["lookback_bars"]),
            float(candidate["entry_threshold"]),
        )]
        for candidate in candidates
    ], dim=1).contiguous()


def simulate_scalping_candidates_cuda(
    bars: pd.DataFrame,
    candidates: list[dict[str, Any]],
    *,
    cost_per_side_bps: float,
    benchmark_bars: pd.DataFrame | None,
    signal_matrix=None,
    risk_fraction=None,
    volatility_exits: bool = False,
    max_trades_per_session: int | list[int] | None = None,
    return_curves: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
    """Run exact candidate state machines on CUDA and aggregate on the host."""
    import torch

    global _SIMULATION_KERNEL
    if _SIMULATION_KERNEL is None:
        _SIMULATION_KERNEL = _kernel()

    open_px = torch.as_tensor(
        bars["open"].to_numpy(dtype=np.float64), device="cuda",
    )
    high_px = torch.as_tensor(
        bars["high"].to_numpy(dtype=np.float64), device="cuda",
    )
    low_px = torch.as_tensor(
        bars["low"].to_numpy(dtype=np.float64), device="cuda",
    )
    close_px = torch.as_tensor(
        bars["close"].to_numpy(dtype=np.float64), device="cuda",
    )
    sessions = torch.as_tensor(
        bars["session"].to_numpy(dtype=np.int64), device="cuda",
    )
    minutes = torch.as_tensor(
        bars["minute_of_day"].to_numpy(dtype=np.int32), device="cuda",
    )
    last_bar = torch.as_tensor(
        bars["last_bar"].to_numpy(dtype=np.uint8), device="cuda",
    )
    signals = (
        _signal_matrix(close_px, sessions, candidates)
        if signal_matrix is None
        else signal_matrix
    ).to(device="cuda", dtype=torch.int8).contiguous()
    count = len(candidates)
    if signals.shape != (len(bars), count):
        raise ValueError("signal matrix shape must be (bars, candidates)")
    risk = (
        torch.zeros(len(bars), dtype=torch.float64, device="cuda")
        if risk_fraction is None
        else torch.as_tensor(risk_fraction, dtype=torch.float64, device="cuda")
    )
    if risk.shape != (len(bars),):
        raise ValueError("risk fraction shape must be (bars,)")
    max_hold = torch.tensor(
        [int(candidate["max_hold_bars"]) for candidate in candidates],
        dtype=torch.int32, device="cuda",
    )
    if volatility_exits:
        take_values = [float(candidate["take_profit_atr"]) for candidate in candidates]
        stop_values = [float(candidate["stop_loss_atr"]) for candidate in candidates]
    else:
        take_values = [
            float(candidate["take_profit_bps"]) / 10_000.0
            for candidate in candidates
        ]
        stop_values = [
            float(candidate["stop_loss_bps"]) / 10_000.0
            for candidate in candidates
        ]
    take = torch.tensor(take_values, dtype=torch.float64, device="cuda")
    stop = torch.tensor(stop_values, dtype=torch.float64, device="cuda")
    caps = (
        [2_147_483_647] * count
        if max_trades_per_session is None
        else (
            [int(max_trades_per_session)] * count
            if isinstance(max_trades_per_session, int)
            else [int(value) for value in max_trades_per_session]
        )
    )
    if len(caps) != count or any(value <= 0 for value in caps):
        raise ValueError("trade caps must be positive and match candidates")
    trade_cap = torch.tensor(caps, dtype=torch.int32, device="cuda")
    session_count = int(last_bar[1:].sum().item())
    if not session_count:
        raise ValueError("simulation produced no complete sessions")
    curves = torch.empty(
        (session_count, count), dtype=torch.float64, device="cuda",
    )
    trades = torch.empty(count, dtype=torch.int32, device="cuda")
    wins = torch.empty(count, dtype=torch.int32, device="cuda")
    gross_win = torch.empty(count, dtype=torch.float64, device="cuda")
    gross_loss = torch.empty(count, dtype=torch.float64, device="cuda")
    cost = torch.tensor(
        [cost_per_side_bps / 10_000.0], dtype=torch.float64, device="cuda",
    )
    _SIMULATION_KERNEL[(count,)](
        open_px, high_px, low_px, close_px, sessions, minutes, last_bar, signals, risk,
        max_hold, take, stop, trade_cap, curves, trades, wins, gross_win,
        gross_loss, cost, len(bars), n_candidates=count,
        use_atr=volatility_exits,
    )
    torch.cuda.synchronize()
    curves_np = curves.cpu().numpy()
    daily_returns = curves_np[1:] / curves_np[:-1] - 1.0
    trades_np = trades.cpu().numpy()
    wins_np = wins.cpu().numpy()
    gross_win_np = gross_win.cpu().numpy()
    gross_loss_np = gross_loss.cpu().numpy()

    benchmark_source = bars if benchmark_bars is None else benchmark_bars
    traded_dates = set(bars["session_date"])
    benchmark_close = benchmark_source.loc[
        benchmark_source["last_bar"]
        & benchmark_source["session_date"].isin(traded_dates),
        "close",
    ].to_numpy(dtype=float)
    if len(benchmark_close) < 2:
        raise ValueError("benchmark has fewer than two aligned sessions")
    benchmark_curve = benchmark_close / benchmark_close[0]
    benchmark_returns = pd.Series(benchmark_curve).pct_change().dropna()
    benchmark_metrics = {
        "benchmark_total_return": float(benchmark_curve[-1] - 1.0),
        "benchmark_sharpe": float(sharpe(benchmark_returns)),
        "benchmark_max_drawdown": float(max_drawdown(pd.Series(benchmark_curve))),
    }

    rows = []
    for idx, candidate in enumerate(candidates):
        returns = pd.Series(daily_returns[:, idx])
        curve = pd.Series(curves_np[:, idx])
        total_return = float(curve.iloc[-1] - 1.0)
        rows.append({
            **candidate,
            "sharpe": float(sharpe(returns)),
            "cagr": float(cagr(returns)),
            "max_drawdown": float(max_drawdown(curve)),
            "total_return": total_return,
            "n_trades": int(trades_np[idx]),
            "win_rate": (
                float(wins_np[idx] / trades_np[idx])
                if trades_np[idx] else float("nan")
            ),
            "profit_factor": (
                float(gross_win_np[idx] / gross_loss_np[idx])
                if gross_loss_np[idx] > 0 else float("nan")
            ),
            "excess_total_return": (
                total_return - benchmark_metrics["benchmark_total_return"]
            ),
            "cost_per_side_bps": cost_per_side_bps,
            "simulation_device": "cuda",
            **benchmark_metrics,
        })
    result = pd.DataFrame(rows)
    return (result, curves_np) if return_curves else result
