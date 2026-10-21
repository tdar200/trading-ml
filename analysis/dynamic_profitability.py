"""Dynamic-universe construction for causal intraday profitability research.

Unlike a global inner join, one missing bar removes only that stock for that
session. This preserves history while making availability explicit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.profitability import DecisionDataset
from analysis.scalping import prepare_intraday_bars
from data.event_calendar import event_clear_mask
from data.quotes import quote_features_at
from data.universe_membership import members_on
from data.tickers import SCALPING_STOCK_SECTORS


def _sector_demean(values: np.ndarray, tickers: tuple[str, ...]) -> np.ndarray:
    result = values.copy()
    groups = {SCALPING_STOCK_SECTORS.get(ticker, "other") for ticker in tickers}
    for group in groups:
        columns = np.asarray([
            SCALPING_STOCK_SECTORS.get(ticker, "other") == group for ticker in tickers
        ])
        subset = result[:, columns]
        counts = np.isfinite(subset).sum(axis=1)
        means = np.divide(
            np.nansum(subset, axis=1), counts,
            out=np.zeros(len(result)), where=counts > 0,
        )
        result[:, columns] = subset - means[:, None]
    return result


def _features(
    frame: pd.DataFrame, benchmark: pd.DataFrame, *, bar_minutes: int = 5,
) -> pd.DataFrame:
    bars = prepare_intraday_bars(frame).copy()
    eastern = bars["ts"].dt.tz_convert("America/New_York")
    bars["bar"] = ((eastern.dt.hour * 60 + eastern.dt.minute - 570) // bar_minutes).astype(int)
    indexed = bars.set_index("ts")
    benchmark_close = benchmark.set_index("ts")["close"].reindex(indexed.index)
    same_session = indexed["session_date"].eq(indexed["session_date"].shift(1))
    stock_return = indexed["close"].pct_change(fill_method=None).where(same_session)
    market_return = benchmark_close.pct_change(fill_method=None).where(same_session)
    bars_per_session = int(np.ceil(390 / bar_minutes))
    beta_window = 5 * bars_per_session
    beta_minimum = max(10, 2 * bars_per_session)
    variance = market_return.rolling(beta_window, min_periods=beta_minimum).var(ddof=0)
    covariance = stock_return.rolling(beta_window, min_periods=beta_minimum).cov(market_return, ddof=0)
    indexed["beta"] = (covariance / variance.where(variance > 1e-14)).clip(0.2, 3.0).fillna(1.0)
    indexed["relative_volume"] = indexed["volume"] / indexed.groupby("bar")["volume"].transform(
        lambda values: values.shift(1).rolling(20, min_periods=5).median()
    ).where(lambda values: values > 0)
    return indexed.reset_index()


def _session_rows(frame: pd.DataFrame) -> dict[object, pd.DataFrame]:
    return {date: rows.set_index("bar").sort_index() for date, rows in frame.groupby("session_date")}


def prepare_dynamic_decision_dataset(
    bars_by_ticker: dict[str, pd.DataFrame], benchmark_bars: pd.DataFrame,
    decision_bar: int, *, min_assets: int = 15,
    bar_minutes: int = 5, signal_bar: int | None = None,
    exit_after_bars: int | None = None,
    quotes_by_ticker: dict[str, pd.DataFrame] | None = None,
    membership: pd.DataFrame | None = None,
    event_calendar: pd.DataFrame | None = None,
) -> DecisionDataset:
    """Build session snapshots without requiring every stock bar to intersect."""
    if len(bars_by_ticker) < 4 or not 4 <= min_assets <= len(bars_by_ticker):
        raise ValueError("min_assets must be between four and the universe size")
    if bar_minutes <= 0 or 390 % bar_minutes != 0 and bar_minutes != 60:
        raise ValueError("bar_minutes must describe a supported regular-session bar")
    if exit_after_bars is not None and exit_after_bars < 1:
        raise ValueError("exit_after_bars must be positive")
    signal_bar = max(0, int(np.ceil(30 / bar_minutes)) - 1) if signal_bar is None else signal_bar
    final_bar = int(np.ceil(390 / bar_minutes)) - 1
    tickers = tuple(bars_by_ticker)
    benchmark = _features(
        benchmark_bars, prepare_intraday_bars(benchmark_bars), bar_minutes=bar_minutes,
    )
    benchmark_sessions = _session_rows(benchmark)
    asset_sessions = {
        ticker: _session_rows(_features(frame, benchmark, bar_minutes=bar_minutes))
        for ticker, frame in bars_by_ticker.items()
    }
    dates = sorted(benchmark_sessions)
    prior_stock_sessions = {}
    market_history = []
    rows = []
    decision_timestamps = []
    for date in dates:
        market = benchmark_sessions[date]
        required = {0, signal_bar, decision_bar, decision_bar + 1}
        if not required.issubset(market.index) or market.index.max() <= decision_bar + 1:
            continue
        market_exit = int(market.index[market.index <= final_bar].max())
        market_first = float(market.loc[signal_bar, "close"] / market.loc[0, "open"] - 1)
        previous_market_close = market_history[-1][1] if market_history else np.nan
        market_gap = float(market.loc[0, "open"] / previous_market_close - 1) if market_history else np.nan
        trailing_market = (
            float(previous_market_close / market_history[-5][1] - 1)
            if len(market_history) >= 5 else 0.0
        )
        matrices = {name: np.full(len(tickers), np.nan) for name in (
            "first", "gap", "rvol", "realized", "pressure", "beta", "future",
            "vwap", "prior_position", "opening_break", "rejection",
        )}
        for column, ticker in enumerate(tickers):
            stock = asset_sessions[ticker].get(date)
            if stock is None or not required.issubset(stock.index) or stock.index.max() <= decision_bar + 1:
                continue
            session_exit = int(stock.index[stock.index <= final_bar].max())
            exit_bar = (
                session_exit if exit_after_bars is None
                else decision_bar + int(exit_after_bars)
            )
            if exit_bar not in stock.index or exit_bar < decision_bar + 1:
                continue
            beta = float(stock.loc[decision_bar, "beta"])
            prior = prior_stock_sessions.get(ticker)
            matrices["first"][column] = float(stock.loc[signal_bar, "close"] / stock.loc[0, "open"] - 1 - beta * market_first)
            if (
                prior is not None and market_history and prior[0] == market_history[-1][0]
                and np.isfinite(market_gap)
            ):
                matrices["gap"][column] = float(
                    stock.loc[0, "open"] / prior[1] - 1 - beta * market_gap
                )
            known = stock.loc[stock.index <= decision_bar]
            history = known["close"].to_numpy(dtype=float)
            log_returns = np.diff(np.log(history))
            matrices["rvol"][column] = float(stock.loc[decision_bar, "relative_volume"])
            matrices["realized"][column] = float(np.sqrt(np.mean(log_returns * log_returns))) if len(log_returns) else np.nan
            matrices["pressure"][column] = float(stock.loc[decision_bar, "close"] / stock.loc[decision_bar, "open"] - 1)
            matrices["beta"][column] = beta
            matrices["future"][column] = float(stock.loc[exit_bar, "close"] / stock.loc[decision_bar + 1, "open"] - 1)
            typical = (known["high"] + known["low"] + known["close"]) / 3
            volume = known["volume"].clip(lower=0)
            vwap = float((typical * volume).sum() / volume.sum()) if volume.sum() > 0 else np.nan
            decision_close = float(stock.loc[decision_bar, "close"])
            matrices["vwap"][column] = decision_close / vwap - 1 if vwap > 0 else np.nan
            opening = stock.loc[stock.index <= signal_bar]
            opening_high = float(opening["high"].max())
            opening_low = float(opening["low"].min())
            matrices["opening_break"][column] = (
                1.0 if decision_close > opening_high
                else -1.0 if decision_close < opening_low else 0.0
            )
            if prior is not None and market_history and prior[0] == market_history[-1][0]:
                prior_high, prior_low = prior[2], prior[3]
                prior_range = max(prior_high - prior_low, 1e-12)
                matrices["prior_position"][column] = np.clip(
                    (decision_close - (prior_high + prior_low) / 2) / prior_range, -2, 2,
                )
                touched_support = float(known["low"].min()) <= prior_low
                touched_resistance = float(known["high"].max()) >= prior_high
                matrices["rejection"][column] = (
                    1.0 if touched_support and decision_close > prior_low + 0.1 * prior_range
                    else -1.0 if touched_resistance and decision_close < prior_high - 0.1 * prior_range
                    else 0.0
                )
            prior_stock_sessions[ticker] = (
                date, float(stock.loc[exit_bar, "close"]),
                float(stock["high"].max()), float(stock["low"].min()),
            )
        available = np.isfinite(matrices["future"]) & np.isfinite(matrices["first"])
        if available.sum() < min_assets:
            market_history.append((date, float(market.loc[market_exit, "close"])))
            continue
        finite_rvol = matrices["rvol"][np.isfinite(matrices["rvol"])]
        rows.append((
            date, matrices,
            bool(len(finite_rvol) and np.median(finite_rvol) >= 1.0 and np.nanstd(matrices["first"]) >= 0.003),
            bool(trailing_market < -0.02 and market_first > 0.002), available,
        ))
        decision_timestamps.append(pd.Timestamp(market.loc[decision_bar, "ts"]) + pd.Timedelta(minutes=5))
        market_history.append((date, float(market.loc[market_exit, "close"])))
    if len(rows) < 60:
        raise ValueError("fewer than 60 dynamic-universe sessions")
    session_dates = np.asarray([row[0] for row in rows], dtype=object)
    target = np.stack([row[1]["future"] for row in rows])
    seasonality = {
        days: pd.DataFrame(target).shift(1).rolling(days, min_periods=max(10, days // 2)).mean().to_numpy()
        for days in (20, 40)
    }
    active = np.stack([row[4] for row in rows])
    if membership is not None:
        for index, date in enumerate(session_dates):
            members = set(members_on(membership, pd.Timestamp(date, tz="UTC")))
            active[index] &= np.asarray([ticker in members for ticker in tickers])
    clear = (
        np.ones(target.shape, dtype=bool) if event_calendar is None
        else event_clear_mask(event_calendar, session_dates, tickers)
    )
    quote_imbalance = spread = None
    if quotes_by_ticker and all(ticker in quotes_by_ticker for ticker in tickers):
        times = pd.DatetimeIndex(decision_timestamps)
        quote_items = [quote_features_at(quotes_by_ticker[ticker], times) for ticker in tickers]
        candidate_spread = np.column_stack([item["spread_bps"] for item in quote_items])
        covered = np.isfinite(candidate_spread) & active
        if covered.sum() / max(1, active.sum()) >= 0.80:
            spread = candidate_spread
            quote_imbalance = np.column_stack([item["quote_imbalance"] for item in quote_items])
    first_residual = np.stack([row[1]["first"] for row in rows])
    return DecisionDataset(
        decision_bar=decision_bar, session_dates=session_dates, tickers=tickers,
        first_residual=first_residual,
        gap_residual=np.stack([row[1]["gap"] for row in rows]),
        relative_volume=np.stack([row[1]["rvol"] for row in rows]),
        realized_volatility=np.stack([row[1]["realized"] for row in rows]),
        bar_pressure=np.stack([row[1]["pressure"] for row in rows]),
        beta=np.stack([row[1]["beta"] for row in rows]), forward_return=target,
        seasonality=seasonality, high_information=np.asarray([row[2] for row in rows]),
        panic_rebound=np.asarray([row[3] for row in rows]), active_mask=active,
        event_clear=clear, quote_imbalance=quote_imbalance, spread_bps=spread,
        sector_residual=_sector_demean(first_residual, tickers),
        vwap_deviation=np.stack([row[1]["vwap"] for row in rows]),
        prior_range_position=np.stack([row[1]["prior_position"] for row in rows]),
        opening_range_break=np.stack([row[1]["opening_break"] for row in rows]),
        support_rejection=np.stack([row[1]["rejection"] for row in rows]),
        bar_minutes=bar_minutes,
    )
