"""Canonical rule-based cross-sectional rotation strategy.

The structure is data, not generated Python: signal_family chooses a
registered scoring rule and the remaining fields enable explicit filters.
This keeps discovered strategies reproducible and safe to deserialize.
"""
from __future__ import annotations

from statistics import fmean

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.model.objects import Quantity

from data.instruments import VENUE_MIC
from strategies.momentum_topn import MomentumTopNStrategy


SUPPORTED_SIGNAL_FAMILIES = ("momentum", "mean_reversion")


def score_rule(
    closes: list[float],
    *,
    signal_family: str,
    lookback_days: int,
    skip_days: int,
) -> float | None:
    """Return one registered signal score, or None before warm-up."""
    if signal_family not in SUPPORTED_SIGNAL_FAMILIES:
        raise ValueError(f"unsupported signal family: {signal_family!r}")
    if lookback_days < 1 or skip_days < 0:
        raise ValueError("lookback_days must be positive and skip_days non-negative")
    end = len(closes) - 1 - skip_days
    start = end - lookback_days
    if start < 0 or closes[start] <= 0:
        return None
    raw_return = closes[end] / closes[start] - 1.0
    return raw_return if signal_family == "momentum" else -raw_return


class RuleRotationConfig(StrategyConfig, frozen=True):
    """Serializable structure and parameters for a discovered strategy."""

    universe: tuple[str, ...]
    benchmark: str = "SPY"
    signal_family: str = "momentum"
    trend_filter_days: int = 0
    require_positive: bool = False
    top_n: int = 3
    lookback_days: int = 126
    skip_days: int = 21


class RuleRotationStrategy(MomentumTopNStrategy):
    """Monthly long-only rotation driven by a registered rule structure."""

    def __init__(self, config: RuleRotationConfig) -> None:
        if config.signal_family not in SUPPORTED_SIGNAL_FAMILIES:
            raise ValueError(f"unsupported signal family: {config.signal_family!r}")
        if config.trend_filter_days < 0:
            raise ValueError("trend_filter_days must be non-negative")
        super().__init__(config)

    def _history_cap(self) -> int:
        return max(
            self.config.lookback_days + self.config.skip_days + 5,
            self.config.trend_filter_days + self.config.skip_days + 5,
        )

    def _is_warmed_up(self) -> bool:
        needed = max(
            self.config.lookback_days + self.config.skip_days + 1,
            self.config.trend_filter_days + self.config.skip_days,
        )
        return all(len(self._closes.get(t, [])) >= needed for t in self.config.universe)

    def _ranked_targets(self) -> list[str]:
        scored: list[tuple[str, float]] = []
        for ticker in self.config.universe:
            closes = self._closes.get(ticker, [])
            score = score_rule(
                closes,
                signal_family=self.config.signal_family,
                lookback_days=self.config.lookback_days,
                skip_days=self.config.skip_days,
            )
            if score is None:
                continue
            end = len(closes) - 1 - self.config.skip_days
            if self.config.trend_filter_days:
                start = end - self.config.trend_filter_days + 1
                if start < 0 or closes[end] <= fmean(closes[start : end + 1]):
                    continue
            if self.config.require_positive and score <= 0:
                continue
            scored.append((ticker, score))
        scored.sort(key=lambda item: (-item[1], self.config.universe.index(item[0])))
        return [ticker for ticker, _ in scored[: self.config.top_n]]

    def _rebalance(self, now_ts_ns: int) -> None:
        self._rebalance_count += 1
        target_tickers = set(self._ranked_targets())

        venue = Venue(VENUE_MIC)
        account = self.portfolio.account(venue)
        equity_money = self.portfolio.equity(venue)[account.base_currency]
        equity_usd = float(equity_money.as_decimal())
        per_name_usd = (equity_usd * 0.98) / max(len(target_tickers), 1)

        sells: list[tuple[InstrumentId, int]] = []
        buys: list[tuple[InstrumentId, int]] = []
        for ticker in self.config.universe:
            instrument_id = InstrumentId.from_str(f"{ticker}.{VENUE_MIC}")
            current = self._current_position_qty(instrument_id)
            if ticker in target_tickers:
                last_close = self._closes[ticker][-1]
                target = int(per_name_usd / last_close) if last_close > 0 else 0
            else:
                target = 0
            delta = target - current
            if delta > 0:
                buys.append((instrument_id, delta))
            elif delta < 0:
                sells.append((instrument_id, -delta))

        for instrument_id, quantity in sells:
            self.submit_order(self.order_factory.market(
                instrument_id=instrument_id,
                order_side=OrderSide.SELL,
                quantity=Quantity.from_int(quantity),
            ))
        for instrument_id, quantity in buys:
            self.submit_order(self.order_factory.market(
                instrument_id=instrument_id,
                order_side=OrderSide.BUY,
                quantity=Quantity.from_int(quantity),
            ))
