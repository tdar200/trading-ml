"""
MomentumMetaStrategy — Stage-5 ML overlay on top of the Stage-2 momentum strategy.

For each (ticker, rebalance-date) pair the primary signal would consider, a
LightGBM meta-model predicts P(win) (will THIS position be profitable over
the next holding period?). The result is used as a filter + sizer:

    if p_win < filter_threshold:
        target_weight = 0.0
    else:
        confidence = (p_win - filter_threshold) / (1.0 - filter_threshold)
        target_weight = (equity / top_n) * confidence

The strategy loads the most-recent ModelStore artifact whose training
cutoff is STRICTLY before the rebalance date. During warm-up (no model
yet available), the strategy falls back to pure baseline behaviour
(equivalent to Stage-2).

The feature extraction at rebalance time uses ``models.features.extract_features_for_date``,
which is the SAME function the offline trainer calls — one source of truth.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.model.objects import Quantity

from data.instruments import VENUE_MIC
from models.features import extract_features_for_date
from models.store import ModelStore
from strategies.momentum_topn import MomentumTopNConfig, MomentumTopNStrategy
from strategies.ranking import rank_by_momentum


# The feature extractor needs ~126 business days of history for mom_6m and
# ~60 days for vol_60d. The parent's ring buffer cap is sized for the
# momentum ranker only and may be too small for our features. Keep enough
# bars for the longest feature lookback plus a small margin.
_MIN_FEATURE_HISTORY = 130


class MomentumMetaConfig(MomentumTopNConfig, frozen=True):
    """Config = Stage-2 MomentumTopNConfig fields + Stage-5 model parameters."""
    model_store_path: str = ""          # filesystem path to ModelStore cache_dir
    model_store_label: str = "default"  # ModelStore label (one per hyperparam combo)
    filter_threshold: float = 0.55      # P(win) below this → skip position


class MomentumMetaStrategy(MomentumTopNStrategy):
    """Stage-5 meta-labeled wrapper around Stage-2 momentum top-N."""

    def __init__(self, config: MomentumMetaConfig) -> None:
        super().__init__(config)
        # Resolve the ModelStore. Stays None if no path configured or
        # the path doesn't exist on disk.
        self._model_store: ModelStore | None = None
        if config.model_store_path:
            cache_dir = Path(config.model_store_path)
            if cache_dir.exists():
                self._model_store = ModelStore(
                    cache_dir=cache_dir,
                    label=config.model_store_label,
                )
        # Latest bar timestamp seen — used to align the synthetic index in
        # _bars_to_series_dict so rebalance_date is locatable in the index.
        self._last_bar_ts_ns: int = 0

    # ---- on_bar override: maintain a feature-sized history buffer --------

    def on_bar(self, bar: Bar) -> None:
        """Replicate parent's on_bar but with a buffer cap sized for our
        feature extractor (up to 126 business days for mom_6m)."""
        from nautilus_trader.model.identifiers import Venue
        ticker = self._ticker_for(bar)
        history = self._closes.setdefault(ticker, [])
        history.append(float(bar.close))
        # Cap = max(parent cap, feature minimum). Keeps the larger of the
        # two so deployments configured with a long lookback aren't shrunk.
        parent_cap = self.config.lookback_days + self.config.skip_days + 5
        cap = max(parent_cap, _MIN_FEATURE_HISTORY)
        if len(history) > cap:
            del history[0 : len(history) - cap]

        # Track latest seen bar timestamp (used to align the synthetic index
        # built in _bars_to_series_dict with the rebalance_date passed to
        # the feature extractor).
        self._last_bar_ts_ns = max(self._last_bar_ts_ns, int(bar.ts_event))

        # Daily equity snapshot — only on benchmark bars (mirrors parent).
        if ticker == self.config.benchmark:
            year, month, day = self._year_month_day(bar.ts_event)
            if self._last_equity_snapshot_date != (year, month, day):
                self._last_equity_snapshot_date = (year, month, day)
                try:
                    venue = Venue(VENUE_MIC)
                    account = self.portfolio.account(venue)
                    if account is not None:
                        eq_money = self.portfolio.equity(venue)[account.base_currency]
                        self._equity_history.append(
                            (bar.ts_event, float(eq_money.as_decimal()))
                        )
                except (KeyError, AttributeError, TypeError):
                    pass

        # Rebalance check (mirrors parent).
        if ticker not in self.config.universe:
            return
        if not self._is_warmed_up():
            return
        year, month = self._year_month(bar.ts_event)
        if self._last_rebalance_month == (year, month):
            return
        self._last_rebalance_month = (year, month)
        self._rebalance(bar.ts_event)

    # ---- _rebalance override ----------------------------------------------

    def _rebalance(self, now_ts_ns: int) -> None:
        """Stage-5 rebalance: same as Stage-2 except positions go through the
        meta-model filter+sizer before order generation."""
        self._rebalance_count += 1

        # 1. Primary ranking (identical to Stage 2).
        ranked = rank_by_momentum(
            {t: self._closes.get(t, []) for t in self.config.universe},
            lookback_days=self.config.lookback_days,
            skip_days=self.config.skip_days,
        )
        if not ranked:
            return
        target_tickers = ranked[: self.config.top_n]

        # 2. Equity → base per-name USD.
        venue = Venue(VENUE_MIC)
        account = self.portfolio.account(venue)
        equity_usd = float(self.portfolio.equity(venue)[account.base_currency].as_decimal())
        # 2% cash buffer (Stage 2): leave headroom for slippage + integer rounding.
        per_name_usd = (equity_usd * 0.98) / max(self.config.top_n, 1)

        # 3. Compute per-ticker target USD via filter+sizer.
        target_usd_by_ticker: dict[str, float] = {}
        rebalance_date = pd.Timestamp(now_ts_ns, unit="ns", tz="UTC").tz_convert(None)

        # Load a model for this date; None → baseline behaviour.
        model_payload = None
        if self._model_store is not None:
            try:
                model_payload = self._model_store.load_for(rebalance_date)
            except Exception:
                model_payload = None

        if model_payload is None:
            # Baseline (warm-up): top_n at full base_weight.
            for t in target_tickers:
                target_usd_by_ticker[t] = per_name_usd
        else:
            model, feature_names = model_payload
            # Align the synthetic index's last entry with rebalance_date so
            # the feature extractor's _pos_or_none(series, rebalance_date)
            # succeeds.
            bars_by_ticker = self._bars_to_series_dict(end_date=rebalance_date)
            for t in target_tickers:
                feats = extract_features_for_date(
                    bars_by_ticker, ticker=t, date=rebalance_date,
                    benchmark=self.config.benchmark,
                )
                if not feats:
                    continue
                if any(
                    (isinstance(v, float) and math.isnan(v))
                    for v in feats.values()
                ):
                    # Insufficient history for some feature → skip this ticker.
                    continue
                # Build a single-row DataFrame in the model's expected column order.
                row = pd.DataFrame([{k: feats[k] for k in feature_names}])
                proba = model.predict_proba(row)
                p_win = float(proba[0, 1])
                if p_win < self.config.filter_threshold:
                    continue
                confidence = (p_win - self.config.filter_threshold) / (
                    1.0 - self.config.filter_threshold
                )
                target_usd_by_ticker[t] = per_name_usd * confidence

        # 4. Reconcile current positions to target USD (Stage 2: sells before buys).
        sells: list[tuple[Any, Quantity]] = []
        buys: list[tuple[Any, Quantity]] = []
        for t in self.config.universe:
            instr_id = InstrumentId.from_str(f"{t}.{VENUE_MIC}")
            current = self._current_position_qty(instr_id)
            target_usd = target_usd_by_ticker.get(t, 0.0)
            if t in target_tickers and target_usd > 0:
                last_close = self._closes[t][-1] if self._closes.get(t) else 0.0
                target = int(target_usd / last_close) if last_close > 0 else 0
            else:
                target = 0
            delta = target - current
            if delta == 0:
                continue
            qty = Quantity.from_int(abs(delta))
            if delta < 0:
                sells.append((instr_id, qty))
            else:
                buys.append((instr_id, qty))

        # SELL first, then BUY (cash-account safety from Stage 4).
        for instr_id, qty in sells:
            order = self.order_factory.market(
                instrument_id=instr_id, order_side=OrderSide.SELL, quantity=qty,
            )
            self.submit_order(order)
        for instr_id, qty in buys:
            order = self.order_factory.market(
                instrument_id=instr_id, order_side=OrderSide.BUY, quantity=qty,
            )
            self.submit_order(order)

    # ---- helpers -----------------------------------------------------------

    def _bars_to_series_dict(self, *, end_date: pd.Timestamp) -> dict[str, pd.Series]:
        """Convert the ring buffer into the pd.Series-per-ticker shape that
        models.features expects. Builds a synthetic business-day index whose
        LAST entry equals ``end_date`` (the rebalance date) so that
        ``extract_features_for_date`` can locate it via ``index.get_loc``.

        The momentum/vol features are position-relative, so the exact
        synthetic dates don't bias them; the month_sin/cos features
        correctly reflect ``end_date.month``."""
        out: dict[str, pd.Series] = {}
        end = end_date.normalize()
        for ticker, closes in self._closes.items():
            if not closes:
                continue
            idx = pd.date_range(end=end, periods=len(closes), freq="B")
            out[ticker] = pd.Series(closes, index=idx)
        return out
