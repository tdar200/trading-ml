"""
Thin wrapper around alpaca-py's StockHistoricalDataClient for daily bars.

Isolates all network IO into one place so the rest of the pipeline is
mockable. Reads credentials from environment variables; raises a clear
error if they are missing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


class AlpacaCredentialsError(RuntimeError):
    """Raised when ALPACA_API_KEY / ALPACA_API_SECRET are not in the env."""


@dataclass(frozen=True)
class AlpacaClient:
    """Wraps StockHistoricalDataClient. Use AlpacaClient.from_env() in app code."""
    client: StockHistoricalDataClient

    @classmethod
    def from_env(cls) -> "AlpacaClient":
        key = os.environ.get("ALPACA_API_KEY")
        secret = os.environ.get("ALPACA_API_SECRET")
        if not key or not secret:
            raise AlpacaCredentialsError(
                "ALPACA_API_KEY and ALPACA_API_SECRET must be set "
                "(copy .env.example to .env and fill in values)"
            )
        return cls(client=StockHistoricalDataClient(key, secret))

    def fetch_daily_bars(
        self,
        tickers: Iterable[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Fetch adjusted (splits+dividends) daily bars for the given tickers
        in [start, end]. Returns the Alpaca .df result (MultiIndex on
        symbol+timestamp). Empty DataFrame if no bars are returned.
        """
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        request = StockBarsRequest(
            symbol_or_symbols=list(tickers),
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            adjustment=Adjustment.ALL,  # apply splits AND dividends
            feed=DataFeed.IEX,          # free-tier compatible
        )
        bars = self.client.get_stock_bars(request)
        if bars.df is None or bars.df.empty:
            return pd.DataFrame()
        return bars.df
