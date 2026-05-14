"""Alpaca + Polygon market data client with automatic fallback."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class MarketDataClient:
    """Fetches OHLCV bars from Alpaca with Polygon fallback."""

    def __init__(self, config_path: str = "config.yaml"):
        cfg = _load_config(config_path)
        self._alpaca_key = cfg["alpaca"]["api_key"]
        self._alpaca_secret = cfg["alpaca"]["secret_key"]
        self._base_url = cfg["alpaca"]["base_url"]
        self._polygon_key = cfg.get("polygon", {}).get("api_key")

    def get_historical_bars(
        self,
        ticker: str,
        timeframe: str = "1Day",
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV bars.

        Args:
            ticker: Ticker symbol, e.g. "SPY".
            timeframe: Alpaca timeframe string ("1Min", "1Hour", "1Day").
            start: ISO date string "YYYY-MM-DD". Defaults to 1 year ago.
            end: ISO date string "YYYY-MM-DD". Defaults to today.
            limit: Max number of bars to return.

        Returns:
            DataFrame with columns [open, high, low, close, volume, vwap].
        """
        if start is None:
            start = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end is None:
            end = datetime.utcnow().strftime("%Y-%m-%d")

        try:
            return self._fetch_alpaca(ticker, timeframe, start, end, limit)
        except Exception as exc:
            logger.warning("Alpaca fetch failed (%s); falling back to Polygon.", exc)
            return self._fetch_polygon(ticker, start, end, limit)

    def _fetch_alpaca(
        self,
        ticker: str,
        timeframe: str,
        start: str,
        end: str,
        limit: int,
    ) -> pd.DataFrame:
        import alpaca_trade_api as tradeapi

        api = tradeapi.REST(
            self._alpaca_key, self._alpaca_secret, self._base_url, api_version="v2"
        )
        bars = api.get_bars(
            ticker,
            timeframe,
            start=start,
            end=end,
            limit=limit,
            adjustment="all",
        ).df
        bars.index = pd.to_datetime(bars.index, utc=True)
        bars = bars[["open", "high", "low", "close", "volume", "vwap"]].copy()
        logger.info("Alpaca: fetched %d bars for %s", len(bars), ticker)
        return bars

    def _fetch_polygon(
        self,
        ticker: str,
        start: str,
        end: str,
        limit: int,
    ) -> pd.DataFrame:
        from polygon import RESTClient

        client = RESTClient(self._polygon_key)
        aggs = client.get_aggs(ticker, 1, "day", start, end, limit=limit)
        df = pd.DataFrame(
            [
                {
                    "open": a.open,
                    "high": a.high,
                    "low": a.low,
                    "close": a.close,
                    "volume": a.volume,
                    "vwap": a.vwap,
                }
                for a in aggs
            ]
        )
        df.index = pd.to_datetime([a.timestamp for a in aggs], unit="ms", utc=True)
        logger.info("Polygon: fetched %d bars for %s", len(df), ticker)
        return df
