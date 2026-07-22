"""Fetch and cache daily OHLCV price data.

Primary source is Binance via ``ccxt``. If that fails (geo-blocking, downtime,
rate limits) we fall back to the CryptoCompare / CoinDesk daily endpoint, which
optionally uses an API key from the ``CRYPTOCOMPARE_API_KEY`` environment
variable. Whatever is fetched is cached to a CSV so subsequent runs — including
the demo notebook — are fast and network-independent.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .. import config

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# Source 1: ccxt / Binance (primary)

def _fetch_ccxt(
    symbol: str,
    timeframe: str,
    start: str,
) -> pd.DataFrame:
    """Fetch full OHLCV history from Binance via ccxt, paginating as needed.

    Args: 
        symbol: ccxt market symbol, e.g. ``"SOL/USDT"``.
        timeframe: ccxt timeframe string, e.g. ``"1d"``.
        start: ISO date string for the earliest candle to request.

    Returns:
        DataFrame indexed by UTC timestamp with ``OHLCV_COLUMNS``.
    """
    import ccxt

    exchange = ccxt.binance({"enableRateLimit": True})
    ms_per_candle = exchange.parse_timeframe(timeframe) * 1000
    since = exchange.parse8601(f"{start}T00:00:00Z")
    now = exchange.milliseconds()

    rows: list[list[float]] = []
    limit = 1000
    while since < now:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        since = batch[-1][0] + ms_per_candle
        if len(batch) < limit:
            break

    if not rows:
        raise RuntimeError(f"ccxt returned no candles for {symbol} {timeframe}")

    frame = pd.DataFrame(rows, columns=["timestamp", *OHLCV_COLUMNS])
    frame["date"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame = (
        frame.drop(columns=["timestamp"])
        .set_index("date")
        .loc[:, OHLCV_COLUMNS]
    )
    frame = frame[~frame.index.duplicated(keep="first")].sort_index()
    return frame


# Source 2: CryptoCompare / CoinDesk (fallback)

def _fetch_cryptocompare(
    base: str,
    quote: str,
    limit: int = 2000,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch daily OHLCV from the CryptoCompare histoday endpoint (fallback).

    The public endpoint now requires an API key. If one is available (argument
    or ``CRYPTOCOMPARE_API_KEY`` env var) it is used; otherwise the request is
    still attempted and will raise if the service rejects it.

    Args:
        base: base asset, e.g. ``"SOL"``.
        quote: quote asset, e.g. ``"USDT"``.
        limit: number of daily candles (max 2000 per request).
        api_key: optional CryptoCompare API key.

    Returns:
        DataFrame indexed by UTC timestamp with ``OHLCV_COLUMNS``.
    """
    import requests

    url = "https://min-api.cryptocompare.com/data/v2/histoday"
    key = api_key or os.getenv("CRYPTOCOMPARE_API_KEY")
    headers = {"authorization": f"Apikey {key}"} if key else {}
    params = {"fsym": base, "tsym": quote, "limit": limit}

    resp = requests.get(url, params=params, headers=headers, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("Response") == "Error":
        raise RuntimeError(f"CryptoCompare error: {payload.get('Message')}")

    data = payload.get("Data", {}).get("Data", [])
    if not data:
        raise RuntimeError("CryptoCompare returned no data (an API key may be required)")

    frame = pd.DataFrame(data)
    frame["date"] = pd.to_datetime(frame["time"], unit="s", utc=True)
    frame = frame.rename(columns={"volumefrom": "volume"})
    frame = frame.set_index("date").loc[:, OHLCV_COLUMNS]
    frame = frame[frame["close"] > 0]  # drop pre-listing zero-price rows
    frame = frame[~frame.index.duplicated(keep="first")].sort_index()
    return frame


# Public entry point
def load_price_data(
    symbol: str = config.SYMBOL,
    timeframe: str = config.TIMEFRAME,
    *,
    start: str = config.HISTORY_START,
    force_refresh: bool = False,
    cache_dir: Path = config.CACHE_DIR,
) -> pd.DataFrame:
    """Load daily OHLCV, using a CSV cache when available.

    On a cache miss (or ``force_refresh=True``) the data is fetched from Binance
    via ccxt, falling back to CryptoCompare, then written to the cache.

    Args:
        symbol: market symbol, e.g. ``"SOL/USDT"``.
        timeframe: candle timeframe, e.g. ``"1d"``.
        start: earliest date to request from the primary source.
        force_refresh: if True, ignore any cached file and re-fetch.
        cache_dir: directory holding the CSV cache.

    Returns:
        DataFrame indexed by a UTC ``DatetimeIndex`` with columns
        ``[open, high, low, close, volume]`` and a ``source`` attribute in
        ``frame.attrs["source"]``.
    """
    path = cache_dir / f"{symbol.replace('/', '_')}_{timeframe}.csv"

    if path.exists() and not force_refresh:
        logger.info("Loading %s %s from cache: %s", symbol, timeframe, path)
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        frame.attrs["source"] = "cache"
        return frame

    try:
        logger.info("Fetching %s %s from Binance (ccxt)...", symbol, timeframe)
        frame = _fetch_ccxt(symbol, timeframe, start)
        source = "ccxt/binance"
    except Exception as exc:  # noqa: BLE001 - fall back on any ccxt failure
        logger.warning("ccxt fetch failed (%s); falling back to CryptoCompare", exc)
        base, quote = symbol.split("/")
        frame = _fetch_cryptocompare(base, quote)
        source = "cryptocompare"

    # numeric hygiene
    frame = frame.astype(float)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(how="any")

    cache_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)
    logger.info("Cached %d rows from %s to %s", len(frame), source, path)
    frame.attrs["source"] = source
    return frame
