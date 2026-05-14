"""Technical indicator feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators and lag features on OHLCV data.

    Args:
        df: DataFrame with columns [open, high, low, close, volume].

    Returns:
        DataFrame with original columns plus all indicator features.
        NaN rows (from lookback periods) are dropped.
    """
    out = df.copy()

    close = out["close"]
    high = out["high"]
    low = out["low"]
    volume = out["volume"]

    # --- Trend ---
    for period in [9, 21, 50]:
        out[f"ema_{period}"] = close.ewm(span=period, adjust=False).mean()

    out["ema_cross_9_21"] = out["ema_9"] - out["ema_21"]

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    # --- Momentum ---
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # --- Volatility ---
    out["bb_mid"] = close.rolling(20).mean()
    out["bb_std"] = close.rolling(20).std()
    out["bb_upper"] = out["bb_mid"] + 2 * out["bb_std"]
    out["bb_lower"] = out["bb_mid"] - 2 * out["bb_std"]
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / (out["bb_mid"] + 1e-9)
    out["bb_pct"] = (close - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"] + 1e-9)

    # ATR
    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr_14"] = tr.ewm(span=14, adjust=False).mean()
    out["atr_pct"] = out["atr_14"] / (close + 1e-9)

    # --- Volume ---
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    out["obv"] = obv
    out["obv_ema"] = pd.Series(obv).ewm(span=20, adjust=False).mean().values
    out["volume_ratio"] = volume / (volume.rolling(20).mean() + 1e-9)

    # --- Lag features ---
    for lag in range(1, 6):
        out[f"close_lag_{lag}"] = close.shift(lag)
        out[f"return_lag_{lag}"] = close.pct_change(lag)

    # --- Target: next-bar direction ---
    out["target"] = (close.shift(-1) > close).astype(int)

    out = out.dropna()
    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return names of all feature columns (excludes OHLCV raw and target)."""
    exclude = {"open", "high", "low", "close", "volume", "vwap", "target"}
    return [c for c in df.columns if c not in exclude]
