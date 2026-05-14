"""Tests for feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.features import add_indicators, get_feature_columns


def _make_ohlcv(n: int = 200) -> pd.DataFrame:
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame(
        {
            "open": close * (1 + np.random.uniform(-0.002, 0.002, n)),
            "high": close * (1 + np.random.uniform(0, 0.01, n)),
            "low": close * (1 - np.random.uniform(0, 0.01, n)),
            "close": close,
            "volume": np.random.randint(100_000, 1_000_000, n).astype(float),
        },
        index=pd.date_range("2022-01-01", periods=n, freq="B"),
    )
    return df


def test_add_indicators_shape():
    df = _make_ohlcv()
    out = add_indicators(df)
    assert len(out) < len(df), "NaN rows should be dropped"
    assert "rsi_14" in out.columns
    assert "macd_hist" in out.columns
    assert "atr_14" in out.columns
    assert "target" in out.columns


def test_no_nan_after_indicators():
    df = _make_ohlcv()
    out = add_indicators(df)
    assert not out.isnull().any().any(), "No NaN values expected after dropna"


def test_feature_columns_excludes_target():
    df = _make_ohlcv()
    out = add_indicators(df)
    feat_cols = get_feature_columns(out)
    assert "target" not in feat_cols
    assert "close" not in feat_cols
    assert len(feat_cols) > 10


def test_target_is_binary():
    df = _make_ohlcv()
    out = add_indicators(df)
    assert set(out["target"].unique()).issubset({0, 1})
