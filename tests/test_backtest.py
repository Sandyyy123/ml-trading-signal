"""Tests for backtesting engine."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestEngine, BacktestConfig, BacktestResult


def _make_prices(n: int = 252) -> pd.Series:
    np.random.seed(0)
    returns = np.random.randn(n) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=pd.date_range("2023-01-01", periods=n, freq="B"))


def test_all_hold_signals():
    prices = _make_prices()
    signals = np.zeros(len(prices), dtype=int)
    engine = BacktestEngine()
    result = engine.run(prices, signals)
    assert result.total_trades == 0
    assert result.win_rate == 0.0


def test_equity_curve_length():
    prices = _make_prices()
    signals = np.ones(len(prices), dtype=int)
    engine = BacktestEngine()
    result = engine.run(prices, signals)
    assert len(result.equity_curve) == len(prices) - 1


def test_sharpe_is_finite():
    prices = _make_prices()
    signals = np.random.choice([-1, 0, 1], size=len(prices))
    result = BacktestEngine().run(prices, signals)
    assert np.isfinite(result.sharpe_ratio)


def test_max_drawdown_negative():
    prices = _make_prices()
    signals = np.ones(len(prices), dtype=int)
    result = BacktestEngine().run(prices, signals)
    assert result.max_drawdown <= 0.0


def test_backtest_result_summary_keys():
    prices = _make_prices()
    signals = np.random.choice([-1, 0, 1], size=len(prices))
    result = BacktestEngine().run(prices, signals)
    required_keys = {"sharpe_ratio", "max_drawdown", "win_rate", "total_return", "cagr", "profit_factor", "total_trades"}
    assert required_keys.issubset(result.summary.keys())
