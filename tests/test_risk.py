"""Tests for risk management."""

import numpy as np
import pytest

from src.risk import RiskManager


def test_kelly_size_bounded():
    rm = RiskManager(max_position_pct=0.10)
    size = rm.kelly_size(win_rate=0.6, avg_win=0.02, avg_loss=0.01)
    assert 0.0 <= size <= 0.10


def test_kelly_size_zero_on_losing_strategy():
    rm = RiskManager()
    size = rm.kelly_size(win_rate=0.3, avg_win=0.01, avg_loss=0.02)
    assert size == 0.0


def test_atr_stop_long():
    rm = RiskManager(atr_stop_multiplier=1.5)
    stop = rm.atr_stop_loss(entry_price=100.0, atr=2.0, direction=1)
    assert stop == pytest.approx(100.0 - 1.5 * 2.0)


def test_atr_stop_short():
    rm = RiskManager(atr_stop_multiplier=1.5)
    stop = rm.atr_stop_loss(entry_price=100.0, atr=2.0, direction=-1)
    assert stop == pytest.approx(100.0 + 1.5 * 2.0)


def test_filter_signals_exposure_cap():
    rm = RiskManager(max_portfolio_exposure=0.80)
    signals = np.array([1, 1, -1, 0])
    filtered = rm.filter_signals(signals, current_exposure=0.90)
    assert 1 not in filtered


def test_filter_signals_low_confidence():
    rm = RiskManager()
    signals = np.array([1, -1, 1])
    confidence = np.array([0.50, 0.50, 0.80])
    filtered = rm.filter_signals(signals, current_exposure=0.0, confidence=confidence)
    assert filtered[0] == 0
    assert filtered[1] == 0
    assert filtered[2] == 1
