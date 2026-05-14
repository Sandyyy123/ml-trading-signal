"""Risk management: position sizing, stop-loss, and exposure limits."""

from __future__ import annotations

import numpy as np
import pandas as pd


class RiskManager:
    """
    Applies risk controls to raw model signals before execution.

    Controls:
    - Kelly criterion position sizing (fractional)
    - ATR-based dynamic stop-loss
    - Max single-position size as % of portfolio
    - Max total portfolio exposure cap
    """

    def __init__(
        self,
        max_position_pct: float = 0.10,
        max_portfolio_exposure: float = 0.80,
        kelly_fraction: float = 0.25,
        atr_stop_multiplier: float = 1.5,
    ):
        self.max_position_pct = max_position_pct
        self.max_portfolio_exposure = max_portfolio_exposure
        self.kelly_fraction = kelly_fraction
        self.atr_stop_multiplier = atr_stop_multiplier

    def kelly_size(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Fractional Kelly position size.

        Args:
            win_rate: Historical win rate [0, 1].
            avg_win: Average winning trade return.
            avg_loss: Average losing trade return (positive value).

        Returns:
            Position size as fraction of capital, capped at max_position_pct.
        """
        b = avg_win / (avg_loss + 1e-9)
        kelly = (win_rate * b - (1 - win_rate)) / (b + 1e-9)
        kelly = max(0.0, kelly) * self.kelly_fraction
        return min(kelly, self.max_position_pct)

    def atr_stop_loss(self, entry_price: float, atr: float, direction: int) -> float:
        """
        ATR-based stop-loss price.

        Args:
            entry_price: Trade entry price.
            atr: Current ATR value.
            direction: 1 for long, -1 for short.

        Returns:
            Stop-loss trigger price.
        """
        offset = self.atr_stop_multiplier * atr
        return entry_price - direction * offset

    def filter_signals(
        self,
        signals: np.ndarray,
        current_exposure: float,
        confidence: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Apply exposure cap and optional confidence gating to raw signals.

        Args:
            signals: Array of -1/0/1 signals.
            current_exposure: Current portfolio exposure fraction [0, 1].
            confidence: Optional probability array; signals below 0.55 are zeroed.

        Returns:
            Filtered signal array.
        """
        filtered = signals.copy()

        if current_exposure >= self.max_portfolio_exposure:
            filtered = np.where(filtered == 1, 0, filtered)

        if confidence is not None:
            low_conf = np.abs(confidence - 0.5) < 0.05
            filtered = np.where(low_conf, 0, filtered)

        return filtered
