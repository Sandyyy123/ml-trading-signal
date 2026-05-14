"""Walk-forward backtesting engine with performance metrics."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    commission: float = 0.001
    slippage: float = 0.001
    position_size: float = 1.0  # fraction of available capital per trade


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trade_log: pd.DataFrame
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_return: float
    cagr: float
    profit_factor: float
    total_trades: int
    summary: dict = field(default_factory=dict)

    def __post_init__(self):
        self.summary = {
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "win_rate": round(self.win_rate, 4),
            "total_return": round(self.total_return, 4),
            "cagr": round(self.cagr, 4),
            "profit_factor": round(self.profit_factor, 4),
            "total_trades": self.total_trades,
        }

    def print_summary(self) -> None:
        print("\n--- Backtest Results ---")
        for k, v in self.summary.items():
            print(f"  {k:22s}: {v}")


class BacktestEngine:
    """Event-driven backtester with commission, slippage, and walk-forward support."""

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def run(self, prices: pd.Series, signals: np.ndarray) -> BacktestResult:
        """
        Simulate trades on a price series given an integer signal array.

        Args:
            prices: Close price series (DatetimeIndex).
            signals: Array of -1 (SELL), 0 (HOLD), 1 (BUY) aligned to prices.

        Returns:
            BacktestResult with full metrics.
        """
        cfg = self.config
        capital = cfg.initial_capital
        position = 0.0
        entry_price = 0.0
        equity = []
        trades = []

        prices_arr = prices.values
        n = len(prices_arr)

        for i in range(1, n):
            sig = int(signals[i - 1])
            price = prices_arr[i]
            fill_price = price * (1 + cfg.slippage * np.sign(sig) if sig != 0 else 1)

            # Close existing position on opposite signal
            if position != 0 and sig != 0 and np.sign(sig) != np.sign(position):
                pnl = position * (fill_price - entry_price) - abs(position) * fill_price * cfg.commission
                capital += pnl
                trades.append(
                    {"date": prices.index[i], "side": "close", "price": fill_price, "pnl": pnl}
                )
                position = 0.0

            # Open new position
            if sig != 0 and position == 0:
                size = (capital * cfg.position_size) / fill_price
                position = size * sig
                entry_price = fill_price
                capital -= abs(position) * fill_price * cfg.commission
                trades.append(
                    {"date": prices.index[i], "side": "buy" if sig > 0 else "sell", "price": fill_price, "pnl": 0.0}
                )

            unrealized = position * (price - entry_price) if position != 0 else 0.0
            equity.append(capital + unrealized)

        equity_series = pd.Series(equity, index=prices.index[1:])
        trade_log = pd.DataFrame(trades)

        return BacktestResult(
            equity_curve=equity_series,
            trade_log=trade_log,
            sharpe_ratio=self._sharpe(equity_series),
            max_drawdown=self._max_drawdown(equity_series),
            win_rate=self._win_rate(trade_log),
            total_return=(equity_series.iloc[-1] / cfg.initial_capital) - 1,
            cagr=self._cagr(equity_series, cfg.initial_capital),
            profit_factor=self._profit_factor(trade_log),
            total_trades=len(trade_log[trade_log["pnl"] != 0]) if not trade_log.empty else 0,
        )

    def _sharpe(self, equity: pd.Series, risk_free: float = 0.04) -> float:
        returns = equity.pct_change().dropna()
        excess = returns - risk_free / 252
        return float(excess.mean() / (excess.std() + 1e-9) * np.sqrt(252))

    def _max_drawdown(self, equity: pd.Series) -> float:
        roll_max = equity.cummax()
        drawdown = (equity - roll_max) / (roll_max + 1e-9)
        return float(drawdown.min())

    def _win_rate(self, trades: pd.DataFrame) -> float:
        closed = trades[trades["pnl"] != 0] if not trades.empty else trades
        if len(closed) == 0:
            return 0.0
        return float((closed["pnl"] > 0).mean())

    def _cagr(self, equity: pd.Series, initial: float) -> float:
        years = len(equity) / 252
        if years < 0.01:
            return 0.0
        return float((equity.iloc[-1] / initial) ** (1 / years) - 1)

    def _profit_factor(self, trades: pd.DataFrame) -> float:
        closed = trades[trades["pnl"] != 0] if not trades.empty else trades
        gross_win = closed[closed["pnl"] > 0]["pnl"].sum()
        gross_loss = abs(closed[closed["pnl"] < 0]["pnl"].sum())
        return float(gross_win / (gross_loss + 1e-9))
