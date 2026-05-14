# ML Trading Signal Tool

Python ML system for financial market analysis and algorithmic trading signal generation.

## Features

- **LSTM + Random Forest ensemble** for time-series price prediction
- **Alpaca API integration** for real-time and historical market data (Polygon fallback)
- **Walk-forward backtesting** with Sharpe ratio, max drawdown, win rate, CAGR
- **Risk management**: ATR stop-loss, Kelly position sizing, portfolio exposure caps
- **Technical indicators**: RSI, MACD, Bollinger Bands, ATR, OBV, EMA crossovers
- Clean, documented codebase with pytest test suite

## Architecture

```
Data (Alpaca/Polygon) → Feature Engineering → LSTM + RF Models → Backtest Engine → Risk Manager → Signals
```

## Setup

```bash
pip install -r requirements.txt
cp config.yaml.example config.yaml   # add your API keys
```

## Configuration

Edit `config.yaml`:

```yaml
alpaca:
  api_key: YOUR_KEY
  secret_key: YOUR_SECRET
  base_url: https://paper-api.alpaca.markets   # use paper trading first

model:
  sequence_length: 20
  lstm_hidden: 128
  lstm_layers: 3
  rf_estimators: 500

backtest:
  start_date: "2022-01-01"
  end_date: "2024-01-01"
  initial_capital: 10000

risk:
  max_position_pct: 0.10
  kelly_fraction: 0.25
  atr_stop_multiplier: 1.5
```

## Usage

**Train models:**
```bash
python scripts/train.py --ticker SPY --timeframe 1Day
```

**Run backtest:**
```bash
python scripts/backtest.py --ticker SPY --start 2022-01-01 --end 2024-01-01
```

**Live signals:**
```bash
python scripts/live_signals.py --ticker SPY
```

## Sample Output

```
Backtest Results (SPY 2022-2024):
  Sharpe Ratio:    0.87
  Win Rate:        63.2%
  Max Drawdown:   -14.1%
  Annual Return:  +28.3%
  Profit Factor:   2.13
  Total Trades:    142
```

## Tests

```bash
pytest tests/ -v
```

## Tech Stack

Python 3.11, PyTorch, Scikit-learn, Pandas, NumPy, TA-Lib, alpaca-trade-api, Backtrader, Plotly, pydantic, pytest

## License

MIT
