"""Run walk-forward backtest and print performance report."""

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import MarketDataClient
from src.features import add_indicators, get_feature_columns
from src.models import LSTMPredictor, RFPredictor, EnsemblePredictor
from src.backtest import BacktestEngine, BacktestConfig
from src.risk import RiskManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Backtest the ensemble model.")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model-dir", default="models/")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    client = MarketDataClient(args.config)
    df = client.get_historical_bars(args.ticker, "1Day", args.start, args.end)
    df = add_indicators(df)
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values

    model_dir = Path(args.model_dir)
    lstm = LSTMPredictor.load(str(model_dir / f"lstm_{args.ticker}.pt"), input_size=len(feature_cols))
    rf = RFPredictor.load(str(model_dir / f"rf_{args.ticker}.pkl"))

    lstm_proba = lstm.predict_proba(X)
    rf_proba = rf.predict_proba(X)

    ensemble = EnsemblePredictor(lstm_weight=0.5, rf_weight=0.5, threshold=0.55)
    signals = ensemble.predict(lstm_proba, rf_proba)

    risk = RiskManager(**cfg.get("risk", {}))
    signals = risk.filter_signals(signals, current_exposure=0.0, confidence=ensemble.predict_proba(lstm_proba, rf_proba))

    bt_cfg_raw = cfg.get("backtest", {})
    bt_cfg = BacktestConfig(
        initial_capital=bt_cfg_raw.get("initial_capital", 10_000),
        commission=bt_cfg_raw.get("commission", 0.001),
        slippage=bt_cfg_raw.get("slippage", 0.001),
    )
    engine = BacktestEngine(bt_cfg)
    result = engine.run(df["close"], signals)
    result.print_summary()

    result.trade_log.to_csv(f"backtest_{args.ticker}.csv", index=False)
    logger.info("Trade log saved to backtest_%s.csv", args.ticker)


if __name__ == "__main__":
    main()
