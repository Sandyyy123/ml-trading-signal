"""Generate live trading signals from latest market data."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import MarketDataClient
from src.features import add_indicators, get_feature_columns
from src.models import LSTMPredictor, RFPredictor, EnsemblePredictor
from src.risk import RiskManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SIGNAL_LABELS = {1: "BUY", -1: "SELL", 0: "HOLD"}


def main():
    parser = argparse.ArgumentParser(description="Generate live trading signals.")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model-dir", default="models/")
    args = parser.parse_args()

    client = MarketDataClient(args.config)
    df = client.get_historical_bars(args.ticker, "1Day", limit=200)
    df = add_indicators(df)
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values

    model_dir = Path(args.model_dir)
    lstm = LSTMPredictor.load(str(model_dir / f"lstm_{args.ticker}.pt"), input_size=len(feature_cols))
    rf = RFPredictor.load(str(model_dir / f"rf_{args.ticker}.pkl"))

    lstm_proba = lstm.predict_proba(X)
    rf_proba = rf.predict_proba(X)

    ensemble = EnsemblePredictor(threshold=0.55)
    blended = ensemble.predict_proba(lstm_proba, rf_proba)
    signals = ensemble.predict(lstm_proba, rf_proba)

    risk = RiskManager()
    signals = risk.filter_signals(signals, current_exposure=0.0, confidence=blended)

    latest_signal = int(signals[-1])
    latest_confidence = float(blended[-1])
    latest_price = float(df["close"].iloc[-1])
    latest_atr = float(df["atr_14"].iloc[-1])

    stop = risk.atr_stop_loss(latest_price, latest_atr, latest_signal if latest_signal != 0 else 1)

    print(f"\n=== Live Signal: {args.ticker} ===")
    print(f"  Signal     : {SIGNAL_LABELS[latest_signal]}")
    print(f"  Confidence : {latest_confidence:.2%}")
    print(f"  Last Close : ${latest_price:.2f}")
    print(f"  ATR Stop   : ${stop:.2f}")
    print(f"  As of      : {df.index[-1].date()}")


if __name__ == "__main__":
    main()
