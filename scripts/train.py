"""Train LSTM and Random Forest models on historical market data."""

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import MarketDataClient
from src.features import add_indicators, get_feature_columns
from src.models import LSTMPredictor, RFPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train ML trading models.")
    parser.add_argument("--ticker", default="SPY", help="Ticker symbol")
    parser.add_argument("--timeframe", default="1Day", help="Alpaca timeframe")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default="models/")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Fetch data
    client = MarketDataClient(args.config)
    df = client.get_historical_bars(args.ticker, args.timeframe, args.start, args.end)
    logger.info("Fetched %d bars for %s", len(df), args.ticker)

    # Feature engineering
    df = add_indicators(df)
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values
    y = df["target"].values
    logger.info("Features: %d columns, %d samples", len(feature_cols), len(X))

    # Walk-forward split (80/20)
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]

    model_cfg = cfg.get("model", {})
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Train LSTM
    logger.info("Training LSTM...")
    lstm = LSTMPredictor(
        sequence_length=model_cfg.get("sequence_length", 20),
        hidden_size=model_cfg.get("lstm_hidden", 128),
        num_layers=model_cfg.get("lstm_layers", 3),
        dropout=model_cfg.get("lstm_dropout", 0.2),
    )
    lstm.fit(X_train, y_train)
    lstm.save(str(out / f"lstm_{args.ticker}.pt"))
    logger.info("LSTM saved to %s", out / f"lstm_{args.ticker}.pt")

    # Train Random Forest
    logger.info("Training Random Forest...")
    rf = RFPredictor(
        n_estimators=model_cfg.get("rf_estimators", 500),
        max_depth=model_cfg.get("rf_max_depth", 10),
        random_state=model_cfg.get("random_seed", 42),
    )
    rf.fit(X_train, y_train, feature_names=feature_cols)
    rf.save(str(out / f"rf_{args.ticker}.pkl"))
    logger.info("RF saved to %s", out / f"rf_{args.ticker}.pkl")

    # Feature importance
    fi = rf.feature_importance().head(10)
    logger.info("Top 10 features:\n%s", fi.to_string(index=False))


if __name__ == "__main__":
    main()
