import argparse
import os
import time
import joblib
import numpy as np
import pandas as pd

# Data / indicators / ML / exchange
import yfinance as yf
from ta.momentum import rsi
from ta.trend import macd_diff, ema_indicator
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# optional: for live trading
try:
    import ccxt
except Exception:
    ccxt = None

MODEL_FILE = "rf_model.joblib"

# -------------------------
# 1) Data + Indicators
# -------------------------
def fetch_ohlcv_yfinance(symbol, start, end, interval="1d"):
    # symbol examples: "BTC-USD", "AAPL"
    df = yf.download(symbol, start=start, end=end, interval=interval, progress=False)
    if df.empty:
        raise ValueError("No data returned for symbol: " + symbol)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df

def add_indicators(df):
    df = df.copy()
    # EMA short/long
    df["ema8"] = ema_indicator(df["Close"], window=8)
    df["ema21"] = ema_indicator(df["Close"], window=21)
    # MACD diff
    df["macd_diff"] = macd_diff(df["Close"])
    # RSI
    df["rsi14"] = rsi(df["Close"], window=14)
    # Bollinger Bands (width)
    bb = BollingerBands(df["Close"], window=20, window_dev=2)
    df["bb_h"] = bb.bollinger_hband()
    df["bb_l"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_h"] - df["bb_l"]) / df["Close"]
    # Price returns
    df["ret1"] = df["Close"].pct_change(1)
    df["ret5"] = df["Close"].pct_change(5)
    df = df.dropna()
    return df

# -------------------------
# 2) Features / Labels
# -------------------------
def prepare_features(df):
    df = df.copy()
    # Features we will use
    features = [
        "ema8", "ema21", "macd_diff", "rsi14", "bb_width", "ret1", "ret5", "Volume"
    ]
    X = df[features].fillna(method="ffill").fillna(0)
    # Label: whether next day's return is > threshold (e.g., 0.5%)
    threshold = 0.005
    df["future_ret1"] = df["Close"].pct_change(1).shift(-1)
    y = (df["future_ret1"] > threshold).astype(int)
    # Drop last row where y is NaN
    X = X.iloc[:-1]
    y = y.iloc[:-1]
    return X, y

# -------------------------
# 3) Train model
# -------------------------
def train_model(X, y, save_path=MODEL_FILE):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )  # time series: no shuffle
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    joblib.dump(model, save_path)
    print("Model saved to", save_path)
    return model

# -------------------------
# 4) Simple backtest engine
# -------------------------
def backtest(df, model, initial_capital=10000, position_size=0.1, verbose=True):
    """
    Simple backtest: if model predicts 1 => go long with position_size fraction of capital at next open price.
    Exit when model predicts 0 (or at trailing stop / take profit - not implemented).
    This is a simple example for illustration only.
    """
    df = add_indicators(df)
    X, y = prepare_features(df)
    # align model predictions with df index (X index)
    preds = model.predict(X)
    # Build simulation
    cash = initial_capital
    position = 0.0  # number of units (asset)
    last_entry_price = None
    equity_curve = []

    for i, idx in enumerate(X.index):
        price_open = df.loc[idx, "Open"]
        # If prediction says buy and we have no position -> buy at open using position_size*cash
        if preds[i] == 1 and position == 0:
            buy_amount = cash * position_size
            position = buy_amount / price_open
            cash -= buy_amount
            last_entry_price = price_open
            if verbose:
                print(f"{idx.date()}: BUY {position:.6f} @ {price_open:.2f} (used {buy_amount:.2f})")
        # If prediction says 0 and we have position -> sell at open
        elif preds[i] == 0 and position > 0:
            sell_value = position * price_open
            cash += sell_value
            if verbose:
                print(f"{idx.date()}: SELL {position:.6f} @ {price_open:.2f} (rec {sell_value:.2f})")
            position = 0.0
            last_entry_price = None
        # compute equity
        market_val = position * df.loc[idx, "Close"]
        equity = cash + market_val
        equity_curve.append({"date": idx, "equity": equity, "cash": cash, "position": position})
    eq_df = pd.DataFrame(equity_curve).set_index("date")
    if verbose:
        print("Final equity:", eq_df["equity"].iloc[-1])
    return eq_df

# -------------------------
# 5) Live trading (basic)
# -------------------------
def connect_exchange(name, api_key, api_secret):
    if ccxt is None:
        raise RuntimeError("ccxt not installed. pip install ccxt")
    exchange_cls = getattr(ccxt, name)
    ex = exchange_cls({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
    })
    return ex

def fetch_ohlcv_ccxt(exchange, symbol, timeframe="1h", limit=200):
    # returns DataFrame with Open, High, Low, Close, Volume
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp")
    return df

def live_loop(exchange_name, api_key, api_secret, symbol, model, timeframe="1h", poll=60):
    ex = connect_exchange(exchange_name, api_key, api_secret)
    print("Connected to", exchange_name)
    while True:
        try:
            df = fetch_ohlcv_ccxt(ex, symbol, timeframe=timeframe, limit=300)
            df = add_indicators(df)
            X, _ = prepare_features(df)
            # use last row prediction
            x_last = X.iloc[[-1]]
            pred = model.predict(x_last)[0]
            print(pd.Timestamp.now(), "prediction:", pred)
            # Place market order example (BE CAREFUL) - here we just print action
            if pred == 1:
                print("Signal to BUY", symbol)
                # Example to create real order (uncomment after verifying):
                # amount = 0.001
                # order = ex.create_market_buy_order(symbol, amount)
            else:
                print("Signal to SELL or HOLD", symbol)
            time.sleep(poll)
        except Exception as e:
            print("Live loop error:", e)
            time.sleep(5)

# -------------------------
# 6) CLI / main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "backtest", "live"], required=True)
    parser.add_argument("--symbol", required=True, help="yfinance symbol for train/backtest (e.g., BTC-USD) or ccxt (BTC/USDT)")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--exchange", help="ccxt exchange id (binance, kraken, etc.) for live mode")
    parser.add_argument("--api_key", help="API key for exchange (live)")
    parser.add_argument("--api_secret", help="API secret for exchange (live)")
    args = parser.parse_args()

    if args.mode in ["train", "backtest"]:
        if not args.start or not args.end:
            raise SystemExit("start and end required for train/backtest")
        df = fetch_ohlcv_yfinance(args.symbol, args.start, args.end, interval="1d")
        df = add_indicators(df)
        X, y = prepare_features(df)
        if args.mode == "train":
            model = train_model(X, y)
            return
        elif args.mode == "backtest":
            if os.path.exists(MODEL_FILE):
                model = joblib.load(MODEL_FILE)
                print("Loaded model", MODEL_FILE)
            else:
                print("No saved model found - training a new one")
                model = train_model(X, y)
            eq = backtest(df, model, verbose=True)
            # Save equity curve
            eq.to_csv("equity_curve.csv")
            print("Saved equity_curve.csv")
            return

    if args.mode == "live":
        if ccxt is None:
            raise SystemExit("ccxt not installed. Install with: pip install ccxt")
        if not args.exchange or not args.api_key or not args.api_secret:
            raise SystemExit("exchange, api_key, api_secret required for live mode")
        if os.path.exists(MODEL_FILE):
            model = joblib.load(MODEL_FILE)
            print("Loaded model", MODEL_FILE)
        else:
            raise SystemExit("Model file not found. Train and save model first.")
        live_loop(args.exchange, args.api_key, args.api_secret, args.symbol, model)

if __name__ == "__main__":
    main()
