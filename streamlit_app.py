# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from datetime import datetime
import yfinance as yf
from ta.trend import ema_indicator, macd_diff
from ta.momentum import rsi
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px

# optional Alpaca for US equities live/paper trading (optional)
try:
    import alpaca_trade_api as tradeapi
except Exception:
    tradeapi = None

MODEL_FILE = "rf_5m_model.joblib"

st.set_page_config(layout="wide", page_title="AI 5m Intraday Trader (US)")

st.title("AI 5m Intraday Trader — ตลาดสหรัฐฯ (ตัวอย่าง)")

# Sidebar inputs
st.sidebar.header("Settings")
symbol = st.sidebar.text_input("Symbol (US, e.g. AAPL, SPY)", value="AAPL").upper()
period = st.sidebar.selectbox("Period for 5m data (yfinance limits intraday history)", ["7d", "14d", "30d", "60d"], index=2)
threshold = st.sidebar.number_input("Label threshold (future return >)", value=0.001, format="%.4f")  # 0.1%
n_estimators = st.sidebar.slider("RF n_estimators", 50, 500, 200)
max_depth = st.sidebar.slider("RF max_depth", 2, 20, 6)
position_size_pct = st.sidebar.slider("Position size per trade (%)", 1, 100, 10) / 100.0
initial_capital = st.sidebar.number_input("Initial capital (USD)", value=10000)
commission = st.sidebar.number_input("Commission per trade (USD)", value=1.0)
slippage = st.sidebar.number_input("Slippage per trade (USD)", value=0.01)

st.sidebar.markdown("---")
st.sidebar.header("Alpaca (optional for paper/live)")
alpaca_key = st.sidebar.text_input("ALPACA_API_KEY", type="password")
alpaca_secret = st.sidebar.text_input("ALPACA_SECRET_KEY", type="password")
alpaca_base_url = st.sidebar.selectbox("Alpaca base_url", ["https://paper-api.alpaca.markets", "https://api.alpaca.markets"])
use_alpaca = st.sidebar.checkbox("Enable Alpaca (place orders)", value=False)

# Helper functions
@st.cache_data(show_spinner=False)
def fetch_5m_yf(symbol: str, period: str):
    # yfinance supports intraday: interval='5m' but history window limited
    df = yf.download(symbol, period=period, interval="5m", progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema8"] = ema_indicator(df["Close"], window=8)
    df["ema21"] = ema_indicator(df["Close"], window=21)
    df["macd_diff"] = macd_diff(df["Close"])
    df["rsi14"] = rsi(df["Close"], window=14)
    bb = BollingerBands(df["Close"], window=20, window_dev=2)
    df["bb_h"] = bb.bollinger_hband()
    df["bb_l"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_h"] - df["bb_l"]) / df["Close"]
    df["ret1"] = df["Close"].pct_change(1)
    df["ret5"] = df["Close"].pct_change(5)
    df = df.dropna()
    return df

def prepare_features(df: pd.DataFrame, threshold: float):
    df = df.copy()
    features = ["ema8", "ema21", "macd_diff", "rsi14", "bb_width", "ret1", "ret5", "Volume"]
    X = df[features].fillna(method="ffill").fillna(0)
    df["future_ret1"] = df["Close"].pct_change(1).shift(-1)
    y = (df["future_ret1"] > threshold).astype(int)
    X = X.iloc[:-1]
    y = y.iloc[:-1]
    return X, y

def train_rf(X, y, n_estimators=200, max_depth=6):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return model, acc, classification_report(y_test, preds, zero_division=0)

def backtest_intraday(df_raw: pd.DataFrame, model, position_size_pct=0.1, initial_capital=10000, commission=1.0, slippage=0.01):
    df = add_indicators(df_raw)
    X, y = prepare_features(df, threshold)
    preds = model.predict(X)
    cash = initial_capital
    position = 0.0
    last_entry_price = None
    equity_curve = []
    trades = []
    for i, idx in enumerate(X.index):
        next_open = df.loc[idx, "Open"]  # assuming we act at this bar open
        # Buy signal and no position
        if preds[i] == 1 and position == 0:
            buy_amount = cash * position_size_pct
            if buy_amount <= 0:
                continue
            units = (buy_amount - commission - slippage) / next_open
            if units <= 0:
                continue
            cash -= buy_amount
            position = units
            last_entry_price = next_open
            trades.append({"date": idx, "side": "BUY", "price": next_open, "units": units})
        # Sell signal and have position
        elif preds[i] == 0 and position > 0:
            sell_value = position * next_open
            cash += sell_value - commission - slippage
            trades.append({"date": idx, "side": "SELL", "price": next_open, "units": position})
            position = 0.0
            last_entry_price = None
        market_val = position * df.loc[idx, "Close"]
        equity = cash + market_val
        equity_curve.append({"date": idx, "equity": equity})
    eq_df = pd.DataFrame(equity_curve).set_index("date")
    trades_df = pd.DataFrame(trades)
    return eq_df, trades_df

st.markdown("## 1) Load 5m data")
with st.expander("Fetch data & preview"):
    df = fetch_5m_yf(symbol, period)
    if df.empty:
        st.warning("ไม่พบข้อมูล 5m สำหรับ symbol/period ที่เลือก — ลองลด period หรือเช็คชื่อ symbol (ตัวอย่าง: AAPL, SPY).")
        st.stop()
    st.write(f"Loaded {len(df)} rows for {symbol} ({period}, interval=5m)")
    st.dataframe(df.tail(10))

st.markdown("## 2) Indicators & Features")
if st.button("Add indicators & preview"):
    df_ind = add_indicators(df)
    st.dataframe(df_ind.tail(10))
    fig = px.line(df_ind.tail(200).reset_index(), x="index", y=["Close", "ema8", "ema21"])
    fig.update_layout(height=350, legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("## 3) Train model")
if st.button("Train RandomForest"):
    with st.spinner("Training..."):
        df_ind = add_indicators(df)
        X, y = prepare_features(df_ind, threshold)
        if X.empty or y.sum() == 0:
            st.error("ข้อมูลไม่พอสำหรับเทรน หรือ label เป็น 0 ทั้งหมด — ปรับ threshold หรือ period ให้มากขึ้น")
        else:
            model, acc, creport = train_rf(X, y, n_estimators=n_estimators, max_depth=max_depth)
            joblib.dump(model, MODEL_FILE)
            st.success(f"Training done — test accuracy: {acc:.4f}")
            st.text(creport)

st.markdown("## 4) Backtest")
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Run backtest"):
        if not os.path.exists := False:  # placeholder to satisfy static analyzers
            pass
        try:
            model = joblib.load(MODEL_FILE)
        except Exception:
            st.warning("Model not found — training automatically")
            df_ind = add_indicators(df)
            X, y = prepare_features(df_ind, threshold)
            model, acc, _ = train_rf(X, y, n_estimators=n_estimators, max_depth=max_depth)
            joblib.dump(model, MODEL_FILE)
            st.info(f"Trained model with acc {acc:.4f} and saved to {MODEL_FILE}")
        df_ind = add_indicators(df)
        eq_df, trades_df = backtest_intraday(df_ind, model, position_size_pct, initial_capital, commission, slippage)
        if eq_df.empty:
            st.error("Backtest produced no trades/equity — ปรับ settings")
        else:
            st.success(f"Final equity: {eq_df['equity'].iloc[-1]:.2f}")
            fig = px.line(eq_df.reset_index(), x="date", y="equity", title="Equity Curve")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("Trades:")
            st.dataframe(trades_df.tail(20))

with col2:
    st.markdown("Backtest settings")
    st.write(f"- Position size: {position_size_pct*100:.0f}% per trade")
    st.write(f"- Initial capital: ${initial_capital:.2f}")
    st.write(f"- Commission per trade: ${commission:.2f}")
    st.write(f"- Slippage per trade: ${slippage:.2f}")

st.markdown("## 5) (Optional) Place paper order via Alpaca — single-step execution")
st.write("จะสั่งซื้อจริงผ่าน Alpaca หรือไม่? ให้ใช้เฉพาะ paper account และทดสอบให้ละเอียดก่อน")

if use_alpaca:
    if alpaca_key == "" or alpaca_secret == "":
        st.warning("กรุณาใส่ Alpaca API keys ใน sidebar")
    elif tradeapi is None:
        st.error("alpaca-trade-api ไม่ได้ติดตั้ง (pip install alpaca-trade-api)")
    else:
        if st.button("Run one-step Alpaca decision & (paper) order"):
            try:
                api = tradeapi.REST(alpaca_key, alpaca_secret, base_url=alpaca_base_url)
                # fetch latest bar via yfinance (for decision)
                df_ind = add_indicators(df)
                X, y = prepare_features(df_ind, threshold)
                last_x = X.iloc[[-1]]
                model = joblib.load(MODEL_FILE)
                pred = model.predict(last_x)[0]
                st.write("Model prediction (next bar):", pred)
                if pred == 1:
                    # compute allocation
                    account = api.get_account()
                    cash = float(account.cash)
                    size_usd = cash * position_size_pct
                    last_price = df_ind["Close"].iloc[-1]
                    qty = max(1, int((size_usd - commission - slippage) / last_price))
                    st.write(f"Placing MARKET BUY for {qty} shares of {symbol} (approx ${size_usd:.2f})")
                    # place order (paper) - uncomment if you want to execute
                    order = api.submit_order(symbol=symbol, qty=qty, side='buy', type='market', time_in_force='day')
                    st.write("Order submitted:", order.__dict__)
                else:
                    # Try to close any position
                    positions = {p.symbol: p for p in api.list_positions()}
                    if symbol in positions:
                        pos = positions[symbol]
                        qty = int(float(pos.qty))
                        st.write(f"Placing MARKET SELL for {qty} shares of {symbol} to close position")
                        order = api.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='day')
                        st.write("Order submitted:", order.__dict__)
                    else:
                        st.info("No open position to close. Model says HOLD/SELL.")
            except Exception as e:
                st.error(f"Alpaca error: {e}")

st.markdown("----")
st.caption("หมายเหตุ: โค้ดเป็นตัวอย่างศึกษาการทำระบบ Intraday AI. ตรวจสอบ logic, risk และ latency ก่อนใช้งานจริง.")
