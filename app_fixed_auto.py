AI Intraday Desktop Trader (PySimpleGUIQt + PyQt5)
Features:
# 5m intraday data (yfinance by default). Optional: Alpaca, Polygon, IEX if API keys provided.
# indicators: EMA, MACD diff, RSI, Bollinger width
# RandomForest training
# Backtest intraday
# Walk-forward (expanding-window) CV
# Metrics: Sharpe, MaxDrawdown, WinRate, CAGR
# Desktop GUI (PySimpleGUIQt) with matplotlib plots
Notes:
# Study/demo code. Test in paper account before live trading.
# Install dependencies from requirements.txt
"""

import os
import io
import time
import math
import json
from datetime import datetime, timedelta
import threading
import PySimpleGUI as sg
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from ta.trend import ema_indicator, macd_diff
from ta.momentum import rsi
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import PySimpleGUIQt as sg

# Matplotlib for plotting inside GUI
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import joblib

# Optional: Alpaca
try:
    import alpaca_trade_api as tradeapi
except Exception:
    tradeapi = None

# ----------------------------
# Helper: plotting in PySimpleGUIQt
# ----------------------------
def draw_figure(canvas, figure):
    if canvas.layout() is None:
        # PySimpleGUIQt: the element is itself a QWidget; we put a FigureCanvas inside
        fc = FigureCanvas(figure)
        canvas.addWidget(fc)
        return fc
    else:
        # fallback
        fc = FigureCanvas(figure)
        canvas.addWidget(fc)
        return fc

def clear_canvas(canvas):
    # remove all widgets
    for i in reversed(range(canvas.layout().count())):
        w = canvas.layout().itemAt(i).widget()
        w.setParent(None)

# ----------------------------
# Data providers
# ----------------------------
def fetch_yfinance(symbol, interval='5m', period='30d'):
    # interval='5m' and limited period (7d/30d/60d)
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

def fetch_polygon(ticker, from_dt, to_dt, multiplier=5, timespan='minute', api_key=None):
    # using Polygon Aggregates endpoint
    # from_dt/to_dt must be YYYY-MM-DD
    if api_key is None:
        raise ValueError("Polygon API key required")
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_dt}/{to_dt}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "results" not in data:
        return pd.DataFrame()
    rows = data["results"]
    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
    df = df.set_index('timestamp')
    df = df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'})[['Open','High','Low','Close','Volume']]
    return df

def fetch_alpaca(symbol, timeframe='5Min', limit=1000, alpaca_key=None, alpaca_secret=None, base_url=None):
    if tradeapi is None:
        raise RuntimeError("alpaca-trade-api not installed")
    if not alpaca_key or not alpaca_secret:
        raise ValueError("Alpaca API keys required")
    api = tradeapi.REST(alpaca_key, alpaca_secret, base_url=base_url)
    # Newer alpaca: get_barset deprecated; use get_bars
    try:
        bars = api.get_barset(symbol, timeframe, limit=limit)
        # get_barset returns dict of Bar objects
        arr = []
        for b in bars[symbol]:
            arr.append({"t": b.t, "o": b.o, "h": b.h, "l": b.l, "c": b.c, "v": b.v})
        df = pd.DataFrame(arr)
        if df.empty:
            return pd.DataFrame()
        df['t'] = pd.to_datetime(df['t'])
        df = df.set_index('t').rename(columns={'o':'Open','h':'High','l':'Low','c':'Close','v':'Volume'})
        return df
    except Exception as e:
        # fallback to data API if available
        raise

# ----------------------------
# Indicators, features, labels
# ----------------------------
def add_indicators(df):
    df = df.copy()
    df['ema8'] = ema_indicator(df['Close'], window=8)
    df['ema21'] = ema_indicator(df['Close'], window=21)
    df['macd_diff'] = macd_diff(df['Close'])
    df['rsi14'] = rsi(df['Close'], window=14)
    bb = BollingerBands(df['Close'], window=20, window_dev=2)
    df['bb_h'] = bb.bollinger_hband()
    df['bb_l'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_h'] - df['bb_l']) / df['Close']
    df['ret1'] = df['Close'].pct_change(1)
    df['ret5'] = df['Close'].pct_change(5)
    df = df.dropna()
    return df

def prepare_features(df, threshold=0.001):
    df = df.copy()
    features = ['ema8','ema21','macd_diff','rsi14','bb_width','ret1','ret5','Volume']
    X = df[features].fillna(method='ffill').fillna(0)
    df['future_ret1'] = df['Close'].pct_change(1).shift(-1)
    y = (df['future_ret1'] > threshold).astype(int)
    # align
    X = X.iloc[:-1]
    y = y.iloc[:-1]
    return X, y

# ----------------------------
# Backtest & metrics
# ----------------------------
def backtest_intraday(df, model, position_size_pct=0.1, initial_capital=10000, commission=1.0, slippage=0.0):
    df = add_indicators(df)
    X, y = prepare_features(df)
    preds = model.predict(X)
    cash = initial_capital
    position = 0.0
    trades = []
    equity_ts = []
    for i, idx in enumerate(X.index):
        price_open = df.loc[idx, 'Open']
        # BUY
        if preds[i] == 1 and position == 0:
            allocation = cash * position_size_pct
            qty = (allocation - commission - slippage) / price_open
            if qty > 0:
                position = qty
                cash -= allocation
                trades.append({'date': idx, 'side':'BUY','price':price_open,'qty':qty})
        elif preds[i] == 0 and position > 0:
            # SELL
            proceeds = position * price_open
            cash += proceeds - commission - slippage
            trades.append({'date': idx, 'side':'SELL','price':price_open,'qty':position})
            position = 0.0
        market_val = position * df.loc[idx,'Close']
        equity = cash + market_val
        equity_ts.append({'date': idx, 'equity': equity})
    eq = pd.DataFrame(equity_ts).set_index('date')
    trades_df = pd.DataFrame(trades)
    return eq, trades_df

def compute_metrics(equity_series, trading_days_per_year=252*6.5*12):  # not exact for intraday; user can adjust
    # equity_series: pd.Series indexed by datetime
    returns = equity_series.pct_change().dropna()
    if returns.empty:
        return {}
    avg_ret = returns.mean()
    std_ret = returns.std()
    # Annualize factor approximate: number of bars per year
    bars_per_year = len(returns) * (252 / (len(equity_series)/ (252))) if len(equity_series) > 1 else 252
    # simpler: assume 252 trading days * 78 bars per day (for 5m ~ 78 bars)
    bars_per_year = 252 * 78
    ann_ret = (1 + returns.mean()) ** bars_per_year - 1 if returns.mean() > -1 else np.nan
    ann_vol = returns.std() * np.sqrt(bars_per_year)
    sharpe = (ann_ret / ann_vol) if ann_vol != 0 else np.nan
    # Max drawdown
    cum = equity_series
    roll_max = cum.cummax()
    drawdown = (cum - roll_max) / roll_max
    max_dd = drawdown.min()
    # CAGR (approx)
    total_period_days = (equity_series.index[-1] - equity_series.index[0]).days if len(equity_series) > 1 else 1
    years = total_period_days / 365.25
    cagr = (equity_series.iloc[-1] / equity_series.iloc[0]) ** (1/years) - 1 if years > 0 else np.nan
    return {
        'sharpe': float(sharpe),
        'max_drawdown': float(max_dd),
        'cagr': float(cagr),
        'start_equity': float(equity_series.iloc[0]),
        'end_equity': float(equity_series.iloc[-1])
    }

def winrate_from_trades(trades_df):
    if trades_df.empty:
        return np.nan
    # compute trade P&L: pair buy-sell sequentially
    pairs = []
    buy = None
    for _, r in trades_df.iterrows():
        if r['side'] == 'BUY':
            buy = r
        elif r['side'] == 'SELL' and buy is not None:
            pnl = (r['price'] - buy['price']) * r['qty']
            pairs.append(pnl)
            buy = None
    if len(pairs) == 0:
        return np.nan
    wins = sum(1 for p in pairs if p > 0)
    return wins / len(pairs)

# ----------------------------
# Walk-forward / Time-series CV
# ----------------------------
def walk_forward_cv(df, n_splits=5, test_window_bars=500, model_fn=None, threshold=0.001, verbose=False):
    """
    Expanding window: start with initial training size, then iterate:
# train on [0:train_end]
# test on [train_end: train_end+test_window_bars]
# expand train_end by test_window_bars
    model_fn: function that returns a fitted model given X_train, y_train
    """
    df = add_indicators(df)
    X_all, y_all = prepare_features(df, threshold)
    splits = []
    N = len(X_all)
    if N < test_window_bars*2:
        raise ValueError("Not enough data for walk-forward with given test_window_bars")
    train_end = int(N * 0.2)  # start with 20% as initial training
    while train_end + test_window_bars <= N:
        X_train = X_all.iloc[:train_end]
        y_train = y_all.iloc[:train_end]
        X_test = X_all.iloc[train_end: train_end + test_window_bars]
        y_test = y_all.iloc[train_end: train_end + test_window_bars]
        model = model_fn(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        splits.append({
            'train_range': (X_train.index[0], X_train.index[-1]),
            'test_range': (X_test.index[0], X_test.index[-1]),
            'model': model,
            'acc': acc,
            'X_test': X_test,
            'y_test': y_test,
            'preds': preds
        })
        if verbose:
            print(f"WF step train_end={train_end}, acc={acc:.4f}")
        train_end += test_window_bars
    return splits

def rf_factory(n_estimators=200, max_depth=6):
    def train(X, y):
        m = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
        m.fit(X, y)
        return m
    return train

# ----------------------------
# GUI Layout
# ----------------------------
sg.theme('DarkAmber')

layout = [
    [sg.QLabel("<h2>AI Intraday Desktop Trader (5m) â€” PySimpleGUIQt</h2>")],
    [sg.QLabel("Symbol (US):"), sg.InputText("AAPL", key="-SYMBOL-", size=(12,1)),
     sg.Combo(['yfinance','alpaca','polygon'], default_value='yfinance', key='-PROVIDER-'),
     sg.Text("Period (yfinance):"), sg.Combo(['7d','14d','30d','60d'], default_value='30d', key='-PERIOD-')],
    [sg.Frame("API keys (optional)",[
        [sg.Text("Alpaca Key:"), sg.InputText("", key='-ALPACA_KEY-', password_char='*')],
        [sg.Text("Alpaca Secret:"), sg.InputText("", key='-ALPACA_SECRET-', password_char='*'), sg.Text("Base URL:"), sg.InputCombo(['https://paper-api.alpaca.markets','https://api.alpaca.markets'], default_value='https://paper-api.alpaca.markets', key='-ALPACA_URL-')],
        [sg.Text("Polygon Key:"), sg.InputText("", key='-POLY_KEY-')]
    ])],
    [sg.Frame("Model / Backtest Settings",[
        [sg.Text("RF n_estimators:"), sg.Slider(range=(10,500), default_value=200, orientation='h', key='-NEST-')],
        [sg.Text("RF max_depth:"), sg.Slider(range=(2,30), default_value=6, orientation='h', key='-MAXDEP-')],
        [sg.Text("Label threshold (future ret >):"), sg.InputText("0.001", key='-THRESH-')],
        [sg.Text("Position size (%):"), sg.Slider(range=(1,100), default_value=10, orientation='h', key='-POS-%')],
        [sg.Text("Initial capital ($):"), sg.InputText("10000", key='-CAP-')],
        [sg.Text("Commission ($):"), sg.InputText("1.0", key='-COM-'), sg.Text("Slippage ($):"), sg.InputText("0.0", key='-SLIP-')]
    ])],
    [sg.Button("Fetch Data"), sg.Button("Add Indicators Preview"), sg.Button("Train RF"), sg.Button("Backtest"), sg.Button("Walk-Forward CV"), sg.Button("Save Model"), sg.Button("Load Model")],
    [sg.Multiline("", size=(100,8), key='-LOG-')],
    [sg.Canvas(key='-CANVAS-', size=(900,450))],
    [sg.Button("Export equity CSV"), sg.Button("Exit")]
]

window = sg.Window("AI Intraday Trader", layout, finalize=True, keep_on_top=False, resizable=True)

# figure canvas holder
canvas_elem = window['-CANVAS-']
fig_canvas_agg = None

# state
state = {
    'df': None,
    'model': None,
    'last_equity': None,
    'last_trades': None
}

def log(s):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    window['-LOG-'].update(f"[{now}] {s}\n", append=True)

# ----------------------------
# Event loop
# ----------------------------
while True:
    event, values = window.read(timeout=100)
    if event == sg.WIN_CLOSED or event == "Exit":
        break

    if event == "Fetch Data":
        symbol = values['-SYMBOL-'].strip().upper()
        provider = values['-PROVIDER-']
        log(f"Fetching {symbol} from {provider} ...")
        try:
            if provider == 'yfinance':
                period = values['-PERIOD-']
                df = fetch_yfinance(symbol, interval='5m', period=period)
            elif provider == 'polygon':
                poly_key = values['-POLY_KEY-'].strip()
                if poly_key == '':
                    sg.popup("Polygon key empty")
                    continue
                # for polygon need from/to dates - use last 30 days
                to_dt = datetime.utcnow().date()
                from_dt = to_dt - timedelta(days=30)
                df = fetch_polygon(symbol, from_dt.isoformat(), to_dt.isoformat(), multiplier=5, timespan='minute', api_key=poly_key)
            elif provider == 'alpaca':
                ak = values['-ALPACA_KEY-']; sk = values['-ALPACA_SECRET-']; base = values['-ALPACA_URL-']
                df = fetch_alpaca(symbol, timeframe='5Min', limit=2000, alpaca_key=ak, alpaca_secret=sk, base_url=base)
            else:
                df = pd.DataFrame()
            if df is None or df.empty:
                log("No data returned.")
                sg.popup("No data returned. Try different period/provider/symbol.")
            else:
                state['df'] = df
                log(f"Loaded {len(df)} rows. From {df.index[0]} to {df.index[-1]}")
        except Exception as e:
            log(f"Error fetching: {e}")
            sg.popup("Error fetching data:", str(e))

    if event == "Add Indicators Preview":
        if state['df'] is None:
            sg.popup("Fetch data first")
            continue
        dfi = add_indicators(state['df'])
        # plot last 200 bars
        fig, ax = plt.subplots(figsize=(10,4))
        dfp = dfi.tail(200)
        ax.plot(dfp.index, dfp['Close'], label='Close')
        ax.plot(dfp.index, dfp['ema8'], label='EMA8')
        ax.plot(dfp.index, dfp['ema21'], label='EMA21')
        ax.legend()
        ax.set_title("Price + EMA (last 200 bars)")
        if fig_canvas_agg:
            clear_canvas(canvas_elem.Widget)
        fig_canvas_agg = draw_figure(canvas_elem.Widget, fig)
        plt.close(fig)

    if event == "Train RF":
        if state['df'] is None:
            sg.popup("Fetch data first")
            continue
        try:
            thr = float(values['-THRESH-'])
            n_est = int(values['-NEST-'])
            maxd = int(values['-MAXDEP-'])
            df_ind = add_indicators(state['df'])
            X, y = prepare_features(df_ind, threshold=thr)
            if X.empty or y.sum()==0:
                sg.popup("Not enough data or labels are all zero. Adjust threshold/period.")
                continue
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            model = RandomForestClassifier(n_estimators=n_est, max_depth=maxd, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            state['model'] = model
            log(f"Trained RF. Test acc: {acc:.4f}")
            sg.popup("Training finished", f"Test accuracy: {acc:.4f}")
        except Exception as e:
            log(f"Train error: {e}")
            sg.popup("Train error", str(e))

    if event == "Backtest":
        if state['model'] is None:
            sg.popup("Train or load a model first")
            continue
        try:
            pos_pct = float(values['-POS-%'])/100.0
            cap = float(values['-CAP-'])
            com = float(values['-COM-'])
            slip = float(values['-SLIP-'])
            eq_df, trades_df = backtest_intraday(state['df'], state['model'], position_size_pct=pos_pct, initial_capital=cap, commission=com, slippage=slip)
            if eq_df.empty:
                sg.popup("Backtest returned no equity. Check model/trades.")
                continue
            state['last_equity'] = eq_df
            state['last_trades'] = trades_df
            metrics = compute_metrics(eq_df['equity'])
            wr = winrate_from_trades(trades_df)
            s = f"Backtest done. Final equity: {metrics.get('end_equity'):.2f}\nSharpe: {metrics.get('sharpe'):.4f}, MaxDD: {metrics.get('max_drawdown'):.4f}, CAGR: {metrics.get('cagr'):.4f}, Winrate: {wr:.3f}"
            log(s)
            sg.popup("Backtest finished", s)
            # plot equity
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(eq_df.index, eq_df['equity'], label='Equity')
            ax.set_title("Equity Curve")
            ax.legend()
            if fig_canvas_agg:
                clear_canvas(canvas_elem.Widget)
            fig_canvas_agg = draw_figure(canvas_elem.Widget, fig)
            plt.close(fig)
        except Exception as e:
            log(f"Backtest error: {e}")
            sg.popup("Backtest error", str(e))

    if event == "Walk-Forward CV":
        if state['df'] is None:
            sg.popup("Fetch data first")
            continue
        try:
            thr = float(values['-THRESH-'])
            n_splits = 5
            test_window = 500  # bars per test window
            n_est = int(values['-NEST-']); maxd = int(values['-MAXDEP-'])
            model_fn = rf_factory(n_estimators=n_est, max_depth=maxd)
            splits = walk_forward_cv(state['df'], n_splits=n_splits, test_window_bars=test_window, model_fn=model_fn, threshold=thr, verbose=False)
            # summarize
            txt = "Walk-forward results:\n"
            for i, sp in enumerate(splits):
                txt += f"Step {i+1}: test {sp['test_range'][0]} to {sp['test_range'][1]} acc={sp['acc']:.4f}\n"
            log(txt)
            sg.popup("Walk-forward done", txt)
            # show last test predictions equity (simulate per-step using last model)
            last = splits[-1]
            model_last = last['model']
            X_test = last['X_test']
            # backtest on last test slice
            # rebuild df slice for this X_test indexes
            slice_idx = X_test.index
            df_slice = state['df'].loc[slice_idx[0]:slice_idx[-1]]
            eq_df, trades_df = backtest_intraday(df_slice, model_last, position_size_pct=float(values['-POS-%'])/100.0, initial_capital=float(values['-CAP-']), commission=float(values['-COM-']), slippage=float(values['-SLIP-']))
            state['last_equity'] = eq_df
            state['last_trades'] = trades_df
            # plot equity
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(eq_df.index, eq_df['equity'], label='Equity (last WF test)')
            ax.legend()
            if fig_canvas_agg:
                clear_canvas(canvas_elem.Widget)
            fig_canvas_agg = draw_figure(canvas_elem.Widget, fig)
            plt.close(fig)
        except Exception as e:
            log(f"Walk-forward error: {e}")
            sg.popup("Walk-forward error", str(e))

    if event == "Save Model":
        if state['model'] is None:
            sg.popup("No model to save")
            continue
        fname = sg.popup_get_file("Save model as", save_as=True, no_window=True, file_types=(("Joblib", "*.joblib"),("All","*.*")))
        if fname:
            joblib.dump(state['model'], fname)
            log(f"Model saved to {fname}")
            sg.popup("Saved", f"Model saved to {fname}")

    if event == "Load Model":
        fname = sg.popup_get_file("Load model", no_window=True, file_types=(("Joblib", "*.joblib"),("All","*.*")))
        if fname:
            try:
                state['model'] = joblib.load(fname)
                log(f"Model loaded from {fname}")
                sg.popup("Loaded", f"Model loaded from {fname}")
            except Exception as e:
                log(f"Load model error: {e}")
                sg.popup("Load model error", str(e))

    if event == "Export equity CSV":
        if state['last_equity'] is None:
            sg.popup("No equity to export")
            continue
        fname = sg.popup_get_file("Save equity CSV", save_as=True, no_window=True, file_types=(("CSV","*.csv"),))
        if fname:
            state['last_equity'].to_csv(fname)
            log(f"Equity exported to {fname}")
            sg.popup("Exported", fname)

window.close()
# --- Additional performance metrics & utilities ---
import numpy as np

def annualize_returns(returns, bars_per_year=252*78):
    # returns: series of per-bar returns
    mean = returns.mean()
    ann = (1 + mean) ** bars_per_year - 1
    return ann

def sharpe_ratio(returns, bars_per_year=252*78):
    ann_ret = annualize_returns(returns, bars_per_year)
    ann_vol = returns.std() * np.sqrt(bars_per_year)
    return ann_ret / ann_vol if ann_vol != 0 else np.nan

def sortino_ratio(returns, bars_per_year=252*78, target=0.0):
    # downside deviation only
    neg = returns[returns < target]
    if len(neg) == 0:
        return np.nan
    downside_dev = np.sqrt((neg**2).mean()) * np.sqrt(bars_per_year)
    ann_ret = annualize_returns(returns, bars_per_year)
    return (ann_ret - target) / downside_dev if downside_dev != 0 else np.nan

def max_drawdown(series):
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max
    return drawdown.min()

def calmar_ratio(series):
    # CAGR / |MaxDrawdown|
    total_days = (series.index[-1] - series.index[0]).days if len(series) > 1 else 1
    years = total_days / 365.25
    if years <= 0:
        return np.nan
    cagr = (series.iloc[-1] / series.iloc[0]) ** (1/years) - 1
    mdd = abs(max_drawdown(series))
    return cagr / mdd if mdd > 0 else np.nan

def drawdown_plot(series, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,3))
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max
    ax.fill_between(series.index, drawdown, 0, color='red', alpha=0.3)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    return ax

def add_trade_markers(ax, trades_df, price_series):
    # trades_df with columns: date, side, price
    for _, r in trades_df.iterrows():
        t = r['date']
        if t in price_series.index:
            p = r['price']
            if r['side'] == 'BUY':
                ax.scatter([t], [p], marker='^', color='green')
            else:
                ax.scatter([t], [p], marker='v', color='red')

# Confirmation dialog wrapper for Alpaca order
def confirm_and_place_alpaca(api, symbol, side, qty, paper=True):
    # return order object or raise/None
    ans = sg.popup_yes_no(f"Confirm {side} {qty} {symbol}? (paper={paper})")
    if ans != 'Yes':
        return None
    try:
        order = api.submit_order(symbol=symbol, qty=qty, side=side.lower(), type='market', time_in_force='day')
        sg.popup("Order submitted", f"Order id: {order.id}")
        return order
    except Exception as e:
        sg.popup("Order error", str(e))
        return None

"""
