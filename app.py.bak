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

import numpy as np
import pandas as pd

def compute_metrics(equity_series, bars_per_year=None, trading_days_per_year=252, trading_day_seconds=6.5*3600):
    if not isinstance(equity_series, pd.Series):
        equity_series = pd.Series(equity_series)
    equity_series = equity_series.dropna()
    returns = equity_series.pct_change().dropna()
    if returns.empty:
        return {}

    avg_ret = returns.mean()
    std_ret = returns.std()

    # infer bars_per_year if not provided (try from DatetimeIndex; fallback to 5-min assumption)
    if bars_per_year is None:
        if isinstance(equity_series.index, pd.DatetimeIndex) and len(equity_series.index) > 1:
            deltas = equity_series.index.to_series().diff().dropna().dt.total_seconds()
            median_delta = deltas.median()
            if median_delta > 0:
                bars_per_day = trading_day_seconds / median_delta
                bars_per_year = int(round(bars_per_day * trading_days_per_year))
            else:
                bars_per_year = trading_days_per_year * 78
        else:
            bars_per_year = trading_days_per_year * 78

    ann_ret = (1 + avg_ret) ** bars_per_year - 1 if avg_ret > -1 else np.nan
    ann_vol = std_ret * np.sqrt(bars_per_year) if std_ret is not None else np.nan
    sharpe = ann_ret / ann_vol if ann_vol and not np.isnan(ann_vol) else np.nan

    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    max_dd = drawdown.min()

    return {
        "avg_return_per_bar": avg_ret,
        "std_return_per_bar": std_ret,
        "annualized_return": ann_ret,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "bars_per_year": bars_per_year,
    }
