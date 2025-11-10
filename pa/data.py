from __future__ import annotations
import pandas as pd
import yfinance as yf

def load_ohlcv(symbol: str, interval: str = "1d", period: str = "3y") -> pd.DataFrame:
    df = yf.download(symbol, interval=interval, period=period, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError("No data returned. Check symbol/interval/period.")
    df = df.rename(columns=str.title)  # Open, High, Low, Close, Adj Close, Volume
    df = df.dropna(subset=["Open","High","Low","Close","Volume"])
    df.index.name = "Date"
    return df
