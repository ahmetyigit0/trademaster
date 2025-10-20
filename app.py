import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.subplots as sp

st.set_page_config(page_title="Strategy Starter – EMA/MACD/RSI/Bollinger", layout="wide")
st.title("📈 Strategy Starter – EMA/MACD/RSI/Bollinger")
st.caption("Eğitim amaçlıdır; yatırım tavsiyesi değildir.")

@st.cache_data(show_spinner=True, ttl=3600)
def load_yf(ticker: str, period: str = "1y", interval: str = "1d"):
    try:
        import yfinance as yf
    except Exception:
        st.error("yfinance kurulu değil. requirements.txt ile yükleyin.")
        return None
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False)
    if df is None or df.empty:
        return None
    df = df.rename(columns=str.title)
    df.index.name = "Date"
    return df

src = st.sidebar.radio("Veri Kaynağı", ["YFinance", "CSV Yükle"], index=0)
df = None
if src == "YFinance":
    tkr = st.sidebar.text_input("Ticker", "BTC-USD")
    period = st.sidebar.selectbox("Periyot", ["3mo","6mo","1y","2y","5y","max"], index=2)
    interval = st.sidebar.selectbox("Zaman Dilimi", ["1d","4h","1h"], index=0)
    if st.sidebar.button("Veriyi Çek"):
        df = load_yf(tkr, period, interval)
else:
    up = st.sidebar.file_uploader("CSV (Date,Open,High,Low,Close,Volume)", type=["csv"])
    if up:
        try:
            raw = pd.read_csv(up)
            cols = [c.lower() for c in raw.columns]
            mapping = dict(zip(cols, raw.columns))
            need = ["date","open","high","low","close","volume"]
            if all(c in mapping for c in need):
                df = raw[[mapping[c] for c in need]].copy()
                df.columns = ["Date","Open","High","Low","Close","Volume"]
            else:
                df = raw[["Date","Open","High","Low","Close","Volume"]].copy()
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").set_index("Date")
        except Exception as e:
            st.error(f"CSV okunamadı: {e}")

if df is None:
    st.info("Soldan veri kaynağını seçin ve yükleyin.")
    st.stop()

def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(close, fast=12, slow=26, signal=9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(close, n=20, k=2.0):
    mid = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=0)
    up = mid + k * std
    down = mid - k * std
    return mid, up, down

ema_short = st.sidebar.number_input("EMA Kısa", 3, 50, 9)
ema_long  = st.sidebar.number_input("EMA Uzun", 10, 300, 21)
ema_trend = st.sidebar.number_input("EMA Trend", 50, 400, 200, step=10)
rsi_len   = st.sidebar.number_input("RSI Periyot", 5, 50, 14)
bb_len    = st.sidebar.number_input("BB Periyot", 5, 100, 20)
bb_std    = st.sidebar.slider("BB Std", 1.0, 3.5, 2.0, 0.1)
m_fast    = st.sidebar.number_input("MACD Hızlı", 3, 50, 12)
m_slow    = st.sidebar.number_input("MACD Yavaş", 6, 200, 26)
m_sig     = st.sidebar.number_input("MACD Sinyal", 3, 50, 9)

data = df.copy()
data["EMA_Short"] = ema(data["Close"], ema_short)
data["EMA_Long"]  = ema(data["Close"], ema_long)
data["EMA_Trend"] = ema(data["Close"], ema_trend)
data["RSI"] = rsi(data["Close"], rsi_len)
data["MACD"], data["MACD_Signal"], data["MACD_Hist"] = macd(data["Close"], m_fast, m_slow, m_sig)
data["BB_Mid"], data["BB_Up"], data["BB_Down"] = bollinger(data["Close"], bb_len, bb_std)

bull = data["EMA_Short"] > data["EMA_Long"]
buy  = bull & bull.ne(bull.shift(1))
sell = (~bull) & bull.ne(bull.shift(1))

fig = sp.make_subplots(rows=3, cols=1, shared_xaxes=True,
                       row_heights=[0.6,0.2,0.2], vertical_spacing=0.04)
fig.add_candlestick(x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"], name="OHLC", row=1, col=1)
fig.add_scatter(x=data.index, y=data["EMA_Short"], name=f"EMA {ema_short}", row=1, col=1)
fig.add_scatter(x=data.index, y=data["EMA_Long"],  name=f"EMA {ema_long}", row=1, col=1)
fig.add_scatter(x=data.index, y=data["EMA_Trend"], name=f"EMA {ema_trend}", row=1, col=1)
fig.add_scatter(x=data.index, y=data["BB_Up"],   name="BB Upper", row=1, col=1)
fig.add_scatter(x=data.index, y=data["BB_Mid"],  name="BB Mid",   row=1, col=1)
fig.add_scatter(x=data.index, y=data["BB_Down"], name="BB Lower", row=1, col=1)
fig.add_scatter(x=data.index[buy], y=data["Close"][buy], mode="markers", marker_symbol="triangle-up", marker_size=10, name="BUY", row=1, col=1)
fig.add_scatter(x=data.index[sell], y=data["Close"][sell], mode="markers", marker_symbol="triangle-down", marker_size=10, name="SELL", row=1, col=1)
fig.add_bar(x=data.index, y=data["MACD_Hist"], name="MACD Hist", row=2, col=1)
fig.add_scatter(x=data.index, y=data["MACD"], name="MACD", row=2, col=1)
fig.add_scatter(x=data.index, y=data["MACD_Signal"], name="Signal", row=2, col=1)
fig.add_scatter(x=data.index, y=data["RSI"], name=f"RSI {rsi_len}", row=3, col=1)
fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
fig.update_layout(height=900, xaxis_rangeslider_visible=False, legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

st.subheader("🔎 Son Değerler")
st.write({
    "Close": float(data["Close"].iloc[-1]),
    "EMA_Short": float(data["EMA_Short"].iloc[-1]),
    "EMA_Long": float(data["EMA_Long"].iloc[-1]),
    "RSI": float(data["RSI"].iloc[-1]),
})