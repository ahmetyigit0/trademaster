
import streamlit as st
import pandas as pd
import numpy as np
import plotly.subplots as sp
import plotly.graph_objects as go

st.set_page_config(page_title="Trading Strategy – TP/SL Signals", layout="wide")
st.title("📈 Trading Strategy – TP/SL Signals (Safe Version v2)")
st.caption("Eğitim amaçlıdır; yatırım tavsiyesi değildir.")

# ---------- Helpers ----------
def dedup_columns(cols):
    seen = {}
    new = []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            new.append(c)
        else:
            seen[c] += 1
            new.append(f"{c}.{seen[c]}")
    return new

@st.cache_data(ttl=1800, show_spinner=True)
def load_yf(ticker: str, period="1y", interval="1d"):
    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=False)
        if df.empty:
            return None
        df = df.rename(columns=str.title)
        df.index.name = "Date"
        return df
    except Exception:
        return None

def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Deduplicate possible duplicate column names (CSV kaynaklı olabilir)
    df.columns = dedup_columns([str(c).title() for c in df.columns])

    targets = ["Open", "High", "Low", "Close", "Volume"]
    for base in targets:
        # aynı isimli birden fazla kolon varsa ilkini kullan
        candidates = [c for c in df.columns if c == base or c.split(".")[0] == base]
        if not candidates:
            continue
        col = candidates[0]
        s = df[col]
        # Eğer DataFrame döndüyse (duplicate isimden), ilk kolonu al
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        # Güvenli dönüştürme
        s = s.astype(str)
        s = (s.str.replace(",", "", regex=False)
               .str.replace(" ", "", regex=False)
               .replace({"": np.nan, "None": np.nan, "nan": np.nan, "NaN": np.nan}))
        df[base] = pd.to_numeric(s, errors="coerce")
    return df

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

def atr(df, n=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def fmt(x, n=6):
    try:
        x = float(x)
        if np.isfinite(x):
            return f"{x:.{n}f}"
    except Exception:
        pass
    return "—"

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Parametreler")
source = st.sidebar.radio("Veri Kaynağı", ["YFinance", "CSV Yükle"], index=0)
df = None

if source == "YFinance":
    ticker = st.sidebar.text_input("Ticker", "BTC-USD")
    period = st.sidebar.selectbox("Periyot", ["3mo","6mo","1y","2y","5y"], index=2)
    interval = st.sidebar.selectbox("Zaman Dilimi", ["1d","4h","1h"], index=0)
    if st.sidebar.button("Veriyi Çek"):
        df = load_yf(ticker, period, interval)
else:
    up = st.sidebar.file_uploader("CSV (Date,Open,High,Low,Close,Volume)", type=["csv"])
    if up:
        try:
            raw = pd.read_csv(up)
            df = ensure_numeric(raw)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.set_index("Date").sort_index()
        except Exception as e:
            st.error(f"CSV okunamadı: {e}")

if df is None or df.empty:
    st.info("Soldan veri kaynağını seçin ve yükleyin.")
    st.stop()

# YF verisini de normalize et
df = ensure_numeric(df)

ema_short = st.sidebar.number_input("EMA Kısa", 3, 50, 9)
ema_long  = st.sidebar.number_input("EMA Uzun", 10, 300, 21)
ema_trend = st.sidebar.number_input("EMA Trend", 50, 400, 200, step=10)
rsi_len   = st.sidebar.number_input("RSI Periyot", 5, 50, 14)
bb_len    = st.sidebar.number_input("Bollinger Periyot", 5, 100, 20)
bb_std    = st.sidebar.slider("BB Std", 1.0, 3.5, 2.0, 0.1)
atr_len   = st.sidebar.number_input("ATR Periyot", 5, 50, 14)

stop_mode = st.sidebar.selectbox("Stop Tipi", ["ATR x K", "Sabit %"], index=0)
atr_k     = st.sidebar.slider("ATR çarpanı", 0.5, 5.0, 2.0, 0.1)
stop_pct  = st.sidebar.slider("Zarar %", 0.5, 10.0, 2.0, 0.1)

# ---------- Calculate ----------
data = df.copy()
for col in ["Open","High","Low","Close"]:
    if col not in data.columns:
        st.error(f"Veride '{col}' kolonu bulunamadı.")
        st.stop()

data["EMA_Short"] = ema(data["Close"], ema_short)
data["EMA_Long"]  = ema(data["Close"], ema_long)
data["EMA_Trend"] = ema(data["Close"], ema_trend)
data["RSI"] = rsi(data["Close"], rsi_len)
data["BB_Mid"], data["BB_Up"], data["BB_Down"] = bollinger(data["Close"], bb_len, bb_std)
data["ATR"] = atr(data, atr_len)

bull = data["EMA_Short"] > data["EMA_Long"]
buy  = bull & bull.ne(bull.shift(1))
sell = (~bull) & bull.ne(bull.shift(1))
data["BUY"], data["SELL"] = buy, sell

last_buy = data.index[data["BUY"]].max() if data["BUY"].any() else None
entry_price, stop_price, tp_levels = None, None, []

def _hline(x, y, name, color):
    return go.Scatter(x=x, y=[y]*len(x), mode="lines", name=name,
                      line=dict(width=1.5, dash="dash", color=color), hoverinfo="skip")

if last_buy is not None and pd.notnull(last_buy):
    entry_price = float(data.loc[last_buy, "Close"])
    atr_val = float(data.loc[last_buy, "ATR"])
    if stop_mode == "ATR x K":
        stop_price = entry_price - atr_k * atr_val
        risk = entry_price - stop_price
    else:
        stop_price = entry_price * (1 - stop_pct/100.0)
        risk = entry_price - stop_price
    tp_levels = [("TP1", entry_price + 1*risk),
                 ("TP2", entry_price + 2*risk),
                 ("TP3", entry_price + 3*risk)]

# ---------- Plot ----------
fig = sp.make_subplots(rows=3, cols=1, shared_xaxes=True,
                       row_heights=[0.6,0.2,0.2], vertical_spacing=0.04)
fig.add_candlestick(x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"], name="OHLC", row=1, col=1)
fig.add_scatter(x=data.index, y=data["EMA_Short"], name=f"EMA {ema_short}", row=1, col=1)
fig.add_scatter(x=data.index, y=data["EMA_Long"],  name=f"EMA {ema_long}",  row=1, col=1)
fig.add_scatter(x=data.index, y=data["EMA_Trend"], name=f"EMA {ema_trend}", row=1, col=1)
fig.add_scatter(x=data.index, y=data["BB_Up"],   name="BB Upper", row=1, col=1)
fig.add_scatter(x=data.index, y=data["BB_Mid"],  name="BB Mid",   row=1, col=1)
fig.add_scatter(x=data.index, y=data["BB_Down"], name="BB Lower", row=1, col=1)
fig.add_scatter(x=data.index[buy], y=data["Close"][buy], mode="markers", marker_symbol="triangle-up", marker_size=12, name="BUY", row=1, col=1)
fig.add_scatter(x=data.index[sell], y=data["Close"][sell], mode="markers", marker_symbol="triangle-down", marker_size=12, name="SELL", row=1, col=1)

if entry_price is not None:
    fig.add_trace(_hline(data.index, entry_price, f"ENTRY {entry_price:.4f}", "blue"), row=1, col=1)
    fig.add_trace(_hline(data.index, stop_price, f"STOP {stop_price:.4f}", "red"), row=1, col=1)
    for name, lvl in tp_levels:
        fig.add_trace(_hline(data.index, lvl, f"{name} {lvl:.4f}", "green"), row=1, col=1)

fig.add_scatter(x=data.index, y=data["RSI"], name=f"RSI {rsi_len}", row=3, col=1)
fig.add_trace(_hline(data.index, 70, "RSI 70", "red"), row=3, col=1)
fig.add_trace(_hline(data.index, 30, "RSI 30", "green"), row=3, col=1)

fig.update_layout(height=900, xaxis_rangeslider_visible=False, legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

# ---------- Info ----------
col1, col2 = st.columns([2,1])
latest = data.iloc[-1]
with col2:
    st.subheader("🔔 Sinyal Durumu")
    st.write(f"Son Kapanış: **{fmt(latest.get('Close'))}**")
    st.write(f"EMA{ema_short}: **{fmt(latest.get('EMA_Short'))}**, EMA{ema_long}: **{fmt(latest.get('EMA_Long'))}**")
    st.write(f"RSI: **{fmt(latest.get('RSI'),2)}**, ATR: **{fmt(latest.get('ATR'))}**")
    if entry_price is not None:
        try:
            st.success(f"Son BUY: {pd.to_datetime(last_buy).strftime('%Y-%m-%d %H:%M:%S')} @ {entry_price:.4f}")
        except Exception:
            st.success(f"Son BUY: {str(last_buy)} @ {entry_price:.4f}")
    else:
        st.info("Aktif alım sinyali yok.")
with col1:
    st.subheader("🧾 İşlem Listesi")
    trades = data.loc[data["BUY"] | data["SELL"], ["Close","BUY","SELL"]].copy()
    if not trades.empty:
        trades["Type"] = np.where(trades["BUY"], "BUY", "SELL")
        st.dataframe(trades[["Close","Type"]].tail(10))
    else:
        st.write("Son dönemde işlem sinyali yok.")

st.markdown("---")
st.caption("CSV kolonlarınız 'Date,Open,High,Low,Close,Volume' olmalı. Aynı isimli kolonlar varsa otomatik çözümlenir. TP/SL çizgileri son BUY sinyali baz alınır.")
