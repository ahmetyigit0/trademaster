
# app_with_tp_sl.py
# Run: streamlit run app_with_tp_sl.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.subplots as sp
import plotly.graph_objects as go

st.set_page_config(page_title="Strategy + TP/SL Signals", layout="wide")

st.title("📈 Strategy + TP/SL Signals")
st.caption("Eğitim amaçlıdır; yatırım tavsiyesi değildir.")

# ---------------- Data Loading ----------------
@st.cache_data(show_spinner=True, ttl=1800)
def load_yf(ticker: str, period: str = "1y", interval: str = "1d"):
    try:
        import yfinance as yf
    except Exception:
        st.error("yfinance kurulu değil. requirements.txt ile yükleyin veya CSV kullanın.")
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
    if up is not None:
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

# ---------------- Indicators ----------------
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

# Sidebar params
st.sidebar.header("⚙️ Parametreler")
ema_short = st.sidebar.number_input("EMA Kısa", 3, 50, 9)
ema_long  = st.sidebar.number_input("EMA Uzun", 10, 300, 21)
ema_trend = st.sidebar.number_input("EMA Trend", 50, 400, 200, step=10)

m_fast = st.sidebar.number_input("MACD Hızlı", 3, 50, 12)
m_slow = st.sidebar.number_input("MACD Yavaş", 6, 200, 26)
m_sig  = st.sidebar.number_input("MACD Sinyal", 3, 50, 9)

rsi_len = st.sidebar.number_input("RSI Periyot", 5, 50, 14)
bb_len  = st.sidebar.number_input("BB Periyot", 5, 100, 20)
bb_std  = st.sidebar.slider("BB Std", 1.0, 3.5, 2.0, 0.1)

atr_len = st.sidebar.number_input("ATR Periyot", 5, 50, 14)

st.sidebar.header("🎯 Sinyal Kuralları")
combine = st.sidebar.selectbox("Sinyal modu", ["EMA Crossover", "EMA + MACD", "EMA + MACD + RSI"], index=2)
rsi_min = st.sidebar.slider("RSI alt sınır (alım için)", 10, 60, 40)
rsi_max = st.sidebar.slider("RSI üst sınır (alım için)", 40, 90, 70)

st.sidebar.header("🛡️ Stop & TP")
stop_mode = st.sidebar.selectbox("Stop tipi", ["ATR x K", "Sabit %"], index=0)
atr_k     = st.sidebar.slider("ATR çarpanı (K)", 0.5, 5.0, 2.0, 0.1)
stop_pct  = st.sidebar.slider("Sabit zarar %", 0.5, 10.0, 2.0, 0.1)

tp_mode   = st.sidebar.selectbox("TP seviyesi", ["R-multiple", "Sabit %"], index=0)
tp1_r     = st.sidebar.slider("TP1 (R)", 0.5, 5.0, 1.0, 0.1)
tp2_r     = st.sidebar.slider("TP2 (R)", 1.0, 8.0, 2.0, 0.1)
tp3_r     = st.sidebar.slider("TP3 (R)", 1.5, 12.0, 3.0, 0.1)
tp_pct    = st.sidebar.slider("TP %", 0.5, 50.0, 5.0, 0.5)

# Calculate indicators
data = df.copy()
data["EMA_Short"] = ema(data["Close"], ema_short)
data["EMA_Long"]  = ema(data["Close"], ema_long)
data["EMA_Trend"] = ema(data["Close"], ema_trend)

data["MACD"], data["MACD_Signal"], data["MACD_Hist"] = macd(data["Close"], m_fast, m_slow, m_sig)
data["RSI"] = rsi(data["Close"], rsi_len)
data["BB_Mid"], data["BB_Up"], data["BB_Down"] = bollinger(data["Close"], bb_len, bb_std)
data["ATR"] = atr(data, atr_len)

# Entry/exit logic
bull = data["EMA_Short"] > data["EMA_Long"]
macd_cross_up = (data["MACD"] > data["MACD_Signal"]) & (data["MACD"].shift(1) <= data["MACD_Signal"].shift(1))
rsi_ok = (data["RSI"] >= rsi_min) & (data["RSI"] <= rsi_max)

if combine == "EMA Crossover":
    buy = bull & bull.ne(bull.shift(1))
elif combine == "EMA + MACD":
    buy = bull & macd_cross_up
else:
    buy = bull & macd_cross_up & rsi_ok

sell = (~bull) & bull.ne(bull.shift(1))

data["BUY"] = buy
data["SELL"] = sell

# Build latest signal and TP/SL lines based on the last BUY
last_buy_idx = data.index[data["BUY"]].max() if data["BUY"].any() else None
tp_lines = []
sl_line = None
entry_price = None
if last_buy_idx is not None:
    entry_price = float(data.loc[last_buy_idx, "Close"])
    atr_val = float(data.loc[last_buy_idx, "ATR"])
    if stop_mode == "ATR x K":
        stop_price = entry_price - atr_k * atr_val
        risk_per_unit = entry_price - stop_price
    else:  # Sabit %
        stop_price = entry_price * (1 - stop_pct/100.0)
        risk_per_unit = entry_price - stop_price

    sl_line = stop_price

    if tp_mode == "R-multiple":
        tp1 = entry_price + tp1_r * risk_per_unit
        tp2 = entry_price + tp2_r * risk_per_unit
        tp3 = entry_price + tp3_r * risk_per_unit
    else:
        tp1 = entry_price * (1 + tp_pct/100.0)
        tp2 = entry_price * (1 + 2*tp_pct/100.0)
        tp3 = entry_price * (1 + 3*tp_pct/100.0)

    tp_lines = [("TP1", tp1), ("TP2", tp2), ("TP3", tp3)]

# Chart
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

fig.add_bar(x=data.index, y=data["MACD_Hist"], name="MACD Hist", row=2, col=1)
fig.add_scatter(x=data.index, y=data["MACD"], name="MACD", row=2, col=1)
fig.add_scatter(x=data.index, y=data["MACD_Signal"], name="Signal", row=2, col=1)

fig.add_scatter(x=data.index, y=data["RSI"], name=f"RSI {rsi_len}", row=3, col=1)
fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)

# Draw TP/SL horizontal lines for the latest BUY
if entry_price is not None:
    fig.add_hline(y=entry_price, line_color="blue", line_dash="dot", annotation_text=f"ENTRY {entry_price:.4f}", row=1, col=1)
    fig.add_hline(y=sl_line, line_color="red", line_dash="dash", annotation_text=f"STOP {sl_line:.4f}", row=1, col=1)
    for label, lvl in tp_lines:
        fig.add_hline(y=lvl, line_color="green", line_dash="dash", annotation_text=f"{label} {lvl:.4f}", row=1, col=1)

fig.update_layout(height=900, xaxis_rangeslider_visible=False, legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

# Right panel: latest status
col1, col2 = st.columns([2,1])
with col2:
    st.subheader("🔔 Sinyal Durumu")
    latest = data.iloc[-1]
    st.write(f"Son Kapanış: **{latest['Close']:.6f}**")
    st.write(f"EMA{ema_short}: **{latest['EMA_Short']:.6f}**  |  EMA{ema_long}: **{latest['EMA_Long']:.6f}**")
    st.write(f"RSI: **{latest['RSI']:.2f}**  |  ATR: **{latest['ATR']:.6f}**")
    if entry_price is None:
        st.warning("Aktif LONG sinyali bulunamadı (son BUY yok). Parametreleri değiştir veya veri aralığını artır.")
    else:
        st.success(f"Son LONG sinyali: **{last_buy_idx.strftime('%Y-%m-%d %H:%M:%S')}** @ **{entry_price:.4f}**")
        # Check TP/SL status with the latest candle
        low = float(latest["Low"]); high = float(latest["High"])
        hit = []
        if sl_line is not None and low <= sl_line:
            hit.append(("STOP", sl_line))
        for label, lvl in tp_lines:
            if high >= lvl:
                hit.append((label, lvl))
        if hit:
            st.write("🎯 **Gerçekleşen hedefler:**")
            for name, p in hit:
                st.write(f"- {name}: **{p:.4f}**")
        else:
            st.info("Henüz TP/SL gerçekleşmedi.")

with col1:
    st.subheader("🧾 İşlem İşaretleri")
    sigs = data.loc[data["BUY"] | data["SELL"], ["Close","BUY","SELL"]].copy()
    sigs["Type"] = np.where(sigs["BUY"], "BUY", "SELL")
    sigs = sigs[["Close","Type"]]
    st.dataframe(sigs.tail(15))

st.markdown("---")
st.caption("Not: TP/SL çizgileri en **son** BUY sinyali baz alınarak hesaplanır. Stop tipi ATR×K veya sabit yüzde olarak ayarlanabilir; TP ise R-multiple veya sabit yüzde.")
