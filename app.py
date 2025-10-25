# app.py
# ───────────────────────────────────────────────
# 4H • EMA50 Trend • RSI14 • S/R Bölgeleri
# Retest + Wick reddi • 1D EMA200 Rejim filtresi
# Cooldown • Ayarlanabilir R/R • 90g Backtest
# 🎬 Backtest zaman çizelgesi (aktif işlemler)
# Şifre: efe
# ───────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from dataclasses import dataclass

st.set_page_config(page_title="4H Pro TA", layout="wide")

# ────────────────────────────────
# Şifre
# ────────────────────────────────
def check_password():
    def entered():
        if st.session_state.get("password") == "efe":
            st.session_state["ok"] = True
            del st.session_state["password"]
        else:
            st.session_state["ok"] = False
    if "ok" not in st.session_state:
        st.text_input("Şifre", type="password", key="password", on_change=entered)
        st.stop()
    elif not st.session_state["ok"]:
        st.text_input("Şifre", type="password", key="password", on_change=entered)
        st.error("❌ Yanlış şifre")
        st.stop()
check_password()

# ────────────────────────────────
# Yardımcılar
# ────────────────────────────────
@st.cache_data
def get_4h(symbol, days):
    if "-" not in symbol:
        symbol += "-USD"
    df = yf.download(symbol, period=f"{days}d", interval="4h", progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    return df.dropna()

@st.cache_data
def get_1d(symbol, days=400):
    if "-" not in symbol:
        symbol += "-USD"
    df = yf.download(symbol, period=f"{days}d", interval="1d", progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.dropna().copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")]
    return df

def compute_indicators(df):
    df["EMA"] = df["Close"].ewm(span=50, adjust=False).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss.replace(0, np.nan))
    df["RSI"] = 100 - (100 / (1 + rs))
    tr = pd.concat([
        (df["High"] - df["Low"]),
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    return df.dropna()

def compute_regime(df1d):
    """1D EMA200 rejim filtresi"""
    if df1d is None or df1d.empty or len(df1d) < 210:
        return pd.Series(dtype="object")

    close = df1d["Close"].astype(float)
    ema200 = close.ewm(span=200, adjust=False).mean()
    slope = ema200 > ema200.shift(1)

    up = ((close > ema200) & slope).to_numpy().ravel()
    down = ((close < ema200) & (~slope)).to_numpy().ravel()
    arr = np.where(up, "UP", np.where(down, "DOWN", "RANGE")).ravel()
    return pd.Series(arr, index=df1d.index)

# ────────────────────────────────
# S/R bölgeleri
# ────────────────────────────────
@dataclass
class Zone:
    low: float
    high: float
    kind: str  # support/resistance
    score: int

def find_zones(df):
    cur = df["Close"].iloc[-1]
    atr = df["ATR"].iloc[-1]
    bins = np.arange(df["Low"].min(), df["High"].max(), atr)
    zones = []
    for b in bins:
        c = df[(df["Low"] < b + atr) & (df["High"] > b)]
        if len(c) > 5:
            if b < cur:
                zones.append(Zone(b, b + atr, "support", len(c)))
            else:
                zones.append(Zone(b, b + atr, "resistance", len(c)))
    s = sorted([z for z in zones if z.kind == "support"], key=lambda z: z.score, reverse=True)[:3]
    r = sorted([z for z in zones if z.kind == "resistance"], key=lambda z: z.score, reverse=True)[:3]
    return s, r

# ────────────────────────────────
# Sinyal
# ────────────────────────────────
@dataclass
class Signal:
    typ: str
    entry: float
    sl: float
    tp1: float
    tp2: float

def make_signal(df, s_zones, r_zones, rr, regime):
    c = df["Close"].iloc[-1]
    ema = df["EMA"].iloc[-1]
    rsi = df["RSI"].iloc[-1]
    atr = df["ATR"].iloc[-1]
    trend = "bull" if c > ema else "bear"
    sigs = []
    # long
    if trend == "bull" and regime in ("UP", "ANY") and s_zones:
        s = s_zones[0]
        entry = s.high
        sl = s.low - 0.25 * atr
        tp1 = entry + rr * 0.5 * (entry - sl)
        tp2 = entry + rr * (entry - sl)
        sigs.append(Signal("BUY", entry, sl, tp1, tp2))
    # short
    if trend == "bear" and regime in ("DOWN", "ANY") and r_zones:
        r = r_zones[0]
        entry = r.low
        sl = r.high + 0.25 * atr
        tp1 = entry - rr * 0.5 * (sl - entry)
        tp2 = entry - rr * (sl - entry)
        sigs.append(Signal("SELL", entry, sl, tp1, tp2))
    return sigs

# ────────────────────────────────
# Grafik
# ────────────────────────────────
def chart(df, s_zones, r_zones, sigs, sym):
    view = df.tail(20)
    f = go.Figure()
    for z in s_zones:
        f.add_hrect(y0=z.low, y1=z.high, fillcolor="rgba(0,255,0,0.1)", line=dict(color="green"))
    for z in r_zones:
        f.add_hrect(y0=z.low, y1=z.high, fillcolor="rgba(255,0,0,0.1)", line=dict(color="red"))
    f.add_trace(go.Candlestick(
        x=view.index, open=view["Open"], high=view["High"], low=view["Low"], close=view["Close"],
        increasing_line_color="#00C805", increasing_fillcolor="#00C805",
        decreasing_line_color="#FF4D4D", decreasing_fillcolor="#FF4D4D"
    ))
    if "EMA" in view:
        f.add_trace(go.Scatter(x=view.index, y=view["EMA"], line=dict(color="orange"), name="EMA"))
    if sigs:
        s = sigs[0]
        f.add_hline(y=s.entry, line_color="#00FFFF", annotation_text="Entry")
        f.add_hline(y=s.sl, line_color="#FF4444", annotation_text="SL")
        f.add_hline(y=s.tp2, line_color="#00FF00", annotation_text="TP")
    f.update_layout(title=f"{sym} • 4H", xaxis_rangeslider_visible=False,
                    plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font=dict(color="white"))
    return f

# ────────────────────────────────
# Backtest
# ────────────────────────────────
@dataclass
class Trade:
    open_time: pd.Timestamp
    side: str
    entry: float
    exit_time: pd.Timestamp
    pnl: float
    reason: str = ""

def backtest(df, rr):
    bal = 10000
    eq = [bal]
    trades = []
    for i in range(100, len(df) - 2):
        s_z, r_z = find_zones(df.iloc[:i])
        sigs = make_signal(df.iloc[:i], s_z, r_z, rr, "ANY")
        if not sigs:
            eq.append(bal)
            continue
        s = sigs[0]
        side = s.typ
        entry = df["Open"].iloc[i + 1]
        sl, tp = s.sl, s.tp2
        reason = ""
        for j in range(i + 1, min(i + 40, len(df))):
            hi, lo = df["High"].iloc[j], df["Low"].iloc[j]
            if side == "BUY" and lo <= sl:
                pnl = (sl - entry)
                reason = "SL"
            elif side == "BUY" and hi >= tp:
                pnl = (tp - entry)
                reason = "TP"
            elif side == "SELL" and hi >= sl:
                pnl = (entry - sl)
                reason = "SL"
            elif side == "SELL" and lo <= tp:
                pnl = (entry - tp)
                reason = "TP"
            else:
                continue
            bal += pnl * 10
            trades.append(Trade(df.index[i + 1], side, entry, df.index[j], pnl, reason))
            break
        eq.append(bal)
    eqdf = pd.DataFrame({"time": df.index[:len(eq)], "equity": eq})
    tdf = pd.DataFrame([t.__dict__ for t in trades])
    return eqdf, tdf

# ────────────────────────────────
# UI
# ────────────────────────────────
st.title("📈 4H Pro TA (Backtest'li)")

with st.sidebar:
    sym = st.text_input("Sembol", "BTC-USD")
    rr = st.slider("Min R/R", 1.0, 3.0, 1.8, 0.1)
    run_bt = st.button("Backtest (90g)")

df = get_4h(sym, 30)
if df.empty:
    st.stop()
df = compute_indicators(df)
d1 = get_1d(sym)
reg = compute_regime(d1)
regime = "ANY" if reg.empty else reg.iloc[-1]
s_z, r_z = find_zones(df)
sigs = make_signal(df, s_z, r_z, rr, regime)
st.plotly_chart(chart(df, s_z, r_z, sigs, sym), use_container_width=True)

# ─────────────── backtest
if run_bt:
    st.header("📊 Backtest 90g")
    df2 = get_4h(sym, 90)
    df2 = compute_indicators(df2)
    eqdf, tdf = backtest(df2, rr)
    st.metric("Toplam İşlem", len(tdf))
    st.line_chart(eqdf.set_index("time")["equity"], height=300)

    # 🎬 zaman çizelgesi
    st.subheader("🎬 Zaman Çizelgesi")
    maxbar = len(eqdf)
    bar = st.slider("Bar seç", 0, maxbar - 1, maxbar - 1)
    curtime = eqdf["time"].iloc[bar]
    cureq = eqdf["equity"].iloc[bar]
    st.metric("Zaman", str(curtime))
    st.metric("Bakiye", f"${cureq:,.0f}")
    if not tdf.empty:
        ongoing = tdf[tdf["exit_time"] > curtime]
        closed = tdf[tdf["exit_time"] <= curtime]
        st.write(f"🟢 Aktif: {len(ongoing)} | 🔴 Kapalı: {len(closed)}")
        if len(ongoing) > 0:
            st.dataframe(ongoing, use_container_width=True)