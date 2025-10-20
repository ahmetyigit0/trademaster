
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="TradeMaster", layout="wide")
st.title("🧭 TradeMaster")
st.caption("Eğitim amaçlıdır; yatırım tavsiyesi değildir.")

# =========================
# Helpers
# =========================
@st.cache_data(ttl=1800, show_spinner=True)
def load_yf(ticker: str, period="6mo", interval="1d"):
    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=False)
        if df is None or df.empty:
            return None
        df = df.rename(columns=str.title)
        df.index.name = "Date"
        return df
    except Exception:
        return None

def to_num(s):
    s = s.astype(str).str.replace(",", "", regex=False).str.replace(" ", "", regex=False)
    return pd.to_numeric(s, errors="coerce")

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

def fib_levels(a, b):
    return {
        "0.0": b,
        "0.236": b - (b - a) * 0.236,
        "0.382": b - (b - a) * 0.382,
        "0.5": b - (b - a) * 0.5,
        "0.618": b - (b - a) * 0.618,
        "0.786": b - (b - a) * 0.786,
        "1.0": a,
    }

# =========================
# Sidebar — Data & Settings
# =========================
st.sidebar.header("📥 Veri Kaynağı")

src = st.sidebar.radio("Kaynak", ["YFinance (internet)", "CSV Yükle"], index=0)

# --- Crypto suggestions ---
COMMON_TICKERS = [
    "BTC-USD","ETH-USD","BNB-USD","SOL-USD","XRP-USD","ADA-USD","DOGE-USD","TRX-USD","TON-USD","AVAX-USD",
    "LINK-USD","DOT-USD","MATIC-USD","LTC-USD","APT-USD","ATOM-USD","FIL-USD","XLM-USD","ICP-USD","ETC-USD",
    "HBAR-USD","OKB-USD","NEAR-USD","ALGO-USD","INJ-USD","SUI-USD","AR-USD","AAVE-USD","OP-USD","THETA-USD",
    "RUNE-USD","FTM-USD","GRT-USD","EGLD-USD","KAS-USD","SAND-USD","MANA-USD","APE-USD","IMX-USD","SEI-USD",
    "PEPE-USD","BONK-USD","WIF-USD","PYTH-USD","JUP-USD","JASMY-USD","ORDI-USD","TAO-USD","TIA-USD","ENA-USD"
]

# Kripto Değer (search box)
query = st.sidebar.text_input("🔎 Kripto Değer", value="THETA-USD", help="Örn: THETA-USD, BTC-USD, ETH-USD")
# Suggestions based on query
if query:
    q = query.lower()
    suggestions = [t for t in COMMON_TICKERS if q in t.lower()][:15]
else:
    suggestions = COMMON_TICKERS[:15]

sel = st.sidebar.selectbox("Öneriler", options=suggestions, index=min( suggestions.index("THETA-USD") if "THETA-USD" in suggestions else 0, len(suggestions)-1 ))

period = st.sidebar.selectbox("Periyot", ["3mo","6mo","1y","2y","5y"], index=1)
interval = st.sidebar.selectbox("Zaman Dilimi", ["1d","4h","1h"], index=0)

uploaded = None
if src == "CSV Yükle":
    uploaded = st.sidebar.file_uploader("CSV (Date,Open,High,Low,Close,Volume)", type=["csv"])

st.sidebar.header("⚙️ Strateji Ayarları")
ema_short = st.sidebar.number_input("EMA Kısa", 3, 50, 9)
ema_long  = st.sidebar.number_input("EMA Uzun", 10, 300, 21)
ema_trend = st.sidebar.number_input("EMA Trend (uzun)", 50, 400, 200, step=10)

rsi_len   = st.sidebar.number_input("RSI Periyot", 5, 50, 14)
rsi_buy_min = st.sidebar.slider("RSI Alım Alt Sınır", 10, 60, 40)
rsi_buy_max = st.sidebar.slider("RSI Alım Üst Sınır", 40, 90, 70)

macd_fast = st.sidebar.number_input("MACD Hızlı", 3, 50, 12)
macd_slow = st.sidebar.number_input("MACD Yavaş", 6, 200, 26)
macd_sig  = st.sidebar.number_input("MACD Sinyal", 3, 50, 9)

bb_len    = st.sidebar.number_input("Bollinger Periyot", 5, 100, 20)
bb_std    = st.sidebar.slider("Bollinger Std", 1.0, 3.5, 2.0, 0.1)

fib_lookback = st.sidebar.number_input("Fibonacci Lookback (gün)", 20, 400, 120, 5)

st.sidebar.header("💰 Risk / Pozisyon")
equity      = st.sidebar.number_input("Sermaye (USD)", min_value=100.0, value=10000.0, step=100.0)
risk_pct    = st.sidebar.slider("İşlem başına risk (%)", 0.2, 5.0, 1.0, 0.1)
max_alloc   = st.sidebar.slider("Maks. Pozisyon (%)", 1.0, 100.0, 20.0, 1.0)
stop_mode   = st.sidebar.selectbox("Stop Tipi", ["ATR x K","Sabit %"], index=0)
atr_k       = st.sidebar.slider("ATR çarpanı", 0.5, 5.0, 2.0, 0.1)
stop_pct    = st.sidebar.slider("Sabit Zarar %", 0.5, 20.0, 3.0, 0.5)
tp_mode     = st.sidebar.selectbox("TP modu", ["R-multiple","Sabit %"], index=0)
tp1_r       = st.sidebar.slider("TP1", 0.5, 5.0, 1.0, 0.1)
tp2_r       = st.sidebar.slider("TP2", 0.5, 10.0, 2.0, 0.1)
tp3_r       = st.sidebar.slider("TP3", 0.5, 15.0, 3.0, 0.1)
tp_pct      = st.sidebar.slider("TP %", 0.5, 50.0, 5.0, 0.5)

# =========================
# Load data
# =========================
if src == "YFinance (internet)":
    df = load_yf(sel if query == "" else (query if query.upper().endswith("-USD") else sel), period, interval)
else:
    df = None
    if uploaded:
        try:
            raw = pd.read_csv(uploaded)
            # normalize columns
            cols = {c.lower(): c for c in raw.columns}
            need = ["date","open","high","low","close","volume"]
            if all(n in cols for n in need):
                df = raw[[cols[n] for n in need]].copy()
                df.columns = ["Date","Open","High","Low","Close","Volume"]
            else:
                df = raw.copy()
                df.columns = [c.title() for c in df.columns]
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
            for c in ["Open","High","Low","Close","Volume"]:
                if c in df.columns:
                    df[c] = to_num(df[c])
        except Exception as e:
            st.error(f"CSV okunamadı: {e}")

if df is None or df.empty:
    st.warning("Veri alınamadı. Farklı period/interval deneyin veya CSV yükleyin.")
    st.stop()

# Ensure numeric
for c in ["Open","High","Low","Close","Volume"]:
    if c in df.columns:
        df[c] = to_num(df[c])

have_ohlc = all(c in df.columns for c in ["Open","High","Low"])
data = df.copy()

# Indicators
data["EMA_Short"] = ema(data["Close"], ema_short)
data["EMA_Long"]  = ema(data["Close"], ema_long)
data["EMA_Trend"] = ema(data["Close"], ema_trend)
data["RSI"] = rsi(data["Close"], rsi_len)
data["MACD"], data["MACD_Signal"], data["MACD_Hist"] = macd(data["Close"], macd_fast, macd_slow, macd_sig)
data["BB_Mid"], data["BB_Up"], data["BB_Down"] = bollinger(data["Close"], bb_len, bb_std)
if have_ohlc:
    data["ATR"] = atr(data, 14)
else:
    data["ATR"] = np.nan

# Fib from lookback window
recent = data.tail(int(fib_lookback))
swing_high = float(recent["High"].max()) if "High" in recent else float(recent["Close"].max())
swing_low  = float(recent["Low"].min())  if "Low" in recent else float(recent["Close"].min())
trend_up = recent["Close"].iloc[-1] > (swing_low + swing_high) / 2
fa, fb = (swing_low, swing_high) if trend_up else (swing_high, swing_low)
fibs = fib_levels(fa, fb)

# Signal rules
bull = data["EMA_Short"] > data["EMA_Long"]
macd_cross_up = (data["MACD"] > data["MACD_Signal"]) & (data["MACD"].shift(1) <= data["MACD_Signal"].shift(1))
rsi_ok = (data["RSI"] >= rsi_buy_min) & (data["RSI"] <= rsi_buy_max)

# composite signal
buy_now = bull.iloc[-1] and (macd_cross_up.iloc[-1] or rsi_ok.iloc[-1])
sell_now = (not bull.iloc[-1]) and (data["MACD"].iloc[-1] < data["MACD_Signal"].iloc[-1])

last_close = float(data["Close"].iloc[-1])
entry_price = last_close  # Plan: piyasa/limit, kullanıcı dilerse değiştirir
atr_val = float(data["ATR"].iloc[-1]) if np.isfinite(data["ATR"].iloc[-1]) else np.nan

# Stop & TP
if stop_mode == "ATR x K" and np.isfinite(atr_val):
    stop_price = entry_price - atr_k * atr_val
    per_unit_risk = max(entry_price - stop_price, 1e-9)
else:
    stop_price = entry_price * (1 - stop_pct/100.0)
    per_unit_risk = max(entry_price - stop_price, 1e-9)

if tp_mode == "R-multiple":
    tp1 = entry_price + tp1_r * per_unit_risk
    tp2 = entry_price + tp2_r * per_unit_risk
    tp3 = entry_price + tp3_r * per_unit_risk
else:
    tp1 = entry_price * (1 + tp_pct/100.0)
    tp2 = entry_price * (1 + 2*tp_pct/100.0)
    tp3 = entry_price * (1 + 3*tp_pct/100.0)

# Position sizing
risk_amount = equity * (risk_pct/100.0)
qty = risk_amount / per_unit_risk
position_value = qty * entry_price
max_value = equity * (max_alloc/100.0)
if position_value > max_value:
    scale = max_value / max(position_value, 1e-9)
    qty *= scale
    position_value = qty * entry_price

# Decision text
if buy_now:
    headline = "✅ SİNYAL: AL (long)"
elif sell_now:
    headline = "❌ SİNYAL: SAT / LONG kapat"
else:
    headline = "⏸ SİNYAL: BEKLE"

# =========================
# Output (no charts)
# =========================
st.subheader("📌 Özet")
st.markdown(f"""
**Kripto:** `{sel if query=='' else query}`  
**Son Fiyat:** **{fmt(last_close)}**  
**Durum:** **{headline}**
""")

st.subheader("🎯 Sinyal (Öneri)")
st.markdown(f"""
- **Giriş Fiyatı:** **{fmt(entry_price)}**
- **Önerilen Miktar:** ~ **{qty:.4f}** birim (≈ **${position_value:,.2f}**)
- **Stop:** **{fmt(stop_price)}**
- **TP1 / TP2 / TP3:** **{fmt(tp1)}** / **{fmt(tp2)}** / **{fmt(tp3)}**
- **Risk:** Sermayenin %{risk_pct:.1f}’i (max pozisyon %{max_alloc:.0f})
""")

with st.expander("🔍 Detaylar (Gerekçeler)"):
    bb_pos = "alt band yakınında" if last_close <= recent["BB_Down"].iloc[-1] else ("üst band yakınında" if last_close >= recent["BB_Up"].iloc[-1] else "band içinde")
    ema_state = "EMA kısa > EMA uzun (trend ↑)" if bull.iloc[-1] else "EMA kısa < EMA uzun (trend ↓)"
    macd_state = "MACD signal üstü" if data["MACD"].iloc[-1] > data["MACD_Signal"].iloc[-1] else "MACD signal altı"
    macd_x = "Son bar kesişim ↑ var" if macd_cross_up.iloc[-1] else "Son bar kesişim yok"
    rsi_state = f"RSI {data['RSI'].iloc[-1]:.2f} ({'alım için uygun' if rsi_ok.iloc[-1] else 'nötr/aşırı'})"
    fib_side = "yukarı trend fib seti" if trend_up else "aşağı trend fib seti"
    fib_near = min(fibs.items(), key=lambda kv: abs(kv[1]-last_close))
    st.write(f"- **Bollinger:** {bb_pos}")
    st.write(f"- **EMA:** {ema_state}")
    st.write(f"- **MACD:** {macd_state} | {macd_x}")
    st.write(f"- **RSI:** {rsi_state}")
    st.write(f"- **Fibonacci:** {fib_side}; fiyata en yakın seviye: **{fib_near[0]} = {fmt(fib_near[1])}**")
    if np.isfinite(atr_val):
        st.write(f"- **ATR(14):** {fmt(atr_val)} (stop için ATR×{atr_k})")
    else:
        st.write(f"- **ATR:** — (OHLC eksik; stop sabit % ile)")

st.markdown("---")
st.caption("Not: Bu çıktı yalnızca eğitim amaçlıdır. İşlem kararı risk içerir; finansal tavsiye değildir.")
