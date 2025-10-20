
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# =========================
# App Meta
# =========================
BUILD_VERSION = "v6.0"
BUILD_TIME = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

st.set_page_config(page_title="TradeMaster", layout="wide")
st.title(f"🧭 TradeMaster  {BUILD_VERSION}")
st.caption(f"Eğitim amaçlıdır; yatırım tavsiyesi değildir. • Build: {BUILD_TIME}")

# =========================
# Quick cache clear
# =========================
def _clear_all_caches():
    try:
        st.cache_data.clear()
    except Exception:
        pass
    try:
        st.cache_resource.clear()
    except Exception:
        pass

with st.sidebar:
    if st.button("🔄 Cache Temizle ve Yenile"):
        _clear_all_caches()
        st.experimental_rerun()

# =========================
# Helpers
# =========================
@st.cache_data(ttl=1800, show_spinner=True)
def load_yf(ticker: str, period="1y", interval="1d"):
    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
        if df is None or df.empty:
            return None
        df = normalize_ohlc(df)
        df.index.name = "Date"
        return df
    except Exception:
        return None

def to_num(s):
    """Accept Series or DataFrame; always return numeric Series safely."""
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    try:
        s = s.astype(str)
    except Exception:
        s = s.apply(lambda x: str(x) if x is not None else "")
    s = (s.str.replace(",", "", regex=False)
           .str.replace(" ", "", regex=False)
           .replace({"": np.nan, "None": np.nan, "none": np.nan, "NaN": np.nan, "nan": np.nan}))
    return pd.to_numeric(s, errors="coerce")

def get_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a 1-D Series for a given column name even if duplicates exist."""
    if col in df.columns:
        s = df[col]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s
    return pd.Series(index=df.index, dtype=float)

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

def colored(text, kind="neutral"):

def _titleize_cols(cols):
    out = []
    for c in cols:
        try:
            out.append(str(c).title())
        except Exception:
            out.append(str(c))
    return out

def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Return a copy with stringified Title() columns and ensure Close exists (fallback to Adj Close).\"\"\"
    if df is None or df.empty:
        return df
    df = df.copy()
    try:
        df.columns = _titleize_cols(df.columns)
    except Exception:
        # Fallback: brute force mapping
        df.columns = [str(c) for c in df.columns]
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    return df

    if kind == "pos":
        color = "#0f9d58"; dot = "🟢"
    elif kind == "neg":
        color = "#d93025"; dot = "🔴"
    else:
        color = "#f29900"; dot = "🟠"
    return f"{dot} <span style='color:{color}; font-weight:600;'>{text}</span>"

@st.cache_data(ttl=1800)
def load_macro_series():
    """Load VIX, DXY, BTC dominance (best-effort), daily resolution for context."""
    vix = load_yf("^VIX", period="2y", interval="1d")
    dxy = load_yf("DX-Y.NYB", period="2y", interval="1d")
    if dxy is None or dxy.empty:
        dxy = load_yf("DX=F", period="2y", interval="1d")
    dom = None
    for t in ["BTC-DOM", "BTC.D", "BTCDOM-INDEX", "BTCDOM", "CRYPTOCAP:BTC.D"]:
        dom = load_yf(t, period="2y", interval="1d")
        if dom is not None and not dom.empty:
            break
    return vix, dxy, dom

def market_score(btc_trend_up: bool|None, btc_rsi: float|None, vix_last: float|None, dxy_last: float|None, dom_last: float|None):
    """Return 0-100 simple risk-on score based on heuristics."""
    score = 50.0
    # BTC trend & momentum
    if btc_trend_up is True: score += 15
    elif btc_trend_up is False: score -= 15
    if btc_rsi is not None and np.isfinite(btc_rsi):
        if btc_rsi >= 55: score += 10
        elif btc_rsi <= 45: score -= 10
    # VIX
    if vix_last is not None and np.isfinite(vix_last):
        if vix_last <= 15: score += 10
        elif vix_last >= 25: score -= 10
    # DXY
    if dxy_last is not None and np.isfinite(dxy_last):
        if dxy_last <= 100: score += 7
        elif dxy_last >= 105: score -= 7
    # Dominance (lower better for alts)
    if dom_last is not None and np.isfinite(dom_last):
        if dom_last <= 45: score += 8
        elif dom_last >= 55: score -= 8
    return float(np.clip(score, 0, 100))


def last_scalar(x):
    """Return a float scalar from Series/DataFrame/ndarray; np.nan on failure."""
    try:
        v = x.iloc[-1]
    except Exception:
        try:
            v = x[-1]
        except Exception:
            v = x
    if isinstance(v, (pd.Series, pd.DataFrame)):
        v = v.to_numpy().ravel()
    if isinstance(v, (list, tuple)):
        v = np.array(v)
    if isinstance(v, np.ndarray):
        v = v.ravel()[0] if v.size else np.nan
    try:
        return float(v)
    except Exception:
        return np.nan


def last_close_of(df_):
    try:
        return float(df_["Close"].iloc[-1])
    except Exception:
        return np.nan

# =========================
# Sidebar — Inputs (common across tabs)
# =========================
st.sidebar.header("📥 Veri Kaynağı")
src = st.sidebar.radio("Kaynak", ["YFinance (internet)", "CSV Yükle"], index=0,
                       help="YFinance: İnternetten fiyatları çeker. CSV: Kendi verinizi yükleyin.")
ticker = st.sidebar.text_input("🪙 Kripto Değer", value="THETA-USD",
                               help="Yahoo Finance formatı: BTC-USD, ETH-USD, THETA-USD vb.")

interval_preset = st.sidebar.selectbox(
    "Zaman Dilimi",
    ["Kısa (4h)", "Orta (1 gün)", "Uzun (1 hafta)"],
    index=1,
    help="Veri çözünürlüğü: kısa=4 saat, orta=günlük, uzun=haftalık."
)
if interval_preset.startswith("Kısa"):
    yf_interval, yf_period = "4h", "180d"
elif interval_preset.startswith("Orta"):
    yf_interval, yf_period = "1d", "1y"
else:
    yf_interval, yf_period = "1wk", "5y"

uploaded = st.sidebar.file_uploader("CSV (opsiyonel)", type=["csv"],
                                    help="Kolonlar: Date, Open, High, Low, Close, Volume") if src == "CSV Yükle" else None

st.sidebar.header("⚙️ Strateji Ayarları")
ema_short = st.sidebar.number_input("EMA Kısa", 3, 50, 9, help="Kısa vadeli trend ortalaması.")
ema_long  = st.sidebar.number_input("EMA Uzun", 10, 300, 21, help="Orta vadeli trend ortalaması.")
ema_trend = st.sidebar.number_input("EMA Trend (uzun)", 50, 400, 200, step=10, help="Uzun vadeli trend referansı.")
rsi_len   = st.sidebar.number_input("RSI Periyot", 5, 50, 14, help="Momentum göstergesi RSI'ın periyodu.")
# default: 35–60 band
rsi_buy_min = st.sidebar.slider("RSI Alım Alt Sınır", 10, 60, 35, help="RSI bu değerin üzerindeyse alım için daha uygun.")
rsi_buy_max = st.sidebar.slider("RSI Alım Üst Sınır", 40, 90, 60, help="RSI bu değerin altındaysa aşırı alım değildir.")
macd_fast = st.sidebar.number_input("MACD Hızlı", 3, 50, 12, help="MACD hızlı EMA periyodu.")
macd_slow = st.sidebar.number_input("MACD Yavaş", 6, 200, 26, help="MACD yavaş EMA periyodu.")
macd_sig  = st.sidebar.number_input("MACD Sinyal", 3, 50, 9, help="MACD sinyal çizgisi periyodu.")
bb_len    = st.sidebar.number_input("Bollinger Periyot", 5, 100, 20, help="Ortalama için periyot.")
bb_std    = st.sidebar.slider("Bollinger Std", 1.0, 3.5, 2.0, 0.1, help="Bant genişliği katsayısı.")
fib_lookback = st.sidebar.number_input("Fibonacci Lookback (gün)", 20, 400, 120, 5,
                                       help="Swing high/low’u bulmak için bakılacak son gün sayısı.")

st.sidebar.header("💰 Risk / Pozisyon")
equity      = st.sidebar.number_input("Sermaye (USD)", min_value=100.0, value=10000.0, step=100.0,
                                      help="Toplam portföy büyüklüğü.")
risk_pct    = st.sidebar.slider("İşlem başına risk (%)", 0.2, 5.0, 1.0, 0.1,
                                help="Tek işlemde riske edilecek sermaye yüzdesi.")
max_alloc   = st.sidebar.slider("Maks. Pozisyon (%)", 1.0, 100.0, 20.0, 1.0,
                                help="Tek işlem için maksimum portföy payı.")
stop_mode   = st.sidebar.selectbox("Stop Tipi", ["ATR x K","Sabit %"], index=0,
                                   help="ATR tabanlı dinamik stop veya sabit yüzde stop.")
atr_k       = st.sidebar.slider("ATR çarpanı", 0.5, 5.0, 2.0, 0.1,
                                help="ATR stop kullanıyorsanız çarpan. Örn: 2×ATR.")
stop_pct    = st.sidebar.slider("Sabit Zarar %", 0.5, 20.0, 3.0, 0.5,
                                help="Sabit yüzdesel stop (ATR devre dışıysa).")
tp_mode     = st.sidebar.selectbox("TP modu", ["R-multiple","Sabit %"], index=0,
                                   help="R-multiple: (TP=Entry + k×Risk) / Sabit %: fiyatın yüzdesi.")
tp1_r       = st.sidebar.slider("TP1 (R)", 0.5, 5.0, 1.0, 0.1, help="R-multiple modunda TP1 katsayısı.")
tp2_r       = st.sidebar.slider("TP2 (R)", 0.5, 10.0, 2.0, 0.1, help="R-multiple modunda TP2 katsayısı.")
tp3_r       = st.sidebar.slider("TP3 (R)", 0.5, 15.0, 3.0, 0.1, help="R-multiple modunda TP3 katsayısı.")
tp_pct      = st.sidebar.slider("TP %", 0.5, 50.0, 5.0, 0.5, help="Sabit yüzde TP (R-multiple devre dışı).")

# =========================
# Tabs
# =========================
tab_an, tab_guide, tab_watch, tab_regime, tab_risk, tab_bt, tab_corr, tab_news = st.tabs(
    ["📈 Analiz","📘 Rehber","📋 Watchlist","🧭 Rejim","🧮 Risk","🧪 Backtest","📈 Korelasyon","📰 Haberler"]
)

# ========== Common data loaders for multiple tabs ==========
def load_asset_and_macro():
    if src == "YFinance (internet)":
        df = load_yf(ticker, yf_period, yf_interval)
        btc_df = load_yf("BTC-USD", yf_period, yf_interval)
    else:
        df = None
        if uploaded:
            try:
                raw = pd.read_csv(uploaded)
                cols = {str(c).lower(): c for c in raw.columns}
                need = ["date","open","high","low","close","volume"]
                if all(n in cols for n in need):
                    df = raw[[cols[n] for n in need]].copy()
                    df.columns = ["Date","Open","High","Low","Close","Volume"]
                else:
                    df = raw.copy()
                    df.columns = [str(c).title() for c in df.columns]
                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                    df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
                for c in ["Open","High","Low","Close","Volume"]:
                    if c in df.columns:
                        df[c] = to_num(df[c])
            except Exception as e:
                st.error(f"CSV okunamadı: {e}")
        btc_df = load_yf("BTC-USD", "1y", "1d")
    vix_df, dxy_df, dom_df = load_macro_series()
    return df, btc_df, vix_df, dxy_df, dom_df

# ========== ANALIZ ==========
with tab_an:
    df, btc_df, vix_df, dxy_df, dom_df = load_asset_and_macro()

    if df is None or df.empty:
        st.warning("Veri alınamadı. Ticker'ı tam (ör. BTC-USD) yazdığınızdan emin olun veya CSV yükleyin.")
        st.stop()

    for c in ["Open","High","Low","Close","Volume"]:
        if c in df.columns:
            df[c] = to_num(df[c])

    have_ohlc = all(c in df.columns for c in ["Open","High","Low"])
    data = df.copy()

    # Indicators (asset)
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

    # BTC context indicators
    btc_ctx = {}
    if btc_df is not None and not btc_df.empty:
        b = normalize_ohlc(btc_df).copy()
        b["EMA_Short"] = ema(b["Close"], ema_short)
        b["EMA_Long"]  = ema(b["Close"], ema_long)
        b["RSI"] = rsi(b["Close"], rsi_len)
        btc_ctx["last"] = float(b["Close"].iloc[-1])
        btc_ctx["trend_up"] = bool((b["EMA_Short"] > b["EMA_Long"]).iloc[-1])
        btc_ctx["rsi"] = float(b["RSI"].iloc[-1])
    else:
        btc_ctx["last"] = np.nan
        btc_ctx["trend_up"] = None
        btc_ctx["rsi"] = np.nan

    vix_last = last_close_of(vix_df) if vix_df is not None else np.nan
    dxy_last = last_close_of(dxy_df) if dxy_df is not None else np.nan
    dom_last = last_close_of(dom_df) if dom_df is not None else np.nan

    # Fibonacci (safe 1-D)
    lb = max(int(fib_lookback), 20)
    recent = data.tail(lb)
    hi_series = get_series(recent, "High") if "High" in list(recent.columns) else get_series(recent, "Close")
    lo_series = get_series(recent, "Low")  if "Low"  in list(recent.columns) else get_series(recent, "Close")
    cl_series = get_series(recent, "Close")
    hi_series = pd.to_numeric(hi_series, errors="coerce").dropna()
    lo_series = pd.to_numeric(lo_series, errors="coerce").dropna()
    cl_series = pd.to_numeric(cl_series, errors="coerce").dropna()
    if hi_series.empty or lo_series.empty or cl_series.empty:
        swing_high = swing_low = last_close = np.nan
    else:
        swing_high = float(hi_series.max())
        swing_low  = float(lo_series.min())
        last_close = float(cl_series.iloc[-1])

    # Signals
    bull = data["EMA_Short"] > data["EMA_Long"]
    macd_cross_up = (data["MACD"] > data["MACD_Signal"]) & (data["MACD"].shift(1) <= data["MACD_Signal"].shift(1))
    rsi_ok = (data["RSI"] >= rsi_buy_min) & (data["RSI"] <= rsi_buy_max)
    buy_now = bool(bull.iloc[-1] and (macd_cross_up.iloc[-1] or rsi_ok.iloc[-1]))
    sell_now = bool((not bull.iloc[-1]) and (data["MACD"].iloc[-1] < data["MACD_Signal"].iloc[-1]))

    if not np.isfinite(last_close):
        last_close = float(data["Close"].iloc[-1])
    entry_price = last_close
    atr_val = float(data["ATR"].iloc[-1]) if np.isfinite(data["ATR"].iloc[-1]) else np.nan

    # Stop & TP (long baseline)
    if stop_mode == "ATR x K" and np.isfinite(atr_val):
        stop_price_long = entry_price - max(atr_val * atr_k, 1e-9)
    else:
        stop_price_long = entry_price * (1 - stop_pct/100.0)
    risk_long = max(entry_price - stop_price_long, 1e-9)

    if tp_mode == "R-multiple":
        tp1_long = entry_price + tp1_r * risk_long
        tp2_long = entry_price + tp2_r * risk_long
        tp3_long = entry_price + tp3_r * risk_long
    else:
        tp1_long = entry_price * (1 + tp_pct/100.0)
        tp2_long = entry_price * (1 + 2*tp_pct/100.0)
        tp3_long = entry_price * (1 + 3*tp_pct/100.0)

    # Position sizing
    risk_amount = equity * (risk_pct/100.0)
    qty = risk_amount / risk_long
    position_value = qty * entry_price
    max_value = equity * (max_alloc/100.0)
    if position_value > max_value:
        scale = max_value / max(position_value, 1e-9)
        qty *= scale
        position_value = qty * entry_price

    # Metrics for summary
    atr_pct = (atr_val / entry_price * 100.0) if np.isfinite(atr_val) and entry_price > 0 else np.nan
    stop_dist_pct = (entry_price - stop_price_long) / entry_price * 100.0 if entry_price > 0 else np.nan
    rr_tp1 = (tp1_long - entry_price) / (entry_price - stop_price_long) if (entry_price - stop_price_long) > 0 else np.nan
    pos_ratio_pct = (position_value / equity * 100.0) if equity > 0 else np.nan
    momentum = float(data["RSI"].iloc[-1]) - 50.0
    trend_txt = "🟢 Yükseliş" if bool(bull.iloc[-1]) else "🔴 Düşüş"

    # Headline
    if buy_now:
        headline = "✅ SİNYAL: AL (long)"
    elif sell_now:
        headline = "❌ SİNYAL: SAT / LONG kapat"
    else:
        headline = "⏸ SİNYAL: BEKLE"

    # Output
    st.subheader("📌 Özet")
    colA, colB, colC = st.columns([1.2,1,1])
    with colA:
        st.markdown(f"**Kripto:** `{ticker}`")
        st.markdown(f"**Son Fiyat:** **{fmt(entry_price)}**")
        st.markdown(f"**Durum:** **{headline}**")
    with colB:
        st.markdown(f"**Trend:** {trend_txt}")
        st.markdown(f"**Momentum (RSI−50):** {momentum:+.2f}")
        st.markdown(f"**Volatilite (ATR):** {fmt(atr_val)}  ({fmt(atr_pct,2)}%)")
    with colC:
        st.markdown(f"**R:R (TP1):** {fmt(rr_tp1,2)}")
        st.markdown(f"**Stop Mesafesi:** {fmt(stop_dist_pct,2)}%")
        st.markdown(f"**Pozisyon Oranı:** {fmt(pos_ratio_pct,2)}%")

    st.subheader("🎯 Sinyal (Öneri)")
    if buy_now:
        st.markdown(f"""
- **Giriş (Long):** **{fmt(entry_price)}**
- **Önerilen Miktar:** ~ **{qty:.4f}** birim (≈ **${position_value:,.2f}**)
- **Stop (Long):** **{fmt(stop_price_long)}**
- **TP1 / TP2 / TP3 (Long):** **{fmt(tp1_long)}** / **{fmt(tp2_long)}** / **{fmt(tp3_long)}**
- **Risk:** Sermayenin %{risk_pct:.1f}’i (max pozisyon %{max_alloc:.0f})
        """)
    elif sell_now:
        st.markdown(f"""
- **Aksiyon:** **Long kapat / short düşün**
- **Kılavuz Stop (short için):** **{fmt(entry_price + risk_long)}** _(örnek: long riskine simetrik)_
- **Not:** SAT sinyalinde long taraf TP seviyeleri **gösterilmez**.
        """)
    else:
        st.markdown("Şu anda belirgin bir al/sat sinyali yok; parametreleri veya zaman dilimini değiştirerek tekrar değerlendiriniz.")

    st.subheader("🧠 Gerekçeler")
    def line(text, kind="neutral"):
        st.markdown(colored(text, kind), unsafe_allow_html=True)

    if bool(bull.iloc[-1]):
        line("EMA kısa > EMA uzun (trend ↑)", "pos")
    else:
        line("EMA kısa < EMA uzun (trend ↓)", "neg")

    macd_now = float(data["MACD"].iloc[-1]); macd_sig_now = float(data["MACD_Signal"].iloc[-1])
    if macd_now > macd_sig_now and bool(macd_cross_up.iloc[-1]):
        line("MACD sinyal üstünde ve son bar kesişim ↑", "pos")
    elif macd_now > macd_sig_now:
        line("MACD sinyal üstünde (pozitif momentum)", "neutral")
    else:
        line("MACD sinyal altında (momentum zayıf)", "neg")

    rsi_now = float(data["RSI"].iloc[-1])
    if rsi_now < 30: line(f"RSI {rsi_now:.2f} (aşırı satım – tepki gelebilir, trend zayıf)", "neg")
    elif 30 <= rsi_now < 35: line(f"RSI {rsi_now:.2f} (dip; onay beklenmeli)", "neutral")
    elif 35 <= rsi_now <= 45: line(f"RSI {rsi_now:.2f} (alım bölgesi; onaylıysa olumlu)", "pos")
    elif 45 < rsi_now < 60: line(f"RSI {rsi_now:.2f} (nötr-olumlu)", "neutral")
    elif 60 <= rsi_now <= 70: line(f"RSI {rsi_now:.2f} (güçlü momentum)", "pos")
    else: line(f"RSI {rsi_now:.2f} (aşırı alım – dikkat)", "neg")

    if "BB_Up" in data and "BB_Down" in data and np.isfinite(entry_price):
        try:
            if entry_price <= data["BB_Down"].iloc[-1]: line("Fiyat alt banda yakın (tepki potansiyeli)", "pos")
            elif entry_price >= data["BB_Up"].iloc[-1]: line("Fiyat üst banda yakın (ısınma)", "neg")
            else: line("Fiyat bant içinde (nötr)", "neutral")
        except Exception:
            line("Bollinger hesaplanamadı", "neutral")

    # Market context
    st.subheader("🛰️ Piyasa Bağlamı (BTC, Dominance, VIX, DXY)")
    is_alt = ticker.upper().endswith("-USD") and ticker.upper() != "BTC-USD"
    if is_alt:
        if btc_ctx["trend_up"] is True:
            line(f"BTC trendi **yukarı** ve RSI {btc_ctx['rsi']:.1f} → altcoinler için **pozitif**.", "pos")
        elif btc_ctx["trend_up"] is False:
            line(f"BTC trendi **aşağı** ve RSI {btc_ctx['rsi']:.1f} → altcoinler üzerinde **baskı**.", "neg")
        else:
            line("BTC durumu alınamadı.", "neutral")
        if np.isfinite(dom_last):
            if dom_last >= 55: line(f"BTC Dominance {dom_last:.2f} → Altlar üzerinde **baskı**.", "neg")
            elif dom_last <= 45: line(f"BTC Dominance {dom_last:.2f} → Altlar için **destekleyici**.", "pos")
            else: line(f"BTC Dominance {dom_last:.2f} → Nötr.", "neutral")
        else:
            line("BTC Dominance verisi bulunamadı (geliştiriliyor).", "neutral")
    else:
        line("Seçili varlık BTC; dominance yorumu altcoinler için daha anlamlıdır.", "neutral")

    if np.isfinite(vix_last):
        if vix_last >= 25: line(f"VIX {vix_last:.2f} → **Risk-off** ortam.", "neg")
        elif vix_last <= 15: line(f"VIX {vix_last:.2f} → **Risk-on** ortam.", "pos")
        else: line(f"VIX {vix_last:.2f} → Nötr volatilite.", "neutral")
    else:
        line("VIX verisi alınamadı.", "neutral")

    if np.isfinite(dxy_last):
        if dxy_last >= 105: line(f"DXY {dxy_last:.2f} → Güçlü dolar; kripto için **negatif**.", "neg")
        elif dxy_last <= 100: line(f"DXY {dxy_last:.2f} → Zayıf dolar; kripto için **pozitif**.", "pos")
        else: line(f"DXY {dxy_last:.2f} → Nötr seviye.", "neutral")
    else:
        line("DXY verisi alınamadı.", "neutral")

    st.markdown("---")
    st.markdown("**Not:** Dalgalanmalardan etkilenmemek için her zaman kademeli alım yapın. "
                "Bu uygulamada özet sinyal ve gerekçeler gösterilir.")

# ========== REHBER ==========
with tab_guide:
    st.subheader("📘 Rehber – Kapsamlı Açıklamalar")
    st.markdown("""
### EMA – Üssel Hareketli Ortalama
- **Amaç:** Trend yönünü ve dönüşleri görmek.
- **Sinyal:** **EMA Kısa > EMA Uzun → yükseliş**, tersi **düşüş**.
- **İpucu:** Kısa periyot daha hızlı, ama daha fazla yalancı sinyal üretir.

### RSI – Göreceli Güç Endeksi (0–100)
- **30 altı:** Aşırı satım (erken alım riski yüksek).
- **35–45:** **Alım bölgesi** (EMA/MACD onayıyla güçlenir).
- **50:** Nötr eşik; üzeri momentum artışı.
- **70 üstü:** Aşırı alım (kâr realizasyonu / temkin).
- **Formül:** RSI = 100 − 100 / (1 + RS); RS = Ortalama Kazanç / Ortalama Kayıp

### MACD – Momentum ve Kesişimler
- **Tanım:** MACD = EMA(12) − EMA(26), **Sinyal** = EMA(9) of MACD
- **Yorum:** MACD > Sinyal → Pozitif momentum. Kesişimler başlatıcı tetikleyici olabilir.

### Bollinger Bantları
- **Orta bant:** MA(20), **Üst/Alt:** MA ± k·σ (genelde k=2)
- **Üste yakın:** Aşırı alım riski, **Alta yakın:** Aşırı satım / tepki.
- **Sıkışma:** Volatilite düşüktür; kırılım potansiyeli artar.

### Fibonacci Düzeltmeleri
- **Seviyeler:** 0.236, 0.382, 0.5, 0.618, 0.786
- **Kullanım:** Trend yönünde düzeltmelerde potansiyel giriş/tepki bölgeleri.
- **Not:** Tek başına sinyal değil; trend/momentumla birlikte düşünün.

### ATR – Ortalama Gerçek Aralık (Volatilite)
- **Amaç:** Günlük ortalama gerçek hareketi ölçer (volatilite).
- **Formül:** TR = max(High−Low, |High−PrevClose|, |Low−PrevClose|); **ATR = EMA(TR, n)**
- **Stop için:** **Stop = Entry − K × ATR** (Long), **Stop = Entry + K × ATR** (Short). K tipik olarak 1.5–3.
- **ATR%:** `ATR / Fiyat × 100` → Göreceli volatilite.

### Risk/Ödül (R-Multiple) ve Hedefler
- **Risk (R):** **R = Entry − Stop** (Long için).
- **Hedefler:** **TP1 = Entry + 1R**, **TP2 = Entry + 2R**, **TP3 = Entry + 3R**.
- **Yorum:** **R:R ≥ 1:2** genelde tercih edilir; 1:1 zayıftır.
- **Kısmi kâr alma:** Örn. **%50 TP1**, **%30 TP2**, **%20 TP3**.

### Pozisyon Boyutu ve Sermaye Riski
- **İşlem riski $:** `Risk Tutarı = Sermaye × Risk%`
- **Pozisyon (adet):** `Miktar = Risk Tutarı / (Entry − Stop)`
- **Maks. Pozisyon:** `Sermaye × Maks%` sınırını aşmayın.
- **Kademeli Alım / Kısmi TP:** Dalgalanmayı yumuşatır; stratejiye disiplin katar.

### Piyasa Bağlamı (BTC, Dominance, VIX, DXY)
- **Altcoinler** çoğunlukla **BTC** ile koreledir → BTC’nin **trend** ve **momentumu** kritik.
- **BTC Dominance:** Yüksekse altlar baskılanır; düşükse altlara para akışı artabilir.
- **VIX:** Yüksek VIX = Risk-off (kripto için olumsuz); düşük VIX = Risk-on.
- **DXY:** Güçlü Dolar kriptoya baskı, zayıf Dolar destekleyicidir.
""")

# ========== WATCHLIST ==========
with tab_watch:
    st.subheader("📋 Watchlist")
    wl = st.text_area("Ticker listesi (virgülle ayırın)", "BTC-USD,ETH-USD,THETA-USD,SOL-USD,BNB-USD", key="watchlist_tickers")
    tickers = [t.strip().upper() for t in wl.split(",") if t.strip()][:20]
    rows = []
    for t in tickers:
        d = load_yf(t, yf_period, yf_interval)
        if d is None or d.empty:
            rows.append({"Ticker": t, "Son Fiyat": "—", "RSI": "—", "Trend": "—", "Sinyal": "—"})
            continue
        d = normalize_ohlc(d)
        close = d["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:,0]
        e1 = ema(close, ema_short); e2 = ema(close, ema_long)
        r = rsi(close, rsi_len)
        macd_l, macd_s, _ = macd(close, macd_fast, macd_slow, macd_sig)
        bull = bool((e1 > e2).iloc[-1])
        macd_ok = bool(last_scalar(macd_l) > last_scalar(macd_s))
        r_last = last_scalar(r)
        rsi_okay = (r_last >= rsi_buy_min) and (r_last <= rsi_buy_max)
        buy_sig = bool(bull and (macd_ok or rsi_okay))
        rows.append({
            "Ticker": t,
            "Son Fiyat": fmt(last_scalar(close)),
            "RSI": f"{r_last:.2f}",
            "Trend": "↑" if bull else "↓",
            "Sinyal": "AL" if buy_sig else "—"
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ========== REJIM (Market score) ==========
with tab_regime:
    st.subheader("🧭 Piyasa Rejimi / Skor")
    _, btc_df, vix_df, dxy_df, dom_df = load_asset_and_macro()
    # BTC context
    btc_trend_up = None; btc_rsi = np.nan
    if btc_df is not None and not btc_df.empty:
        b = normalize_ohlc(btc_df).copy()
        close_s = get_series(b, "Close")
        e_s = ema(close_s, ema_short); e_l = ema(close_s, ema_long)
        btc_trend_up = bool(float((e_s - e_l).iloc[-1]) > 0)
        btc_rsi = float(rsi(close_s, rsi_len).iloc[-1])
    vix_last = last_close_of(vix_df) if vix_df is not None else np.nan
    dxy_last = last_close_of(dxy_df) if dxy_df is not None else np.nan
    dom_last = last_close_of(dom_df) if dom_df is not None else np.nan
    score = market_score(btc_trend_up, btc_rsi, vix_last, dxy_last, dom_last)
    st.markdown(f"**Piyasa Skoru:** **{score:.0f}/100**  —  *(0: risk-off, 100: risk-on)*")
    st.markdown("**BTC Trend:** " + ("🟢 Yükseliş" if btc_trend_up else "🔴 Düşüş" if btc_trend_up is False else "—"))
    st.markdown(f"**BTC RSI:** {fmt(btc_rsi,2)}  |  **VIX:** {fmt(vix_last,2)}  |  **DXY:** {fmt(dxy_last,2)}  |  **Dominance:** {fmt(dom_last,2)}")
    st.caption("Not: Dominance Yahoo'da resmi olmayabilir; best‑effort çekilir. (geliştiriliyor)")

# ========== RISK TOOL ==========
with tab_risk:
    st.subheader("🧮 Pozisyon / Risk Hesaplayıcı")
    col1, col2, col3 = st.columns(3)
    with col1:
        entry = st.number_input("Giriş Fiyatı", min_value=0.0, value=100.0)
        stop  = st.number_input("Stop Fiyatı", min_value=0.0, value=95.0)
    with col2:
        eq    = st.number_input("Sermaye ($)", min_value=10.0, value=float(equity))
        riskp = st.slider("Risk (%)", 0.1, 10.0, float(risk_pct), 0.1)
    with col3:
        tp_mode_risk = st.selectbox("TP Modu", ["R-multiple","Sabit %"], index=0)
        tp_pct_risk  = st.slider("TP %", 0.5, 50.0, 5.0, 0.5)
        tpR = st.slider("TP (R)", 0.5, 10.0, 2.0, 0.1)
    risk_per_unit = max(entry - stop, 1e-9)
    risk_amt = eq * (riskp/100.0)
    qty = risk_amt / risk_per_unit
    if tp_mode_risk == "R-multiple":
        tp_calc = entry + tpR * risk_per_unit
    else:
        tp_calc = entry * (1 + tp_pct_risk/100.0)
    rr = (tp_calc - entry) / (entry - stop) if (entry - stop) > 0 else np.nan
    st.markdown(f"- **Miktar:** ~ **{qty:.4f}**\n- **Hedef (TP):** **{fmt(tp_calc)}**\n- **R:R:** **{fmt(rr,2)}**")
    st.caption("İpucu: Kademeli TP dağılımı (ör. %50/%30/%20) ile dalgalanmayı yumuşatabilirsiniz.")

# ========== SIMPLE BACKTEST ==========
with tab_bt:
    st.subheader("🧪 Basit Backtest (long-only)")
    df_bt = load_yf(ticker, yf_period, yf_interval)
    if df_bt is None or df_bt.empty:
        st.warning("Veri alınamadı.")
    else:
        d = normalize_ohlc(df_bt).copy()
        d["EMA_S"] = ema(d["Close"], ema_short)
        d["EMA_L"] = ema(d["Close"], ema_long)
        d["RSI"]   = rsi(d["Close"], rsi_len)
        d["MACD"], d["MACD_S"], _ = macd(d["Close"], macd_fast, macd_slow, macd_sig)
        buy = (d["EMA_S"] > d["EMA_L"]) & ((d["MACD"] > d["MACD_S"]) | ((d["RSI"]>=rsi_buy_min) & (d["RSI"]<=rsi_buy_max)))
        sell = (d["EMA_S"] < d["EMA_L"]) & (d["MACD"] < d["MACD_S"])
        position = False; entry_px = 0.0; trades = []
        for i in range(1, len(d)):
            if (not position) and buy.iloc[i-1] and not buy.iloc[i-2 if i>=2 else 0]:
                position = True
                entry_px = float(d["Close"].iloc[i])
            elif position and sell.iloc[i-1] and not sell.iloc[i-2 if i>=2 else 0]:
                exit_px = float(d["Close"].iloc[i])
                trades.append(exit_px/entry_px - 1.0)
                position = False
        if position:
            exit_px = float(d["Close"].iloc[-1])
            trades.append(exit_px/entry_px - 1.0)
        if trades:
            tr = np.array(trades)
            win_rate = (tr>0).mean()*100.0
            total_ret = (tr + 1.0).prod() - 1.0
            avg_tr = tr.mean()
            df_out = pd.DataFrame({
                "İşlem": list(range(1, len(tr)+1)),
                "Getiri %": np.round(tr*100, 2)
            })
            st.markdown(f"- **İşlem sayısı:** {len(tr)}")
            st.markdown(f"- **Kazanma oranı:** **{win_rate:.1f}%**")
            st.markdown(f"- **Toplam getiri:** **{total_ret*100:.1f}%**")
            st.markdown(f"- **Ortalama işlem:** **{avg_tr*100:.2f}%**")
            st.dataframe(df_out, use_container_width=True)
        else:
            st.info("Sinyal tabanlı kapanışla işlem oluşmadı. Parametreleri değiştirip tekrar deneyin.")

# ========== CORRELATION ==========
with tab_corr:
    st.subheader("📈 Korelasyon (BTC ile ve aralarında)")
    wl2 = st.text_area("Ticker listesi (virgülle ayırın)", "BTC-USD,ETH-USD,THETA-USD,SOL-USD,BNB-USD", key="corr_tickers")
    tickers2 = [t.strip().upper() for t in wl2.split(",") if t.strip()][:15]
    # Load and align
    prices = []
    names = []
    for t in tickers2:
        d = load_yf(t, "1y", "1d")
        if d is None or d.empty: continue
        d = normalize_ohlc(d)
        close = get_series(d, "Close").rename(t)
        prices.append(close)
        names.append(t)
    if len(prices) >= 2:
        dfp = pd.concat(prices, axis=1).dropna(how="any")
        rets = np.log(dfp).diff().dropna()
        corr = rets.corr()
        st.dataframe(corr.style.format("{:.2f}"), use_container_width=True)
        st.caption("Not: Günlük log getiri korelasyonu.")
    else:
        st.info("Yeterli veri yok. En az iki geçerli ticker girin.")

# ========== NEWS (stub) ==========
with tab_news:
    st.subheader("🗞️ Haberler (geliştiriliyor)")
    st.write("• Seçili varlık ve piyasa geneli haberleri burada özetlenecek. (geliştiriliyor)")
    st.write("• BTC/Dominance/VIX/DXY etkisine göre etiketli kısa özetler planlanıyor. (geliştiriliyor)")
