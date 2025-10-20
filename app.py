
import streamlit as st

st.set_page_config(page_title="TradeMaster", layout="wide")
st.title("🧭 TradeMaster (Lite Safe)")
st.caption("Eğitim amaçlıdır; yatırım tavsiyesi değildir. • Bu sürüm, eksik paketlerde çökmemek için güvenli modda çalışır.")

# -------- Lazy import helpers --------
def need_libs():
    """Try to import third‑party libs lazily; return dict of modules or None if missing."""
    mods = {}
    missing = []
    try:
        import pandas as pd  # type: ignore
        mods["pd"] = pd
    except Exception:
        missing.append("pandas")
    try:
        import numpy as np  # type: ignore
        mods["np"] = np
    except Exception:
        missing.append("numpy")
    try:
        import yfinance as yf  # type: ignore
        mods["yf"] = yf
    except Exception:
        missing.append("yfinance")
    return mods, missing

def show_missing(missing):
    st.error("Gerekli bazı paketler eksik: " + ", ".join(missing))
    st.info("Repo köküne bir **requirements.txt** ekleyin veya aşağıdaki komutla kurun:")
    st.code("pip install " + " ".join(missing), language="bash")

# -------- Safe numeric helpers (require pandas/numpy) --------
def ema(pd, s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(pd, np, close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(pd, close, fast=12, slow=26, signal=9):
    macd_line = ema(pd, close, fast) - ema(pd, close, slow)
    signal_line = ema(pd, macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(close, n=20, k=2.0):
    mid = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=0)
    up = mid + k * std
    down = mid - k * std
    return mid, up, down

def atr(pd, df, n=14):
    if df is None or df.empty or not all(c in df.columns for c in ["High","Low","Close"]):
        return df["Close"] * 0 if df is not None else None
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def normalize_ohlc(pd, df):
    df = df.copy()
    df.columns = [str(c).title() for c in df.columns]
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    return df

def num(pd, np, s):
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = s.astype(str).str.replace(",", "", regex=False).str.replace(" ", "", regex=False)
    s = s.replace({"": np.nan, "None": np.nan, "NaN": np.nan, "nan": np.nan})
    return pd.to_numeric(s, errors="coerce")

def last_scalar(pd, np, x):
    try:
        v = x.iloc[-1]
    except Exception:
        try:
            v = x[-1]
        except Exception:
            v = x
    try:
        import numpy as _np  # may exist even if np missing earlier
        if hasattr(v, "to_numpy"):
            a = v.to_numpy().ravel()
            v = a[0] if a.size else _np.nan
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        return float("nan")

# -------- Sidebar --------
st.sidebar.header("📥 Veri Kaynağı")
src = st.sidebar.radio("Kaynak", ["YFinance (internet)", "CSV Yükle"], index=0)
ticker = st.sidebar.text_input("🪙 Kripto Değer", value="THETA-USD")

interval = st.sidebar.selectbox("Zaman Dilimi (interval)", ["1h","4h","1d","1wk"], index=2)
period   = st.sidebar.selectbox("Periyot (period)", ["1mo","3mo","6mo","1y","2y","5y","max"], index=3)

uploaded = st.sidebar.file_uploader("CSV (opsiyonel)", type=["csv"]) if src == "CSV Yükle" else None

st.sidebar.header("⚙️ Strateji Ayarları")
ema_short = st.sidebar.number_input("EMA Kısa", 3, 50, 9)
ema_long  = st.sidebar.number_input("EMA Uzun", 10, 300, 21)
ema_trend = st.sidebar.number_input("EMA Trend (uzun)", 50, 400, 200, step=10)
rsi_len   = st.sidebar.number_input("RSI Periyot", 5, 50, 14)
rsi_buy_min = st.sidebar.slider("RSI Alım Alt Sınır", 10, 60, 35)
rsi_buy_max = st.sidebar.slider("RSI Alım Üst Sınır", 40, 90, 60)
macd_fast = st.sidebar.number_input("MACD Hızlı", 3, 50, 12)
macd_slow = st.sidebar.number_input("MACD Yavaş", 6, 200, 26)
macd_sig  = st.sidebar.number_input("MACD Sinyal", 3, 50, 9)
bb_len    = st.sidebar.number_input("Bollinger Periyot", 5, 100, 20)
bb_std    = st.sidebar.slider("Bollinger Std", 1.0, 3.5, 2.0, 0.1)

st.sidebar.header("💰 Risk / Pozisyon")
equity      = st.sidebar.number_input("Sermaye (USD)", min_value=100.0, value=10000.0, step=100.0)
risk_pct    = st.sidebar.slider("İşlem başına risk (%)", 0.2, 5.0, 1.0, 0.1)
max_alloc   = st.sidebar.slider("Maks. Pozisyon (%)", 1.0, 100.0, 20.0, 1.0)
stop_mode   = st.sidebar.selectbox("Stop Tipi", ["ATR x K","Sabit %"], index=0)
atr_k       = st.sidebar.slider("ATR çarpanı", 0.5, 5.0, 2.0, 0.1)
stop_pct    = st.sidebar.slider("Sabit Zarar %", 0.5, 20.0, 3.0, 0.5)
tp_mode     = st.sidebar.selectbox("TP modu", ["R-multiple","Sabit %"], index=0)
tp1_r       = st.sidebar.slider("TP1 (R)", 0.5, 5.0, 1.0, 0.1)
tp2_r       = st.sidebar.slider("TP2 (R)", 0.5, 10.0, 2.0, 0.1)
tp3_r       = st.sidebar.slider("TP3 (R)", 0.5, 15.0, 3.0, 0.1)
tp_pct      = st.sidebar.slider("TP %", 0.5, 50.0, 5.0, 0.5)

# -------- Tabs --------
tab_an, tab_graf, tab_guide = st.tabs(["📈 Analiz","📊 Grafik","📘 Rehber"])

def load_asset(mods):
    pd = mods["pd"]; yf = mods["yf"]
    if src == "YFinance (internet)":
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
        if df is None or df.empty:
            return None
        df = normalize_ohlc(pd, df)
        # Ensure 'Close' exists
        if "Close" not in df.columns:
            # try Adj Close
            if "Adj Close" in df.columns:
                df["Close"] = df["Adj Close"]
            else:
                # case-insensitive search
                for c in df.columns:
                    if str(c).strip().lower() == "close":
                        df["Close"] = df[c]
                        break
        if "Close" not in df.columns:
            # fallback: first numeric column
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if num_cols:
                df["Close"] = df[num_cols[0]]
        df.index.name = "Date"
        return df
    else:
        import io
        if not uploaded:
            return None
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
                df[c] = num(pd, mods["np"], df[c])
        df = normalize_ohlc(pd, df)
        if "Close" not in df.columns:
            if "Adj Close" in df.columns:
                df["Close"] = df["Adj Close"]
            else:
                for c in df.columns:
                    if str(c).strip().lower() == "close":
                        df["Close"] = df[c]
                        break
        if "Close" not in df.columns:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if num_cols:
                df["Close"] = df[num_cols[0]]
        return df

# ---------- ANALIZ ----------
with tab_an:
    mods, missing = need_libs()
    if missing:
        show_missing(missing)
        st.stop()

    pd, np = mods["pd"], mods["np"]
    df = load_asset(mods)
    if df is None or df.empty:
        st.warning("Veri alınamadı. Ticker (örn. BTC-USD) ya da CSV kontrol edin.")
        st.stop()

    for c in ["Open","High","Low","Close","Volume"]:
        if c in df.columns:
            df[c] = num(pd, np, df[c])

    if "Close" not in df.columns:
        st.error("Veride 'Close' kolonu bulunamadı. CSV kullanıyorsanız 'Close' veya 'Adj Close' kolonu ekleyin.")
        st.stop()
    data = df.copy()
    data["EMA_Short"] = ema(pd, data["Close"], ema_short)
    data["EMA_Long"]  = ema(pd, data["Close"], ema_long)
    data["EMA_Trend"] = ema(pd, data["Close"], ema_trend)
    data["RSI"] = rsi(pd, np, data["Close"], rsi_len)
    data["MACD"], data["MACD_Signal"], data["MACD_Hist"] = macd(pd, data["Close"], macd_fast, macd_slow, macd_sig)
    data["BB_Mid"], data["BB_Up"], data["BB_Down"] = bollinger(data["Close"], bb_len, bb_std)
    data["ATR"] = atr(pd, data, 14)

    last_price = last_scalar(pd, np, data["Close"])
    bull = (data["EMA_Short"] > data["EMA_Long"]).iloc[-1]
    macd_cross_up = (data["MACD"] > data["MACD_Signal"]) & (data["MACD"].shift(1) <= data["MACD_Signal"].shift(1))
    rsi_ok = (data["RSI"] >= rsi_buy_min) & (data["RSI"] <= rsi_buy_max)

    buy_now = bool(bull and (macd_cross_up.iloc[-1] or rsi_ok.iloc[-1]))
    sell_now = bool((not bull) and (data["MACD"].iloc[-1] < data["MACD_Signal"].iloc[-1]))

    atr_val = last_scalar(pd, np, data["ATR"])
    if stop_mode == "ATR x K" and np.isfinite(atr_val):
        stop_price_long = last_price - max(atr_val * atr_k, 1e-9)
    else:
        stop_price_long = last_price * (1 - stop_pct/100.0)
    risk_long = max(last_price - stop_price_long, 1e-9)

    if tp_mode == "R-multiple":
        tp1_long = last_price + tp1_r * risk_long
        tp2_long = last_price + tp2_r * risk_long
        tp3_long = last_price + tp3_r * risk_long
        rr_tp1 = tp1_r
    else:
        tp1_long = last_price * (1 + tp_pct/100.0)
        tp2_long = last_price * (1 + 2*tp_pct/100.0)
        tp3_long = last_price * (1 + 3*tp_pct/100.0)
        rr_tp1 = (tp1_long - last_price) / (last_price - stop_price_long)

    risk_amount = equity * (risk_pct/100.0)
    qty = risk_amount / risk_long
    position_value = qty * last_price
    max_value = equity * (max_alloc/100.0)
    if position_value > max_value:
        scale = max_value / max(position_value, 1e-9)
        qty *= scale
        position_value = qty * last_price

    atr_pct = (atr_val / last_price * 100.0) if (atr_val == atr_val and last_price > 0) else float("nan")
    stop_dist_pct = (last_price - stop_price_long) / last_price * 100.0 if last_price > 0 else float("nan")
    pos_ratio_pct = (position_value / equity * 100.0) if equity > 0 else float("nan")
    momentum = float(data["RSI"].iloc[-1]) - 50.0
    trend_txt = "🟢 Yükseliş" if bool(bull) else "🔴 Düşüş"

    if buy_now:
        headline = "✅ SİNYAL: AL (long)"
    elif sell_now:
        headline = "❌ SİNYAL: SAT / LONG kapat"
    else:
        headline = "⏸ SİNYAL: BEKLE"

    st.subheader("📌 Özet")
    colA, colB, colC = st.columns([1.2,1,1])
    with colA:
        st.markdown(f"**Kripto:** `{ticker}`")
        st.markdown(f"**Son Fiyat:** **{last_price:.6f}**")
        st.markdown(f"**Durum:** **{headline}**")
    with colB:
        st.markdown(f"**Trend:** {trend_txt}")
        st.markdown(f"**Momentum (RSI−50):** {momentum:+.2f}")
        st.markdown(f"**Volatilite (ATR):** {atr_val:.6f}  ({atr_pct:.2f}%)")
    with colC:
        st.markdown(f"**Risk Oranı (R:R, TP1):** {rr_tp1:.2f}")
        st.markdown(f"**Stop Mesafesi:** {stop_dist_pct:.2f}%")
        st.markdown(f"**Pozisyon Oranı:** {pos_ratio_pct:.2f}%")

    st.subheader("🎯 Sinyal (Öneri)")
    if buy_now:
        st.markdown(f"""
- **Giriş (Long):** **{last_price:.6f}**
- **Önerilen Miktar:** ~ **{qty:.4f}** birim (≈ **${position_value:,.2f}**)
- **Stop (Long):** **{stop_price_long:.6f}**
- **TP1 / TP2 / TP3 (Long):** **{tp1_long:.6f}** / **{tp2_long:.6f}** / **{tp3_long:.6f}**
- **Risk:** Sermayenin %{risk_pct:.1f}’i (max pozisyon %{max_alloc:.0f})
        """)
    elif sell_now:
        st.markdown(f"""
- **Aksiyon:** **Long kapat / short düşün**
- **Kılavuz Stop (short için):** **{(last_price + risk_long):.6f}** _(örnek: long riskine simetrik)_
- **Not:** SAT sinyalinde long taraf TP seviyeleri **gösterilmez**.
        """)
    else:
        st.markdown("Şu anda belirgin bir al/sat sinyali yok; parametreleri veya zaman dilimini değiştirerek tekrar değerlendiriniz.")

    st.subheader("🧠 Gerekçeler")
    def line(text, kind="neutral"):
        if kind == "pos":
            color = "#0f9d58"; dot = "🟢"
        elif kind == "neg":
            color = "#d93025"; dot = "🔴"
        else:
            color = "#f29900"; dot = "🟠"
        st.markdown(f"{dot} <span style='color:{color}; font-weight:600;'>{text}</span>", unsafe_allow_html=True)

    if bool(bull):
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

    line("Haber akışı: Seçili varlığa dair başlıklar burada özetlenecek. (geliştiriliyor)", "neutral")

    st.markdown("---")
    st.markdown("**Not:** Dalgalanmalardan etkilenmemek için her zaman kademeli alım yapın. "
                "Bu uygulamada özet sinyal ve gerekçeler gösterilir.")

# ---------- GRAFIK ----------
with tab_graf:
    mods, missing = need_libs()
    if missing:
        show_missing(missing)
    else:
        pd, np = mods["pd"], mods["np"]
        df = load_asset(mods)
        if df is None or df.empty:
            st.warning("Veri alınamadı.")
        else:
            close = df["Close"]
            # Matplotlib yerine Streamlit çizimi (bağımlılık istemiyor)
            st.line_chart(pd.DataFrame({"Close": close}))
            # Ek olarak EMA çizgilerini tablo ile göster
            e1 = ema(pd, close, ema_short); e2 = ema(pd, close, ema_long); et = ema(pd, close, ema_trend)
            st.dataframe(pd.DataFrame({
                f"EMA{ema_short}": e1.tail(10),
                f"EMA{ema_long}": e2.tail(10),
                f"EMA{ema_trend}": et.tail(10),
            }), use_container_width=True)

# ---------- REHBER ----------
with tab_guide:
    st.subheader("📘 Rehber – Kısa Notlar")
    st.markdown("""
### EMA – Üssel Hareketli Ortalama
- **Sinyal:** **EMA Kısa > EMA Uzun → yükseliş**, tersi **düşüş**.

### RSI (0–100)
- **35–45:** Alım bölgesi (onayla güçlenir). **>70:** Aşırı alım; **<30:** Aşırı satım.

### MACD
- **MACD > Sinyal:** Pozitif momentum. Kesişimler tetikleyici olabilir.

### Bollinger
- **Üst banda yakın:** ısınma riski; **Alt banda yakın:** tepki potansiyeli.

### Risk / R-Multiple
- **R = Entry − Stop** (long).
- **TP1 = Entry + 1R**, **TP2 = +2R**, **TP3 = +3R**.
""")
