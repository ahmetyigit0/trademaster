
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="TradeMaster", layout="wide")
st.title("🧭 TradeMaster")
st.caption("Eğitim amaçlıdır; yatırım tavsiyesi değildir.")

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

def _titleize_cols(cols):
    out = []
    for c in cols:
        try:
            out.append(str(c).title())
        except Exception:
            out.append(str(c))
    return out

def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Title-case columns, ensure Close exists by falling back to Adj Close if needed."""
    if df is None or df.empty:
        return df
    df = df.copy()
    try:
        df.columns = _titleize_cols(df.columns)
    except Exception:
        df.columns = [str(c) for c in df.columns]
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    return df

def to_num(s):
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
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if col in df.columns:
        s = df[col]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s
    return pd.Series(index=df.index, dtype=float)

def last_scalar(x):
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
    if df is None or df.empty or not all(c in df.columns for c in ["High","Low","Close"]):
        return pd.Series(index=df.index if df is not None else None, dtype=float)
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def colored(text, kind="neutral"):
    if kind == "pos":
        color = "#0f9d58"; dot = "🟢"
    elif kind == "neg":
        color = "#d93025"; dot = "🔴"
    else:
        color = "#f29900"; dot = "🟠"
    return f"{dot} <span style='color:{color}; font-weight:600;'>{text}</span>"

def fmt(x, n=6):
    try:
        x = float(x)
        if np.isfinite(x):
            return f"{x:.{n}f}"
    except Exception:
        pass
    return "—"

# =========================
# Sidebar — Inputs
# =========================
st.sidebar.header("📥 Veri Kaynağı")
src = st.sidebar.radio("Kaynak", ["YFinance (internet)", "CSV Yükle"], index=0,
                       help="YFinance: İnternetten fiyatları çeker. CSV: Kendi verinizi yükleyin.")
ticker = st.sidebar.text_input("🪙 Kripto Değer", value="THETA-USD",
                               help="Yahoo Finance formatı: BTC-USD, ETH-USD, THETA-USD vb.")

# Zaman Dilimi (interval) + Periyot (period) geri geldi
interval = st.sidebar.selectbox(
    "Zaman Dilimi (interval)",
    ["1h","4h","1d","1wk"],
    index=2,
    help="Grafik çözünürlüğü"
)
period = st.sidebar.selectbox(
    "Periyot (period)",
    ["1mo","3mo","6mo","1y","2y","5y","max"],
    index=3,
    help="Veri uzunluğu"
)

uploaded = st.sidebar.file_uploader("CSV (opsiyonel)", type=["csv"],
                                    help="Kolonlar: Date, Open, High, Low, Close, Volume") if src == "CSV Yükle" else None

st.sidebar.header("⚙️ Strateji Ayarları")
ema_short = st.sidebar.number_input("EMA Kısa", 3, 50, 9, help="Kısa vadeli trend ortalaması.")
ema_long  = st.sidebar.number_input("EMA Uzun", 10, 300, 21, help="Orta vadeli trend ortalaması.")
ema_trend = st.sidebar.number_input("EMA Trend (uzun)", 50, 400, 200, step=10, help="Uzun vadeli trend referansı.")
rsi_len   = st.sidebar.number_input("RSI Periyot", 5, 50, 14, help="Momentum göstergesi RSI'ın periyodu.")
rsi_buy_min = st.sidebar.slider("RSI Alım Alt Sınır", 10, 60, 35, help="RSI bu değerin üzerindeyse alım için daha uygun.")
rsi_buy_max = st.sidebar.slider("RSI Alım Üst Sınır", 40, 90, 60, help="RSI bu değerin altındaysa aşırı alım değildir.")
macd_fast = st.sidebar.number_input("MACD Hızlı", 3, 50, 12, help="MACD hızlı EMA periyodu.")
macd_slow = st.sidebar.number_input("MACD Yavaş", 6, 200, 26, help="MACD yavaş EMA periyodu.")
macd_sig  = st.sidebar.number_input("MACD Sinyal", 3, 50, 9, help="MACD sinyal çizgisi periyodu.")
bb_len    = st.sidebar.number_input("Bollinger Periyot", 5, 100, 20, help="Ortalama için periyot.")
bb_std    = st.sidebar.slider("Bollinger Std", 1.0, 3.5, 2.0, 0.1, help="Bant genişliği katsayısı.")

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
tab_an, tab_graf, tab_guide = st.tabs(["📈 Analiz","📊 Grafik","📘 Rehber"])

# Common data loader
def load_asset():
    if src == "YFinance (internet)":
        df = load_yf(ticker, period, interval)
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
                df = normalize_ohlc(df)
            except Exception as e:
                st.error(f"CSV okunamadı: {e}")
    return df

# ========== ANALIZ ==========
with tab_an:
    df = load_asset()
    if df is None or df.empty:
        st.warning("Veri alınamadı. Ticker'ı tam (ör. BTC-USD) yazdığınızdan emin olun veya CSV yükleyin.")
        st.stop()

    for c in ["Open","High","Low","Close","Volume"]:
        if c in df.columns:
            df[c] = to_num(df[c])

    data = df.copy()
    data["EMA_Short"] = ema(data["Close"], ema_short)
    data["EMA_Long"]  = ema(data["Close"], ema_long)
    data["EMA_Trend"] = ema(data["Close"], ema_trend)
    data["RSI"] = rsi(data["Close"], rsi_len)
    data["MACD"], data["MACD_Signal"], data["MACD_Hist"] = macd(data["Close"], macd_fast, macd_slow, macd_sig)
    data["BB_Mid"], data["BB_Up"], data["BB_Down"] = bollinger(data["Close"], bb_len, bb_std)
    data["ATR"] = atr(data, 14)

    last_price = last_scalar(data["Close"])
    bull = (data["EMA_Short"] > data["EMA_Long"]).iloc[-1]
    macd_cross_up = (data["MACD"] > data["MACD_Signal"]) & (data["MACD"].shift(1) <= data["MACD_Signal"].shift(1))
    rsi_ok = (data["RSI"] >= rsi_buy_min) & (data["RSI"] <= rsi_buy_max)

    buy_now = bool(bull and (macd_cross_up.iloc[-1] or rsi_ok.iloc[-1]))
    sell_now = bool((not bull) and (data["MACD"].iloc[-1] < data["MACD_Signal"].iloc[-1]))

    # Stop & TP (long baseline)
    atr_val = last_scalar(data["ATR"])
    if stop_mode == "ATR x K" and np.isfinite(atr_val):
        stop_price_long = last_price - max(atr_val * atr_k, 1e-9)
    else:
        stop_price_long = last_price * (1 - stop_pct/100.0)
    risk_long = max(last_price - stop_price_long, 1e-9)

    if tp_mode == "R-multiple":
        tp1_long = last_price + tp1_r * risk_long
        tp2_long = last_price + tp2_r * risk_long
        tp3_long = last_price + tp3_r * risk_long
        rr_tp1 = tp1_r  # by definition
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

    atr_pct = (atr_val / last_price * 100.0) if np.isfinite(atr_val) and last_price > 0 else np.nan
    stop_dist_pct = (last_price - stop_price_long) / last_price * 100.0 if last_price > 0 else np.nan
    pos_ratio_pct = (position_value / equity * 100.0) if equity > 0 else np.nan
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
        st.markdown(f"**Son Fiyat:** **{fmt(last_price)}**")
        st.markdown(f"**Durum:** **{headline}**")
    with colB:
        st.markdown(f"**Trend:** {trend_txt}")
        st.markdown(f"**Momentum (RSI−50):** {momentum:+.2f}")
        st.markdown(f"**Volatilite (ATR):** {fmt(atr_val)}  ({fmt(atr_pct,2)}%)")
    with colC:
        st.markdown(f"**Risk Oranı (R:R, TP1):** {fmt(rr_tp1,2)}")
        st.markdown(f"**Stop Mesafesi:** {fmt(stop_dist_pct,2)}%")
        st.markdown(f"**Pozisyon Oranı:** {fmt(pos_ratio_pct,2)}%")

    st.subheader("🎯 Sinyal (Öneri)")
    if buy_now:
        st.markdown(f"""
- **Giriş (Long):** **{fmt(last_price)}**
- **Önerilen Miktar:** ~ **{qty:.4f}** birim (≈ **${position_value:,.2f}**)
- **Stop (Long):** **{fmt(stop_price_long)}**
- **TP1 / TP2 / TP3 (Long):** **{fmt(tp1_long)}** / **{fmt(tp2_long)}** / **{fmt(tp3_long)}**
- **Risk:** Sermayenin %{risk_pct:.1f}’i (max pozisyon %{max_alloc:.0f})
        """)
    elif sell_now:
        st.markdown(f"""
- **Aksiyon:** **Long kapat / short düşün**
- **Kılavuz Stop (short için):** **{fmt(last_price + risk_long)}** _(örnek: long riskine simetrik)_
- **Not:** SAT sinyalinde long taraf TP seviyeleri **gösterilmez**.
        """)
    else:
        st.markdown("Şu anda belirgin bir al/sat sinyali yok; parametreleri veya zaman dilimini değiştirerek tekrar değerlendiriniz.")

    st.subheader("🧠 Gerekçeler")
    def line(text, kind="neutral"):
        st.markdown(colored(text, kind), unsafe_allow_html=True)

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

    if "BB_Up" in data and "BB_Down" in data and np.isfinite(last_price):
        try:
            if last_price <= data["BB_Down"].iloc[-1]: line("Fiyat alt banda yakın (tepki potansiyeli)", "pos")
            elif last_price >= data["BB_Up"].iloc[-1]: line("Fiyat üst banda yakın (ısınma)", "neg")
            else: line("Fiyat bant içinde (nötr)", "neutral")
        except Exception:
            line("Bollinger hesaplanamadı", "neutral")

    # Haber akışı yeri (geliştiriliyor)
    line("Haber akışı: Seçili varlığa dair başlıklar burada özetlenecek. (geliştiriliyor)", "neutral")

    st.markdown("---")
    st.markdown("**Not:** Dalgalanmalardan etkilenmemek için her zaman kademeli alım yapın. "
                "Bu uygulamada özet sinyal ve gerekçeler gösterilir.")

# ========== GRAFIK ==========
with tab_graf:
    # Lazy import matplotlib (fallback if missing)
    try:
        import matplotlib.pyplot as plt
        _HAS_MPL = True
    except Exception as _e:
        _HAS_MPL = False
        st.warning("Matplotlib yüklü değil. `requirements.txt` içine `matplotlib` ekleyin ya da aşağıdaki basit grafikleri kullanın.")
    df = load_asset()
    if df is None or df.empty:
        st.warning("Veri alınamadı.")
    else:
        d = df.copy()
        close = get_series(d, "Close")
        ema_s = ema(close, ema_short); ema_l = ema(close, ema_long); ema_t = ema(close, ema_trend)
        bb_m, bb_u, bb_d = bollinger(close, bb_len, bb_std)
        r = rsi(close, rsi_len)
        m_l, m_s, m_h = macd(close, macd_fast, macd_slow, macd_sig)

        # Price + EMAs + BB
        if not _HAS_MPL:
            st.line_chart(pd.DataFrame({
                'Close': close,
                f'EMA{ema_short}': ema_s,
                f'EMA{ema_long}': ema_l,
                f'EMA{ema_trend}': ema_t
            }))
        else:
            fig1, ax1 = plt.subplots(figsize=(10,4))
        ax1.plot(d.index, close, label="Close")
        ax1.plot(d.index, ema_s, label=f"EMA{ema_short}")
        ax1.plot(d.index, ema_l, label=f"EMA{ema_long}")
        ax1.plot(d.index, ema_t, label=f"EMA{ema_trend}")
        ax1.plot(d.index, bb_u, label="BB Upper")
        ax1.plot(d.index, bb_m, label="BB Mid")
        ax1.plot(d.index, bb_d, label="BB Lower")
        ax1.set_title(f"{ticker} Fiyat")
        ax1.legend(loc="upper left")
        st.pyplot(fig1) if _HAS_MPL else None

        # RSI
        if not _HAS_MPL:
            st.line_chart(pd.DataFrame({'RSI': r}))
        else:
            fig2, ax2 = plt.subplots(figsize=(10,2.8))
        ax2.plot(d.index, r, label="RSI")
        ax2.axhline(30, linestyle="--")
        ax2.axhline(70, linestyle="--")
        ax2.set_title("RSI")
        ax2.legend(loc="upper left")
        st.pyplot(fig2) if _HAS_MPL else None

        # MACD
        if not _HAS_MPL:
            st.line_chart(pd.DataFrame({'MACD': m_l, 'Signal': m_s}))
        else:
            fig3, ax3 = plt.subplots(figsize=(10,2.8))
        ax3.plot(d.index, m_l, label="MACD")
        ax3.plot(d.index, m_s, label="Signal")
        ax3.bar(d.index, m_h, label="Hist")
        ax3.set_title("MACD")
        ax3.legend(loc="upper left")
        st.pyplot(fig3) if _HAS_MPL else None

# ========== REHBER ==========
with tab_guide:
    st.subheader("📘 Rehber – Kısa Notlar")
    st.markdown("""
### EMA – Üssel Hareketli Ortalama
- **Sinyal:** **EMA Kısa > EMA Uzun → yükseliş**, tersi **düşüş**.
- Kısa periyot hızlı tepki verir; yalancı sinyal riski daha yüksektir.

### RSI (0–100)
- **35–45:** Alım bölgesi (onayla güçlenir). **>70:** Aşırı alım; **<30:** Aşırı satım.
- Formül kısaca: RSI = 100 − 100 / (1 + RS).

### MACD
- **MACD > Sinyal:** Pozitif momentum. Kesişimler tetikleyici olabilir.

### Bollinger
- **Üst banda yakın:** ısınma riski; **Alt banda yakın:** tepki potansiyeli.
- **Sıkışma:** Volatilite düşük; kırılım olasılığı artar.

### Risk / R-Multiple
- **R = Entry − Stop** (long).
- **TP1 = Entry + 1R**, **TP2 = +2R**, **TP3 = +3R**.
- R:R ≥ 1:2 genelde tercih edilir.

**Not:** Kademeli alım ve kısmi TP ile dalgalanma etkisini azaltabilirsiniz.
""")