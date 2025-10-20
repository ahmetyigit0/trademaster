
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="TradeMaster", layout="wide")
st.title("🧭 TradeMaster v2")
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

def color_badge(text, color):
    return f'<span style="color:{color}; font-weight:600;">{text}</span>'

# =========================
# Sidebar — Mode
# =========================
mode = st.sidebar.radio("Sayfa", ["📈 Analiz", "📘 Rehber"], index=0)

# =========================
# Sidebar — Inputs (with tooltips)
# =========================
if mode == "📈 Analiz":
    st.sidebar.header("📥 Veri Kaynağı")
    src = st.sidebar.radio("Kaynak", ["YFinance (internet)", "CSV Yükle"], index=0,
                           help="YFinance: İnternetten fiyatları çeker. CSV: Kendi verinizi yükleyin.")
    ticker = st.sidebar.text_input("🪙 Kripto Değer", value="THETA-USD",
                                   help="Yahoo Finance formatı: BTC-USD, ETH-USD, THETA-USD vb.")
    period = st.sidebar.selectbox("Periyot", ["3mo","6mo","1y","2y","5y"], index=1,
                                  help="Veri aralığı (ör. 6 ay).")
    interval = st.sidebar.selectbox("Zaman Dilimi", ["1d","4h","1h"], index=0,
                                    help="Her barın zaman aralığı.")

    uploaded = st.sidebar.file_uploader("CSV (opsiyonel)", type=["csv"],
                                        help="Kolonlar: Date, Open, High, Low, Close, Volume") if src == "CSV Yükle" else None

    st.sidebar.header("⚙️ Strateji Ayarları")
    ema_short = st.sidebar.number_input("EMA Kısa", 3, 50, 9, help="Kısa vadeli trend ortalaması.")
    ema_long  = st.sidebar.number_input("EMA Uzun", 10, 300, 21, help="Orta vadeli trend ortalaması.")
    ema_trend = st.sidebar.number_input("EMA Trend (uzun)", 50, 400, 200, step=10, help="Uzun vadeli trend referansı.")

    rsi_len   = st.sidebar.number_input("RSI Periyot", 5, 50, 14, help="Momentum göstergesi RSI'ın periyodu.")
    rsi_buy_min = st.sidebar.slider("RSI Alım Alt Sınır", 10, 60, 40, help="RSI bu değerin üzerindeyse aşırı zayıf değildir.")
    rsi_buy_max = st.sidebar.slider("RSI Alım Üst Sınır", 40, 90, 70, help="RSI bu değerin altındaysa aşırı alım değildir.")

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
    # Load data
    # =========================
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
            except Exception as e:
                st.error(f"CSV okunamadı: {e}")

    if df is None or df.empty:
        st.warning("Veri alınamadı. Ticker'ı tam (ör. BTC-USD) yazdığınızdan emin olun veya CSV yükleyin.")
        st.stop()

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
        fibs = {}
        trend_up_bool = False
    else:
        swing_high = float(hi_series.max())
        swing_low  = float(lo_series.min())
        last_close = float(cl_series.iloc[-1])
        mid = (swing_low + swing_high) / 2.0
        trend_up_bool = bool(last_close > mid)
        a, b = (swing_low, swing_high) if trend_up_bool else (swing_high, swing_low)
        fibs = fib_levels(a, b)

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

    # Metrics for summary
    atr_pct = (atr_val / entry_price * 100.0) if np.isfinite(atr_val) and entry_price > 0 else np.nan
    stop_dist_pct = (entry_price - stop_price) / entry_price * 100.0 if entry_price > 0 else np.nan
    rr_tp1 = (tp1 - entry_price) / (entry_price - stop_price) if (entry_price - stop_price) > 0 else np.nan
    pos_ratio_pct = (position_value / equity * 100.0) if equity > 0 else np.nan
    momentum = float(data["RSI"].iloc[-1]) - 50.0
    trend_txt = "🟢 Yükseliş" if bool(bull.iloc[-1]) else "🔴 Düşüş"

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
    st.markdown(f"""
- **Giriş Fiyatı:** **{fmt(entry_price)}**
- **Önerilen Miktar:** ~ **{qty:.4f}** birim (≈ **${position_value:,.2f}**)
- **Stop:** **{fmt(stop_price)}**
- **TP1 / TP2 / TP3:** **{fmt(tp1)}** / **{fmt(tp2)}** / **{fmt(tp3)}**
- **Risk:** Sermayenin %{risk_pct:.1f}’i (max pozisyon %{max_alloc:.0f})
    """)

    # Colored reasons
    st.subheader("🧠 Gerekçeler")
    def line(text, kind="neutral"):
        if kind == "pos":
            color = "#0f9d58"   # green
            dot = "🟢"
        elif kind == "neg":
            color = "#d93025"   # red
            dot = "🔴"
        else:
            color = "#f29900"   # orange
            dot = "🟠"
        st.markdown(f"{dot} <span style='color:{color}; font-weight:600;'>{text}</span>", unsafe_allow_html=True)

    # EMA
    if bool(bull.iloc[-1]):
        line("EMA kısa > EMA uzun (trend ↑)", "pos")
    else:
        line("EMA kısa < EMA uzun (trend ↓)", "neg")

    # MACD
    macd_now = float(data["MACD"].iloc[-1])
    macd_sig_now = float(data["MACD_Signal"].iloc[-1])
    if macd_now > macd_sig_now and bool(macd_cross_up.iloc[-1]):
        line("MACD sinyal üstünde ve son bar kesişim ↑", "pos")
    elif macd_now > macd_sig_now:
        line("MACD sinyal üstünde (pozitif momentum)", "neutral")
    else:
        line("MACD sinyal altında (momentum zayıf)", "neg")

    # RSI
    rsi_now = float(data["RSI"].iloc[-1])
    if rsi_now < 30:
        line(f"RSI {rsi_now:.2f} (aşırı satım – potansiyel tepki)", "pos")
    elif rsi_now > 70:
        line(f"RSI {rsi_now:.2f} (aşırı alım – dikkat)", "neg")
    elif rsi_ok.iloc[-1]:
        line(f"RSI {rsi_now:.2f} (alım için uygun aralık)", "pos")
    else:
        line(f"RSI {rsi_now:.2f} (nötr)", "neutral")

    # Bollinger
    if "BB_Up" in data and "BB_Down" in data and np.isfinite(entry_price):
        try:
            if entry_price <= data["BB_Down"].iloc[-1]:
                line("Fiyat alt banda yakın (aşırı satım riski / tepki gelebilir)", "pos")
            elif entry_price >= data["BB_Up"].iloc[-1]:
                line("Fiyat üst banda yakın (aşırı alım riski)", "neg")
            else:
                line("Fiyat bant içinde (nötr)", "neutral")
        except Exception:
            line("Bollinger hesaplanamadı", "neutral")

    # Fibonacci
    if np.isfinite(swing_high) and np.isfinite(swing_low):
        side = "yukarı trend fib seti" if trend_up_bool else "aşağı trend fib seti"
        fibs_here = fib_levels(swing_low, swing_high) if trend_up_bool else fib_levels(swing_high, swing_low)
        near = min(fibs_here.items(), key=lambda kv: abs(kv[1]-entry_price))
        near_txt = f"{side}; en yakın seviye: {near[0]} = {fmt(near[1])}"
        # Heuristic coloring: yakın seviye 0.382/0.618 ise nötr/pozitif; 0.786 ya da 0.236 uyarı
        key = near[0]
        if key in ("0.382", "0.5", "0.618"):
            line(f"Fibonacci {near_txt}", "neutral")
        elif key in ("0.786", "0.236"):
            line(f"Fibonacci {near_txt}", "neg")
        else:
            line(f"Fibonacci {near_txt}", "neutral")
    else:
        line("Fibonacci için yeterli veri yok", "neutral")

    st.markdown("---")
    st.markdown("**Not:** Dalgalanmalardan etkilenmemek için her zaman kademeli alım yapın. "
                "Bu uygulamada özet sinyal ve gerekçeler gösterilir.")

elif mode == "📘 Rehber":
    st.subheader("📘 Rehber")
    st.markdown("""
**EMA (Üssel Hareketli Ortalama)**  
- Kısa EMA trend değişimlerini hızlı yakalar, Uzun EMA genel yönü gösterir.  
- Kesişim: **EMA Kısa > EMA Uzun → yükseliş**, tersi düşüş.

**RSI (Göreceli Güç Endeksi)**  
- 0–100 arası. 30 altı: **aşırı satım**, 70 üstü: **aşırı alım**.  
- Orta bölge (40–60) nötr; 50 üstü momentum artışı.

**MACD**  
- MACD = EMA(12) − EMA(26), Sinyal = EMA(9)  
- MACD > Sinyal → pozitif momentum. Kesişimler tetikleyici olur.

**Bollinger Bantları**  
- Orta bant = MA(20), Üst/Alt = MA ± k·σ  
- Üste yakın → ısınma; alta yakın → soğuma/tepki potansiyeli.

**Fibonacci Seviyeleri**  
- 0.382/0.5/0.618 takip edilir. Trend yönünde düzeltmelerde giriş/tepki yerleri verir.

**ATR (Ortalama Gerçek Aralık)**  
- Volatilite ölçer. Stop için **Stop = Entry − K × ATR** sık kullanılır.

**Risk Yönetimi**  
- İşlem başına risk: **Sermaye × Risk%**  
- Pozisyon = **Risk Tutarı / (Entry−Stop)**  
- Maks. pozisyon = **Sermaye × Maks%**
""")
    st.markdown("---")
    st.markdown("İpuçları: Parametrelerin üzerine gelince küçük yardım metinlerini görebilirsiniz (tooltip).")
