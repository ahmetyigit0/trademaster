
import streamlit as st

st.set_page_config(page_title="TradeMaster | Kripto Strateji Analizörü", layout="wide")
st.title("🧭 TradeMaster")
st.caption("Eğitim amaçlıdır; yatırım tavsiyesi değildir.")

# =========================
# Lazy imports
# =========================
def need_libs():
    mods = {}; missing = []
    try:
        import pandas as pd; mods["pd"] = pd
    except Exception: missing.append("pandas")
    try:
        import numpy as np; mods["np"] = np
    except Exception: missing.append("numpy")
    try:
        import yfinance as yf; mods["yf"] = yf
    except Exception: missing.append("yfinance")
    return mods, missing

def show_missing(missing):
    st.error("Gerekli paketler eksik: " + ", ".join(missing))
    st.info("Kurulum için:")
    st.code("pip install " + " ".join(missing), language="bash")

# =========================
# Utils & Indicators
# =========================
def normalize_ohlc(pd, df):
    df = df.copy()
    df.columns = [str(c).title() for c in df.columns]
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    return df

def num(pd, np, s):
    if hasattr(s, "astype"):
        s = s.astype(str).str.replace(",", "", regex=False).str.replace(" ", "", regex=False)
        s = s.replace({"": np.nan, "None": np.nan, "NaN": np.nan, "nan": np.nan})
    return pd.to_numeric(s, errors="coerce")

def last_scalar(pd, np, x):
    try:
        v = x.iloc[-1]
    except Exception:
        try: v = x[-1]
        except Exception: v = x
    try:
        if hasattr(v, "to_numpy"):
            a = v.to_numpy().ravel()
            v = a[0] if a.size else float("nan")
    except Exception: pass
    try: return float(v)
    except Exception: return float("nan")

def ema(pd, s, n): return s.ewm(span=n, adjust=False).mean()

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
    req = all(c in df.columns for c in ["High","Low","Close"])
    if not req: return df["Close"]*0
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    import numpy as np
    tr1 = (high-low).abs()
    tr2 = (high-prev_close).abs()
    tr3 = (low-prev_close).abs()
    tr = tr1.combine(tr2, max).combine(tr3, max)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def fmt(x, n=6):
    try:
        x = float(x)
        if x == x: return f"{x:.{n}f}"
    except Exception: pass
    return "-"

# =========================
# Sidebar
# =========================
st.sidebar.header("📥 Veri Kaynağı")
src = st.sidebar.radio("Kaynak", ["YFinance (internet)", "CSV Yükle"], index=0, help="YFinance: İnternetten otomatik veri çeker. CSV: Kendi verinizi yükleyin (Date, Open, High, Low, Close, Volume).")
ticker = st.sidebar.text_input("🪙 Kripto Değer", value="THETA-USD", help="Sembol formatı: BTC-USD, ETH-USD, THETA-USD vb.")

preset = st.sidebar.selectbox(
    "Zaman Dilimi",
    ["Kısa (4h)", "Orta (1g)", "Uzun (1hft)"],
    index=1,
    help="Veri çözünürlüğü ve geçmiş: Kısa=4h/180g, Orta=1d/1y, Uzun=1wk/5y."
)
if preset.startswith("Kısa"):
    interval, period = "4h", "180d"
elif preset.startswith("Orta"):
    interval, period = "1d", "1y"
else:
    interval, period = "1wk", "5y"

uploaded = st.sidebar.file_uploader("CSV (opsiyonel)", type=["csv"], help="Zorunlu kolonlar: Date, Open, High, Low, Close, Volume. Tarihi otomatik algılar.") if src == "CSV Yükle" else None

st.sidebar.header("⚙️ Strateji Ayarları")
ema_short = st.sidebar.number_input("EMA Kısa", 3, 50, 9, help="Kısa vadeli eğilim.")
ema_long  = st.sidebar.number_input("EMA Uzun", 10, 300, 21, help="Orta vadeli eğilim. EMA Kısa bunun üstüne çıkınca yükseliş sinyali.")
ema_trend = st.sidebar.number_input("EMA Trend (uzun)", 50, 400, 200, step=10, help="Uzun dönem eğilim filtresi (200 önerilir).")
rsi_len   = st.sidebar.number_input("RSI Periyot", 5, 50, 14, help="RSI hesap periyodu. 14 klasik değerdir.")
rsi_buy_min = st.sidebar.slider("RSI Alım Alt Sınır", 10, 60, 35, help="RSI bu seviyenin üzerindeyse aşırı satım riski azalır.")
rsi_buy_max = st.sidebar.slider("RSI Alım Üst Sınır", 40, 90, 60, help="RSI bu seviyenin altındaysa aşırı alımda değildir.")
macd_fast = st.sidebar.number_input("MACD Hızlı", 3, 50, 12, help="MACD hızlı EMA süresi.")
macd_slow = st.sidebar.number_input("MACD Yavaş", 6, 200, 26, help="MACD yavaş EMA süresi.")
macd_sig  = st.sidebar.number_input("MACD Sinyal", 3, 50, 9, help="MACD sinyal çizgisi.")
bb_len    = st.sidebar.number_input("Bollinger Periyot", 5, 100, 20, help="Bollinger orta bant periyodu.")
bb_std    = st.sidebar.slider("Bollinger Std", 1.0, 3.5, 2.0, 0.1, help="Bant genişliği katsayısı.")

st.sidebar.header("💰 Risk / Pozisyon")
equity      = st.sidebar.number_input("Sermaye (USD)", min_value=100.0, value=10000.0, step=100.0, help="Toplam hesap büyüklüğü.")
risk_pct    = st.sidebar.slider("İşlem başına risk (%)", 0.2, 5.0, 1.0, 0.1, help="Bir işlemde kaybetmeyi göze aldığınız maksimum oran.")
max_alloc   = st.sidebar.slider("Maks. Pozisyon (%)", 1.0, 100.0, 20.0, 1.0, help="Tek bir pozisyon için üst sınır.")
stop_mode   = st.sidebar.selectbox("Stop Tipi", ["ATR x K","Sabit %"], index=0, help="ATR tabanlı dinamik stop veya sabit yüzde stop.")
atr_k       = st.sidebar.slider("ATR çarpanı", 0.5, 5.0, 2.0, 0.1, help="Stop mesafesi = ATR × K.")
stop_pct    = st.sidebar.slider("Sabit Zarar %", 0.5, 20.0, 3.0, 0.5, help="ATR yoksa devreye girer.")
tp_mode     = st.sidebar.selectbox("TP modu", ["R-multiple","Sabit %"], index=0, help="R-multiple: TP = Entry + k×Risk. Sabit %: Yüzdesel hedef.")
tp1_r       = st.sidebar.slider("TP1 (R)", 0.5, 5.0, 1.0, 0.1, help="İlk hedef için riskin kaç katı (1R).")
tp2_r       = st.sidebar.slider("TP2 (R)", 0.5, 10.0, 2.0, 0.1, help="İkinci hedef (2R).")
tp3_r       = st.sidebar.slider("TP3 (R)", 0.5, 15.0, 3.0, 0.1, help="Üçüncü hedef (3R).")
tp_pct      = st.sidebar.slider("TP %", 0.5, 50.0, 5.0, 0.5, help="Sabit yüzde TP modunda hedef yüzdesi.")

# =========================
# Tabs (Grafik kaldırıldı)
# =========================
tab_an, tab_guide = st.tabs(["📈 Analiz","📘 Rehber"])

# =========================
# Data loader
# =========================
def load_asset(mods):
    pd = mods["pd"]; np = mods["np"]; yf = mods["yf"]
    if src == "YFinance (internet)":
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
        if df is None or df.empty: return None
        df = normalize_ohlc(pd, df)
        if "Close" not in df.columns:
            if "Adj Close" in df.columns: df["Close"] = df["Adj Close"]
            else:
                for c in df.columns:
                    if str(c).strip().lower() == "close":
                        df["Close"] = df[c]; break
        if "Close" not in df.columns:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if num_cols: df["Close"] = df[num_cols[0]]
        df.index.name = "Date"
        return df
    else:
        if not uploaded: return None
        pd = mods["pd"]; np = mods["np"]
        raw = pd.read_csv(uploaded)
        cols = {str(c).lower(): c for c in raw.columns}
        need = ["date","open","high","low","close","volume"]
        if all(n in cols for n in need):
            df = raw[[cols[n] for n in need]].copy()
            df.columns = ["Date","Open","High","Low","Close","Volume"]
        else:
            df = raw.copy(); df.columns = [str(c).title() for c in df.columns]
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
        for c in ["Open","High","Low","Close","Volume"]:
            if c in df.columns: df[c] = num(pd, np, df[c])
        df = normalize_ohlc(pd, df)
        if "Close" not in df.columns:
            if "Adj Close" in df.columns: df["Close"] = df["Adj Close"]
        return df

# =========================
# ANALIZ
# =========================
with tab_an:
    mods, missing = need_libs()
    if missing: show_missing(missing); st.stop()
    pd, np = mods["pd"], mods["np"]
    df = load_asset(mods)
    if df is None or df.empty:
        st.warning("Veri alınamadı. Ticker (örn. BTC-USD) ya da CSV kontrol edin."); st.stop()
    if "Close" not in df.columns:
        st.error("Veride 'Close' kolonu bulunamadı."); st.stop()

    for c in ["Open","High","Low","Close","Volume"]:
        if c in df.columns: df[c] = num(pd, np, df[c])

    data = df.copy()
    data["EMA_Short"] = ema(pd, data["Close"], ema_short)
    data["EMA_Long"]  = ema(pd, data["Close"], ema_long)
    data["EMA_Trend"] = ema(pd, data["Close"], ema_trend)
    data["RSI"] = rsi(pd, np, data["Close"], rsi_len)
    data["MACD"], data["MACD_Signal"], data["MACD_Hist"] = macd(pd, data["Close"], macd_fast, macd_slow, macd_sig)
    data["BB_Mid"], data["BB_Up"], data["BB_Down"] = bollinger(data["Close"], bb_len, bb_std)
    data["ATR"] = atr(pd, data, 14)
    # Fiyat Volatilitesi (30g sigma %)
    vol = data["Close"].pct_change().rolling(30).std() * 100
    vol_now = float(vol.iloc[-1])

    last_price = last_scalar(pd, np, data["Close"])
    bull = (data["EMA_Short"] > data["EMA_Long"]).iloc[-1]
    macd_cross_up = (data["MACD"] > data["MACD_Signal"]) & (data["MACD"].shift(1) <= data["MACD_Signal"].shift(1))
    rsi_ok = (data["RSI"] >= rsi_buy_min) & (data["RSI"] <= rsi_buy_max)

    buy_now = bool(bull and (macd_cross_up.iloc[-1] or rsi_ok.iloc[-1]))
    sell_now = bool((not bull) and (data["MACD"].iloc[-1] < data["MACD_Signal"].iloc[-1]))

    # Stop logic (robust with fallback)
    atr_val = last_scalar(pd, np, data["ATR"])
    if stop_mode == "ATR x K":
        atr_ok = (atr_val == atr_val) and (atr_val > 0)
        if atr_ok:
            stop_price_long = last_price - atr_val * atr_k
        else:
            stop_price_long = last_price * (1 - max(stop_pct, 0.5)/100.0)
    else:
        stop_price_long = last_price * (1 - max(stop_pct, 0.5)/100.0)
    if stop_price_long >= last_price * 0.9999:
        stop_price_long = last_price * (1 - max(stop_pct, 1.0)/100.0)
    risk_long = max(last_price - stop_price_long, 1e-6)

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
        qty *= scale; position_value = qty * last_price

    stop_dist_pct = (last_price - stop_price_long) / last_price * 100.0 if last_price > 0 else float("nan")
    pos_ratio_pct = (position_value / equity * 100.0) if equity > 0 else float("nan")
    momentum = float(data["RSI"].iloc[-1]) - 50.0
    trend_txt = "🟢 Yükseliş" if bool(bull) else "🔴 Düşüş"

    if buy_now: headline = "✅ SİNYAL: AL (long)"
    elif sell_now: headline = "❌ SİNYAL: SAT / LONG kapat"
    else: headline = "⏸ SİNYAL: BEKLE"

    st.subheader("📌 Özet")
    colA, colB, colC = st.columns([1.2,1,1])
    with colA:
        st.markdown(f"**Kripto:** `{ticker}`")
        st.markdown(f"**Son Fiyat:** **{last_price:.6f}**")
        st.markdown(f"**Durum:** **{headline}**")
    with colB:
        st.markdown(f"**Trend:** {trend_txt}")
        st.markdown(f"**Momentum (RSI-50):** {momentum:+.2f}")
        st.markdown(f"**Fiyat Volatilitesi (30g sigma %):** {vol_now:.2f}%")
    with colC:
        st.markdown(f"**Risk Oranı (R:R, TP1):** {fmt(rr_tp1,2)}")
        st.markdown(f"**Stop Mesafesi:** {fmt(stop_dist_pct,2)}%")
        st.markdown(f"**Pozisyon Oranı:** {fmt(pos_ratio_pct,2)}%")

    # ===== Sparkline / Emoji Trend (Son 5 mum)
# ===== Son 5 mum: renkli daireler (yeşil/yukari, kirmizi/asagi, gri/durağan) =====
cl_tail = data["Close"].tail(6)
if len(cl_tail) >= 2:
    chg = cl_tail.diff().tail(5)
    cols = []
    for x in chg:
        try:
            val = float(x)
        except Exception:
            val = 0.0
        if val > 0:
            cols.append("#0f9d58")  # yesil
        elif val < 0:
            cols.append("#d93025")  # kirmizi
        else:
            cols.append("#9aa0a6")  # gri

    dots = "".join([f"<span style='display:inline-block;width:12px;height:12px;border-radius:50%;background-color:{c};margin-right:6px;'></span>" for c in cols])
    trend_lbl = "YUKARI" if bool(bull) else "ASAGI"
    color = "#0f9d58" if bool(bull) else "#d93025"
    st.markdown(f"<div style='font-size:18px'>Son 5 mum kapanis: {dots} <span style='color:{color};font-weight:600'>(Trend filtresi: {trend_lbl})</span></div>", unsafe_allow_html=True)
    posc = int((chg > 0).sum()); negc = int((chg < 0).sum())
    comment = "momentum yukari" if posc >= 3 else ("momentum asagi" if negc >= 3 else "yanal/sikisik")
    st.caption(f"Pozitif: {posc}, Negatif: {negc} -> {comment}")

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

    # =========================
    # GEREKÇELER
    # =========================
    st.subheader("🧠 Gerekçeler")
    def line(text, kind="neutral"):
        if kind == "pos":
            color = "#0f9d58"; dot = "🟢"
        elif kind == "neg":
            color = "#d93025"; dot = "🔴"
        else:
            color = "#f29900"; dot = "🟠"
        st.markdown(f"{dot} <span style='color:{color}; font-weight:600;'>{text}</span>", unsafe_allow_html=True)

    # Trend & güç
    if bool(bull): line("EMA kısa > EMA uzun -> yükseliş trendi", "pos")
    else: line("EMA kısa < EMA uzun -> düşüş trendi", "neg")
    try:
        ema_spread = float((data["EMA_Short"].iloc[-1] - data["EMA_Long"].iloc[-1]) / last_price * 100.0)
        if ema_spread > 1.0: line(f"Trend gücü: EMA farkı {ema_spread:.2f}% (güçlü).", "pos")
        elif ema_spread < -1.0: line(f"Trend gücü: EMA farkı {ema_spread:.2f}% (güçlü düşüş).", "neg")
        else: line(f"Trend gücü: EMA farkı {ema_spread:.2f}% (zayıf).", "neutral")
    except Exception: pass
    try:
        dist_trend = float((last_price - float(data["EMA_Trend"].iloc[-1])) / last_price * 100.0)
        if dist_trend >= 0: line(f"Fiyat uzun dönem EMA'nın {dist_trend:.2f}% üzerinde.", "pos")
        else: line(f"Fiyat uzun dönem EMA'nın {abs(dist_trend):.2f}% altında.", "neg")
    except Exception: pass

    # MACD
    macd_now = float(data["MACD"].iloc[-1]); macd_sig_now = float(data["MACD_Signal"].iloc[-1])
    macd_hist_now = float(data["MACD_Hist"].iloc[-1]) if "MACD_Hist" in data else 0.0
    macd_hist_prev = float(data["MACD_Hist"].iloc[-2]) if "MACD_Hist" in data and len(data) >= 2 else macd_hist_now
    if macd_now > macd_sig_now and bool(macd_cross_up.iloc[-1]):
        line("MACD sinyal üstünde ve yukarı kesişim yeni oldu.", "pos")
    elif macd_now > macd_sig_now:
        line("MACD sinyal üstünde (pozitif momentum).", "pos")
    else:
        line("MACD sinyal altında (momentum zayıf).", "neg")
    if macd_hist_now > macd_hist_prev: line("MACD histogram güçleniyor.", "pos")
    elif macd_hist_now < macd_hist_prev: line("MACD histogram zayıflıyor.", "neg")

    # RSI
    rsi_now = float(data["RSI"].iloc[-1])
    if rsi_now < 30: line(f"RSI {rsi_now:.2f}: aşırı satım - erken alım riski.", "neg")
    elif 30 <= rsi_now < 35: line(f"RSI {rsi_now:.2f}: dip bölge - onay beklenmeli.", "neutral")
    elif 35 <= rsi_now <= 45: line(f"RSI {rsi_now:.2f}: alım bölgesi (EMA/MACD onayıyla).", "pos")
    elif 45 < rsi_now < 60: line(f"RSI {rsi_now:.2f}: nötr-olumlu.", "neutral")
    elif 60 <= rsi_now <= 70: line(f"RSI {rsi_now:.2f}: güçlü momentum.", "pos")
    else: line(f"RSI {rsi_now:.2f}: aşırı alım - temkin.", "neg")

    # Bollinger
    try:
        bb_up = float(data["BB_Up"].iloc[-1]); bb_dn = float(data["BB_Down"].iloc[-1])
        if last_price <= bb_dn: line("Fiyat alt banda yakın (tepki potansiyeli).", "pos")
        elif last_price >= bb_up: line("Fiyat üst banda yakın (ısınma).", "neg")
        else: line("Fiyat bant içinde (nötr).", "neutral")
        import numpy as _np
        bww = (data["BB_Up"] - data["BB_Down"]) / data["BB_Mid"].abs().replace(0, _np.nan)
        pct = float((bww.rank(pct=True).iloc[-1]) * 100.0)
        if pct <= 20: line("Bollinger genişliği düşük (sıkışma) -> kırılım potansiyeli.", "neutral")
    except Exception: pass

    # Volatility reasoning (30g sigma)
    try:
        if vol_now > 5:
            line(f"Fiyat volatilitesi {vol_now:.2f}% -> yüksek dalgalanma, stop aralığı geniş tutulmalı.", "neg")
        elif vol_now < 2:
            line(f"Fiyat volatilitesi {vol_now:.2f}% -> sakin piyasa, kırılım sonrası trend beklenebilir.", "neutral")
        else:
            line(f"Fiyat volatilitesi {vol_now:.2f}% -> orta düzey, strateji normal çalışır.", "pos")
    except Exception: pass

    # R:R
    try:
        if rr_tp1 >= 2.0: line(f"R:R (TP1) {rr_tp1:.2f} -> hedef/riske oran iyi.", "pos")
        elif 1.0 <= rr_tp1 < 2.0: line(f"R:R (TP1) {rr_tp1:.2f} -> orta karar.", "neutral")
        else: line(f"R:R (TP1) {rr_tp1:.2f} -> zayıf.", "neg")
    except Exception: pass

    # BTC context
    try:
        if ticker.upper() != "BTC-USD" and ticker.upper().endswith("-USD"):
            import yfinance as _yf
            _b = _yf.download("BTC-USD", period=period, interval=interval, auto_adjust=False, progress=False)
            if _b is not None and len(_b) > 0:
                _b.columns = [str(c).title() for c in _b.columns]
                _c = _b["Close"]
                _e1 = _c.ewm(span=ema_short, adjust=False).mean()
                _e2 = _c.ewm(span=ema_long, adjust=False).mean()
                import pandas as _pd, numpy as _np
                _r = rsi(_pd, _np, _c, rsi_len)
                r_txt = f" | RSI {_r.iloc[-1]:.1f}"
                if bool((_e1 > _e2).iloc[-1]):
                    line(f"BTC trend yukarı{r_txt} -> altcoinler için destekleyici.", "pos")
                else:
                    line(f"BTC trend aşağı{r_txt} -> altlar üzerinde baskı.", "neg")
    except Exception: pass

    line("Haber akışı: Seçili varlığa dair başlıklar burada özetlenecek. (geliştiriliyor)", "neutral")
    st.markdown("---")
    st.markdown("**Not:** Dalgalanmalardan etkilenmemek için her zaman kademeli alım yapın. Bu uygulamada özet sinyal ve gerekçeler gösterilir.")

# =========================
# REHBER
# =========================
with tab_guide:
    st.subheader("📘 Rehber – Trend & Mean-Reversion")
    st.markdown(r"""
**Trend modu (kırılım takip)**  
- Koşul: EMA kısa>uzun, Rejim boğa (EMA200 üstü), MACD>Signal, **Donchian üstü kırılım**.  
- Artılar: Trendde kalır; whipsaw filtreler.  
- Eksiler: Bazen geç giriş.

**Mean-Reversion modu (geri çekilmede alım)**  
- Koşul: Rejim boğa (EMA200 üstü), **Alt BB teması**, **RSI<35**.  
- Artılar: İndirimli giriş, iyi R:R.  
- Eksiler: Düşüş uzarsa erken alım riski.

**Ortak filtreler**: MTF uyumu, BTC filtresi, OBV eğimi, σ% (volatilite).  
**RR veto**: RR(TP1) < 1.2 ise işlem BEKLE.

**Kademeli alım bölgeleri (SAT sonrası)**: 0.382 / 0.5 / 0.618 fib, EMA200, Alt BB, Anchored VWAP.
    """)

