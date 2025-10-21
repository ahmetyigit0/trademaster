# app.py — TradeMaster (Trend & Mean-Reversion, skor tabanlı sinyal, SAT'ta alım bölgeleri)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="TradeMaster | Kripto Strateji Analizörü", layout="wide")
st.title("💹 TradeMaster")
st.caption("Eğitim amaçlıdır; yatırım tavsiyesi değildir.")

# ──────────────────────────────
# Yardımcılar & Göstergeler
# ──────────────────────────────
def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).title() for c in df.columns]
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    return df

def to_num(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(",", "", regex=False).str.replace(" ", "", regex=False)
    s = s.replace({"": np.nan, "None": np.nan, "NaN": np.nan, "nan": np.nan})
    return pd.to_numeric(s, errors="coerce")

def last_scalar(x):
    try:
        v = x.iloc[-1]
    except Exception:
        try:
            v = x[-1]
        except Exception:
            v = x
    try:
        return float(v)
    except Exception:
        return float("nan")

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(close: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(close: pd.Series, n=20, k=2.0):
    mid = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=0)
    up = mid + k * std
    down = mid - k * std
    return mid, up, down

def atr(df: pd.DataFrame, n=14):
    if not all(c in df.columns for c in ["High", "Low", "Close"]):
        return df["Close"] * 0
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = tr1.combine(tr2, max).combine(tr3, max)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def fmt(x, n=6):
    try:
        x = float(x)
        if x == x:
            return f"{x:.{n}f}"
    except Exception:
        pass
    return "—"

# ── Hacim/MTF/Swing araçları ─────────────────────────
def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    delta = close.diff().fillna(0)
    sign = pd.Series(0, index=close.index)
    sign[delta > 0] = 1
    sign[delta < 0] = -1
    return (sign * volume.fillna(0)).cumsum()

def anchored_vwap(price: pd.Series, volume: pd.Series, anchor_pos: int) -> pd.Series:
    """anchor_pos: iloc pozisyon indexi (0..N-1)"""
    pv = (price * volume).copy()
    pv.iloc[:anchor_pos] = 0
    vol = volume.copy()
    vol.iloc[:anchor_pos] = 0
    cum_vol = vol.cumsum().replace(0, np.nan)
    return pv.cumsum() / cum_vol

def detect_swings(high: pd.Series, low: pd.Series, w: int = 5):
    """Basit swing tespiti → (high_positions, low_positions) döner (iloc pozisyonları)."""
    highs, lows = [], []
    for i in range(w, len(high) - w):
        window_h = high.iloc[i - w : i + w + 1]
        window_l = low.iloc[i - w : i + w + 1]
        if high.iloc[i] == window_h.max():
            highs.append(i)
        if low.iloc[i] == window_l.min():
            lows.append(i)
    return highs, lows

def last_up_swing(high: pd.Series, low: pd.Series, swings):
    """Son yukarı bacağını (low→high) döndür (iloc pozisyonları)."""
    highs, lows = swings
    if not highs or not lows:
        return None, None
    last_low = lows[-1]
    # last_low'tan sonra gelen ilk high'ı bul
    after_lows = [h for h in highs if h > last_low]
    if not after_lows:
        return None, None
    last_high = after_lows[-1]
    return last_low, last_high

def slope(series: pd.Series, n: int = 10) -> float:
    try:
        if len(series) < n:
            return 0.0
        y = series.tail(n).astype(float)
        x = np.arange(len(y))
        xm, ym = x.mean(), y.mean()
        num = ((x - xm) * (y - ym)).sum()
        den = ((x - xm) ** 2).sum() or 1.0
        return float(num / den)
    except Exception:
        return 0.0

def load_big_tf(tk: str, interval: str):
    big_interval = {"4h": "1d", "1d": "1wk", "1wk": "1mo"}.get(interval, "1d")
    big_period = {"4h": "1y", "1d": "5y", "1wk": "10y"}.get(interval, "1y")
    dfb = yf.download(tk, period=big_period, interval=big_interval, auto_adjust=False, progress=False)
    if dfb is None or dfb.empty:
        return None
    dfb = normalize_ohlc(dfb)
    if "Close" not in dfb.columns and "Adj Close" in dfb.columns:
        dfb["Close"] = dfb["Adj Close"]
    return dfb

# ──────────────────────────────
# Sidebar
# ──────────────────────────────
st.sidebar.header("📥 Veri Kaynağı")
src = st.sidebar.radio(
    "Kaynak",
    ["YFinance (internet)", "CSV Yükle"],
    index=0,
    help="YFinance: Otomatik veri. CSV: Kendi veriniz (Date, Open, High, Low, Close, Volume).",
)
ticker = st.sidebar.text_input("🪙 Kripto Değer", value="THETA-USD", help="Örn. BTC-USD, ETH-USD, THETA-USD")

preset = st.sidebar.selectbox(
    "Zaman Dilimi",
    ["Kısa (4h)", "Orta (1g)", "Uzun (1hft)"],
    index=1,
    help="Kısa=4h/180g, Orta=1d/1y, Uzun=1wk/5y",
)
if preset.startswith("Kısa"):
    interval, period = "4h", "180d"
elif preset.startswith("Orta"):
    interval, period = "1d", "1y"
else:
    interval, period = "1wk", "5y"

uploaded = (
    st.sidebar.file_uploader("CSV (opsiyonel)", type=["csv"], help="Zorunlu kolonlar: Date, Open, High, Low, Close, Volume.")
    if src == "CSV Yükle"
    else None
)

st.sidebar.header("⚙️ Strateji Ayarları")
mode = st.sidebar.selectbox("Strateji Modu", ["Otomatik", "Trend", "Mean-Reversion"], help="Trend: kırılım takip. MR: geri çekilmede alım.")
ema_short = st.sidebar.number_input("EMA Kısa", 3, 50, 9)
ema_long = st.sidebar.number_input("EMA Uzun", 10, 300, 21)
ema_trend = st.sidebar.number_input("EMA Trend (uzun)", 50, 400, 200, step=10)
rsi_len = st.sidebar.number_input("RSI Periyot", 5, 50, 14)
rsi_buy_min = st.sidebar.slider("RSI Alım Alt Sınır", 10, 60, 35)
rsi_buy_max = st.sidebar.slider("RSI Alım Üst Sınır", 40, 90, 60)
macd_fast = st.sidebar.number_input("MACD Hızlı", 3, 50, 12)
macd_slow = st.sidebar.number_input("MACD Yavaş", 6, 200, 26)
macd_sig = st.sidebar.number_input("MACD Sinyal", 3, 50, 9)
bb_len = st.sidebar.number_input("Bollinger Periyot", 5, 100, 20)
bb_std = st.sidebar.slider("Bollinger Std", 1.0, 3.5, 2.0, 0.1)
don_n = st.sidebar.slider("Donchian Kırılım (Trend)", 10, 60, 20, 1)

st.sidebar.header("💰 Risk / Pozisyon")
equity = st.sidebar.number_input("Sermaye (USD)", min_value=100.0, value=10000.0, step=100.0)
risk_pct = st.sidebar.slider("İşlem başına risk (%)", 0.2, 5.0, 1.0, 0.1)
max_alloc = st.sidebar.slider("Maks. Pozisyon (%)", 1.0, 100.0, 20.0, 1.0)
stop_mode = st.sidebar.selectbox("Stop Tipi", ["ATR x K", "Sabit %"], index=0)
atr_k = st.sidebar.slider("ATR çarpanı", 0.5, 5.0, 2.0, 0.1)
stop_pct = st.sidebar.slider("Sabit Zarar %", 0.5, 20.0, 3.0, 0.5)
tp_mode = st.sidebar.selectbox("TP modu", ["R-multiple", "Sabit %"], index=0)
tp1_r = st.sidebar.slider("TP1 (R)", 0.5, 5.0, 1.0, 0.1)
tp2_r = st.sidebar.slider("TP2 (R)", 0.5, 10.0, 2.0, 0.1)
tp3_r = st.sidebar.slider("TP3 (R)", 0.5, 15.0, 3.0, 0.1)
tp_pct = st.sidebar.slider("TP %", 0.5, 50.0, 5.0, 0.5)

# ──────────────────────────────
# Sekmeler
# ──────────────────────────────
tab_an, tab_guide = st.tabs(["📈 Analiz", "📘 Rehber"])

# ──────────────────────────────
# Veri Yükleyici
# ──────────────────────────────
def load_asset():
    if src == "YFinance (internet)":
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
        if df is None or df.empty:
            return None
        df = normalize_ohlc(df)
        if "Close" not in df.columns and "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        df.index.name = "Date"
        return df
    else:
        if not uploaded:
            return None
        raw = pd.read_csv(uploaded)
        cols = {str(c).lower(): c for c in raw.columns}
        need = ["date", "open", "high", "low", "close", "volume"]
        if all(n in cols for n in need):
            df = raw[[cols[n] for n in need]].copy()
            df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        else:
            df = raw.copy()
            df.columns = [str(c).title() for c in df.columns]
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c in df.columns:
                df[c] = to_num(df[c])
        df = normalize_ohlc(df)
        if "Close" not in df.columns and "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        return df

# ──────────────────────────────
# ANALİZ
# ──────────────────────────────
with tab_an:
    df = load_asset()
    if df is None or df.empty:
        st.warning("Veri alınamadı. Ticker (örn. BTC-USD) ya da CSV kontrol edin.")
        st.stop()
    if "Close" not in df.columns:
        st.error("Veride 'Close' kolonu bulunamadı.")
        st.stop()

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = to_num(df[c])

    data = df.copy()
    data["EMA_Short"] = ema(data["Close"], ema_short)
    data["EMA_Long"] = ema(data["Close"], ema_long)
    data["EMA_Trend"] = ema(data["Close"], ema_trend)
    data["RSI"] = rsi(data["Close"], rsi_len)
    data["MACD"], data["MACD_Signal"], data["MACD_Hist"] = macd(data["Close"], macd_fast, macd_slow, macd_sig)
    data["BB_Mid"], data["BB_Up"], data["BB_Down"] = bollinger(data["Close"], bb_len, bb_std)
    data["ATR"] = atr(data, 14)
    vol = data["Close"].pct_change().rolling(30).std() * 100  # σ% (30 bar)
    vol_now = float(vol.iloc[-1])

    last_price = last_scalar(data["Close"])
    bull = (data["EMA_Short"] > data["EMA_Long"]).iloc[-1]
    macd_cross_up = (data["MACD"] > data["MACD_Signal"]) & (data["MACD"].shift(1) <= data["MACD_Signal"].shift(1))
    rsi_now = float(data["RSI"].iloc[-1])

    bw = (data["BB_Up"] - data["BB_Down"]) / (data["BB_Mid"].abs() + 1e-9)
    data["OBV"] = obv(data["Close"], data.get("Volume", data["Close"] * 0 + 0))
    regime_bull = bool(data["Close"].iloc[-1] > data["EMA_Trend"].iloc[-1])

    big = load_big_tf(ticker, interval)
    if big is not None and "Close" in big.columns:
        mtf_ok = bool((ema(big["Close"], ema_short) > ema(big["Close"], ema_long)).iloc[-1])
    else:
        mtf_ok = True

    btc_ok = True
    if ticker.upper() != "BTC-USD" and ticker.upper().endswith("-USD"):
        b = yf.download("BTC-USD", period=period, interval=interval, auto_adjust=False, progress=False)
        if b is not None and not b.empty:
            b = normalize_ohlc(b)
            e1b = ema(b["Close"], ema_short)
            e2b = ema(b["Close"], ema_long)
            mb, sb, _ = macd(b["Close"], macd_fast, macd_slow, macd_sig)
            btc_ok = bool((e1b > e2b).iloc[-1] and (mb.iloc[-1] > sb.iloc[-1]))

    obv_up = slope(data["OBV"], n=10) > 0
    don_hi = data["High"].rolling(int(don_n)).max()

    # Skor
    score = 0
    if regime_bull: score += 20
    if mtf_ok: score += 20
    if btc_ok: score += 15
    if (data["MACD"].iloc[-1] > data["MACD_Signal"].iloc[-1]): score += 15
    if 35 <= rsi_now <= 60: score += 10
    if obv_up: score += 10
    try:
        tight = bool((bw.rolling(20).mean().iloc[-1] < bw.median()))
        if tight: score += 10
    except Exception:
        pass

    # Mod seçimi
    if mode == "Mean-Reversion":
        is_mr = True
    elif mode == "Trend":
        is_mr = False
    else:
        # Otomatik: bant genişliğine bak — geniş bant MR eğilim, küçük bant Trend
        is_mr = bool(bw.rolling(20).mean().iloc[-1] > bw.median())

    # Sinyal kuralları
    min_score_trend = 60
    min_score_mr = 50
    if not is_mr:
        buy_raw = bool(bull and (data["MACD"].iloc[-1] > data["MACD_Signal"].iloc[-1]) and regime_bull)
        don_ok = bool(data["Close"].iloc[-1] > don_hi.iloc[-2]) if len(don_hi) >= 2 else True
        buy_now = bool(buy_raw and (score >= min_score_trend) and don_ok)
    else:
        mr_cond = bool(regime_bull and (data["Close"].iloc[-1] <= data["BB_Down"].iloc[-1]) and (rsi_now < 35))
        buy_now = bool(mr_cond and (score >= min_score_mr))
        don_ok = True

    sell_now = bool((not bull) and (data["MACD"].iloc[-1] < data["MACD_Signal"].iloc[-1]))

    # Stop & TP
    atr_val = last_scalar(data["ATR"])
    if stop_mode == "ATR x K" and atr_val == atr_val and atr_val > 0:
        stop_price_long = last_price - atr_val * atr_k
    else:
        stop_price_long = last_price * (1 - max(stop_pct, 0.5) / 100.0)
    if stop_price_long >= last_price * 0.9999:
        stop_price_long = last_price * (1 - max(stop_pct, 1.0) / 100.0)
    risk_long = max(last_price - stop_price_long, 1e-6)

    if tp_mode == "R-multiple":
        tp1_long = last_price + tp1_r * risk_long
        tp2_long = last_price + tp2_r * risk_long
        tp3_long = last_price + tp3_r * risk_long
        rr_tp1 = tp1_r
    else:
        tp1_long = last_price * (1 + tp_pct / 100.0)
        tp2_long = last_price * (1 + 2 * tp_pct / 100.0)
        tp3_long = last_price * (1 + 3 * tp_pct / 100.0)
        rr_tp1 = (tp1_long - last_price) / (last_price - stop_price_long)

    # RR veto
    rr_veto = False
    if buy_now and rr_tp1 < 1.2:
        rr_veto = True
        buy_now = False

    # Volatiliteye göre max pozisyon
    effective_max_alloc = max_alloc * (0.6 if vol_now > 6 else 1.0)

    risk_amount = equity * (risk_pct / 100.0)
    qty = risk_amount / risk_long
    position_value = qty * last_price
    max_value = equity * (effective_max_alloc / 100.0)
    if position_value > max_value:
        scale = max_value / max(position_value, 1e-9)
        qty *= scale
        position_value = qty * last_price

    stop_dist_pct = (last_price - stop_price_long) / last_price * 100.0 if last_price > 0 else float("nan")
    pos_ratio_pct = (position_value / equity * 100.0) if equity > 0 else float("nan")
    momentum = rsi_now - 50.0
    trend_txt = "🟢 Yükseliş" if bool(bull) else "🔴 Düşüş"
    mode_txt = "Mean-Reversion" if is_mr else "Trend"

    if buy_now:
        headline = "✅ SİNYAL: AL (long)"
    elif sell_now:
        headline = "❌ SİNYAL: SAT / LONG kapat"
    else:
        headline = "⏸ SİNYAL: BEKLE"

    # Özet
    st.subheader("📌 Özet")
    colA, colB, colC, colD = st.columns([1.2, 1, 1, 1])
    with colA:
        st.markdown(f"**Kripto:** `{ticker}`")
        st.markdown(f"**Son Fiyat:** **{last_price:.6f}**")
        st.markdown(f"**Durum:** **{headline}**")
    with colB:
        st.markdown(f"**Trend:** {trend_txt}")
        st.markdown(f"**Momentum (RSI−50):** {momentum:+.2f}")
        st.markdown(f"**Fiyat Volatilitesi (30g σ%):** {vol_now:.2f}%")
    with colC:
        st.markdown(f"**Risk Oranı (R:R, TP1):** {fmt(rr_tp1,2)}")
        st.markdown(f"**Stop Mesafesi:** {fmt(stop_dist_pct,2)}%")
        st.markdown(f"**Pozisyon Oranı:** {fmt(pos_ratio_pct,2)}%")
    with colD:
        st.markdown(f"**Strateji Modu:** {mode_txt}")
        st.markdown(f"**Sinyal Skoru:** {int(score)}")
        st.markdown(f"**Eşik:** {min_score_mr if is_mr else min_score_trend}")

    # Son 5 mum kapanışları — renkli daireler
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
                cols.append("#0f9d58")   # yeşil
            elif val < 0:
                cols.append("#d93025")   # kırmızı
            else:
                cols.append("#9aa0a6")   # gri
        dots = "".join(
            [
                f"<span style='display:inline-block;width:12px;height:12px;border-radius:50%;background-color:{c};margin-right:6px;'></span>"
                for c in cols
            ]
        )
        trend_lbl = "YUKARI" if bool(bull) else "ASAGI"
        color = "#0f9d58" if bool(bull) else "#d93025"
        st.markdown(
            f"<div style='font-size:18px'>Son 5 mum kapanis: {dots} "
            f"<span style='color:{color};font-weight:600'>(Trend filtresi: {trend_lbl})</span></div>",
            unsafe_allow_html=True,
        )
        posc = int((chg > 0).sum())
        negc = int((chg < 0).sum())
        comment = "momentum yukari" if posc >= 3 else ("momentum asagi" if negc >= 3 else "yanal/sikisik")
        st.caption(f"Pozitif: {posc}, Negatif: {negc} -> {comment}")

    # === Sinyal (Öneri) ===
    st.subheader("🎯 Sinyal (Öneri)")

    if buy_now:
        st.markdown(
            f"""
- **Giriş (Long):** **{last_price:.6f}**
- **Önerilen Miktar:** ~ **{qty:.4f}** birim (≈ **${position_value:,.2f}**)
- **Stop (Long):** **{stop_price_long:.6f}**
- **TP1 / TP2 / TP3 (Long):** **{tp1_long:.6f}** / **{tp2_long:.6f}** / **{tp3_long:.6f}**
- **Risk:** Sermayenin %{risk_pct:.1f}’i (maks. pozisyon %{effective_max_alloc:.0f})
"""
        )
        st.caption("Not: TP1 görüldüğünde stop'u maliyete çekmeyi değerlendirin (kılavuz).")

    elif sell_now:
        st.markdown(
            """
- **Aksiyon:** **Long kapat / short düşün**
- **Not:** SAT sinyalinde long taraf TP seviyeleri **gösterilmez**.
"""
        )
        # SAT'ta ALIM bölgeleri
        try:
            swings = detect_swings(data["High"], data["Low"], w=5)
            ll_pos, hh_pos = last_up_swing(data["High"], data["Low"], swings)
            if ll_pos is not None and hh_pos is not None:
                low_p = float(data["Low"].iloc[ll_pos])
                high_p = float(data["High"].iloc[hh_pos])

                # Fib seviyeleri
                diff = high_p - low_p
                fibs = {
                    "0.382": high_p - 0.382 * diff,
                    "0.500": high_p - 0.500 * diff,
                    "0.618": high_p - 0.618 * diff,
                }

                # Anchored VWAP (salınım dibinden)
                avwap = anchored_vwap(
                    data["Close"],
                    data.get("Volume", data["Close"] * 0 + 1),
                    anchor_pos=ll_pos,
                )
                avwap_now = float(avwap.iloc[-1]) if avwap.notna().any() else np.nan

                ema200 = float(data["EMA_Trend"].iloc[-1])
                bb_dn = float(data["BB_Down"].iloc[-1])

                st.markdown("**🔎 SAT sonrası olası ALIM bölgeleri (kademeli):**")
                st.write(f"- Fib 0.382: **{fibs['0.382']:.6f}**")
                st.write(f"- Fib 0.500: **{fibs['0.500']:.6f}**")
                st.write(f"- Fib 0.618: **{fibs['0.618']:.6f}**")
                st.write(f"- EMA200: **{ema200:.6f}**")
                st.write(f"- Alt Bollinger: **{bb_dn:.6f}**")
                if avwap_now == avwap_now:
                    st.write(f"- Anchored VWAP: **{avwap_now:.6f}**")
                st.caption("Plan: %40 (0.382) – %40 (0.500) – %20 (0.618). Stop: 0.618 altı / swing düşük altı.")
            else:
                st.info("Uygun salınım tespit edilemedi. Alternatif: EMA200, Alt BB ve yatay destekleri izleyin.")
        except Exception as e:
            st.warning(f"Alım bölgesi üretiminde sorun: {e}")

    else:
        st.markdown("Şu anda belirgin bir al/sat sinyali yok; parametreleri veya zaman dilimini değiştirerek tekrar değerlendirin.")

    # Gerekçeler
    st.subheader("🧠 Gerekçeler")

    def line(text, kind="neutral"):
        if kind == "pos":
            color, dot = "#0f9d58", "🟢"
        elif kind == "neg":
            color, dot = "#d93025", "🔴"
        else:
            color, dot = "#f29900", "🟠"
        st.markdown(f"{dot} <span style='color:{color}; font-weight:600;'>{text}</span>", unsafe_allow_html=True)

    line(f"Strateji modu: {'Mean-Reversion' if is_mr else 'Trend'}", "neutral")
    if not is_mr:
        ok = "geçti" if don_ok else "geçemedi"
        line(f"Donchian {don_n} kırılım filtresi: {ok}", "pos" if don_ok else "neg")
    if rr_veto:
        line("RR(TP1) < 1.2 olduğu için AL sinyali veto edildi (bekle).", "neg")
    if vol_now > 6:
        line("Yüksek volatilite (σ%>6): maks. pozisyon otomatik kısıldı.", "neutral")

    if bool(bull):
        line("EMA kısa > EMA uzun → yükseliş trendi", "pos")
    else:
        line("EMA kısa < EMA uzun → düşüş trendi", "neg")

    try:
        ema_spread = float((data["EMA_Short"].iloc[-1] - data["EMA_Long"].iloc[-1]) / last_price * 100.0)
        if ema_spread > 1.0:
            line(f"Trend gücü: EMA farkı {ema_spread:.2f}% (güçlü).", "pos")
        elif ema_spread < -1.0:
            line(f"Trend gücü: EMA farkı {ema_spread:.2f}% (güçlü düşüş).", "neg")
        else:
            line(f"Trend gücü: EMA farkı {ema_spread:.2f}% (zayıf).", "neutral")
    except Exception:
        pass

    try:
        dist_trend = float((last_price - float(data["EMA_Trend"].iloc[-1])) / last_price * 100.0)
        if dist_trend >= 0:
            line(f"Fiyat uzun dönem EMA'nın {dist_trend:.2f}% üzerinde.", "pos")
        else:
            line(f"Fiyat uzun dönem EMA'nın {abs(dist_trend):.2f}% altında.", "neg")
    except Exception:
        pass

    macd_now = float(data["MACD"].iloc[-1])
    macd_sig_now = float(data["MACD_Signal"].iloc[-1])
    macd_hist_now = float(data["MACD_Hist"].iloc[-1]) if "MACD_Hist" in data else 0.0
    macd_hist_prev = float(data["MACD_Hist"].iloc[-2]) if "MACD_Hist" in data and len(data) >= 2 else macd_hist_now
    if macd_now > macd_sig_now and bool(macd_cross_up.iloc[-1]):
        line("MACD sinyal üstünde ve yukarı kesişim yeni oldu.", "pos")
    elif macd_now > macd_sig_now:
        line("MACD sinyal üstünde (pozitif momentum).", "pos")
    else:
        line("MACD sinyal altında (momentum zayıf).", "neg")
    if macd_hist_now > macd_hist_prev:
        line("MACD histogram güçleniyor.", "pos")
    elif macd_hist_now < macd_hist_prev:
        line("MACD histogram zayıflıyor.", "neg")

    if rsi_now < 30:
        line(f"RSI {rsi_now:.2f}: aşırı satım – erken alım riski.", "neg")
    elif 30 <= rsi_now < 35:
        line(f"RSI {rsi_now:.2f}: dip bölge – onay beklenmeli.", "neutral")
    elif 35 <= rsi_now <= 45:
        line(f"RSI {rsi_now:.2f}: alım bölgesi (EMA/MACD onayıyla).", "pos")
    elif 45 < rsi_now < 60:
        line(f"RSI {rsi_now:.2f}: nötr-olumlu.", "neutral")
    elif 60 <= rsi_now <= 70:
        line(f"RSI {rsi_now:.2f}: güçlü momentum.", "pos")
    else:
        line(f"RSI {rsi_now:.2f}: aşırı alım – temkin.", "neg")

    line(f"Rejim (EMA{ema_trend}): {'boğa (üstünde)' if regime_bull else 'ayı (altında)'}", "pos" if regime_bull else "neg")
    line(f"MTF (büyük TF): {'uyumlu' if mtf_ok else 'uyumsuz'}", "pos" if mtf_ok else "neg")
    if ticker.upper() != "BTC-USD" and ticker.upper().endswith("-USD"):
        line(f"BTC filtresi: {'destekleyici' if btc_ok else 'destekleyici değil'}", "pos" if btc_ok else "neg")
    line(f"OBV eğimi: {'yukarı' if obv_up else 'aşağı/flat'}", "pos" if obv_up else "neutral")

    try:
        bb_up = float(data["BB_Up"].iloc[-1])
        bb_dn = float(data["BB_Down"].iloc[-1])
        if last_price <= bb_dn:
            line("Fiyat alt banda yakın (tepki potansiyeli).", "pos")
        elif last_price >= bb_up:
            line("Fiyat üst banda yakın (ısınma).", "neg")
        else:
            line("Fiyat bant içinde (nötr).", "neutral")
        bww = (data["BB_Up"] - data["BB_Down"]) / data["BB_Mid"].abs().replace(0, np.nan)
        pct = float((bww.rank(pct=True).iloc[-1]) * 100.0)
        if pct <= 20:
            line("Bollinger genişliği düşük (sıkışma) → kırılım potansiyeli.", "neutral")
    except Exception:
        pass

    try:
        if vol_now > 6:
            line(f"σ% {vol_now:.2f} → çok yüksek; temkinli boyutlandırma.", "neg")
        elif vol_now > 5:
            line(f"σ% {vol_now:.2f} → yüksek; stop daha geniş olmalı.", "neg")
        elif vol_now < 2:
            line(f"σ% {vol_now:.2f} → sakin; kırılım sonrası trend beklenebilir.", "neutral")
        else:
            line(f"σ% {vol_now:.2f} → orta; strateji normal çalışır.", "pos")
    except Exception:
        pass

    line("Haber akışı: Seçili varlığa dair başlıklar burada özetlenecek. (geliştiriliyor)", "neutral")
    st.markdown("---")
    st.markdown("**Not:** Dalgalanmalardan etkilenmemek için her zaman kademeli alım yapın. Bu uygulamada özet sinyal ve gerekçeler gösterilir.")

# ──────────────────────────────
# REHBER
# ──────────────────────────────
with tab_guide:
    st.subheader("📘 Rehber – Trend & Mean-Reversion")
    st.markdown(
        """
**Trend modu (kırılım takip)**  
- Koşul: EMA kısa>uzun, Rejim boğa (EMA200 üstü), MACD>Signal, **Donchian üst kırılım**.  
- Artılar: Trendde kalır; whipsaw filtreler.  
- Eksiler: Bazen geç giriş.

**Mean-Reversion modu (geri çekilmede alım)**  
- Koşul: Rejim boğa (EMA200 üstü), **Alt BB teması**, **RSI<35**.  
- Artılar: İndirimli giriş, iyi R:R.  
- Eksiler: Düşüş uzarsa erken alım riski.

**Ortak filtreler**: MTF uyumu, BTC filtresi, OBV eğimi, σ% (volatilite).  
**RR veto**: RR(TP1) < 1.2 ise işlem BEKLE.

**Kademeli alım bölgeleri (SAT sonrası)**: 0.382 / 0.5 / 0.618 fib, EMA200, Alt BB, Anchored VWAP.
"""
    )