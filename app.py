import streamlit as st

st.set_page_config(page_title="TradeMaster | Kripto Strateji Analizoru", layout="wide")
st.title("💹 TradeMaster")
st.caption("Egitim amaclidir; yatirim tavsiyesi degildir.")

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
    st.info("Kurulum icin:")
    st.code("pip install " + " ".join(missing), language="bash")

def normalize_ohlc(pd, df):
    df = df.copy()
    df.columns = [str(c).title() for c in df.columns]
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    if "Close" not in df.columns:
        for c in list(df.columns):
            if str(c).strip().lower() == "close":
                df["Close"] = df[c]; break
    if "Close" not in df.columns:
        for c in df.columns:
            try:
                if hasattr(df[c], "dtype") and str(df[c].dtype) != "object":
                    df["Close"] = df[c]; break
            except Exception:
                pass
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
    tr1 = (high-low).abs()
    tr2 = (high-prev_close).abs()
    tr3 = (low-prev_close).abs()
    tr = tr1.combine(tr2, max).combine(tr3, max)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def fmt(x, n=6):
    try:
        x = float(x)
        if x == x: return ("{0:." + str(n) + "f}").format(x)
    except Exception: pass
    return "-"

def macd_hist_turning_up(df):
    try: return float(df["MACD_Hist"].iloc[-1]) > float(df["MACD_Hist"].iloc[-2])
    except Exception: return False

def macd_hist_turning_down(df):
    try: return float(df["MACD_Hist"].iloc[-1]) < float(df["MACD_Hist"].iloc[-2])
    except Exception: return False

def bb_bounce_from_lower(df):
    try:
        return (float(df["Close"].iloc[-2]) < float(df["BB_Down"].iloc[-2])) and (float(df["Close"].iloc[-1]) > float(df["BB_Down"].iloc[-1]))
    except Exception: return False

def ema_gap_small(df, thr_pct=2.0):
    try:
        num = abs(float(df["EMA_Short"].iloc[-1]) - float(df["EMA_Long"].iloc[-1]))
        den = max(float(df["Close"].iloc[-1]), 1e-9)
        return (num/den*100.0) <= float(thr_pct)
    except Exception: return False

def compute_reentry_zone(df, atr_mult=0.5, lookback=20):
    try:
        low_recent = float(df["Low"].tail(lookback).min()) if "Low" in df.columns else float(df["Close"].tail(lookback).min())
        bb_dn = float(df["BB_Down"].iloc[-1])
        base = max(low_recent, bb_dn)
        atr_val = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else 0.0
        pad = atr_val * float(atr_mult)
        lo = max(base - pad, 0.0)
        hi = base + pad
        return lo, hi, base, atr_val
    except Exception:
        c = float(df["Close"].iloc[-1])
        return c*0.97, c*0.99, c*0.98, 0.0

st.sidebar.header("Veri Kaynagi")
src = st.sidebar.radio("Kaynak", ["YFinance (internet)", "CSV Yukle"], index=0)
ticker = st.sidebar.text_input("Kripto Deger", value="THETA-USD")

preset = st.sidebar.selectbox("Zaman Dilimi", ["Kisa (4h)","Orta (1g)","Uzun (1hft)"], index=1)
if preset.startswith("Kisa"):
    interval, period = "4h", "180d"
elif preset.startswith("Orta"):
    interval, period = "1d", "1y"
else:
    interval, period = "1wk", "5y"

uploaded = st.sidebar.file_uploader("CSV (opsiyonel)", type=["csv"]) if src == "CSV Yukle" else None

st.sidebar.header("Strateji Ayarlari")
ema_short = st.sidebar.number_input("EMA Kisa", 3, 50, 9)
ema_long  = st.sidebar.number_input("EMA Uzun", 10, 300, 21)
ema_trend = st.sidebar.number_input("EMA Trend (uzun)", 50, 400, 200, step=10)
rsi_len   = st.sidebar.number_input("RSI Periyot", 5, 50, 14)
bb_len    = st.sidebar.number_input("Bollinger Periyot", 5, 100, 20)
bb_std    = st.sidebar.slider("Bollinger Std", 1.0, 3.5, 2.0, 0.1)
macd_fast = st.sidebar.number_input("MACD Hizli", 3, 50, 12)
macd_slow = st.sidebar.number_input("MACD Yavas", 6, 200, 26)
macd_sig  = st.sidebar.number_input("MACD Sinyal", 3, 50, 9)

st.sidebar.header("Risk / Pozisyon")
equity      = st.sidebar.number_input("Sermaye (USD)", min_value=100.0, value=10000.0, step=100.0)
risk_pct    = st.sidebar.slider("Islem basina risk (%)", 0.2, 5.0, 1.0, 0.1)
max_alloc   = st.sidebar.slider("Maks. Pozisyon (%)", 1.0, 100.0, 20.0, 1.0)
stop_pct    = st.sidebar.slider("Sabit Zarar % (fallback)", 0.5, 20.0, 3.0, 0.5)
tp1_r       = st.sidebar.slider("TP1 (R)", 0.5, 5.0, 1.0, 0.1)
tp2_r       = st.sidebar.slider("TP2 (R)", 0.5, 10.0, 2.0, 0.1)
tp3_r       = st.sidebar.slider("TP3 (R)", 0.5, 15.0, 3.0, 0.1)

tab_an, tab_guide = st.tabs(["📈 Analiz","📘 Rehber"])

def load_asset(mods):
    pd = mods["pd"]; np = mods["np"]; yf = mods["yf"]
    if src == "YFinance (internet)":
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
        if df is None or df.empty: return None
        df = normalize_ohlc(pd, df)
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
        return df

with tab_an:
    mods, missing = need_libs()
    if missing: show_missing(missing); st.stop()
    pd, np = mods["pd"], mods["np"]
    df = load_asset(mods)
    if df is None or df.empty:
        st.warning("Veri alinamadi. Ticker ya da CSV'yi kontrol edin."); st.stop()
    if "Close" not in df.columns:
        st.error("Veride 'Close' kolonu bulunamadi."); st.stop()

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

    vol = data["Close"].pct_change().rolling(30).std() * 100
    vol_now = float(vol.iloc[-1]) if len(vol) else float("nan")
    last_price = last_scalar(pd, np, data["Close"])

    bull = bool((data["EMA_Short"] > data["EMA_Long"]).iloc[-1])
    trend_buy = bool(bull and (float(data["MACD"].iloc[-1]) > float(data["MACD_Signal"].iloc[-1])))
    bounce_buy = bool(bb_bounce_from_lower(data) and (30.0 <= float(data["RSI"].iloc[-1]) <= 45.0) and macd_hist_turning_up(data) and ema_gap_small(data, 2.0))
    buy_now = bool(trend_buy or bounce_buy)

    cl, mid, up = float(data["Close"].iloc[-1]), float(data["BB_Mid"].iloc[-1]), float(data["BB_Up"].iloc[-1])
    touch_mid_or_upper = (cl >= mid*0.995) and ((cl <= up*1.005) or (float(data["RSI"].iloc[-1]) >= 60.0))
    sell_weakness = bool(touch_mid_or_upper and macd_hist_turning_down(data))
    classic_bear = bool((not bull) and (float(data["MACD"].iloc[-1]) < float(data["MACD_Signal"].iloc[-1])))
    sell_now = bool(sell_weakness or classic_bear)

    atr_val = last_scalar(pd, np, data["ATR"])
    if atr_val == atr_val and atr_val > 0:
        stop_price_long = last_price - atr_val * 2.0
    else:
        stop_price_long = last_price * (1 - max(stop_pct, 0.5)/100.0)
    if stop_price_long >= last_price * 0.9999:
        stop_price_long = last_price * (1 - max(stop_pct, 1.0)/100.0)
    risk_long = max(last_price - stop_price_long, 1e-6)

    tp1_long = last_price + tp1_r * risk_long
    tp2_long = last_price + tp2_r * risk_long
    tp3_long = last_price + tp3_r * risk_long

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
    trend_txt = "🟢 Yukselis" if bool(bull) else "🔴 Dusus"

    if buy_now: headline = "✅ SINYAL: AL (Trend/Bounce)"
    elif sell_now: headline = "❌ SINYAL: SAT / LONG kapat"
    else: headline = "⏸ SINYAL: BEKLE"

    st.subheader("📌 Ozet")
    colA, colB, colC = st.columns([1.2,1,1])
    with colA:
        st.markdown("**Kripto:** `{}`".format(ticker))
        st.markdown("**Son Fiyat:** **{:.6f}**".format(last_price))
        st.markdown("**Durum:** **{}**".format(headline))
    with colB:
        st.markdown("**Trend:** {}".format(trend_txt))
        st.markdown("**Momentum (RSI-50):** {:+.2f}".format(momentum))
        st.markdown("**Fiyat Volatilitesi (30g sigma %):** {:.2f}%".format(vol_now))
    with colC:
        st.markdown("**Stop Mesafesi:** {}%".format(fmt(stop_dist_pct,2)))
        st.markdown("**Pozisyon Orani:** {}%".format(fmt(pos_ratio_pct,2)))

    def _spark_emoji(x):
        try: x = float(x)
        except Exception: return "–"
        if x > 0: return "🟢"
        elif x < 0: return "🔴"
        else: return "🟠"
    cl_tail = data["Close"].tail(6)
    if len(cl_tail) >= 2:
        chg = cl_tail.diff().tail(5)
        emjs = " ".join([_spark_emoji(x) for x in chg])
        pos = int((chg > 0).sum()); neg = int((chg < 0).sum())
        st.markdown("Son 5 mum: {}  | Pozitif: {}, Negatif: {}".format(emjs, pos, neg))

    st.subheader("🎯 Sinyal (Oneri)")
    if buy_now:
        st.markdown("- **Giris (Long):** **{:.6f}**".format(last_price))
        st.markdown("- **Onerilen Miktar:** ~ **{:.4f}** birim (≈ **${:,.2f}**)".format(qty, position_value))
        st.markdown("- **Stop (Long):** **{:.6f}**".format(stop_price_long))
        st.markdown("- **TP1 / TP2 / TP3 (Long):** **{:.6f}** / **{:.6f}** / **{:.6f}**".format(tp1_long, tp2_long, tp3_long))
    elif sell_now:
        st.markdown("- **Aksiyon:** Long kapat / short dusun")
        try:
            lo, hi, base, atr_val = compute_reentry_zone(data, atr_mult=0.5, lookback=20)
            st.markdown("**💡 Yeniden Alim Bolgesi (kademeli):**")
            st.markdown("- **Aralik:** **{:.6f} – {:.6f}**  (taban ~ **{:.6f}**, ATR~{:.6f})".format(lo, hi, base, atr_val))
            st.caption("Plan: %50 (lo–base), %30 (base), %20 (base–hi). Stop: base - 1xATR alti.")
        except Exception as e:
            st.info("Yeniden alim bolgesi hesaplanamadi: {}".format(e))
    else:
        st.markdown("Net sinyal yok; parametreleri veya zaman dilimini degistirin.")

with tab_guide:
    st.subheader("📘 Rehber - Ozet Notlar")
    st.markdown(\"\"\"{}\"\"\".format(""" + repr(guide_text) + """))
