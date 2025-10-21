
import streamlit as st

st.set_page_config(page_title="TradeMaster | Kripto Strateji Analizoru", layout="wide")
st.title("TradeMaster")
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
    st.info("Kurulum icin: pip install " + " ".join(missing))

def normalize_ohlc(pd, df):
    df = df.copy()
    df.columns = [str(c).title() for c in df.columns]
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    return df

def to_num(pd, np, s):
    if hasattr(s, "astype"):
        s = s.astype(str).str.replace(",", "", regex=False).str.replace(" ", "", regex=False)
        s = s.replace({"": np.nan, "None": np.nan, "NaN": np.nan, "nan": np.nan})
    return pd.to_numeric(s, errors="coerce")

def lastf(pd, np, x):
    try:
        v = x.iloc[-1]
    except Exception:
        try: v = x[-1]
        except Exception: v = x
    try: return float(v)
    except Exception: return float("nan")

def ema(pd, s, n): return s.ewm(span=int(n), adjust=False).mean()

def rsi(pd, np, close, n=14):
    d = close.diff()
    up = d.clip(lower=0); dn = -d.clip(upper=0)
    ru = up.ewm(alpha=1/float(n), adjust=False).mean()
    rd = dn.ewm(alpha=1/float(n), adjust=False).mean()
    rs = ru / (rd + 1e-12)
    return 100 - (100/(1+rs))

def macd(pd, close, fast=12, slow=26, sig=9):
    line = ema(pd, close, fast) - ema(pd, close, slow)
    signal = ema(pd, line, sig)
    hist = line - signal
    return line, signal, hist

def bollinger(close, n=20, k=2.0):
    n = int(n)
    mid = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=0)
    up = mid + k*std; dn = mid - k*std
    return mid, up, dn

def atr(pd, df, n=14):
    if not all(c in df.columns for c in ["High","Low","Close"]):
        return df["Close"]*0
    h,l,c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = (h-l).abs().combine((h-pc).abs(), max).combine((l-pc).abs(), max)
    return tr.ewm(alpha=1/float(n), adjust=False).mean()

def obv(pd, close, volume):
    d = close.diff().fillna(0)
    sign = close.copy()*0
    sign[d>0] = 1; sign[d<0] = -1
    return (sign.fillna(0) * volume.fillna(0)).cumsum()

def anchored_vwap(pd, price, volume, anchor_idx):
    pv = (price*volume).copy(); pv.iloc[:anchor_idx] = 0
    vol = volume.copy(); vol.iloc[:anchor_idx] = 0
    cumv = vol.cumsum().replace(0, float("nan"))
    return pv.cumsum()/cumv

def detect_swings(pd, high, low, w=5):
    HH = (high == high.rolling(w*2+1, center=True).max())
    LL = (low  == low .rolling(w*2+1, center=True).min())
    return pd.DataFrame({"HH": HH.fillna(False), "LL": LL.fillna(False)})

def last_up_swing(pd, high, low, swings):
    idx = swings.index
    lows = idx[swings["LL"]].tolist()
    highs= idx[swings["HH"]].tolist()
    if not lows: return None, None
    ll = lows[-1]; hh = None
    for t in highs[::-1]:
        if t > ll:
            hh = t; break
    return (ll, hh) if hh is not None else (None, None)

def fib_levels(low_p, high_p):
    d = high_p - low_p
    return {"0.382": high_p - 0.382*d, "0.500": high_p - 0.500*d, "0.618": high_p - 0.618*d}

def slope(series, n=10):
    try:
        if len(series) < n: return 0.0
        y = series.tail(n).astype(float)
        x = list(range(len(y)))
        xm = sum(x)/len(x); ym = float(y.mean())
        num = sum((xi-xm)*(float(yi)-ym) for xi,yi in zip(x,y))
        den = sum((xi-xm)**2 for xi in x) or 1.0
        return float(num/den)
    except Exception: return 0.0

st.sidebar.header("Veri Kaynagi")
src = st.sidebar.radio("Kaynak", ["YFinance (internet)", "CSV Yukle"], index=0)
ticker = st.sidebar.text_input("Kripto Deger", value="THETA-USD")
preset = st.sidebar.selectbox("Zaman Dilimi", ["Kisa (4h)","Orta (1g)","Uzun (1hft)"], index=1)
if preset.startswith("Kisa"): interval, period = "4h","180d"
elif preset.startswith("Orta"): interval, period = "1d","1y"
else: interval, period = "1wk","5y"
uploaded = st.sidebar.file_uploader("CSV (opsiyonel)", type=["csv"]) if src=="CSV Yukle" else None

st.sidebar.header("Strateji Ayarlari")
mode = st.sidebar.selectbox("Strateji Modu", ["Otomatik","Trend","Mean-Reversion"])
ema_short = st.sidebar.number_input("EMA Kisa", 3, 50, 9)
ema_long  = st.sidebar.number_input("EMA Uzun", 10, 300, 21)
ema_trend = st.sidebar.number_input("EMA Trend (uzun)", 50, 400, 200, step=10)
rsi_len   = st.sidebar.number_input("RSI Periyot", 5, 50, 14)
rsi_buy_min = st.sidebar.slider("RSI Alim Alt", 10, 60, 35)
rsi_buy_max = st.sidebar.slider("RSI Alim Ust", 40, 90, 60)
macd_fast = st.sidebar.number_input("MACD Hizli", 3, 50, 12)
macd_slow = st.sidebar.number_input("MACD Yavas", 6, 200, 26)
macd_sig  = st.sidebar.number_input("MACD Sinyal", 3, 50, 9)
bb_len    = st.sidebar.number_input("Bollinger Periyot", 5, 100, 20)
bb_std    = st.sidebar.slider("Bollinger Std", 1.0, 3.5, 2.0, 0.1)
don_n     = st.sidebar.slider("Donchian (Trend)", 10, 60, 20, 1)

st.sidebar.header("Risk / Pozisyon")
equity    = st.sidebar.number_input("Sermaye (USD)", min_value=100.0, value=10000.0, step=100.0)
risk_pct  = st.sidebar.slider("Islem riski (%)", 0.2, 5.0, 1.0, 0.1)
max_alloc = st.sidebar.slider("Maks. Pozisyon (%)", 1.0, 100.0, 20.0, 1.0)
stop_mode = st.sidebar.selectbox("Stop Tipi", ["ATR x K","Sabit %"], index=0)
atr_k     = st.sidebar.slider("ATR carpani", 0.5, 5.0, 2.0, 0.1)
stop_pct  = st.sidebar.slider("Sabit Zarar %", 0.5, 20.0, 3.0, 0.5)
tp_mode   = st.sidebar.selectbox("TP modu", ["R-multiple","Sabit %"], index=0)
tp1_r     = st.sidebar.slider("TP1 (R)", 0.5, 5.0, 1.0, 0.1)
tp2_r     = st.sidebar.slider("TP2 (R)", 0.5, 10.0, 2.0, 0.1)
tp3_r     = st.sidebar.slider("TP3 (R)", 0.5, 15.0, 3.0, 0.1)
tp_pct    = st.sidebar.slider("TP %", 0.5, 50.0, 5.0, 0.5)

tab_an, tab_guide = st.tabs(["Analiz","Rehber"])

def load_asset(mods):
    pd, np, yf = mods["pd"], mods["np"], mods["yf"]
    if src == "YFinance (internet)":
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
        if df is None or df.empty: return None
        df = normalize_ohlc(pd, df)
        if "Close" not in df.columns and "Adj Close" in df.columns: df["Close"] = df["Adj Close"]
        df.index.name = "Date"; return df
    else:
        if not uploaded: return None
        raw = pd.read_csv(uploaded)
        raw.columns = [str(c).title() for c in raw.columns]
        if "Date" in raw.columns:
            raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
            raw = raw.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
        for c in ["Open","High","Low","Close","Volume"]:
            if c in raw.columns: raw[c] = to_num(pd, np, raw[c])
        raw = normalize_ohlc(pd, raw)
        return raw

def load_big_tf(yf, tk, interval):
    big_interval = {"4h":"1d","1d":"1wk","1wk":"1mo"}.get(interval, "1d")
    big_period   = {"4h":"1y","1d":"5y","1wk":"10y"}.get(interval, "1y")
    dfb = yf.download(tk, period=big_period, interval=big_interval, auto_adjust=False, progress=False)
    if dfb is None or dfb.empty: return None
    import pandas as pd
    dfb.columns = [str(c).title() for c in dfb.columns]
    if "Close" not in dfb.columns and "Adj Close" in dfb.columns:
        dfb["Close"] = dfb["Adj Close"]
    return dfb

with tab_an:
    mods, missing = need_libs()
    if missing: show_missing(missing); st.stop()
    pd, np = mods["pd"], mods["np"]
    df = load_asset(mods)
    if df is None or df.empty: st.warning("Veri alinamadi."); st.stop()
    if "Close" not in df.columns: st.error("Close kolonu yok."); st.stop()

    for c in ["Open","High","Low","Close","Volume"]:
        if c in df.columns: df[c] = to_num(pd, np, df[c])

    data = df.copy()
    data["EMA_Short"] = ema(pd, data["Close"], ema_short)
    data["EMA_Long"]  = ema(pd, data["Close"], ema_long)
    data["EMA_Trend"] = ema(pd, data["Close"], ema_trend)
    data["RSI"] = rsi(pd, np, data["Close"], rsi_len)
    data["MACD"], data["MACD_Signal"], data["MACD_Hist"] = macd(pd, data["Close"], macd_fast, macd_slow, macd_sig)
    data["BB_Mid"], data["BB_Up"], data["BB_Down"] = bollinger(data["Close"], bb_len, bb_std)
    data["ATR"] = atr(pd, data, 14)

    vol = data["Close"].pct_change().rolling(30).std()*100.0
    vol_now = float(vol.iloc[-1]) if len(vol) else float("nan")

    last_price = lastf(pd, np, data["Close"])
    bull = (data["EMA_Short"] > data["EMA_Long"]).iloc[-1]
    rsi_now = float(data["RSI"].iloc[-1])
    bw = (data["BB_Up"] - data["BB_Down"]) / (data["BB_Mid"].abs() + 1e-9)

    data["OBV"] = obv(pd, data["Close"], data.get("Volume", data["Close"]*0+0))
    regime_bull = bool(data["Close"].iloc[-1] > data["EMA_Trend"].iloc[-1])

    big = load_big_tf(mods["yf"], ticker, interval)
    if big is not None and "Close" in big.columns:
        mtf_ok = bool((ema(pd, big["Close"], ema_short) > ema(pd, big["Close"], ema_long)).iloc[-1])
    else:
        mtf_ok = True

    btc_ok = True
    if ticker.upper() != "BTC-USD" and ticker.upper().endswith("-USD"):
        b = mods["yf"].download("BTC-USD", period=period, interval=interval, auto_adjust=False, progress=False)
        if b is not None and not b.empty:
            b.columns = [str(c).title() for c in b.columns]
            e1b = ema(pd, b["Close"], ema_short); e2b = ema(pd, b["Close"], ema_long)
            mb, sb, hb = macd(pd, b["Close"], macd_fast, macd_slow, macd_sig)
            btc_ok = bool((e1b > e2b).iloc[-1] and (mb.iloc[-1] > sb.iloc[-1]))

    obv_up = slope(data["OBV"], n=10) > 0
    don_hi = data["High"].rolling(int(don_n)).max()

    score = 0
    if regime_bull: score += 20
    if mtf_ok: score += 20
    if btc_ok: score += 15
    if (data["MACD"].iloc[-1] > data["MACD_Signal"].iloc[-1]): score += 15
    if 35 <= rsi_now <= 60: score += 10
    if obv_up: score += 10
    try:
        if (bw.rolling(20).mean().iloc[-1] < bw.median()): score += 10
    except Exception: pass

    if mode == "Mean-Reversion": is_mr = True
    elif mode == "Trend": is_mr = False
    else: is_mr = bool(bw.rolling(20).mean().iloc[-1] > bw.median())

    min_score_trend = 60; min_score_mr = 50
    if not is_mr:
        buy_raw = bool(bull and (data["MACD"].iloc[-1] > data["MACD_Signal"].iloc[-1]) and regime_bull)
        don_ok = bool(data["Close"].iloc[-1] > (don_hi.iloc[-2] if len(don_hi)>=2 else data["Close"].iloc[-1]-1))
        buy_now = bool(buy_raw and (score >= min_score_trend) and don_ok)
    else:
        mr_cond = bool(regime_bull and (data["Close"].iloc[-1] <= data["BB_Down"].iloc[-1]) and (rsi_now < 35))
        buy_now = bool(mr_cond and (score >= min_score_mr))
        don_ok = True

    sell_now = bool((not bull) and (data["MACD"].iloc[-1] < data["MACD_Signal"].iloc[-1]))

    atr_val = lastf(pd, np, data["ATR"])
    if stop_mode == "ATR x K" and atr_val == atr_val and atr_val > 0:
        stop_price_long = last_price - atr_val*atr_k
    else:
        stop_price_long = last_price * (1 - max(stop_pct,0.5)/100.0)
    if stop_price_long >= last_price*0.9999:
        stop_price_long = last_price * (1 - max(stop_pct,1.0)/100.0)
    risk_long = max(last_price - stop_price_long, 1e-6)

    if tp_mode == "R-multiple":
        tp1_long = last_price + tp1_r*risk_long
        tp2_long = last_price + tp2_r*risk_long
        tp3_long = last_price + tp3_r*risk_long
        rr_tp1 = tp1_r
    else:
        tp1_long = last_price*(1+tp_pct/100.0)
        tp2_long = last_price*(1+2*tp_pct/100.0)
        tp3_long = last_price*(1+3*tp_pct/100.0)
        rr_tp1 = (tp1_long-last_price)/risk_long

    rr_veto = False
    if buy_now and rr_tp1 < 1.2:
        rr_veto = True; buy_now = False

    effective_max_alloc = max_alloc * (0.6 if vol_now > 6 else 1.0)
    risk_amount = equity*(risk_pct/100.0)
    qty = risk_amount / risk_long
    pos_val = qty*last_price
    max_val = equity*(effective_max_alloc/100.0)
    if pos_val > max_val:
        scale = max_val/max(pos_val,1e-9)
        qty *= scale; pos_val = qty*last_price

    stop_dist_pct = (last_price - stop_price_long)/last_price*100.0 if last_price>0 else float("nan")
    pos_ratio_pct = (pos_val/equity*100.0) if equity>0 else float("nan")
    momentum = rsi_now - 50.0
    trend_txt = "Yukselis" if bool(bull) else "Dusis"
    mode_txt = "Trend" if not is_mr else "Mean-Reversion"

    if buy_now: headline = "SINyAL: AL (long)"
    elif sell_now: headline = "SINyAL: SAT / LONG kapat"
    else: headline = "SINyAL: BEKLE"

    st.subheader("Ozet")
    c1,c2,c3,c4 = st.columns([1.2,1,1,1])
    with c1:
        st.markdown(f"**Kripto:** `{ticker}`")
        st.markdown(f"**Son Fiyat:** **{last_price:.6f}**")
        st.markdown(f"**Durum:** **{headline}**")
    with c2:
        st.markdown(f"**Trend:** {trend_txt}")
        st.markdown(f"**Momentum (RSI-50):** {momentum:+.2f}")
        st.markdown(f"**Fiyat Volatilitesi (30g sig%):** {vol_now:.2f}%")
    with c3:
        st.markdown(f"**R:R (TP1):** {fmt(rr_tp1,2)}")
        st.markdown(f"**Stop Mesafesi:** {fmt(stop_dist_pct,2)}%")
        st.markdown(f"**Pozisyon Orani:** {fmt(pos_ratio_pct,2)}%")
    with c4:
        st.markdown(f"**Mod:** {mode_txt}")
        st.markdown(f"**Skor:** {int(score)}")
        st.markdown(f"**Esik:** {60 if not is_mr else 50}")

    cl = data["Close"].tail(6)
    if len(cl) >= 2:
        chg = cl.diff().tail(5)
        cols = []
        for v in chg:
            try: x = float(v)
            except Exception: x = 0.0
            if x>0: cols.append("#0f9d58")
            elif x<0: cols.append("#d93025")
            else: cols.append("#9aa0a6")
        dots = "".join([f"<span style='display:inline-block;width:12px;height:12px;border-radius:50%;background:{c};margin-right:6px;'></span>" for c in cols])
        tag = "YUKARI" if bool(bull) else "ASAGI"
        color = "#0f9d58" if bool(bull) else "#d93025"
        st.markdown(f"<div>Son 5 mum kapanis: {dots} <span style='color:{color};font-weight:600'>(Trend filtresi: {tag})</span></div>", unsafe_allow_html=True)

    st.subheader("Sinyal (Oneri)")
    if buy_now:
        st.markdown(f"""
- **Giris (Long):** **{last_price:.6f}**
- **Onerilen Miktar:** ~ **{qty:.4f}** birim (â **${pos_val:,.2f}**)
- **Stop (Long):** **{stop_price_long:.6f}**
- **TP1 / TP2 / TP3 (Long):** **{tp1_long:.6f}** / **{tp2_long:.6f}** / **{tp3_long:.6f}**
- **Risk:** Sermayenin %{risk_pct:.1f} (maks. pozisyon %{effective_max_alloc:.0f})
""")
        st.caption("Not: TP1 gorulunce stop'u maliyete cekmeyi dusunebilirsiniz.")
    elif sell_now:
        st.markdown("""
- **Aksiyon:** **Long kapat / short dusun**
- **Not:** SAT sinyalinde long taraf TP seviyeleri gosterilmez.
""")
        try:
            swings = detect_swings(pd, data["High"], data["Low"], w=5)
            ll_idx, hh_idx = last_up_swing(pd, data["High"], data["Low"], swings)
            if ll_idx is not None and hh_idx is not None:
                low_p  = float(data.loc[ll_idx,"Low"]); high_p = float(data.loc[hh_idx,"High"])
                fibs = fib_levels(low_p, high_p)
                anchor_pos = data.index.get_loc(ll_idx)
                vol_series = data.get("Volume", data["Close"]*0+1)
                avwap = anchored_vwap(pd, data["Close"], vol_series, anchor_pos)
                avwap_now = float(avwap.iloc[-1]) if avwap.notna().any() else None
                ema200 = float(data["EMA_Trend"].iloc[-1]); bb_dn = float(data["BB_Down"].iloc[-1])
                st.markdown("**SAT sonrasi olasi ALIM bolgeleri (kademeli):**")
                st.write(f"- Fib 0.382: **{fibs['0.382']:.6f}**")
                st.write(f"- Fib 0.500: **{fibs['0.500']:.6f}**")
                st.write(f"- Fib 0.618: **{fibs['0.618']:.6f}**")
                st.write(f"- EMA200: **{ema200:.6f}**")
                st.write(f"- Alt Bollinger: **{bb_dn:.6f}**")
                if avwap_now == avwap_now: st.write(f"- Anchored VWAP: **{avwap_now:.6f}**")
                st.caption("Plan: %40 (0.382) â %40 (0.500) â %20 (0.618). Stop: 0.618 alti / swing low alti.")
            else:
                st.info("Uygun salinim bulunamadi. Alternatif: EMA200, Alt BB ve yatay destekler.")
        except Exception as e:
            st.warning(f"Alim bolgesi uretiminde sorun: {e}")
    else:
        st.markdown("Su an belirgin bir sinyal yok; zaman dilimini veya parametreleri degistirin.")

    st.subheader("Gerekceler")
    def line(text, kind="neutral"):
        if kind=="pos": color="#0f9d58"; dot="(+)"
        elif kind=="neg": color="#d93025"; dot="(-)"
        else: color="#f29900"; dot="(~)"
        st.markdown(f"<span style='color:{color};font-weight:600'>{dot} {text}</span>", unsafe_allow_html=True)

    line(f"Strateji modu: {mode_txt}", "neutral")
    if not is_mr: line(f"Donchian {don_n} kirilim: {'gecti' if don_ok else 'gecemedi'}", "pos" if don_ok else "neg")
    if rr_veto: line("RR(TP1) < 1.2 nedeniyle AL veto edildi.", "neg")
    if vol_now > 6: line("Yuksek volatilite: pozisyon kisildi.", "neutral")
    if bool(bull): line("EMA kisa > EMA uzun (yukselis).", "pos")
    else: line("EMA kisa < EMA uzun (dusis).", "neg")
    line(f"Rejim (EMA{ema_trend}): {'boga' if regime_bull else 'ayi'}", "pos" if regime_bull else "neg")
    line(f"MTF: {'uyumlu' if mtf_ok else 'uyumsuz'}", "pos" if mtf_ok else "neg")
    if ticker.upper() != "BTC-USD" and ticker.upper().endswith("-USD"):
        line(f"BTC filtresi: {'destekleyici' if btc_ok else 'destekleyici degil'}", "pos" if btc_ok else "neg")
    line(f"OBV egimi: {'yukari' if slope(data['OBV'],10)>0 else 'asagi/flat'}", "neutral")

    st.markdown("---")
    st.markdown("Not: Kademeli alim tercih edin. Bu uygulama ozet sinyal ve gerekceler gosterir.")

with tab_guide:
    st.subheader("Rehber - Trend vs Mean-Reversion")
    st.markdown("""
Trend modu (kirilim takip): EMA kisa>uzun, Rejim boga (EMA200 ustu), MACD>Signal, Donchian ust kirilim.
Mean-Reversion modu (geri cekilmede alim): Rejim boga, Alt BB temasi, RSI<35.
Ortak filtreler: MTF uyumu, BTC filtresi, OBV egimi, sigma%. RR veto: RR(TP1) < 1.2 ise islemi bekle.
SAT'ta alim bolgeleri: 0.382/0.5/0.618 fib, EMA200, Alt BB, Anchored VWAP.
""")
