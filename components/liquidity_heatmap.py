"""
Liquidation Heatmap — Gerçek Binance Tasfiye Verisi
====================================================
Binance fapi/v1/allForceOrders + fapi/v1/klines kullanır.
Streamlit Cloud (AWS us-east-1) üzerinden Binance'e erişim açık.

GÖRSEL: CoinGlass tarzı 2D heatmap
  X: zaman  |  Y: fiyat  |  Renk: tasfiye yoğunluğu
  Üste: fiyat çizgisi overlay
"""

import streamlit as st
import time, random
from datetime import datetime, timedelta
from typing import Optional

try:
    import requests as _req
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import plotly.graph_objects as go
    import numpy as np
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

_BG="#0d1117"; _BG2="#161b22"; _DG="#21262d"
_TX="#e6edf3"; _DT="#8b949e"; _DT2="#6e7681"
_G="#3fb950"; _R="#ff7b72"; _B="#58a6ff"; _Y="#e3b341"; _P="#a371f7"

PAIRS = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT"]
TIMEFRAMES = {
    "12H": ("15m", 48,  "12 Saat"),
    "1D":  ("30m", 48,  "1 Gün"),
    "3D":  ("1h",  72,  "3 Gün"),
    "1W":  ("2h",  84,  "1 Hafta"),
    "1M":  ("4h",  180, "1 Ay"),
    "3M":  ("1d",  90,  "3 Ay"),
}
FAPI = "https://fapi.binance.com"

COINGLASS_COLORS = [
    [0.00, "rgb(10,6,35)"],
    [0.12, "rgb(22,12,75)"],
    [0.28, "rgb(28,55,115)"],
    [0.48, "rgb(18,115,135)"],
    [0.65, "rgb(25,155,95)"],
    [0.82, "rgb(95,195,55)"],
    [0.93, "rgb(195,225,25)"],
    [1.00, "rgb(255,255,0)"],
]


# ── DATA FETCH ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def fetch_klines(symbol, interval, limit):
    if not HAS_REQUESTS: return None
    try:
        r = _req.get(f"{FAPI}/fapi/v1/klines",
                     params={"symbol":symbol,"interval":interval,"limit":limit},
                     headers={"User-Agent":"Mozilla/5.0"}, timeout=10)
        r.raise_for_status(); return r.json()
    except: return None

@st.cache_data(ttl=20, show_spinner=False)
def fetch_force_orders(symbol, limit=1000):
    if not HAS_REQUESTS: return None
    try:
        r = _req.get(f"{FAPI}/fapi/v1/allForceOrders",
                     params={"symbol":symbol,"limit":limit},
                     headers={"User-Agent":"Mozilla/5.0"}, timeout=10)
        r.raise_for_status(); return r.json()
    except: return None

@st.cache_data(ttl=15, show_spinner=False)
def fetch_ticker(symbol):
    if not HAS_REQUESTS: return None
    try:
        r = _req.get(f"{FAPI}/fapi/v1/ticker/24hr",
                     params={"symbol":symbol},
                     headers={"User-Agent":"Mozilla/5.0"}, timeout=8)
        r.raise_for_status(); return r.json()
    except: return None

@st.cache_data(ttl=60, show_spinner=False)
def fetch_oi(symbol):
    if not HAS_REQUESTS: return None
    try:
        r = _req.get(f"{FAPI}/fapi/v1/openInterest",
                     params={"symbol":symbol},
                     headers={"User-Agent":"Mozilla/5.0"}, timeout=8)
        r.raise_for_status(); return r.json()
    except: return None

@st.cache_data(ttl=60, show_spinner=False)
def fetch_ls(symbol):
    if not HAS_REQUESTS: return None
    try:
        r = _req.get(f"{FAPI}/futures/data/globalLongShortAccountRatio",
                     params={"symbol":symbol,"period":"1h","limit":1},
                     headers={"User-Agent":"Mozilla/5.0"}, timeout=8)
        r.raise_for_status(); return r.json()
    except: return None

@st.cache_data(ttl=60, show_spinner=False)
def fetch_funding(symbol):
    if not HAS_REQUESTS: return None
    try:
        r = _req.get(f"{FAPI}/fapi/v1/premiumIndex",
                     params={"symbol":symbol},
                     headers={"User-Agent":"Mozilla/5.0"}, timeout=8)
        r.raise_for_status(); return r.json()
    except: return None


# ── MATRIX BUILDER ────────────────────────────────────────────────────────────

def build_matrix(klines, force_orders, n_bins=80):
    if not klines: return None
    all_h = [float(k[2]) for k in klines]
    all_l = [float(k[3]) for k in klines]
    p_min = min(all_l)*0.997; p_max = max(all_h)*1.003
    p_bins = np.linspace(p_min, p_max, n_bins)
    t_bins = [datetime.fromtimestamp(k[0]/1000) for k in klines]
    n_time = len(klines)
    z_long  = np.zeros((n_bins, n_time))
    z_short = np.zeros((n_bins, n_time))

    candle_ms = (klines[1][0]-klines[0][0]) if len(klines)>1 else 3600000
    t_start = klines[0][0]; t_end = klines[-1][0]+candle_ms

    if force_orders:
        for o in force_orders:
            ts    = int(o.get("time",0))
            price = float(o.get("averagePrice", o.get("price",0)))
            qty   = float(o.get("origQty",0))
            side  = o.get("side","")
            if not (t_start <= ts <= t_end) or price<=0: continue
            t_idx = max(0, min(int((ts-t_start)/candle_ms), n_time-1))
            p_idx = max(0, min(int(np.searchsorted(p_bins,price))-1, n_bins-1))
            notional = price*qty
            for ti in range(t_idx, n_time):
                decay = max(1.0-(ti-t_idx)/(n_time*2), 0.1)
                if side=="BUY":  z_long[p_idx,ti]  += notional*decay
                else:            z_short[p_idx,ti] += notional*decay

    z_total = z_long+z_short
    mx = z_total.max() or 1
    col_sums = z_total.sum(axis=1)
    top_idx  = np.argsort(col_sums)[-10:][::-1]
    tops = []
    for idx in top_idx:
        if col_sums[idx]>0:
            lv = z_long[idx].sum(); sv = z_short[idx].sum()
            tops.append({"price":float(p_bins[idx]),"total":float(col_sums[idx]),
                         "long_usd":float(lv),"short_usd":float(sv),
                         "type":"Long Liq" if lv>sv else "Short Cluster"})
    return {
        "z": z_total/mx, "z_long": z_long/mx, "z_short": z_short/mx,
        "z_raw": z_total, "p_bins": p_bins, "t_bins": t_bins,
        "p_min": p_min, "p_max": p_max, "tops": tops,
        "total_usd": float(z_total.sum()),
        "real": bool(force_orders and len(force_orders)>0),
    }


def make_demo(symbol, tf_key):
    tf = TIMEFRAMES.get(tf_key,("1h",72,"3 Gün"))
    n  = tf[1]
    base = {"BTCUSDT":78500,"ETHUSDT":1600,"SOLUSDT":145,
            "BNBUSDT":600,"XRPUSDT":0.55,"DOGEUSDT":0.16}.get(symbol,1000)
    rng = random.Random(int(time.time()//120))
    p = base; klines=[]
    for i in range(n):
        o=p; c=p+(rng.random()-0.49)*base*0.004
        h=max(o,c)+rng.random()*base*0.003; lo=min(o,c)-rng.random()*base*0.003
        v=rng.random()*8000+500
        ts=int((datetime.now()-timedelta(hours=n-i)).timestamp()*1000)
        klines.append([ts,str(o),str(h),str(lo),str(c),str(v),0,"0",0,0,0,0])
        p=c
    fos=[]
    t0=klines[0][0]; t1=klines[-1][0]
    for _ in range(800):
        price=base+(rng.random()-0.5)*base*0.07
        ts=int(t0+rng.random()*(t1-t0))
        side=rng.choice(["BUY","SELL"])
        for _ in range(int(rng.random()*10)):
            fos.append({"time":ts+int(rng.random()*3600000),
                        "price":str(price+(rng.random()-0.5)*base*0.002),
                        "averagePrice":str(price),
                        "origQty":str(rng.random()*3+0.01),"side":side})
    cur=float(klines[-1][4])
    tkr={"priceChangePercent":str((rng.random()-0.5)*4),"lastPrice":str(cur),
         "highPrice":str(cur*1.02),"lowPrice":str(cur*0.98),
         "quoteVolume":str(cur*n*1000)}
    return klines, fos, cur, tkr


# ── CHART ─────────────────────────────────────────────────────────────────────

def render_chart(mat, klines, cur, symbol, tf_label):
    fig = go.Figure()

    # Heatmap
    fig.add_trace(go.Heatmap(
        x=mat["t_bins"], y=[float(p) for p in mat["p_bins"]], z=mat["z"],
        colorscale=COINGLASS_COLORS, showscale=True,
        colorbar=dict(
            title=dict(text="Liq", font=dict(size=11,color=_DT)),
            tickfont=dict(size=10,color=_DT),
            bgcolor=_BG2, bordercolor=_DG, thickness=12, len=0.7,
        ),
        hovertemplate="<b>$%{y:,.2f}</b><br>%{x|%d %b %H:%M}<br>Intensity:%{z:.3f}<extra></extra>",
        name="Liquidation", zsmooth="best",
    ))

    # Fiyat çizgisi
    closes = [float(k[4]) for k in klines]
    times  = [datetime.fromtimestamp(k[0]/1000) for k in klines]
    fig.add_trace(go.Scatter(
        x=times, y=closes, mode="lines",
        line=dict(color="rgba(255,255,255,0.92)", width=1.8),
        name="Price",
        hovertemplate="<b>$%{y:,.2f}</b><br>%{x|%d %b %H:%M}<extra></extra>",
    ))

    # Anlık fiyat
    pc = _G if len(closes)>=2 and closes[-1]>=closes[-2] else _R
    fig.add_hline(y=cur, line=dict(color=pc, width=1.5, dash="dash"))
    fig.add_annotation(y=cur, x=times[-1],
        text=f"  ${cur:,.2f}",
        font=dict(size=12,color=pc,family="Space Mono"),
        showarrow=False, xanchor="left")

    # Top liq annotations
    for lv in mat["tops"][:4]:
        lc = _G if lv["type"]=="Long Liq" else _R
        usd = lv["total"]
        label = f"  ${lv['price']:,.2f}  ${usd/1e6:.1f}M" if usd>1e6 else f"  ${lv['price']:,.2f}"
        fig.add_annotation(y=lv["price"], x=times[-1],
            text=label, font=dict(size=10,color=lc,family="Space Mono"),
            showarrow=False, xanchor="left")

    fig.update_layout(
        height=580, paper_bgcolor=_BG, plot_bgcolor=_BG,
        font=dict(family="DM Sans,sans-serif",size=12,color=_DT),
        margin=dict(l=10,r=150,t=30,b=40),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)",showgrid=True,zeroline=False,
                   tickfont=dict(size=11,color=_DT),tickformat="%d %b\n%H:%M"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)",showgrid=True,zeroline=False,
                   tickfont=dict(size=11,color=_DT),tickformat="$,.0f",side="right",
                   range=[mat["p_min"]*0.9985, mat["p_max"]*1.0015]),
        legend=dict(bgcolor="rgba(13,17,23,0.85)",bordercolor=_DG,borderwidth=1,
                    font=dict(size=11,color=_DT),orientation="h",y=1.07,x=0),
        hovermode="closest",
        title=dict(text=f"<b>{symbol}</b> · Liquidation Heatmap · {tf_label}",
                   font=dict(size=13,color=_TX,family="Space Mono"),x=0.01,y=0.985),
    )
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar":True,"scrollZoom":True,
                            "modeBarButtonsToRemove":["select2d","lasso2d","autoScale2d"],
                            "displaylogo":False})


# ── RIGHT PANEL ───────────────────────────────────────────────────────────────

def render_right(mat, ticker, funding, ls_data, oi_data, cur):
    def row(label, val, c=_DT):
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;"
            f"padding:5px 0;border-bottom:1px solid {_DG}20'>"
            f"<span style='font-size:13px;color:{_DT}'>{label}</span>"
            f"<span style='font-size:13px;font-weight:700;"
            f"font-family:\"Space Mono\",monospace;color:{c}'>{val}</span>"
            f"</div>", unsafe_allow_html=True)

    # Seviyeler
    st.markdown(f"<div style='background:{_BG2};border:1px solid {_DG};"
                f"border-radius:12px;padding:10px 12px;margin-bottom:8px'>",
                unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:12px;font-weight:700;color:{_DT};"
                f"text-transform:uppercase;letter-spacing:0.12em;margin-bottom:8px'>"
                f"🎯 Top Liq Seviyeleri</div>", unsafe_allow_html=True)
    for lv in mat["tops"][:8]:
        lc = _R if lv["type"]=="Short Cluster" else _G
        is_a = lv["price"]>cur
        dist = abs(lv["price"]-cur)/cur*100
        arrow = "↑" if is_a else "↓"
        usd = lv["total"]
        usd_s = f"${usd/1e6:.1f}M" if usd>1e6 else f"${usd/1e3:.0f}K"
        bw = min(int(lv["total"]/max(mat["tops"][0]["total"],1)*100),100)
        st.markdown(
            f"<div style='margin-bottom:7px'>"
            f"<div style='display:flex;justify-content:space-between;margin-bottom:2px'>"
            f"<span style='font-family:\"Space Mono\",monospace;font-size:13px;"
            f"font-weight:700;color:{lc}'>{arrow} ${lv['price']:,.2f}</span>"
            f"<span style='font-size:11px;color:{_DT2}'>{lv['type']} · {usd_s} · {dist:.2f}%</span>"
            f"</div>"
            f"<div style='height:3px;background:{_DG};border-radius:2px'>"
            f"<div style='width:{bw}%;height:3px;background:{lc};"
            f"border-radius:2px;box-shadow:0 0 6px {lc}60'></div></div></div>",
            unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Market
    st.markdown(f"<div style='background:{_BG2};border:1px solid {_DG};"
                f"border-radius:12px;padding:10px 12px;margin-bottom:8px'>",
                unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:12px;font-weight:700;color:{_DT};"
                f"text-transform:uppercase;letter-spacing:0.12em;margin-bottom:8px'>"
                f"📊 Market Bilgisi</div>", unsafe_allow_html=True)
    if ticker:
        chg=float(ticker.get("priceChangePercent",0))
        row("24H Change",f"{'+'if chg>=0 else ''}{chg:.2f}%",_G if chg>=0 else _R)
        h=float(ticker.get("highPrice",0)); l=float(ticker.get("lowPrice",0))
        if h: row("24H High",f"${h:,.2f}",_G)
        if l: row("24H Low",f"${l:,.2f}",_R)
        v=float(ticker.get("quoteVolume",0))
        row("24H Vol",f"${v/1e9:.2f}B" if v>1e9 else f"${v/1e6:.0f}M",_DT)
    if funding:
        fr=float(funding.get("lastFundingRate",0))*100
        row("Funding",f"{'+'if fr>=0 else ''}{fr:.4f}%",
            _R if fr>0.02 else _G if fr<-0.02 else _Y)
    if oi_data:
        oi=float(oi_data.get("openInterest",0))
        row("OI",f"${oi/1e9:.2f}B" if oi>1e9 else f"${oi/1e6:.1f}M",_B)
    if ls_data and len(ls_data)>0:
        lr=float(ls_data[0].get("longAccount",0.5))*100
        row("L/S",f"L:{lr:.1f}% S:{100-lr:.1f}%",
            _G if lr>52 else _R if lr<48 else _Y)
    t=mat["total_usd"]
    row("Liq (görünür)",f"${t/1e9:.2f}B" if t>1e9 else f"${t/1e6:.1f}M",_Y)
    st.markdown("</div>", unsafe_allow_html=True)

    # Analiz
    st.markdown(f"<div style='background:{_BG2};border:1px solid {_P}20;"
                f"border-radius:12px;padding:10px 12px'>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:12px;font-weight:700;color:{_DT};"
                f"text-transform:uppercase;letter-spacing:0.12em;margin-bottom:8px'>"
                f"🤖 Analiz</div>", unsafe_allow_html=True)
    above=[l for l in mat["tops"] if l["price"]>cur]
    below=[l for l in mat["tops"] if l["price"]<cur]
    lines=[]
    if above:
        ca=min(above,key=lambda x:x["price"]); d=(ca["price"]-cur)/cur*100
        lines.append(f"<b style='color:{_R}'>Short cluster</b> "
                     f"<b style='color:{_TX}'>${ca['price']:,.2f}</b> "
                     f"(+{d:.2f}%). Reclaim → squeeze.")
    if below:
        cb=max(below,key=lambda x:x["price"]); d=(cur-cb["price"])/cur*100
        lines.append(f"<b style='color:{_G}'>Long liq</b> "
                     f"<b style='color:{_TX}'>${cb['price']:,.2f}</b> "
                     f"(-{d:.2f}%). Break → cascade.")
    ai="<br><br>".join(lines) if lines else "Veri bekleniyor..."
    st.markdown(f"<div style='font-size:13px;line-height:1.8;color:{_DT}'>{ai}</div>",
                unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ── MAIN ─────────────────────────────────────────────────────────────────────

def render_liquidity_heatmap():
    st.markdown(
        f"<div style='margin-bottom:12px'>"
        f"<div style='font-family:\"Space Mono\",monospace;font-size:1.25rem;"
        f"font-weight:700;color:#f0f6fc'>🔥 LIQUIDATION HEATMAP</div>"
        f"<div style='font-size:13px;color:{_DT2};margin-top:2px'>"
        f"Gerçek Binance allForceOrders · CoinGlass tarzı 2D harita · Cache 20s"
        f"</div></div>", unsafe_allow_html=True)

    if not HAS_REQUESTS:
        st.error("pip install requests"); return
    if not HAS_PLOTLY:
        st.error("pip install plotly numpy"); return

    c1,c2,c3,c4 = st.columns([2,2,2,4])
    with c1: symbol = st.selectbox("Pair",PAIRS,key="lhm_pair",label_visibility="collapsed")
    with c2: tf_key = st.selectbox("TF",list(TIMEFRAMES.keys()),index=2,key="lhm_tf",label_visibility="collapsed")
    with c3: n_bins = st.select_slider("Bins",options=[40,60,80,100,120],value=80,key="lhm_bins",
                                        label_visibility="collapsed",format_func=lambda x:f"{x} seviye")
    with c4:
        ra,rb = st.columns(2)
        with ra:
            if st.button("🔄 Yenile",key="lhm_refresh"): st.cache_data.clear(); st.rerun()
        with rb: use_demo = st.checkbox("Demo",key="lhm_demo")

    tf_iv, tf_lim, tf_lbl = TIMEFRAMES[tf_key]
    klines=fo=ticker=funding=oi_data=ls_data=None

    if not use_demo:
        with st.spinner(f"Binance {symbol} yükleniyor..."):
            klines  = fetch_klines(symbol, tf_iv, tf_lim)
            fo      = fetch_force_orders(symbol, 1000)
            ticker  = fetch_ticker(symbol)
            funding = fetch_funding(symbol)
            oi_data = fetch_oi(symbol)
            ls_data = fetch_ls(symbol)
        if klines is None:
            st.warning("⚠️ Binance erişilemiyor (TR IP). Streamlit Cloud'da gerçek veri çalışır. Demo açılıyor...")
            use_demo = True

    if use_demo:
        klines, fo, cur, ticker = make_demo(symbol, tf_key)
        st.info("📊 Demo mod — simüle tasfiye verisi")
    else:
        cur = float(klines[-1][4])
        if ticker: cur = float(ticker.get("lastPrice", cur))

    mat = build_matrix(klines, fo or [], n_bins)
    if not mat: st.error("Matris oluşturulamadı."); return

    # KPI
    k1,k2,k3,k4 = st.columns(4)
    chg = float(ticker.get("priceChangePercent",0)) if ticker else 0
    chg_c = _G if chg>=0 else _R
    k1.markdown(
        f"<div style='background:{_BG2};border:1px solid {chg_c}40;"
        f"border-radius:10px;padding:10px;text-align:center'>"
        f"<div style='font-size:11px;color:{_DT};margin-bottom:4px'>{symbol} · {tf_lbl}</div>"
        f"<div style='font-family:\"Space Mono\",monospace;font-size:1.2rem;"
        f"font-weight:700;color:{chg_c}'>${cur:,.2f}</div>"
        f"<div style='font-size:12px;color:{chg_c}'>{'▲'if chg>=0 else '▼'} {abs(chg):.2f}%</div>"
        f"</div>", unsafe_allow_html=True)

    t = mat["total_usd"]
    k2.markdown(
        f"<div style='background:{_BG2};border:1px solid {_Y}40;"
        f"border-radius:10px;padding:10px;text-align:center'>"
        f"<div style='font-size:11px;color:{_DT};margin-bottom:4px'>Toplam Tasfiye</div>"
        f"<div style='font-family:\"Space Mono\",monospace;font-size:1.1rem;"
        f"font-weight:700;color:{_Y}'>{'${:.2f}B'.format(t/1e9) if t>1e9 else '${:.0f}M'.format(t/1e6)}</div>"
        f"<div style='font-size:11px;color:{_DT}'>{'Gerçek' if mat['real'] else 'Demo'}</div>"
        f"</div>", unsafe_allow_html=True)

    zl=mat["z_long"].sum(); zs=mat["z_short"].sum()
    def kpi(col,val,lbl):
        lvl,c = ("HIGH",_R) if val>0.6 else ("MED",_Y) if val>0.35 else ("LOW",_G)
        col.markdown(
            f"<div style='background:{_BG2};border:1px solid {c}40;"
            f"border-radius:10px;padding:10px;text-align:center'>"
            f"<div style='font-size:11px;color:{_DT};margin-bottom:4px'>{lbl}</div>"
            f"<div style='font-size:1.1rem;font-weight:800;color:{c}'>{lvl}</div>"
            f"<div style='height:4px;background:{_DG};border-radius:2px;margin-top:5px'>"
            f"<div style='width:{int(val*100)}%;height:4px;background:{c};"
            f"border-radius:2px'></div></div></div>", unsafe_allow_html=True)
    kpi(k3, min(zs/(zl+zs+1e-9)*2, 1.0), "Short Squeeze Risk")
    kpi(k4, min(zl/(zl+zs+1e-9)*2, 1.0), "Long Flush Risk")
    st.markdown("")

    cc, rc = st.columns([2.2,1], gap="medium")
    with cc:
        st.markdown(f"<div style='background:{_BG2};border:1px solid {_DG};"
                    f"border-radius:14px;padding:8px 8px 2px'>", unsafe_allow_html=True)
        render_chart(mat, klines, cur, symbol, tf_lbl)
        st.markdown("</div>", unsafe_allow_html=True)
        n_fo = len(fo) if fo else 0; src = "Binance" if mat["real"] else "Demo"
        st.markdown(f"<div style='font-size:11px;color:{_DT2};margin-top:4px'>"
                    f"📡 {src} · {n_fo} tasfiye · {len(klines)} mum · {n_bins} bin"
                    f"</div>", unsafe_allow_html=True)
    with rc:
        render_right(mat, ticker, funding, ls_data, oi_data, cur)
