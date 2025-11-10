from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.data import load_ohlcv
from utils.patterns import annotate_patterns
from utils.sr import pivot_zones, tag_price_context, acceptance_rejection
from utils.price_action import price_volume_assessment, wick_stats
from utils.backtest import simulate

st.set_page_config(page_title="Price Action (Streamlit)", layout="wide")

st.title("📈 Price Action – Streamlit App")
st.caption("S/R + Candlestick + Acceptance/Rejection + Hacim yorumu + Basit backtest (swing/trend).")

with st.sidebar:
    symbol = st.text_input("Sembol", value="BTC-USD")
    interval = st.selectbox("Zaman Dilimi", ["1h","4h","1d","1wk"], index=2)
    period = st.selectbox("Veri Aralığı", ["1y","2y","3y","5y","max"], index=2)
    order = st.slider("Pivot order (S/R)", 3, 20, 7)
    band = st.slider("S/R birleştirme bandı (%)", 0.1, 1.0, 0.3, step=0.1) / 100.0
    tol = st.slider("S/R yakınlık toleransı (%)", 0.1, 1.0, 0.3, step=0.1) / 100.0
    vol_win = st.slider("Hacim spike pencere", 10, 60, 20)
    vol_z = st.slider("Hacim spike z-skoru", 1.0, 4.0, 2.0, step=0.1)
    min_rr = st.slider("Min Risk/Ödül", 1.0, 3.0, 1.5, step=0.1)

df = load_ohlcv(symbol, interval=interval, period=period)
pats = annotate_patterns(df)
zones = pivot_zones(df, order=order, band=band)
ctx = tag_price_context(df, zones, tol=tol)
ar = acceptance_rejection(df, zones)
pva = price_volume_assessment(df, win=vol_win, z=vol_z)
w = wick_stats(df)

# Grafik
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
    name="OHLC"
))
# S/R bantları
for _, z in zones.iterrows():
    fig.add_hrect(y0=z.lo, y1=z.hi, line_width=0, fillcolor="rgba(59,130,246,0.08)" if z.kind=="S" else "rgba(239,68,68,0.08)",
                  annotation_text=("S" if z.kind=="S" else "R"), annotation_position="inside top left")
# Acceptance / Rejection işaretleri
markers = []
markers.append({"series": ar=="reject_res", "name":"Reject@Res", "symbol":"triangle-down"})
markers.append({"series": ar=="reject_sup", "name":"Reject@Sup", "symbol":"triangle-up"})
markers.append({"series": ar=="accept_res_break", "name":"Accept Res Break", "symbol":"circle"})
markers.append({"series": ar=="accept_sup_break", "name":"Accept Sup Break", "symbol":"circle-open"})

for m in markers:
    idxs = df.index[m["series"]]
    fig.add_trace(go.Scatter(
        x=idxs, y=df.loc[idxs, "Close"], mode="markers",
        marker_symbol=m["symbol"], marker_size=10, name=m["name"]
    ))

st.plotly_chart(fig, use_container_width=True)

# Hacim ve wick tablo
tab1, tab2, tab3 = st.tabs(["🔎 Pattern & Bağlam", "📊 Hacim Yorumu", "🧪 Basit Backtest"])

with tab1:
    st.subheader("Candlestick Pattern Etiketleri")
    view = pd.concat([df[["Open","High","Low","Close","Volume"]], pats.astype(int), ctx.rename("context"), ar.rename("acc_rej")], axis=1)
    st.dataframe(view.tail(200))

with tab2:
    st.subheader("Hacim Spike + Wick Tespiti")
    vol_df = pd.concat([df["Volume"].rename("Volume"), w, pva.rename("vol_flag")], axis=1)
    st.dataframe(vol_df.tail(200))

with tab3:
    st.subheader("Swing/Trend mantığına uygun basit girişler")
    # Giriş mantığı: S/R bandı yakınında pattern + acceptance/rejection onayı
    entries = []
    close = df["Close"]; high=df["High"]; low=df["Low"]

    for ts in df.index[5:]:
        near_sup = (ctx.loc[ts] == "near_sup")
        near_res = (ctx.loc[ts] == "near_res")
        long_ok = near_sup and (pats.loc[ts, ["hammer","pin_bull","bull_engulf","harami_bull","morning_star"]].any() or ar.loc[ts]=="reject_sup" or ar.loc[ts]=="accept_res_break") and (pva.loc[ts]!="vol_bear")
        short_ok= near_res and (pats.loc[ts, ["hangman","pin_bear","bear_engulf","harami_bear","evening_star"]].any() or ar.loc[ts]=="reject_res" or ar.loc[ts]=="accept_sup_break") and (pva.loc[ts]!="vol_bull")
        if long_ok:
            # stop = son barın dibi, hedef = en yakın direnç bandı (yoksa RRR=1.8)
            stop = low.loc[ts]
            next_res = zones[zones.kind=="R"]
            if not next_res.empty:
                # mevcut fiyattan yukarıdaki en yakın R
                above = next_res[next_res.lo > close.loc[ts]]
                target = (above.lo.min() if not above.empty else close.loc[ts]*(1+0.03))
            else:
                target = close.loc[ts]*(1+0.03)
            entries.append({"ts": ts, "side":"long", "entry_price": close.loc[ts], "stop": stop, "target": float(target)})
        elif short_ok:
            stop = high.loc[ts]
            next_sup = zones[zones.kind=="S"]
            if not next_sup.empty:
                below = next_sup[next_sup.hi < close.loc[ts]]
                target = (below.hi.max() if not below.empty else close.loc[ts]*(1-0.03))
            else:
                target = close.loc[ts]*(1-0.03)
            entries.append({"ts": ts, "side":"short", "entry_price": close.loc[ts], "stop": stop, "target": float(target)})

    if entries:
        ent = pd.DataFrame(entries).set_index("ts")
        res = simulate(df, ent, min_rr=min_rr)
        st.write("Toplam İşlem:", len(res))
        if not res.empty:
            st.write("Toplam PnL:", round(res["pnl"].sum(), 4))
            st.dataframe(res.tail(100))
    else:
        st.info("Bu parametrelerle henüz sinyal yok.")
