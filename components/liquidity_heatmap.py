"""
Liquidity Heatmap — CryptoCompare Verisi
==========================================
Binance TR/Streamlit Cloud'da 403 veriyor.
CryptoCompare: ücretsiz, geo-restriction yok, API key opsiyonel.

HESAPLAMA:
  Her mum için stop loss kümeleri:
    High * (1 + offset) → Short stop cluster
    Low  * (1 - offset) → Long liquidation zone
  Hacimle ağırlıklandırılır, grid hücrelerine snap edilir.
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
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

_BG  = "#0d1117"; _BG2 = "#161b22"
_DG  = "#21262d"; _DG2 = "#30363d"
_TX  = "#e6edf3"; _DT  = "#8b949e"; _DT2 = "#6e7681"
_G   = "#3fb950"; _R   = "#ff7b72"; _B   = "#58a6ff"
_Y   = "#e3b341"; _P   = "#a371f7"

PAIRS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "AVAX", "LINK"]

# Zaman dilimi → (endpoint_type, limit, label)
# endpoint_type: "hour" | "day"
TIMEFRAMES = {
    "12H": ("hour", 12,  "12 Saat"),
    "1D":  ("hour", 24,  "1 Gün"),
    "3D":  ("hour", 72,  "3 Gün"),
    "1W":  ("hour", 168, "1 Hafta"),
    "1M":  ("day",  30,  "1 Ay"),
    "3M":  ("day",  90,  "3 Ay"),
}

_CC_BASE = "https://min-api.cryptocompare.com/data/v2"


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCH — CryptoCompare
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=120, show_spinner=False)
def fetch_ohlcv(symbol: str, ep_type: str, limit: int) -> Optional[list]:
    """
    CryptoCompare OHLCV verisi.
    Returns: [{time, open, high, low, close, volumefrom, volumeto}]
    """
    if not HAS_REQUESTS:
        return None
    endpoint = "histohour" if ep_type == "hour" else "histoday"
    try:
        r = _req.get(
            f"{_CC_BASE}/{endpoint}",
            params={"fsym": symbol, "tsym": "USD", "limit": limit},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("Response") == "Error":
            return None
        return data.get("Data", {}).get("Data", [])
    except Exception:
        return None


@st.cache_data(ttl=30, show_spinner=False)
def fetch_price(symbol: str) -> Optional[dict]:
    """Anlık fiyat + 24H change."""
    if not HAS_REQUESTS:
        return None
    try:
        r = _req.get(
            "https://min-api.cryptocompare.com/data/pricemultifull",
            params={"fsyms": symbol, "tsyms": "USD"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=8,
        )
        r.raise_for_status()
        raw = r.json().get("RAW", {}).get(symbol, {}).get("USD", {})
        return {
            "price":     raw.get("PRICE", 0),
            "change24h": raw.get("CHANGEPCT24HOUR", 0),
            "volume24h": raw.get("TOTALVOLUME24HTO", 0),
            "high24h":   raw.get("HIGH24HOUR", 0),
            "low24h":    raw.get("LOW24HOUR", 0),
        }
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# LIQUIDITY CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

def _to_binance_fmt(cc_candles: list) -> list:
    """CryptoCompare formatını Binance kline formatına çevir."""
    result = []
    for c in cc_candles:
        result.append([
            c["time"] * 1000,   # ms
            str(c["open"]),
            str(c["high"]),
            str(c["low"]),
            str(c["close"]),
            str(c.get("volumefrom", 0)),
            0, "0", 0, 0, 0, 0,
        ])
    return result


def calculate_liquidity_map(klines: list, current_price: float,
                            grid_pct: float = 0.05) -> dict:
    """OHLCV verilerinden stop cluster + likidite haritası üret."""
    all_highs = [float(k[2]) for k in klines]
    all_lows  = [float(k[3]) for k in klines]
    if not all_highs or not all_lows:
        return {"levels":[], "gaps":[], "magnets":[],
                "price_min": current_price*0.95,
                "price_max": current_price*1.05}

    price_max = max(all_highs) * 1.005
    price_min = min(all_lows)  * 0.995
    cell_size = max(current_price * grid_pct / 100, 0.0001)

    levels: dict = {}
    total_vol = sum(float(k[5]) for k in klines) or 1

    def _add(price, short_v, long_v, w):
        gk = round(round(price / cell_size) * cell_size, 8)
        if gk not in levels:
            levels[gk] = {"price": gk, "short_liq": 0.0,
                          "long_liq": 0.0, "total": 0.0}
        levels[gk]["short_liq"] += short_v * w
        levels[gk]["long_liq"]  += long_v  * w
        levels[gk]["total"]     += (short_v + long_v) * w

    for i, k in enumerate(klines):
        h   = float(k[2]); lo = float(k[3])
        vol = float(k[5])
        rw  = 0.4 + 0.6 * (i / max(len(klines) - 1, 1))
        nv  = vol / total_vol * len(klines)

        for offset in [0.001, 0.0025, 0.005, 0.01]:
            strength = nv * max(1 - offset * 40, 0.1) * rw
            sp = h  * (1 + offset)
            lp = lo * (1 - offset)
            if price_min <= sp <= price_max:
                _add(sp, strength, 0, 1.0)
            if price_min <= lp <= price_max:
                _add(lp, 0, strength, 1.0)

    if not levels:
        return {"levels":[], "gaps":[], "magnets":[],
                "price_min": price_min, "price_max": price_max}

    max_total = max(v["total"] for v in levels.values()) or 1
    for lv in levels.values():
        lv["intensity"]       = lv["total"]     / max_total
        lv["short_intensity"] = lv["short_liq"] / max_total
        lv["long_intensity"]  = lv["long_liq"]  / max_total

    sorted_levels = sorted(levels.values(), key=lambda x: x["price"])

    gaps = []
    for i in range(len(sorted_levels) - 1):
        p1, p2  = sorted_levels[i]["price"], sorted_levels[i+1]["price"]
        gap_pct = (p2 - p1) / current_price * 100
        if (gap_pct > grid_pct * 5
                and sorted_levels[i]["total"]   < max_total * 0.05
                and sorted_levels[i+1]["total"] < max_total * 0.05):
            gaps.append({"from": p1, "to": p2, "size_pct": gap_pct})

    magnets = sorted(sorted_levels, key=lambda x: x["total"], reverse=True)[:3]
    magnets = [{"price": m["price"], "intensity": m["intensity"]}
               for m in magnets]

    return {"levels": sorted_levels, "gaps": gaps[:5], "magnets": magnets,
            "price_min": price_min, "price_max": price_max}


# ══════════════════════════════════════════════════════════════════════════════
# CHART
# ══════════════════════════════════════════════════════════════════════════════

def _render_chart(klines, liq_map, current_price, symbol, tf_label):
    if not HAS_PLOTLY:
        st.warning("Plotly gerekli."); return

    fig          = go.Figure()
    levels       = liq_map["levels"]
    price_min    = liq_map["price_min"]
    price_max    = liq_map["price_max"]
    close_prices = [float(k[4]) for k in klines]
    open_times   = [datetime.fromtimestamp(k[0] / 1000) for k in klines]
    threshold    = 0.07

    # ── Likidite bantları ────────────────────────────────────────────────────
    for lv in levels:
        if lv["intensity"] < threshold:
            continue
        is_above  = lv["price"] > current_price
        is_short  = lv["short_intensity"] > lv["long_intensity"]
        is_long   = lv["long_intensity"]  > lv["short_intensity"]
        is_mix    = abs(lv["short_intensity"] - lv["long_intensity"]) < lv["intensity"] * 0.25

        if is_mix:
            r, g, b = 227, 179, 65
        elif is_short and is_above:
            r, g, b = 255, 100, 80
        elif is_long and not is_above:
            r, g, b = 63, 185, 80
        elif is_short:
            r, g, b = 255, 150, 100
        else:
            r, g, b = 100, 200, 120

        af  = lv["intensity"] * 0.15
        al  = min(lv["intensity"] * 0.85, 0.85)
        lw  = 0.6 + lv["intensity"] * 2.5
        bnd = current_price * 0.0004 * (0.5 + lv["intensity"])

        fig.add_hrect(
            y0=lv["price"] - bnd, y1=lv["price"] + bnd,
            fillcolor=f"rgba({r},{g},{b},{af:.3f})",
            line_color="rgba(0,0,0,0)", layer="below",
        )
        fig.add_hline(
            y=lv["price"],
            line=dict(color=f"rgba({r},{g},{b},{al:.3f})",
                      width=lw,
                      dash="dot" if is_mix else "solid"),
        )

    # ── Likidite boşlukları ──────────────────────────────────────────────────
    for gap in liq_map["gaps"][:3]:
        fig.add_hrect(
            y0=gap["from"], y1=gap["to"],
            fillcolor="rgba(163,113,247,0.07)",
            line_color="rgba(163,113,247,0.3)", line_width=0.5, layer="below",
        )
        mid = (gap["from"] + gap["to"]) / 2
        fig.add_annotation(
            y=mid, x=open_times[max(0, int(len(open_times)*0.05))],
            text=f"⚡ Gap {gap['size_pct']:.2f}%",
            font=dict(size=10, color=_P, family="Space Mono"),
            showarrow=False,
        )

    # ── Magnet zones ─────────────────────────────────────────────────────────
    for mag in liq_map["magnets"]:
        if abs(mag["price"] - current_price) / current_price < 0.001:
            continue
        bnd = current_price * 0.001
        fig.add_hrect(
            y0=mag["price"] - bnd, y1=mag["price"] + bnd,
            fillcolor="rgba(227,179,65,0.12)",
            line_color="rgba(227,179,65,0.55)", line_width=1.5,
        )
        fig.add_annotation(
            y=mag["price"], x=open_times[-1],
            text=f"  🧲 ${mag['price']:,.2f}",
            font=dict(size=11, color=_Y, family="Space Mono"),
            showarrow=False, xanchor="left",
        )

    # ── Fiyat çizgisi ────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=open_times, y=close_prices, mode="lines",
        line=dict(color=_B, width=2.5),
        fillcolor="rgba(88,166,255,0.04)",
        name="Price",
        hovertemplate="<b>$%{y:,.4f}</b><br>%{x|%d %b %H:%M}<extra></extra>",
    ))

    # ── Anlık fiyat ──────────────────────────────────────────────────────────
    pc = _G if len(close_prices) >= 2 and close_prices[-1] >= close_prices[-2] else _R
    fig.add_hline(y=current_price,
                  line=dict(color=pc, width=2, dash="dash"))
    fig.add_annotation(
        y=current_price, x=open_times[-1],
        text=f"  ${current_price:,.4f}",
        font=dict(size=13, color=pc, family="Space Mono"),
        showarrow=False, xanchor="left",
    )

    # ── Legend traces ─────────────────────────────────────────────────────────
    for lbl, clr in [
        ("Short Stop Cluster", "rgba(255,100,80,0.9)"),
        ("Long Liq Zone",      "rgba(63,185,80,0.9)"),
        ("Magnet Zone",        "rgba(227,179,65,0.9)"),
        ("Liquidity Gap",      "rgba(163,113,247,0.8)"),
    ]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
            line=dict(color=clr, width=3), name=lbl, showlegend=True))

    fig.update_layout(
        height=560, paper_bgcolor=_BG, plot_bgcolor=_BG,
        font=dict(family="DM Sans,sans-serif", size=13, color=_DT),
        margin=dict(l=10, r=140, t=30, b=50),
        xaxis=dict(gridcolor=_DG, showgrid=True, zeroline=False,
                   tickfont=dict(size=11, color=_DT), tickformat="%d %b\n%H:%M"),
        yaxis=dict(gridcolor=_DG, showgrid=True, zeroline=False,
                   tickfont=dict(size=11, color=_DT),
                   tickformat="$,.2f", side="right",
                   range=[price_min * 0.9985, price_max * 1.0015]),
        legend=dict(bgcolor="rgba(13,17,23,0.92)", bordercolor=_DG, borderwidth=1,
                    font=dict(size=11, color=_DT), orientation="h", y=1.07, x=0),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=_BG2, bordercolor=_DG,
                        font=dict(size=13, color=_TX)),
        title=dict(
            text=f"<b>{symbol}/USD</b> · Liquidity Heatmap · {tf_label}",
            font=dict(size=13, color=_TX, family="Space Mono"), x=0.01, y=0.98,
        ),
    )

    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": True, "scrollZoom": True,
                            "modeBarButtonsToRemove": ["select2d","lasso2d","autoScale2d"],
                            "displaylogo": False})


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL
# ══════════════════════════════════════════════════════════════════════════════

def _render_right_panel(liq_map, price_data, current_price, symbol):

    def _row(label, value, color):
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;"
            f"padding:5px 0;border-bottom:1px solid {_DG}20'>"
            f"<span style='font-size:13px;color:{_DT}'>{label}</span>"
            f"<span style='font-size:13px;font-weight:700;"
            f"font-family:\"Space Mono\",monospace;color:{color}'>{value}</span>"
            f"</div>", unsafe_allow_html=True,
        )

    # Top Liquidity Levels
    st.markdown(
        f"<div style='background:{_BG2};border:1px solid {_DG};"
        f"border-radius:12px;padding:10px 12px;margin-bottom:8px'>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='font-size:12px;font-weight:700;color:{_DT};"
        f"text-transform:uppercase;letter-spacing:0.12em;margin-bottom:8px'>"
        f"🎯 Top Likidite Seviyeleri</div>",
        unsafe_allow_html=True,
    )
    top = sorted([l for l in liq_map["levels"] if l["intensity"] > 0.12],
                 key=lambda x: x["intensity"], reverse=True)[:8]
    for lv in sorted(top, key=lambda x: x["price"], reverse=True):
        is_above = lv["price"] > current_price
        if lv["short_intensity"] > lv["long_intensity"]:
            lc, lt = _R, "Short"
        elif lv["long_intensity"] > lv["short_intensity"]:
            lc, lt = _G, "Long Liq"
        else:
            lc, lt = _Y, "Magnet"
        dist  = abs(lv["price"] - current_price) / current_price * 100
        arrow = "↑" if is_above else "↓"
        bar_w = int(lv["intensity"] * 100)
        st.markdown(
            f"<div style='margin-bottom:7px'>"
            f"<div style='display:flex;justify-content:space-between;margin-bottom:2px'>"
            f"<span style='font-family:\"Space Mono\",monospace;font-size:13px;"
            f"font-weight:700;color:{lc}'>{arrow} ${lv['price']:,.2f}</span>"
            f"<span style='font-size:11px;color:{_DT2}'>{lt} {dist:.2f}%</span>"
            f"</div>"
            f"<div style='height:3px;background:{_DG};border-radius:2px'>"
            f"<div style='width:{bar_w}%;height:3px;background:{lc};"
            f"border-radius:2px;box-shadow:0 0 6px {lc}60'></div></div></div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Market Info
    st.markdown(
        f"<div style='background:{_BG2};border:1px solid {_DG};"
        f"border-radius:12px;padding:10px 12px;margin-bottom:8px'>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='font-size:12px;font-weight:700;color:{_DT};"
        f"text-transform:uppercase;letter-spacing:0.12em;margin-bottom:8px'>"
        f"📊 Market Bilgisi</div>",
        unsafe_allow_html=True,
    )
    if price_data:
        chg  = price_data.get("change24h", 0)
        vol  = price_data.get("volume24h", 0)
        h24  = price_data.get("high24h",   0)
        l24  = price_data.get("low24h",    0)
        _row("24H Change",  f"{'+'if chg>=0 else ''}{chg:.2f}%",  _G if chg>=0 else _R)
        _row("24H High",    f"${h24:,.2f}",  _G)
        _row("24H Low",     f"${l24:,.2f}",  _R)
        _row("24H Volume",  f"${vol/1e9:.2f}B" if vol > 1e9 else f"${vol/1e6:.0f}M", _DT)
    else:
        _row("Veri", "Yükleniyor...", _DT)
    st.markdown("</div>", unsafe_allow_html=True)

    # Analysis
    st.markdown(
        f"<div style='background:{_BG2};border:1px solid {_P}20;"
        f"border-radius:12px;padding:10px 12px'>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='font-size:12px;font-weight:700;color:{_DT};"
        f"text-transform:uppercase;letter-spacing:0.12em;margin-bottom:8px'>"
        f"🤖 Analysis</div>",
        unsafe_allow_html=True,
    )
    above = [l for l in liq_map["levels"]
             if l["price"] > current_price and l["short_intensity"] > 0.25]
    below = [l for l in liq_map["levels"]
             if l["price"] < current_price and l["long_intensity"]  > 0.25]
    lines = []
    if above:
        ca = min(above, key=lambda x: x["price"])
        d  = (ca["price"] - current_price) / current_price * 100
        lines.append(
            f"<b style='color:{_R}'>Short clusters</b> above "
            f"<b style='color:{_TX}'>${ca['price']:,.2f}</b> (+{d:.2f}%). "
            f"Reclaim → squeeze possible."
        )
    if below:
        cb = max(below, key=lambda x: x["price"])
        d  = (current_price - cb["price"]) / current_price * 100
        lines.append(
            f"<b style='color:{_G}'>Long liq zone</b> at "
            f"<b style='color:{_TX}'>${cb['price']:,.2f}</b> (-{d:.2f}%). "
            f"Break → cascade risk."
        )
    if liq_map.get("magnets"):
        m = liq_map["magnets"][0]
        d = abs(m["price"] - current_price) / current_price * 100
        lines.append(
            f"<b style='color:{_Y}'>🧲 Magnet</b> at "
            f"<b style='color:{_TX}'>${m['price']:,.2f}</b> ({d:.2f}% away)."
        )
    if liq_map.get("gaps"):
        g = liq_map["gaps"][0]
        lines.append(
            f"<b style='color:{_P}'>⚡ Gap</b> "
            f"${g['from']:,.2f}–${g['to']:,.2f} ({g['size_pct']:.2f}%)."
        )
    ai_html = "<br><br>".join(lines) if lines else \
        f"Market neutral around ${current_price:,.4f}."
    st.markdown(
        f"<div style='font-size:13px;line-height:1.8;color:{_DT}'>{ai_html}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def render_liquidity_heatmap():
    st.markdown(
        f"<div style='margin-bottom:12px'>"
        f"<div style='font-family:\"Space Mono\",monospace;font-size:1.25rem;"
        f"font-weight:700;color:#f0f6fc'>🔥 LIQUIDITY HEATMAP</div>"
        f"<div style='font-size:13px;color:{_DT2};margin-top:2px'>"
        f"Stop cluster & liquidation zone analysis · "
        f"CryptoCompare · Cache 2min</div></div>",
        unsafe_allow_html=True,
    )

    if not HAS_REQUESTS:
        st.error("`pip install requests` gerekli."); return
    if not HAS_PLOTLY:
        st.error("`pip install plotly` gerekli."); return

    # Kontroller
    c1, c2, c3, c4 = st.columns([2, 2, 2, 4])
    with c1:
        pair = st.selectbox("Pair", PAIRS, key="lhm_pair",
                            label_visibility="collapsed")
    with c2:
        tf_key = st.selectbox("Timeframe", list(TIMEFRAMES.keys()), index=1,
                              key="lhm_tf", label_visibility="collapsed")
    with c3:
        grid_pct = st.select_slider(
            "Grid", options=[0.02, 0.05, 0.1, 0.2, 0.3],
            value=0.05, key="lhm_grid",
            label_visibility="collapsed",
            format_func=lambda x: f"Grid %{x}",
        )
    with c4:
        ra, rb = st.columns(2)
        with ra:
            if st.button("🔄 Yenile", key="lhm_refresh"):
                st.cache_data.clear(); st.rerun()
        with rb:
            demo_mode = st.checkbox("Demo mod", key="lhm_demo",
                                    help="API erişimi yoksa demo veri")

    tf_ep, tf_limit, tf_label = TIMEFRAMES[tf_key]

    # Veri çek
    klines      = None
    price_data  = None
    use_demo    = demo_mode

    if not use_demo:
        with st.spinner(f"CryptoCompare {pair}/USD · {tf_label}..."):
            cc_data    = fetch_ohlcv(pair, tf_ep, tf_limit)
            price_data = fetch_price(pair)

        if cc_data and len(cc_data) >= 3:
            klines = _to_binance_fmt(cc_data)
        else:
            st.warning(
                f"⚠️ CryptoCompare verisi alınamadı. "
                f"Demo mod açılıyor..."
            )
            use_demo = True

    if use_demo:
        klines, current_price = _make_demo(pair, tf_ep, tf_limit)
        price_data = None
        st.info("📊 Demo mod — simüle veri")
    else:
        current_price = float(klines[-1][4])
        if price_data and price_data.get("price"):
            current_price = float(price_data["price"])

    liq_map = calculate_liquidity_map(klines, current_price, grid_pct)

    # KPI Bar
    k1, k2, k3, k4 = st.columns(4)
    chg   = price_data.get("change24h", 0) if price_data else 0
    chg_c = _G if chg >= 0 else _R

    k1.markdown(
        f"<div style='background:{_BG2};border:1px solid {chg_c}40;"
        f"border-radius:10px;padding:10px;text-align:center'>"
        f"<div style='font-size:11px;color:{_DT};margin-bottom:4px'>"
        f"{pair}/USD · {tf_label}</div>"
        f"<div style='font-family:\"Space Mono\",monospace;font-size:1.2rem;"
        f"font-weight:700;color:{chg_c}'>${current_price:,.4f}</div>"
        f"<div style='font-size:12px;color:{chg_c}'>"
        f"{'▲'if chg>=0 else '▼'} {abs(chg):.2f}%</div></div>",
        unsafe_allow_html=True,
    )

    above_lev = [l for l in liq_map["levels"] if l["price"] > current_price]
    below_lev = [l for l in liq_map["levels"] if l["price"] < current_price]
    sp = (sum(l["short_intensity"] for l in above_lev)
          / max(len(above_lev), 1))
    lp = (sum(l["long_intensity"]  for l in below_lev)
          / max(len(below_lev), 1))

    def _risk_kpi(col, val, label):
        if val > 0.5:    lvl, c = "HIGH",   _R
        elif val > 0.25: lvl, c = "MEDIUM", _Y
        else:            lvl, c = "LOW",    _G
        col.markdown(
            f"<div style='background:{_BG2};border:1px solid {c}40;"
            f"border-radius:10px;padding:10px;text-align:center'>"
            f"<div style='font-size:11px;color:{_DT};margin-bottom:4px'>{label}</div>"
            f"<div style='font-size:1.1rem;font-weight:800;color:{c}'>{lvl}</div>"
            f"<div style='margin-top:5px;height:4px;background:{_DG};border-radius:2px'>"
            f"<div style='width:{int(val*100)}%;height:4px;background:{c};"
            f"border-radius:2px'></div></div></div>",
            unsafe_allow_html=True,
        )
    _risk_kpi(k2, sp, "Short Squeeze Risk")
    _risk_kpi(k3, lp, "Long Flush Risk")

    with k4:
        n_lev = len([l for l in liq_map["levels"] if l["intensity"] > 0.12])
        n_gap = len(liq_map["gaps"])
        src_label = "Demo" if use_demo else "CryptoCompare"
        st.markdown(
            f"<div style='background:{_BG2};border:1px solid {_DG};"
            f"border-radius:10px;padding:10px;text-align:center'>"
            f"<div style='font-size:11px;color:{_DT};margin-bottom:4px'>"
            f"Harita · {src_label}</div>"
            f"<div style='font-size:13px;font-weight:700;color:{_TX}'>"
            f"{n_lev} seviye</div>"
            f"<div style='font-size:12px;color:{_P}'>{n_gap} gap</div>"
            f"</div>", unsafe_allow_html=True,
        )

    st.markdown("")

    chart_col, right_col = st.columns([2.2, 1], gap="medium")
    with chart_col:
        st.markdown(
            f"<div style='background:{_BG2};border:1px solid {_DG};"
            f"border-radius:14px;padding:8px 8px 2px'>",
            unsafe_allow_html=True,
        )
        _render_chart(klines, liq_map, current_price, pair, tf_label)
        st.markdown("</div>", unsafe_allow_html=True)
        last_ts = datetime.fromtimestamp(klines[-1][0] / 1000)
        st.markdown(
            f"<div style='font-size:11px;color:{_DT2};margin-top:4px'>"
            f"📡 {len(klines)} mum · Son: {last_ts.strftime('%d %b %H:%M')} "
            f"· Grid %{grid_pct} · {src_label}</div>",
            unsafe_allow_html=True,
        )
    with right_col:
        _render_right_panel(liq_map, price_data, current_price, pair)


# ══════════════════════════════════════════════════════════════════════════════
# DEMO FALLBACK
# ══════════════════════════════════════════════════════════════════════════════

def _make_demo(pair: str, ep_type: str, limit: int):
    base = {"BTC":94500,"ETH":3200,"SOL":175,"BNB":620,
            "XRP":0.62,"DOGE":0.18,"AVAX":38,"LINK":15}.get(pair, 100)
    rng  = random.Random(int(time.time() // 60))
    p    = base
    klines = []
    for i in range(limit):
        o  = p
        c  = p + (rng.random() - 0.49) * base * 0.004
        h  = max(o, c) + rng.random() * base * 0.002
        lo = min(o, c) - rng.random() * base * 0.002
        v  = rng.random() * 5000 + 500
        hrs = limit - i
        ts = int((datetime.now() - timedelta(
            hours=hrs if ep_type == "hour" else hrs * 24
        )).timestamp() * 1000)
        klines.append([ts, str(o), str(h), str(lo), str(c), str(v),
                       0, "0", 0, 0, 0, 0])
        p = c
    return klines, float(klines[-1][4])
