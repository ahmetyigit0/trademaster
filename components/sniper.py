"""
Sniper Terminal
==============
Manuel seviye girişi → otomatik trade planı + Plotly chart
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_BG  = "#0a0a0a"; _BG2 = "#111111"; _BG3 = "#161b22"
_DG  = "#1e1e1e"; _DG2 = "#2a2a2a"
_TX  = "#e6edf3"; _DT  = "#8b949e"; _DT2 = "#6e7681"
_G   = "#00ff88"; _R   = "#ff4444"; _Y   = "#f0b429"
_B   = "#58a6ff"; _O   = "#ff8c00"; _P   = "#a371f7"
_W   = "#ffffff"

# ── Yardımcılar ───────────────────────────────────────────────────────────────

def _parse_zones(raw: str) -> list[tuple[float, float]]:
    """'73000-74000 71000-71500' → [(73000,74000),(71000,71500)]"""
    zones = []
    for part in raw.replace(",", " ").split():
        part = part.strip()
        if "-" in part:
            halves = part.split("-")
            try:
                lo = float(halves[0].strip())
                hi = float(halves[1].strip())
                if lo > 0 and hi > 0:
                    zones.append((min(lo, hi), max(lo, hi)))
            except ValueError:
                pass
        else:
            try:
                v = float(part)
                if v > 0:
                    zones.append((v * 0.999, v * 1.001))
            except ValueError:
                pass
    return zones

def _parse_levels(raw: str) -> list[float]:
    """'78500 80200 81500' → [78500.0, 80200.0, 81500.0]"""
    levels = []
    for part in raw.replace(",", " ").split():
        try:
            v = float(part.strip())
            if v > 0:
                levels.append(v)
        except ValueError:
            pass
    return levels

def _calc_rr(entry, sl, tp):
    risk   = abs(entry - sl)
    reward = abs(tp - entry)
    if risk == 0:
        return 0.0
    return round(reward / risk, 2)

def _fmt(v):
    if v >= 1000:
        return f"{v:,.2f}"
    return f"{v:.4f}"

def _closest_zone(price: float, zones: list[tuple]) -> tuple | None:
    if not zones:
        return None
    return min(zones, key=lambda z: abs((z[0]+z[1])/2 - price))

def _closest_level(price: float, levels: list[float]) -> float | None:
    if not levels:
        return None
    return min(levels, key=lambda l: abs(l - price))

def _generate_plan(price, direction, supports, resistances, liq_levels, sl_pct):
    """Trade planı üret."""
    plan = {}

    if direction == "LONG":
        sup = _closest_zone(price, supports)
        if not sup:
            return None
        mid_sup = (sup[0] + sup[1]) / 2
        entry   = sup[1]                      # destek üst sınırı
        sl      = sup[0] * (1 - sl_pct/100)  # destek altı
        res1    = _closest_zone(price, resistances)
        liq_up  = [l for l in liq_levels if l > entry]

        tp1 = res1[0] if res1 else entry * 1.03
        tp2_candidates = sorted([l for l in liq_up if l > tp1])
        tp2 = tp2_candidates[0] if tp2_candidates else (
              res1[1] if res1 else entry * 1.06)

        plan = dict(direction="LONG", entry=entry, sl=sl, tp1=tp1, tp2=tp2,
                    rr1=_calc_rr(entry,sl,tp1), rr2=_calc_rr(entry,sl,tp2),
                    zone=sup)

    else:  # SHORT
        res = _closest_zone(price, resistances)
        if not res:
            return None
        entry  = res[0]                       # direnç alt sınırı
        sl     = res[1] * (1 + sl_pct/100)   # direnç üstü
        sup1   = _closest_zone(price, supports)
        liq_dn = [l for l in liq_levels if l < entry]

        tp1 = sup1[1] if sup1 else entry * 0.97
        tp2_candidates = sorted([l for l in liq_dn if l < tp1], reverse=True)
        tp2 = tp2_candidates[0] if tp2_candidates else (
              sup1[0] if sup1 else entry * 0.94)

        plan = dict(direction="SHORT", entry=entry, sl=sl, tp1=tp1, tp2=tp2,
                    rr1=_calc_rr(entry,sl,tp1), rr2=_calc_rr(entry,sl,tp2),
                    zone=res)

    return plan


def _build_chart(price, supports, resistances, liq_levels, plan,
                 show_ema20, show_ema50, show_ema200,
                 ema20_val, ema50_val, ema200_val):
    """Plotly chart oluştur."""

    # Fiyat aralığı belirle
    all_prices = [price]
    for lo,hi in supports + resistances:
        all_prices += [lo, hi]
    all_prices += liq_levels
    if plan:
        all_prices += [plan["entry"], plan["sl"], plan["tp1"], plan["tp2"]]

    p_min = min(all_prices) * 0.992
    p_max = max(all_prices) * 1.008
    x_range = [0, 100]

    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        margin=dict(l=10, r=10, t=10, b=10),
        height=520,
        xaxis=dict(visible=False, range=x_range),
        yaxis=dict(
            gridcolor="#1a1a1a", gridwidth=0.5,
            tickfont=dict(color=_DT, size=11, family="monospace"),
            tickformat=",.0f",
            range=[p_min, p_max],
            side="right",
        ),
        showlegend=False,
        font=dict(family="monospace", color=_TX),
        hovermode="y unified",
    )

    # ── Destek kutuları ───────────────────────────────────────────────────────
    for i, (lo, hi) in enumerate(supports):
        mid = (lo + hi) / 2
        fig.add_shape(type="rect",
            x0=0, x1=100, y0=lo, y1=hi,
            fillcolor="rgba(0,255,136,0.08)",
            line=dict(color="rgba(0,255,136,0.5)", width=1))
        fig.add_annotation(
            x=2, y=mid, text=f"SUP {_fmt(lo)}–{_fmt(hi)}",
            showarrow=False, font=dict(color=_G, size=10),
            xanchor="left", bgcolor="rgba(0,0,0,0.5)")

    # ── Direnç kutuları ───────────────────────────────────────────────────────
    for i, (lo, hi) in enumerate(resistances):
        mid = (lo + hi) / 2
        fig.add_shape(type="rect",
            x0=0, x1=100, y0=lo, y1=hi,
            fillcolor="rgba(255,68,68,0.08)",
            line=dict(color="rgba(255,68,68,0.5)", width=1))
        fig.add_annotation(
            x=2, y=mid, text=f"RES {_fmt(lo)}–{_fmt(hi)}",
            showarrow=False, font=dict(color=_R, size=10),
            xanchor="left", bgcolor="rgba(0,0,0,0.5)")

    # ── Likidasyon çizgileri ──────────────────────────────────────────────────
    for lv in liq_levels:
        fig.add_shape(type="line",
            x0=0, x1=100, y0=lv, y1=lv,
            line=dict(color=_O, width=2, dash="dot"))
        fig.add_annotation(
            x=98, y=lv, text=f"LIQ {_fmt(lv)}",
            showarrow=False, font=dict(color=_O, size=10),
            xanchor="right", bgcolor="rgba(0,0,0,0.5)")

    # ── EMA çizgileri ────────────────────────────────────────────────────────
    if show_ema20 and ema20_val > 0:
        fig.add_shape(type="line", x0=0, x1=100, y0=ema20_val, y1=ema20_val,
            line=dict(color=_W, width=1.5))
        fig.add_annotation(x=50, y=ema20_val,
            text="EMA20", showarrow=False,
            font=dict(color=_W, size=10), bgcolor="rgba(0,0,0,0.6)")

    if show_ema50 and ema50_val > 0:
        fig.add_shape(type="line", x0=0, x1=100, y0=ema50_val, y1=ema50_val,
            line=dict(color=_Y, width=1.5))
        fig.add_annotation(x=50, y=ema50_val,
            text="EMA50", showarrow=False,
            font=dict(color=_Y, size=10), bgcolor="rgba(0,0,0,0.6)")

    if show_ema200 and ema200_val > 0:
        fig.add_shape(type="line", x0=0, x1=100, y0=ema200_val, y1=ema200_val,
            line=dict(color=_R, width=2))
        fig.add_annotation(x=50, y=ema200_val,
            text="EMA200", showarrow=False,
            font=dict(color=_R, size=10), bgcolor="rgba(0,0,0,0.6)")

    # ── Güncel fiyat ──────────────────────────────────────────────────────────
    fig.add_shape(type="line", x0=0, x1=100, y0=price, y1=price,
        line=dict(color=_B, width=1.5, dash="dash"))
    fig.add_annotation(x=98, y=price,
        text=f"▶ {_fmt(price)}", showarrow=False,
        font=dict(color=_B, size=12, family="monospace"),
        xanchor="right",
        bgcolor="rgba(88,166,255,0.15)",
        bordercolor=_B, borderwidth=1)

    # ── Trade plan çizgileri ──────────────────────────────────────────────────
    if plan:
        is_long = plan["direction"] == "LONG"
        e_c  = _G if is_long else _R
        sl_c = _R
        tp_c = _G

        # Entry zone
        fig.add_shape(type="rect",
            x0=0, x1=100,
            y0=plan["entry"] * 0.9995, y1=plan["entry"] * 1.0005,
            fillcolor=f"rgba(0,255,136,0.15)" if is_long else "rgba(255,68,68,0.15)",
            line=dict(color=e_c, width=1.5))
        fig.add_annotation(x=98, y=plan["entry"],
            text=f"ENTRY {_fmt(plan['entry'])}",
            showarrow=False, font=dict(color=e_c, size=11, family="monospace"),
            xanchor="right", bgcolor="rgba(0,0,0,0.7)",
            bordercolor=e_c, borderwidth=1)

        # Stop loss
        fig.add_shape(type="line", x0=0, x1=100,
            y0=plan["sl"], y1=plan["sl"],
            line=dict(color=sl_c, width=2, dash="dash"))
        fig.add_annotation(x=98, y=plan["sl"],
            text=f"SL {_fmt(plan['sl'])}",
            showarrow=False, font=dict(color=sl_c, size=11, family="monospace"),
            xanchor="right", bgcolor="rgba(0,0,0,0.7)",
            bordercolor=sl_c, borderwidth=1)

        # TP1
        fig.add_shape(type="line", x0=0, x1=100,
            y0=plan["tp1"], y1=plan["tp1"],
            line=dict(color=tp_c, width=1.5, dash="dash"))
        fig.add_annotation(x=98, y=plan["tp1"],
            text=f"TP1 {_fmt(plan['tp1'])}  RR {plan['rr1']}",
            showarrow=False, font=dict(color=tp_c, size=11, family="monospace"),
            xanchor="right", bgcolor="rgba(0,0,0,0.7)",
            bordercolor=tp_c, borderwidth=1)

        # TP2
        fig.add_shape(type="line", x0=0, x1=100,
            y0=plan["tp2"], y1=plan["tp2"],
            line=dict(color=_P, width=1.5, dash="dash"))
        fig.add_annotation(x=98, y=plan["tp2"],
            text=f"TP2 {_fmt(plan['tp2'])}  RR {plan['rr2']}",
            showarrow=False, font=dict(color=_P, size=11, family="monospace"),
            xanchor="right", bgcolor="rgba(0,0,0,0.7)",
            bordercolor=_P, borderwidth=1)

    return fig


# ── Ana render ────────────────────────────────────────────────────────────────

def render_sniper():

    # ── Global CSS ─────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    .sniper-header {
        font-family: 'Space Mono', monospace;
        font-size: 1.1rem;
        font-weight: 700;
        color: #00ff88;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .sniper-label {
        font-size: 11px;
        color: #6e7681;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 3px;
    }
    .info-box {
        background: #111111;
        border: 1px solid #1e1e1e;
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 6px;
    }
    .info-val { font-family: monospace; font-size: 14px; font-weight: 700; }
    .plan-box {
        background: #0a0a0a;
        border: 1px solid #1e1e1e;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ─────────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='display:flex;align-items:center;justify-content:space-between;"
        f"margin-bottom:12px'>"
        f"<div>"
        f"<div class='sniper-header'>🎯 Sniper Terminal</div>"
        f"<div style='font-size:12px;color:{_DT}'>Manuel seviye analizi & trade plan üretici</div>"
        f"</div>"
        f"<div style='background:#0a1a0a;border:1px solid #00ff8830;"
        f"border-radius:8px;padding:4px 14px;"
        f"font-size:11px;color:#00ff88;letter-spacing:0.1em'>ACTIVE</div>"
        f"</div>",
        unsafe_allow_html=True)

    # ── Sol panel + Chart ──────────────────────────────────────────────────
    left, chart_col = st.columns([1, 2.8])

    with left:
        st.markdown(f"<div style='font-size:11px;color:{_DT2};text-transform:uppercase;"
                    f"letter-spacing:0.08em;margin-bottom:6px'>Sembol & Fiyat</div>",
                    unsafe_allow_html=True)

        symbol = st.text_input("Sembol", value="BTCUSDT",
                               key="sn_symbol", label_visibility="collapsed",
                               placeholder="BTCUSDT").upper().strip()
        price  = st.number_input("Güncel Fiyat", value=78350.0,
                                 step=10.0, format="%.2f",
                                 key="sn_price", label_visibility="collapsed")

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        # Yön
        direction = st.radio("Yön", ["LONG", "SHORT"],
                             key="sn_dir", horizontal=True,
                             label_visibility="collapsed")

        st.markdown(f"<div style='height:6px'></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:11px;color:{_G};text-transform:uppercase;"
                    f"letter-spacing:0.08em;margin-bottom:4px'>Destek Bölgeleri</div>",
                    unsafe_allow_html=True)
        sup_raw = st.text_area("Destek", value="73000-74000\n71000-71500",
                               key="sn_sup", height=80,
                               label_visibility="collapsed",
                               placeholder="73000-74000\n71000-71500")

        st.markdown(f"<div style='font-size:11px;color:{_R};text-transform:uppercase;"
                    f"letter-spacing:0.08em;margin:6px 0 4px'>Direnç Bölgeleri</div>",
                    unsafe_allow_html=True)
        res_raw = st.text_area("Direnç", value="79000-81250\n83000-84000",
                               key="sn_res", height=80,
                               label_visibility="collapsed",
                               placeholder="79000-81250")

        st.markdown(f"<div style='font-size:11px;color:{_O};text-transform:uppercase;"
                    f"letter-spacing:0.08em;margin:6px 0 4px'>Likidasyon Seviyeleri</div>",
                    unsafe_allow_html=True)
        liq_raw = st.text_input("Likidasyon", value="78500 80200 82500",
                                key="sn_liq", label_visibility="collapsed",
                                placeholder="78500 80200 82500")

        st.markdown(f"<div style='font-size:11px;color:{_DT};text-transform:uppercase;"
                    f"letter-spacing:0.08em;margin:8px 0 4px'>EMA Göster</div>",
                    unsafe_allow_html=True)
        ec1, ec2, ec3 = st.columns(3)
        with ec1: show_ema20  = st.checkbox("EMA20",  value=True, key="sn_e20")
        with ec2: show_ema50  = st.checkbox("EMA50",  value=True, key="sn_e50")
        with ec3: show_ema200 = st.checkbox("EMA200", value=True, key="sn_e200")

        ev1, ev2, ev3 = st.columns(3)
        with ev1: ema20  = st.number_input("20",  value=77200.0, step=10.0, key="sn_ev20",  label_visibility="collapsed")
        with ev2: ema50  = st.number_input("50",  value=75800.0, step=10.0, key="sn_ev50",  label_visibility="collapsed")
        with ev3: ema200 = st.number_input("200", value=68000.0, step=10.0, key="sn_ev200", label_visibility="collapsed")

        st.markdown(f"<div style='font-size:11px;color:{_DT};text-transform:uppercase;"
                    f"letter-spacing:0.08em;margin:6px 0 4px'>SL Mesafesi %</div>",
                    unsafe_allow_html=True)
        sl_pct = st.slider("SL %", 0.2, 3.0, 1.0, 0.1,
                           key="sn_slpct", label_visibility="collapsed")

        generate = st.button("⚡ Plan Oluştur", key="sn_gen",
                             type="primary", use_container_width=True)

    # Parse
    supports    = _parse_zones(sup_raw)
    resistances = _parse_zones(res_raw)
    liq_levels  = _parse_levels(liq_raw)

    # Plan state
    if "sn_plan" not in st.session_state:
        st.session_state.sn_plan = None

    if generate:
        plan = _generate_plan(price, direction, supports, resistances,
                              liq_levels, sl_pct)
        st.session_state.sn_plan = plan
        if not plan:
            st.warning("Plan üretilemedi — seviye aralıklarını kontrol et.")

    plan = st.session_state.get("sn_plan")

    # Chart
    with chart_col:
        fig = _build_chart(
            price, supports, resistances, liq_levels, plan,
            show_ema20, show_ema50, show_ema200,
            ema20, ema50, ema200)
        st.plotly_chart(fig, use_container_width=True, config={
            "displayModeBar": False, "scrollZoom": True})

    # ── Alt panel: Plan detayları + Bilgi kutusu + Manuel düzenleme ────────
    if plan:
        is_long = plan["direction"] == "LONG"
        dir_c   = _G if is_long else _R
        dir_lbl = "LONG ▲" if is_long else "SHORT ▼"

        plan_col, info_col = st.columns([2, 1])

        with plan_col:
            st.markdown(
                f"<div style='background:#0a0a0a;border:1px solid {dir_c}30;"
                f"border-top:2px solid {dir_c};border-radius:10px;"
                f"padding:12px 16px;margin-bottom:8px'>"
                f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:10px'>"
                f"<span style='font-family:monospace;font-size:14px;"
                f"font-weight:700;color:{dir_c}'>{dir_lbl}</span>"
                f"<span style='font-size:12px;color:{_DT}'>{symbol}</span>"
                f"</div>"
                # Grid
                f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px'>"
                f"<div style='background:#111;border-radius:6px;padding:7px 10px'>"
                f"<div style='font-size:10px;color:{_DT2};margin-bottom:2px'>ENTRY</div>"
                f"<div style='font-family:monospace;font-size:13px;font-weight:700;"
                f"color:{dir_c}'>{_fmt(plan['entry'])}</div></div>"
                f"<div style='background:#111;border-radius:6px;padding:7px 10px'>"
                f"<div style='font-size:10px;color:{_DT2};margin-bottom:2px'>STOP LOSS</div>"
                f"<div style='font-family:monospace;font-size:13px;font-weight:700;"
                f"color:{_R}'>{_fmt(plan['sl'])}</div></div>"
                f"<div style='background:#111;border-radius:6px;padding:7px 10px'>"
                f"<div style='font-size:10px;color:{_DT2};margin-bottom:2px'>TP1  <span style='color:{_Y}'>RR {plan['rr1']}</span></div>"
                f"<div style='font-family:monospace;font-size:13px;font-weight:700;"
                f"color:{_G}'>{_fmt(plan['tp1'])}</div></div>"
                f"<div style='background:#111;border-radius:6px;padding:7px 10px'>"
                f"<div style='font-size:10px;color:{_DT2};margin-bottom:2px'>TP2  <span style='color:{_Y}'>RR {plan['rr2']}</span></div>"
                f"<div style='font-family:monospace;font-size:13px;font-weight:700;"
                f"color:{_P}'>{_fmt(plan['tp2'])}</div></div>"
                f"</div></div>",
                unsafe_allow_html=True)

            # Manuel düzenleme
            with st.expander("✏️ Seviyeleri Manuel Düzenle"):
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    new_entry = st.number_input("Entry", value=float(plan["entry"]),
                                                format="%.2f", key="sn_m_entry")
                with mc2:
                    new_sl = st.number_input("Stop Loss", value=float(plan["sl"]),
                                             format="%.2f", key="sn_m_sl")
                with mc3:
                    new_tp1 = st.number_input("TP1", value=float(plan["tp1"]),
                                              format="%.2f", key="sn_m_tp1")
                with mc4:
                    new_tp2 = st.number_input("TP2", value=float(plan["tp2"]),
                                              format="%.2f", key="sn_m_tp2")

                rr1 = _calc_rr(new_entry, new_sl, new_tp1)
                rr2 = _calc_rr(new_entry, new_sl, new_tp2)
                st.markdown(
                    f"<div style='font-size:13px;color:{_DT};margin-top:4px'>"
                    f"RR1: <b style='color:{_G}'>{rr1}</b> &nbsp;·&nbsp; "
                    f"RR2: <b style='color:{_P}'>{rr2}</b></div>",
                    unsafe_allow_html=True)

                if st.button("Grafiği Güncelle", key="sn_update"):
                    st.session_state.sn_plan.update({
                        "entry": new_entry, "sl": new_sl,
                        "tp1": new_tp1, "tp2": new_tp2,
                        "rr1": rr1, "rr2": rr2,
                    })
                    st.rerun()

        with info_col:
            # Bias hesapla
            sup_mids = sorted([(lo+hi)/2 for lo,hi in supports])
            res_mids = sorted([(lo+hi)/2 for lo,hi in resistances])
            closest_sup = _closest_zone(price, supports)
            closest_res = _closest_zone(price, resistances)
            closest_liq = _closest_level(price, liq_levels)

            dist_sup = abs(price - closest_sup[1]) if closest_sup else 999999
            dist_res = abs(price - closest_res[0]) if closest_res else 999999

            if dist_sup < dist_res * 0.5:
                bias, bias_c = "LONG", _G
            elif dist_res < dist_sup * 0.5:
                bias, bias_c = "SHORT", _R
            else:
                bias, bias_c = "WAIT", _Y

            ema_trend = "—"
            if show_ema20 and show_ema50:
                if ema20 > ema50:
                    ema_trend = "Yukarı ↑"
                else:
                    ema_trend = "Aşağı ↓"

            rows = [
                ("CURRENT PRICE", _fmt(price), _B),
                ("TREND (EMA20/50)", ema_trend, _G if "Yukarı" in ema_trend else _R),
                ("CLOSEST SUPPORT",
                 f"{_fmt(closest_sup[0])}–{_fmt(closest_sup[1])}" if closest_sup else "—", _G),
                ("CLOSEST RESIST.",
                 f"{_fmt(closest_res[0])}–{_fmt(closest_res[1])}" if closest_res else "—", _R),
                ("CLOSEST LIQ.",
                 _fmt(closest_liq) if closest_liq else "—", _O),
                ("SUGGESTED BIAS", bias, bias_c),
            ]

            st.markdown(
                f"<div style='background:#0a0a0a;border:1px solid #1e1e1e;"
                f"border-radius:10px;padding:10px 12px'>",
                unsafe_allow_html=True)
            for label, val, color in rows:
                st.markdown(
                    f"<div style='border-bottom:1px solid #1a1a1a;padding:6px 0'>"
                    f"<div style='font-size:10px;color:{_DT2};text-transform:uppercase;"
                    f"letter-spacing:0.06em'>{label}</div>"
                    f"<div style='font-family:monospace;font-size:13px;"
                    f"font-weight:700;color:{color}'>{val}</div>"
                    f"</div>",
                    unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    elif not plan and generate:
        pass  # warning zaten gösterildi
    else:
        st.markdown(
            f"<div style='text-align:center;padding:1.5rem;color:{_DT};"
            f"font-size:13px;border:1px dashed #1e1e1e;border-radius:10px'>"
            f"Seviyeleri gir ve <b>⚡ Plan Oluştur</b> butonuna bas</div>",
            unsafe_allow_html=True)
