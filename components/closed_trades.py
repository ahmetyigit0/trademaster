"""Kapalı İşlemler — Accordion kart düzeni (mobil uyumlu)"""

import streamlit as st
from datetime import datetime
from utils.data_manager import save_data
from utils.calculations import calculate_r_multiple

_G="#3fb950"; _R="#ff7b72"; _B="#58a6ff"; _Y="#e3b341"
_TX="#e6edf3"; _DT="#8b949e"; _DT2="#6e7681"
_BG="#0d1117"; _BG2="#161b22"; _BG3="#1c2128"
_DG="#21262d"; _DBG="#0d1117"

def format_pnl(v):
    s = "+" if v >= 0 else ""
    return f"{s}${v:,.2f}"

def _parse_dt(s):
    if not s: raise ValueError
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f","%Y-%m-%dT%H:%M:%S","%Y-%m-%dT%H:%M","%Y-%m-%d"):
        try: return datetime.strptime(s, fmt)
        except: pass
    raise ValueError


def _delete_closed(tid):
    data = st.session_state.data
    data["closed_trades"] = [t for t in data["closed_trades"] if t.get("id") != tid]
    save_data(data)
    st.session_state.data = data


def render_closed_trades(data: dict):
    trades = data.get("closed_trades", [])

    # ── Özet bar ─────────────────────────────────────────────────────────────
    if trades:
        total_pnl = sum(t.get("pnl", 0) for t in trades)
        wins      = [t for t in trades if t.get("pnl", 0) > 0]
        wr        = len(wins) / len(trades) * 100
        r_vals    = [t.get("r_multiple", 0) for t in trades if t.get("r_multiple")]
        avg_r     = sum(r_vals) / len(r_vals) if r_vals else 0
        pnl_c     = _G if total_pnl >= 0 else _R
        r_c       = _G if avg_r >= 0 else _R
        st.markdown(
            f"<div style='display:flex;gap:24px;flex-wrap:wrap;padding:8px 0;"
            f"border-bottom:1px solid {_DG};margin-bottom:12px'>"
            f"<span style='font-size:14px;color:{_DT}'>{len(trades)} işlem</span>"
            f"<span style='font-size:14px;font-weight:700;color:{pnl_c}'>"
            f"{'+'if total_pnl>=0 else ''}{total_pnl:,.2f}$</span>"
            f"<span style='font-size:14px;color:{_DT}'>Win Rate: "
            f"<b style='color:{_G if wr>=50 else _R}'>{wr:.1f}%</b></span>"
            f"<span style='font-size:14px;color:{_DT}'>Avg R: "
            f"<b style='color:{r_c}'>{'+'if avg_r>=0 else ''}{avg_r:.2f}R</b></span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Filtreler ─────────────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns([2, 2, 2])
    with fc1:
        search = st.text_input("🔍 Ara", placeholder="Sembol...",
                               key="ct_search", label_visibility="collapsed")
    with fc2:
        dir_f = st.selectbox("Yön", ["Tümü","LONG","SHORT"],
                             key="ct_dir_f", label_visibility="collapsed")
    with fc3:
        res_f = st.selectbox("Sonuç", ["Tümü","WIN","LOSS"],
                             key="ct_res_f", label_visibility="collapsed")

    filtered = list(reversed(trades))
    if search:
        filtered = [t for t in filtered if search.upper() in t.get("symbol","").upper()]
    if dir_f != "Tümü":
        filtered = [t for t in filtered if t.get("direction") == dir_f]
    if res_f != "Tümü":
        filtered = [t for t in filtered if t.get("result") == res_f]

    st.markdown("")

    # ── Accordion kartlar ─────────────────────────────────────────────────────
    for trade in filtered:
        _render_accordion(trade)

    if not filtered:
        st.markdown(
            f"<div style='text-align:center;padding:2.5rem;color:{_DT};"
            f"font-size:15px;border:1.5px dashed {_DG};border-radius:12px'>"
            f"İşlem yok</div>",
            unsafe_allow_html=True,
        )


def _render_accordion(trade: dict):
    tid       = trade.get("id", 0)
    pnl       = trade.get("pnl", 0)
    symbol    = trade.get("symbol", "?")
    direction = trade.get("direction", "LONG")
    result    = trade.get("result", "—")
    r_mult    = trade.get("r_multiple", 0) or 0
    rr_str    = trade.get("rr_display", "1:?")
    closed_at = trade.get("closed_at", "")
    avg_entry = trade.get("avg_entry", 0) or 0
    close_px  = trade.get("close_price", trade.get("exit", 0)) or 0

    pnl_c  = _G if pnl >= 0 else _R
    dir_c  = _G if direction == "LONG" else _R
    dir_bg = "#071a0e" if direction == "LONG" else "#1c0505"
    res_c  = _G if result == "WIN" else _R
    res_bg = "#071a0e" if result == "WIN" else "#1c0505"
    r_c    = _G if r_mult >= 0 else _R
    r_sign = "+" if r_mult >= 0 else ""

    try:    closed_str = _parse_dt(closed_at).strftime("%d %b %y")
    except: closed_str = "—"

    open_key = f"ct_open_{tid}"
    st.session_state.setdefault(open_key, False)
    is_open = st.session_state[open_key]

    border_top = f"border-top:2px solid {_B}50;" if is_open else ""
    border_col = _B if is_open else _DG

    # ── Header satırı ────────────────────────────────────────────────────────
    hc1, hc2 = st.columns([14, 1])

    with hc1:
        entry_html = f"<span style='font-size:12px;color:{_DT2}'>E: ${avg_entry:,.2f}</span>" if avg_entry else ""
        close_html = f"<span style='font-size:12px;color:{_DT}'>C: ${close_px:,.2f}</span>" if close_px else ""

        st.markdown(
            f"<div style='background:{_BG2};border:1px solid {border_col};"
            f"border-radius:{'12px 12px 0 0' if is_open else '12px'};"
            f"{border_top}padding:10px 14px;cursor:pointer'>"
            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:5px'>"
            f"<span style='font-size:16px;font-weight:700;color:{_TX}'>{symbol}</span>"
            f"<span style='background:{dir_bg};color:{dir_c};padding:2px 8px;"
            f"border-radius:5px;font-size:11px;font-weight:700'>{direction}</span>"
            f"<span style='background:{res_bg};color:{res_c};padding:2px 8px;"
            f"border-radius:5px;font-size:11px;font-weight:700'>{result}</span>"
            f"<span style='flex:1'></span>"
            f"<span style='font-family:\"Space Mono\",monospace;font-size:15px;"
            f"font-weight:700;color:{pnl_c}'>{format_pnl(pnl)}</span>"
            f"</div>"
            f"<div style='display:flex;align-items:center;gap:14px;flex-wrap:wrap'>"
            f"<span style='font-size:12px;color:{r_c};font-family:\"Space Mono\",monospace'>"
            f"{r_sign}{r_mult:.2f}R</span>"
            f"<span style='font-size:12px;color:{_DT}'>{rr_str}</span>"
            f"{entry_html}{close_html}"
            f"<span style='font-size:12px;color:{_DT2};margin-left:auto'>{closed_str}</span>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with hc2:
        lbl = "▲" if is_open else "▼"
        if st.button(lbl, key=f"ct_toggle_{tid}", use_container_width=True):
            st.session_state[open_key] = not is_open
            st.rerun()

    # ── Açık detay paneli ─────────────────────────────────────────────────────
    if is_open:
        _render_detail_body(trade, tid)


def _render_detail_body(trade: dict, tid: int):
    """Accordion açıldığında görünen detay + butonlar."""
    pnl       = trade.get("pnl", 0)
    symbol    = trade.get("symbol", "?")
    direction = trade.get("direction", "LONG")
    avg_entry = trade.get("avg_entry", 0) or 0
    stop_loss = trade.get("stop_loss", 0) or 0
    close_px  = trade.get("close_price", trade.get("exit", 0)) or 0
    r_mult    = trade.get("r_multiple", 0) or 0
    rr_str    = trade.get("rr_display","1:?")
    tps       = trade.get("take_profits", [])
    setup     = trade.get("setup_type","—")
    mkt       = trade.get("market_condition","—")
    emo       = trade.get("emotion","—")
    plan_f    = trade.get("plan_followed", True)
    exec_s    = trade.get("execution_score","—")
    notes     = trade.get("notes","")
    comment   = trade.get("comment","")
    mistakes  = trade.get("mistakes",[])
    pnl_c     = _G if pnl >= 0 else _R
    r_c       = _G if r_mult >= 0 else _R

    st.markdown(
        f"<div style='background:{_BG3};border:1px solid {_B}50;"
        f"border-top:none;border-radius:0 0 12px 12px;"
        f"padding:12px 14px;margin-bottom:8px'>",
        unsafe_allow_html=True,
    )

    # Metrik grid
    mc = st.columns(4)
    pairs = [
        ("PnL", format_pnl(pnl), pnl_c),
        ("R Multiple", f"{'+'if r_mult>=0 else ''}{r_mult:.2f}R", r_c),
        ("R:R", rr_str, _Y),
        ("Execution", f"{exec_s}/10" if exec_s != "—" else "—", _B),
    ]
    for col, (label, val, color) in zip(mc, pairs):
        with col:
            st.markdown(
                f"<div style='background:{_BG2};border-radius:8px;"
                f"padding:7px 10px;text-align:center'>"
                f"<div style='font-size:10px;color:{_DT};text-transform:uppercase;"
                f"letter-spacing:0.08em;margin-bottom:2px'>{label}</div>"
                f"<div style='font-size:14px;font-weight:700;color:{color};"
                f"font-family:\"Space Mono\",monospace'>{val}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Fiyat bilgileri
    price_info = []
    if avg_entry: price_info.append(f"Entry: <b>${avg_entry:,.4f}</b>")
    if stop_loss: price_info.append(f"SL: <b style='color:{_R}'>${stop_loss:,.4f}</b>")
    if close_px:  price_info.append(f"Çıkış: <b>${close_px:,.4f}</b>")
    if tps:
        tp_strs = [f"TP{i+1}: <b style='color:{_G}'>${t.get('price',0):,.4f}</b>"
                   for i,t in enumerate(tps) if t.get("price")]
        price_info.extend(tp_strs)

    if price_info:
        st.markdown(
            f"<div style='font-size:13px;color:{_DT};line-height:2;"
            f"padding:4px 0;border-bottom:1px solid {_DG};margin-bottom:8px'>"
            + "  ·  ".join(price_info) + "</div>",
            unsafe_allow_html=True,
        )

    # Journal bilgileri
    journal_items = []
    if setup != "—":  journal_items.append(f"Setup: <b>{setup}</b>")
    if mkt != "—":    journal_items.append(f"Piyasa: <b>{mkt}</b>")
    if emo != "—":    journal_items.append(f"Psikoloji: <b>{emo}</b>")
    journal_items.append(f"Plan: <b style='color:{'#3fb950' if plan_f else '#ff7b72'}'>{'✓' if plan_f else '✗'}</b>")

    if journal_items:
        st.markdown(
            f"<div style='font-size:13px;color:{_DT};line-height:2;"
            f"padding:4px 0'>"
            + "  ·  ".join(journal_items) + "</div>",
            unsafe_allow_html=True,
        )

    if mistakes:
        badges = "".join(
            f"<span style='background:#1c0505;color:{_R};border:1px solid {_R}30;"
            f"border-radius:4px;padding:2px 7px;font-size:11px;margin-right:4px'>"
            f"{m}</span>" for m in mistakes
        )
        st.markdown(
            f"<div style='margin-top:6px'>{badges}</div>",
            unsafe_allow_html=True,
        )

    if notes or comment:
        txt = notes or comment
        st.markdown(
            f"<div style='background:{_BG2};border-left:2px solid {_B}60;"
            f"border-radius:0 8px 8px 0;padding:7px 10px;font-size:13px;"
            f"color:{_DT};margin-top:8px;line-height:1.6'>{txt}</div>",
            unsafe_allow_html=True,
        )

    # Sil butonu
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    _, del_col = st.columns([5, 1])
    with del_col:
        if st.button("🗑️ Sil", key=f"ct_del_{tid}", use_container_width=True):
            _delete_closed(tid)
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
