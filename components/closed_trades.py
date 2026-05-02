"""Kapalı İşlemler — Accordion kart düzeni (mobil uyumlu)"""

import streamlit as st
from datetime import datetime
from utils.data_manager import save_data
from utils.calculations import calculate_r_multiple

_G="#3fb950"; _R="#ff7b72"; _B="#58a6ff"; _Y="#e3b341"
_TX="#e6edf3"; _DT="#8b949e"; _DT2="#6e7681"
_BG="#0d1117"; _BG2="#161b22"; _BG3="#1c2128"; _DG="#21262d"

EMOTIONS = ["—","calm","fomo","revenge","anxious","confident"]
SETUPS   = ["—","Liquidity Sweep","Breakout","Trend Devamı","Range","Haber","Diğer"]
MARKETS  = ["—","Trend","Range","Volatil","Düşük Hacim","Haber"]
MISTAKES = ["early exit","late entry","no stop","oversize","revenge trade","ignored plan"]

def _fmt_pnl(v):
    return f"{'+'if v>=0 else ''}${v:,.2f}"

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

    if trades:
        total  = sum(t.get("pnl",0) for t in trades)
        wins   = [t for t in trades if t.get("pnl",0) > 0]
        wr     = len(wins)/len(trades)*100
        r_vals = [t.get("r_multiple",0) for t in trades if t.get("r_multiple")]
        avg_r  = sum(r_vals)/len(r_vals) if r_vals else 0
        pc     = _G if total >= 0 else _R
        rc     = _G if avg_r  >= 0 else _R
        st.markdown(
            f"<div style='display:flex;gap:20px;flex-wrap:wrap;"
            f"padding:6px 0;border-bottom:1px solid {_DG};margin-bottom:10px'>"
            f"<span style='color:{_DT};font-size:13px'>{len(trades)} işlem</span>"
            f"<span style='color:{pc};font-size:13px;font-weight:700'>{_fmt_pnl(total)}</span>"
            f"<span style='color:{_DT};font-size:13px'>WR: <b style='color:{'#3fb950' if wr>=50 else '#ff7b72'}'>{wr:.1f}%</b></span>"
            f"<span style='color:{_DT};font-size:13px'>Avg R: <b style='color:{rc}'>{'+'if avg_r>=0 else ''}{avg_r:.2f}R</b></span>"
            f"</div>", unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns([2,2,2])
    with fc1:
        search = st.text_input("Ara", placeholder="Sembol...",
                               key="ct_search", label_visibility="collapsed")
    with fc2:
        dir_f = st.selectbox("Yön", ["Tümü","LONG","SHORT"],
                             key="ct_dir_f", label_visibility="collapsed")
    with fc3:
        res_f = st.selectbox("Sonuç", ["Tümü","WIN","LOSS"],
                             key="ct_res_f", label_visibility="collapsed")

    filtered = list(reversed(trades))
    if search:        filtered = [t for t in filtered if search.upper() in t.get("symbol","").upper()]
    if dir_f != "Tümü": filtered = [t for t in filtered if t.get("direction") == dir_f]
    if res_f != "Tümü": filtered = [t for t in filtered if t.get("result") == res_f]

    st.markdown("")
    for trade in filtered:
        _render_accordion(trade)

    if not filtered:
        st.markdown(
            f"<div style='text-align:center;padding:2.5rem;color:{_DT};"
            f"border:1.5px dashed {_DG};border-radius:12px'>İşlem yok</div>",
            unsafe_allow_html=True)


def _render_accordion(trade: dict):
    tid       = trade.get("id", 0)
    pnl       = trade.get("pnl", 0)
    symbol    = trade.get("symbol", "?")
    direction = trade.get("direction", "LONG")
    result    = trade.get("result", "—")
    r_mult    = trade.get("r_multiple", 0) or 0
    rr_str    = trade.get("rr_display", "—")
    closed_at = trade.get("closed_at", "")
    avg_entry = trade.get("avg_entry", 0) or 0
    close_px  = trade.get("close_price", trade.get("exit", 0)) or 0

    pnl_c  = _G if pnl  >= 0 else _R
    dir_c  = _G if direction == "LONG" else _R
    dir_bg = "#071a0e" if direction == "LONG" else "#1c0505"
    res_c  = _G if result == "WIN" else _R
    res_bg = "#071a0e" if result == "WIN" else "#1c0505"
    r_c    = _G if r_mult >= 0 else _R
    r_sign = "+" if r_mult >= 0 else ""

    try:    date_str = _parse_dt(closed_at).strftime("%d %b %y")
    except: date_str = "—"

    open_key = f"ct_open_{tid}"
    st.session_state.setdefault(open_key, False)
    is_open  = st.session_state[open_key]

    brd  = _B if is_open else _DG
    brad = "12px 12px 0 0" if is_open else "12px"
    top_border = f"border-top:2px solid {_B}60;" if is_open else ""

    sub_parts = [f"<span style='color:{r_c};font-family:monospace'>{r_sign}{r_mult:.2f}R</span>"]
    if rr_str and rr_str != "—": sub_parts.append(f"<span style='color:{_DT}'>{rr_str}</span>")
    if avg_entry: sub_parts.append(f"<span style='color:{_DT2}'>E ${avg_entry:,.2f}</span>")
    if close_px:  sub_parts.append(f"<span style='color:{_DT2}'>C ${close_px:,.2f}</span>")
    sub_parts.append(f"<span style='color:{_DT2};margin-left:auto'>{date_str}</span>")
    sep = f"<span style='color:{_DG};margin:0 5px'>·</span>"

    hc1, hc2 = st.columns([14, 1])
    with hc1:
        st.markdown(
            f"<div style='background:{_BG2};border:1px solid {brd};"
            f"border-radius:{brad};{top_border}padding:9px 14px'>"
            f"<div style='display:flex;align-items:center;gap:7px;margin-bottom:4px'>"
            f"<span style='font-size:11px;color:{_DT2};font-family:monospace'>#{tid}</span>"
            f"<span style='font-size:15px;font-weight:700;color:{_TX}'>{symbol}</span>"
            f"<span style='background:{dir_bg};color:{dir_c};padding:2px 7px;"
            f"border-radius:4px;font-size:11px;font-weight:700'>{direction}</span>"
            f"<span style='background:{res_bg};color:{res_c};padding:2px 7px;"
            f"border-radius:4px;font-size:11px;font-weight:700'>{result}</span>"
            f"<span style='flex:1'></span>"
            f"<span style='font-family:monospace;font-size:14px;"
            f"font-weight:700;color:{pnl_c}'>{_fmt_pnl(pnl)}</span>"
            f"</div>"
            f"<div style='display:flex;align-items:center;flex-wrap:wrap;font-size:12px'>"
            f"{sep.join(sub_parts)}</div>"
            f"</div>", unsafe_allow_html=True)

    with hc2:
        if st.button("▲" if is_open else "▼",
                     key=f"ct_toggle_{tid}", use_container_width=True):
            st.session_state[open_key] = not is_open
            st.rerun()

    if is_open:
        _render_detail(trade, tid)


def _render_detail(trade: dict, tid: int):
    pnl       = trade.get("pnl", 0)
    symbol    = trade.get("symbol", "?")
    direction = trade.get("direction", "LONG")
    avg_entry = trade.get("avg_entry", 0) or 0
    stop_loss = trade.get("stop_loss", 0) or 0
    close_px  = trade.get("close_price", trade.get("exit", 0)) or 0
    r_mult    = trade.get("r_multiple", 0) or 0
    rr_str    = trade.get("rr_display", "—")
    tps       = trade.get("take_profits", []) or []
    setup     = trade.get("setup_type", "—") or "—"
    mkt       = trade.get("market_condition", "—") or "—"
    emo       = trade.get("emotion", "—") or "—"
    plan_f    = trade.get("plan_followed", True)
    exec_s    = trade.get("execution_score", "—")
    notes     = trade.get("notes", "") or trade.get("comment", "") or ""
    mistakes  = trade.get("mistakes", []) or []

    edit_key = f"ct_edit_{tid}"
    st.session_state.setdefault(edit_key, False)
    is_edit  = st.session_state[edit_key]

    st.markdown(
        f"<div style='background:{_BG3};border:1px solid {_B}40;"
        f"border-top:none;border-radius:0 0 12px 12px;"
        f"padding:10px 14px 10px;margin-bottom:6px'>",
        unsafe_allow_html=True)

    if is_edit:
        st.markdown(
            f"<div style='font-size:12px;font-weight:600;color:{_B};"
            f"margin-bottom:8px'>✏️ Düzenleme Modu</div>",
            unsafe_allow_html=True)

        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        with r1c1:
            new_sym   = st.text_input("Sembol", value=symbol, key=f"ce_sym_{tid}")
            new_dir   = st.selectbox("Yön", ["LONG","SHORT"],
                                     index=0 if direction=="LONG" else 1,
                                     key=f"ce_dir_{tid}")
        with r1c2:
            new_pnl   = st.number_input("PnL ($)", value=float(pnl),
                                        step=0.01, key=f"ce_pnl_{tid}")
            new_entry = st.number_input("Entry", value=float(avg_entry),
                                        format="%.4f", key=f"ce_entry_{tid}")
        with r1c3:
            new_sl    = st.number_input("Stop Loss", value=float(stop_loss),
                                        format="%.4f", key=f"ce_sl_{tid}")
            new_close = st.number_input("Çıkış", value=float(close_px),
                                        format="%.4f", key=f"ce_close_{tid}")
        with r1c4:
            new_exec  = st.slider("Execution", 0, 10,
                                  int(exec_s) if str(exec_s).isdigit() else 7,
                                  key=f"ce_exec_{tid}")
            new_plan  = st.checkbox("Plana uyuldu", value=bool(plan_f),
                                    key=f"ce_plan_{tid}")

        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            new_setup = st.selectbox("Setup", SETUPS,
                                     index=SETUPS.index(setup) if setup in SETUPS else 0,
                                     key=f"ce_setup_{tid}")
        with r2c2:
            new_mkt   = st.selectbox("Piyasa", MARKETS,
                                     index=MARKETS.index(mkt) if mkt in MARKETS else 0,
                                     key=f"ce_mkt_{tid}")
        with r2c3:
            new_emo   = st.selectbox("Psikoloji", EMOTIONS,
                                     index=EMOTIONS.index(emo) if emo in EMOTIONS else 0,
                                     key=f"ce_emo_{tid}")

        new_mist  = st.multiselect("Hatalar", MISTAKES,
                                   default=[m for m in mistakes if m in MISTAKES],
                                   key=f"ce_mist_{tid}")
        new_notes = st.text_area("Notlar", value=notes, key=f"ce_notes_{tid}", height=60)

        sa1, sa2, _ = st.columns([1, 1, 5])
        with sa1:
            if st.button("💾 Kaydet", key=f"ce_save_{tid}",
                         type="primary", use_container_width=True):
                data = st.session_state.data
                idx  = next((i for i,t in enumerate(data["closed_trades"])
                             if t.get("id") == tid), None)
                if idx is not None:
                    risk_amt = data["closed_trades"][idx].get(
                        "risk_calc", {}).get("risk_amount", 0)
                    new_r = calculate_r_multiple(new_pnl, risk_amt)
                    data["closed_trades"][idx].update({
                        "symbol":           new_sym.upper().strip(),
                        "direction":        new_dir,
                        "avg_entry":        new_entry,
                        "stop_loss":        new_sl,
                        "close_price":      new_close,
                        "pnl":              new_pnl,
                        "r_multiple":       new_r,
                        "result":           "WIN" if new_pnl > 0 else "LOSS",
                        "setup_type":       new_setup,
                        "market_condition": new_mkt,
                        "emotion":          new_emo,
                        "plan_followed":    new_plan,
                        "execution_score":  new_exec,
                        "mistakes":         new_mist,
                        "notes":            new_notes,
                    })
                    save_data(data)
                    st.session_state.data = data
                st.session_state[edit_key] = False
                st.rerun()
        with sa2:
            if st.button("İptal", key=f"ce_cancel_{tid}", use_container_width=True):
                st.session_state[edit_key] = False
                st.rerun()

    else:
        sep    = f"<span style='color:{_DG};margin:0 5px'>·</span>"
        r_sign = "+" if r_mult >= 0 else ""
        r_c    = _G if r_mult >= 0 else _R
        exec_str  = f"{exec_s}/10" if str(exec_s).isdigit() else "—"
        plan_html = (f"<span style='color:{_G}'>✓ plan</span>" if plan_f
                     else f"<span style='color:{_R}'>✗ plan</span>")

        fparts = []
        if avg_entry: fparts.append(f"E <b style='color:{_TX}'>${avg_entry:,.4f}</b>")
        if stop_loss: fparts.append(f"SL <b style='color:{_R}'>${stop_loss:,.4f}</b>")
        if close_px:  fparts.append(f"C <b style='color:{_TX}'>${close_px:,.4f}</b>")
        for i, tp in enumerate(tps[:3]):
            if tp.get("price"):
                fparts.append(f"TP{i+1} <b style='color:{_G}'>${tp['price']:,.4f}</b>")

        jparts = []
        if setup != "—": jparts.append(f"<span style='color:{_DT}'>{setup}</span>")
        if mkt   != "—": jparts.append(f"<span style='color:{_DT}'>{mkt}</span>")
        if emo   != "—": jparts.append(f"<span style='color:{_DT}'>{emo}</span>")
        jparts.append(plan_html)
        jparts.append(f"<span style='color:{_B}'>exec {exec_str}</span>")
        jparts.append(f"<span style='color:{r_c};font-family:monospace'>{r_sign}{r_mult:.2f}R</span>")

        html = f"<div style='font-size:12px;color:{_DT2};line-height:1.9'>"
        if fparts: html += f"<div style='margin-bottom:1px'>{sep.join(fparts)}</div>"
        html += f"<div>{sep.join(jparts)}</div>"
        if mistakes:
            badges = " ".join(
                f"<span style='background:#1c0505;color:{_R};border-radius:4px;"
                f"padding:1px 5px;font-size:11px'>{m}</span>"
                for m in mistakes)
            html += f"<div style='margin-top:4px'>{badges}</div>"
        if notes:
            short = notes.replace("\n"," ")[:100]
            dots  = "…" if len(notes) > 100 else ""
            html += (f"<div style='margin-top:4px;border-left:2px solid {_B}40;"
                     f"padding-left:8px;color:{_DT}'>{short}{dots}</div>")
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        ba, bb, _ = st.columns([1, 1, 7])
        with ba:
            if st.button("✏️ Düzenle", key=f"ct_edit_btn_{tid}",
                         use_container_width=True):
                st.session_state[edit_key] = True
                st.rerun()
        with bb:
            if st.button("🗑️ Sil", key=f"ct_del_{tid}", use_container_width=True):
                _delete_closed(tid)
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
