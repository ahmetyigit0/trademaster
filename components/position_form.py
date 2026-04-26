import streamlit as st
from datetime import datetime
from utils.data_manager import save_data
from utils.calculations import (
    calculate_avg_entry, calculate_position_size, calculate_rr,
    calculate_position_heat, rr_color
)

SETUP_TYPES  = ["liquidity", "breakout", "trend", "range", "diğer"]
EMOTIONS     = ["calm", "fomo", "revenge", "anxious", "confident"]
MISTAKES     = ["early exit", "late entry", "no stop", "oversize", "revenge trade", "ignored plan"]
MARKET_CONDS = ["trend", "range", "news", "volatile", "choppy"]

_SZ     = "pf__pos_size"
_SZ_VER = "pf__ps_ver"

# ── Renk paleti ───────────────────────────────────────────────────────────────
_G = "#3fb950"; _R = "#ff7b72"; _B = "#58a6ff"
_Y = "#e3b341"; _DG = "#21262d"; _DT = "#b1bac4"


def render_position_form():
    _render_form()

def render_edit_form(pos_id: int):
    _render_form(edit_id=pos_id)


def _section(icon: str, title: str):
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:8px;"
        f"margin:1.1rem 0 0.6rem'>"
        f"<span style='font-size:16px'>{icon}</span>"
        f"<span style='font-size:13px;font-weight:700;color:{_DT};"
        f"text-transform:uppercase;letter-spacing:0.1em'>{title}</span>"
        f"<span style='flex:1;height:1px;background:{_DG}'></span></div>",
        unsafe_allow_html=True,
    )


def _render_form(edit_id=None):
    editing = edit_id is not None
    px      = f"pf_{edit_id}_" if editing else "pf_"
    pos     = None

    if editing:
        pos = next((p for p in st.session_state.data["active_positions"]
                    if p["id"] == edit_id), None)
        if pos is None:
            st.error("Pozisyon bulunamadı.")
            return
        st.markdown(
            f"<div style='font-size:1.1rem;font-weight:700;color:#f0f6fc;"
            f"padding-bottom:8px;border-bottom:1px solid {_DG};margin-bottom:12px'>"
            f"✏️ Düzenle — #{edit_id} {pos['symbol']}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='font-size:1.1rem;font-weight:700;color:#f0f6fc;"
            "margin-bottom:6px'>📋 İşlem Detayları</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='font-style:italic;color:#6e7681;font-size:13px;"
            f"border-left:2px solid {_B};padding:4px 10px;"
            f"margin-bottom:16px;line-height:1.6'>"
            f"\"Herkes teknik analiz bilebilir, ancak çok az kişi "
            f"belirli bir plana sadık kalabilir.\"</div>",
            unsafe_allow_html=True,
        )

    sz_key  = f"{_SZ}_{px}"
    ver_key = f"{_SZ_VER}_{px}"

    # ══ BÖLÜM 1: Temel ═══════════════════════════════════════════════════════
    _section("🎯", "Temel Bilgiler")

    col_sym, col_dir, col_lev = st.columns([3, 2, 2])
    with col_sym:
        symbol = st.text_input(
            "Sembol",
            value=pos["symbol"] if editing else "",
            placeholder="BTC, ETH, SOL...",
            key=f"{px}sym",
        ).upper().strip()
    with col_dir:
        dir_idx   = 0 if not editing or pos["direction"] == "LONG" else 1
        direction = st.selectbox(
            "Yön", ["LONG 📈", "SHORT 📉"],
            index=dir_idx, key=f"{px}dir"
        )
        direction = direction.split()[0]  # "LONG" veya "SHORT"
    with col_lev:
        lev_def  = float(pos.get("leverage", 1)) if editing else 1.0
        leverage = st.number_input(
            "Kaldıraç ×",
            min_value=1.0, max_value=500.0,
            value=lev_def, step=1.0,
            key=f"{px}lev",
        )

    col_cap, col_rp = st.columns(2)
    with col_cap:
        capital = st.number_input(
            "💰 Sermaye (USDT)",
            min_value=1.0,
            value=float(pos["capital"]) if editing else 10000.0,
            step=100.0, key=f"{px}cap",
        )
    with col_rp:
        risk_pct = st.number_input(
            "🎲 Risk %",
            min_value=0.1, max_value=100.0,
            value=float(pos["risk_pct"]) if editing else 2.0,
            step=0.1, key=f"{px}rp",
            help="Sermayenin yüzde kaçını riske atıyorsun?",
        )

    if leverage > 1:
        st.markdown(
            f"<div style='background:#0a1220;border:1px solid {_B}30;"
            f"border-radius:7px;padding:6px 10px;font-size:13px;margin-top:2px'>"
            f"💹 Kaldıraçlı notional: "
            f"<b style='color:{_B}'>${capital*leverage:,.0f}</b> USDT"
            f"</div>", unsafe_allow_html=True,
        )

    # ══ BÖLÜM 2: Entry ═══════════════════════════════════════════════════════
    _section("📍", "Entry Noktaları")

    ne_def      = len(pos["entries"]) if editing and pos.get("entries") else 1
    num_entries = int(st.number_input(
        "Entry sayısı", min_value=1, max_value=10,
        value=ne_def, step=1, key=f"{px}ne",
    ))
    default_ew  = round(100.0 / num_entries, 1)
    entries     = []
    total_ew    = 0.0

    ecols = st.columns(min(num_entries, 4))
    for i in range(num_entries):
        with ecols[i % len(ecols)]:
            ep = float(pos["entries"][i]["price"])  if editing and i < len(pos.get("entries",[])) else 0.0
            ew = float(pos["entries"][i]["weight"]) if editing and i < len(pos.get("entries",[])) else default_ew
            p  = st.number_input(f"Fiyat {i+1}", min_value=0.0, value=ep,
                                 format="%.4f", key=f"{px}ep{i}",
                                 label_visibility="visible")
            w  = st.number_input(f"Ağırlık {i+1} %", min_value=0.0, value=ew,
                                 step=1.0, key=f"{px}ew{i}",
                                 label_visibility="visible")
            usdt = capital * (w / 100.0)
            st.markdown(
                f"<div style='font-size:12px;color:#6e7681;margin:-4px 0 6px'>"
                f"<b style='color:{_B}'>${usdt:,.0f}</b> USDT ({w:.0f}%)</div>",
                unsafe_allow_html=True,
            )
            total_ew += w
            if p > 0:
                entries.append({"price": p, "weight": w})

    if num_entries > 1:
        tw_c = _G if abs(total_ew - 100) <= 1 else _Y
        st.markdown(
            f"<div style='font-size:12px;color:{tw_c};margin-bottom:4px'>"
            f"Toplam ağırlık: <b>{total_ew:.0f}%</b>"
            f"{'  ✓' if abs(total_ew-100)<1 else '  — 100% önerilir'}</div>",
            unsafe_allow_html=True,
        )

    # ══ BÖLÜM 3: Stop / TP ═══════════════════════════════════════════════════
    _section("🛡️", "Stop Loss & Take Profit")

    sl_col, tp_n_col = st.columns([3, 1])
    with sl_col:
        sl_def    = float(pos["stop_loss"]) if editing else 0.0
        stop_loss = st.number_input(
            "🛑 Stop Loss (zorunlu)",
            min_value=0.0, value=sl_def,
            format="%.4f", key=f"{px}sl",
        )
        # Stop altı anlık risk
        if stop_loss > 0 and entries:
            ae = calculate_avg_entry(entries)
            if ae > 0:
                pdiff    = abs(ae - stop_loss)
                trade_r  = pdiff / ae * 100
                cur_sz   = float(st.session_state.get(sz_key, capital))
                act_loss = cur_sz * (pdiff / ae)
                act_pct  = act_loss / capital * 100
                rc = _R if act_pct > risk_pct * 1.2 else _Y if act_pct > risk_pct else _G
                st.markdown(
                    f"<div style='background:#120a0a;border:1px solid #3a1a1a;"
                    f"border-radius:7px;padding:5px 9px;font-size:13px;"
                    f"display:flex;gap:12px;flex-wrap:wrap'>"
                    f"<span style='color:{rc};font-weight:700'>-${act_loss:,.2f}</span>"
                    f"<span style='color:{rc}'>%{trade_r:.2f} fiyat hareketi</span>"
                    f"<span style='color:{_DT}'>Sermaye riski: "
                    f"<b style='color:{rc}'>%{act_pct:.2f}</b></span>"
                    f"</div>", unsafe_allow_html=True,
                )
    with tp_n_col:
        ntp_def = len(pos["take_profits"]) if editing and pos.get("take_profits") else 1
        num_tp  = int(st.number_input(
            "TP sayısı", min_value=1, max_value=5,
            value=ntp_def, step=1, key=f"{px}ntp",
        ))

    take_profits = []
    default_tpw  = round(100.0 / num_tp, 1)
    tp_cols      = st.columns(min(num_tp, 5))
    for i in range(num_tp):
        with tp_cols[i % len(tp_cols)]:
            tp_pd = float(pos["take_profits"][i]["price"])  if editing and i < len(pos.get("take_profits",[])) else 0.0
            tp_wd = float(pos["take_profits"][i]["weight"]) if editing and i < len(pos.get("take_profits",[])) else default_tpw
            tp_p  = st.number_input(f"TP {i+1} Fiyat", min_value=0.0, value=tp_pd,
                                    format="%.4f", key=f"{px}tpp{i}")
            tp_w  = st.number_input(f"TP {i+1} %", min_value=0.0, value=tp_wd,
                                    step=1.0, key=f"{px}tpw{i}")
            # TP kâr hint
            if tp_p > 0 and entries:
                ae_tp  = calculate_avg_entry(entries)
                cur_tp = float(st.session_state.get(sz_key, capital))
                if ae_tp > 0 and cur_tp > 0:
                    move   = abs(tp_p - ae_tp) / ae_tp
                    profit = cur_tp * (tp_w / 100.0) * move
                    st.markdown(
                        f"<div style='font-size:12px;color:#071a0e;"
                        f"background:#071a0e;border:1px solid #1a3a20;"
                        f"border-radius:6px;padding:4px 8px;margin-top:2px'>"
                        f"<b style='color:{_G}'>+${profit:,.2f}</b> "
                        f"<span style='color:#6e7681'>({tp_w:.0f}% pos)</span></div>",
                        unsafe_allow_html=True,
                    )
            if tp_p > 0:
                take_profits.append({"price": tp_p, "weight": tp_w})

    # ══ BÖLÜM 4: Risk Analizi ═════════════════════════════════════════════════
    calc      = {}
    avg_entry = 0.0
    rr        = None

    if entries and stop_loss > 0:
        avg_entry = calculate_avg_entry(entries)
        calc      = calculate_position_size(capital, risk_pct, avg_entry, stop_loss)
        rr        = calculate_rr(avg_entry, stop_loss, take_profits, direction)

        if calc:
            _section("📐", "Risk Analizi")
            rr_str = f"1:{rr}" if rr else "1:?"
            rrc    = rr_color(rr)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Ort. Entry",      f"${avg_entry:,.4f}")
            m2.metric("Risk Tutarı",     f"${calc['risk_amount']:,.2f}")
            m3.metric("Sermaye Riski",   f"%{calc['full_capital_risk_pct']:.2f}")
            m4.markdown(
                f"<div style='padding-top:0.3rem'>"
                f"<div style='font-size:12px;color:{_DT};margin-bottom:3px'>R:R</div>"
                f"<div style='font-size:1.5rem;font-weight:700;color:{rrc}'>{rr_str}</div>"
                f"</div>", unsafe_allow_html=True,
            )

            if calc["can_use_full"]:
                st.success(f"✅ Tüm sermaye kullanılabilir — Risk %{calc['full_capital_risk_pct']:.2f} ≤ %{risk_pct:.1f}")
                if sz_key not in st.session_state:
                    st.session_state[sz_key]  = float(capital)
                    st.session_state[ver_key] = 0
            else:
                rec = calc["recommended_size"]
                st.warning(
                    f"⚠️ Tam sermaye riski **%{calc['full_capital_risk_pct']:.2f}** "
                    f"— Önerilen: **${rec:,.2f}**"
                )
                ba, _ = st.columns([1, 3])
                with ba:
                    if st.button("💡 Öneriyi Uygula", key=f"{px}apply", type="primary"):
                        st.session_state[sz_key]  = float(rec)
                        st.session_state[ver_key] = st.session_state.get(ver_key, 0) + 1
                        st.rerun()

    if sz_key not in st.session_state:
        st.session_state[sz_key]  = float(pos.get("position_size", capital)) if editing else float(capital)
        st.session_state[ver_key] = 0

    ver        = st.session_state.get(ver_key, 0)
    widget_key = f"{px}psw_v{ver}"

    def _sync():
        st.session_state[sz_key] = float(st.session_state.get(widget_key, 0.0))

    _section("💰", "Pozisyon Büyüklüğü")
    position_size = st.number_input(
        "Yatırılacak Tutar (USDT)",
        min_value=0.0, max_value=float(capital) * 200,
        value=float(st.session_state[sz_key]),
        step=10.0, key=widget_key, on_change=_sync,
    )

    if leverage > 1:
        st.markdown(
            f"<div style='font-size:13px;color:{_DT};margin-top:-4px'>"
            f"Efektif: <b style='color:{_B}'>${position_size*leverage:,.0f}</b> USDT "
            f"({leverage:.0f}× kaldıraç)</div>", unsafe_allow_html=True,
        )

    if calc.get("risk_per_unit") and position_size > 0:
        heat = calculate_position_heat(position_size, capital, calc["risk_per_unit"])
        hc   = _R if heat > risk_pct * 1.5 else _Y if heat > risk_pct else _G
        st.markdown(
            f"<span style='font-size:13px;color:{_DT}'>Position Heat: "
            f"<b style='color:{hc}'>{heat:.2f}%</b></span>",
            unsafe_allow_html=True,
        )

    # ══ İşlem Özeti ══════════════════════════════════════════════════════════
    _cur = float(st.session_state.get(sz_key, position_size))
    if entries and (stop_loss > 0 or take_profits):
        _ae    = calculate_avg_entry(entries)
        rows   = ""
        total_tp_pnl = 0.0

        # Entries
        rows += f"<tr style='background:#0a1220'><td colspan='3' style='padding:5px 8px;font-size:11px;color:{_DT};text-transform:uppercase;letter-spacing:0.1em'>📥 Girişler</td></tr>"
        for i, e in enumerate(entries):
            usdt = _cur * (e["weight"] / 100.0)
            lbl  = f"Entry {i+1} ({e['weight']:.0f}%)" if len(entries)>1 else "Entry"
            rows += (f"<tr style='border-bottom:1px solid #161b22'>"
                     f"<td style='padding:5px 8px;color:{_DT}'>{lbl}</td>"
                     f"<td style='padding:5px 8px;color:#e6edf3;font-family:\"Space Mono\",monospace'>${e['price']:,.4f}</td>"
                     f"<td style='padding:5px 8px;text-align:right;color:{_B};font-family:\"Space Mono\",monospace'>${usdt:,.2f}</td></tr>")
        if len(entries) > 1:
            rows += (f"<tr style='border-bottom:2px solid #21262d;background:#0a1220'>"
                     f"<td style='padding:5px 8px;color:{_DT};font-style:italic'>Ort. Entry</td>"
                     f"<td style='padding:5px 8px;font-family:\"Space Mono\",monospace;font-weight:700;color:#f0f6fc'>${_ae:,.4f}</td>"
                     f"<td style='padding:5px 8px;text-align:right;font-family:\"Space Mono\",monospace;font-weight:700;color:{_B}'>${_cur:,.2f}</td></tr>")

        # Stop
        if stop_loss > 0 and _ae > 0:
            sl_loss = _cur * abs(_ae - stop_loss) / _ae
            sl_pct  = sl_loss / capital * 100
            rows += (f"<tr style='background:#0a0808'><td colspan='3' style='padding:5px 8px;font-size:11px;color:{_DT};text-transform:uppercase;letter-spacing:0.1em'>🛑 Stop</td></tr>"
                     f"<tr style='border-bottom:2px solid #21262d'>"
                     f"<td style='padding:5px 8px;color:{_DT}'>Stop Loss</td>"
                     f"<td style='padding:5px 8px;color:{_R};font-family:\"Space Mono\",monospace'>${stop_loss:,.4f}</td>"
                     f"<td style='padding:5px 8px;text-align:right;color:{_R};font-family:\"Space Mono\",monospace'>-${sl_loss:,.2f} <span style='font-size:11px;opacity:.8'>(%{sl_pct:.2f})</span></td></tr>")

        # TPs
        if take_profits and _ae > 0:
            rows += f"<tr style='background:#080a08'><td colspan='3' style='padding:5px 8px;font-size:11px;color:{_DT};text-transform:uppercase;letter-spacing:0.1em'>🎯 Take Profit</td></tr>"
            for i, t in enumerate(take_profits):
                move    = abs(t["price"] - _ae) / _ae
                portion = _cur * (t["weight"] / 100.0)
                profit  = portion * move
                total_tp_pnl += profit
                lbl = f"TP {i+1} ({t['weight']:.0f}%)" if len(take_profits)>1 else "Take Profit"
                rows += (f"<tr style='border-bottom:1px solid #161b22'>"
                         f"<td style='padding:5px 8px;color:{_DT}'>{lbl}</td>"
                         f"<td style='padding:5px 8px;color:{_G};font-family:\"Space Mono\",monospace'>${t['price']:,.4f}</td>"
                         f"<td style='padding:5px 8px;text-align:right;color:{_G};font-family:\"Space Mono\",monospace'>+${profit:,.2f}</td></tr>")

        if total_tp_pnl > 0:
            rows += (f"<tr style='background:#071a0e'>"
                     f"<td colspan='2' style='padding:6px 8px;color:{_G};font-weight:700'>🏆 Hedef PnL</td>"
                     f"<td style='padding:6px 8px;text-align:right;font-family:\"Space Mono\",monospace;font-weight:700;color:{_G}'>+${total_tp_pnl:,.2f}</td></tr>")

        st.markdown(
            f"<div style='background:#0d1117;border:1px solid {_DG};"
            f"border-radius:10px;overflow:hidden;margin-top:8px'>"
            f"<div style='padding:6px 8px;background:#0a1220;border-bottom:1px solid {_DG};"
            f"font-size:11px;color:{_DT};text-transform:uppercase;letter-spacing:0.1em'>"
            f"📋 İşlem Özeti</div>"
            f"<table style='width:100%;border-collapse:collapse;font-size:13px'>"
            f"<thead><tr style='border-bottom:1px solid {_DG};background:#0a1220'>"
            f"<th style='padding:4px 8px;text-align:left;color:{_DT};font-size:11px'>Seviye</th>"
            f"<th style='padding:4px 8px;text-align:left;color:{_DT};font-size:11px'>Fiyat</th>"
            f"<th style='padding:4px 8px;text-align:right;color:{_DT};font-size:11px'>USDT</th>"
            f"</tr></thead><tbody>{rows}</tbody></table></div>",
            unsafe_allow_html=True,
        )

    # ══ BÖLÜM 5: Journal ═════════════════════════════════════════════════════
    _section("📓", "Journal Bilgileri")

    j1, j2, j3 = st.columns(3)
    with j1:
        si          = SETUP_TYPES.index(pos["setup_type"]) if editing and pos.get("setup_type") in SETUP_TYPES else 0
        setup_type  = st.selectbox("Setup Tipi",    SETUP_TYPES,  index=si, key=f"{px}setup")
        mi          = MARKET_CONDS.index(pos["market_condition"]) if editing and pos.get("market_condition") in MARKET_CONDS else 0
        market_cond = st.selectbox("Piyasa Koşulu", MARKET_CONDS, index=mi, key=f"{px}market")
    with j2:
        ei      = EMOTIONS.index(pos["emotion"]) if editing and pos.get("emotion") in EMOTIONS else 0
        emotion = st.selectbox("Psikoloji", EMOTIONS, index=ei, key=f"{px}emo")
        pf_def        = pos.get("plan_followed", True) if editing else True
        plan_followed = st.checkbox("✅ Plana uyuldu", value=pf_def, key=f"{px}plan")
    with j3:
        es_def          = int(pos.get("execution_score", 7)) if editing else 7
        execution_score = st.slider("Execution (0–10)", 0, 10, es_def, key=f"{px}exec")
        mk_def          = pos.get("mistakes", []) if editing else []
        mistakes        = st.multiselect("Hata Etiketleri", MISTAKES, default=mk_def, key=f"{px}mist")

    notes_def = pos.get("notes", "") if editing else ""
    notes = st.text_area(
        "📝 Notlar / Gerekçe",
        value=notes_def,
        placeholder="Neden bu kurulumu seçtin? Setup analizi...",
        key=f"{px}notes", height=70,
    )

    # ══ KAYDET / İPTAL ════════════════════════════════════════════════════════
    st.markdown("")
    ac1, ac2, ac3 = st.columns([2, 1, 3])
    with ac1:
        save_clicked = st.button(
            "💾 Güncelle" if editing else "✅ Pozisyonu Kaydet",
            type="primary", use_container_width=True, key=f"{px}save",
        )
    with ac2:
        if st.button("İptal", key=f"{px}cancel", use_container_width=True):
            if editing:
                st.session_state[f"edit_mode_{edit_id}"] = False
            _clear(px, sz_key, ver_key)
            st.rerun()

    if save_clicked:
        if not symbol:
            st.error("Symbol giriniz."); return
        if not entries:
            st.error("En az bir entry fiyatı giriniz."); return
        if stop_loss <= 0:
            st.error("Stop loss zorunludur."); return

        avg_e  = calculate_avg_entry(entries)
        calc_s = calculate_position_size(capital, risk_pct, avg_e, stop_loss)
        rr_s   = calculate_rr(avg_e, stop_loss, take_profits, direction)
        heat_s = calculate_position_heat(position_size, capital, calc_s.get("risk_per_unit",0)) if calc_s else 0

        data   = st.session_state.data
        record = dict(
            symbol=symbol, direction=direction, leverage=leverage,
            capital=capital, risk_pct=risk_pct,
            entries=entries, avg_entry=avg_e, stop_loss=stop_loss,
            take_profits=take_profits, position_size=position_size,
            effective_size=position_size * leverage,
            risk_calc=calc_s, rr=rr_s, heat=heat_s,
            setup_type=setup_type, market_condition=market_cond,
            emotion=emotion, plan_followed=plan_followed,
            execution_score=execution_score, mistakes=mistakes, notes=notes,
        )

        if editing:
            idx = next((i for i, p in enumerate(data["active_positions"]) if p["id"] == edit_id), None)
            if idx is not None:
                record["id"]         = edit_id
                record["created_at"] = data["active_positions"][idx].get("created_at", datetime.now().isoformat())
                record["updated_at"] = datetime.now().isoformat()
                data["active_positions"][idx] = record
            st.session_state[f"edit_mode_{edit_id}"] = False
        else:
            record["id"]         = data["next_id"]
            record["created_at"] = datetime.now().isoformat()
            data["next_id"] += 1
            data["active_positions"].append(record)

        save_data(data)
        st.session_state.data = data
        _clear(px, sz_key, ver_key)
        st.rerun()


def _clear(px, sz_key, ver_key):
    for k in list(st.session_state.keys()):
        if k.startswith(px) or k in (sz_key, ver_key):
            del st.session_state[k]
