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
LEVERAGES    = [1, 2, 3, 5, 10, 15, 20, 25, 50, 75, 100, 125]

_SZ     = "pf__pos_size"
_SZ_VER = "pf__ps_ver"


def render_position_form():
    _render_form()


def render_edit_form(pos_id: int):
    _render_form(edit_id=pos_id)


def _render_form(edit_id=None):
    editing = edit_id is not None
    px      = f"pf_{edit_id}_" if editing else "pf_"
    pos     = None

    if editing:
        pos = next((p for p in st.session_state.data["active_positions"] if p["id"] == edit_id), None)
        if pos is None:
            st.error("Pozisyon bulunamadı.")
            return
        st.markdown(f"#### ✏️ Pozisyonu Düzenle — #{edit_id} {pos['symbol']}")
    else:
        st.markdown("#### ➕ Yeni Pozisyon")

    sz_key  = f"{_SZ}_{px}"
    ver_key = f"{_SZ_VER}_{px}"

    # ── Temel bilgiler ────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 2, 2])
    with c1:
        symbol = st.text_input("Symbol", value=pos["symbol"] if editing else "",
                               placeholder="BTC, ETH...", key=f"{px}sym").upper().strip()
    with c2:
        dir_idx   = 0 if not editing or pos["direction"] == "LONG" else 1
        direction = st.selectbox("Yön", ["LONG", "SHORT"], index=dir_idx, key=f"{px}dir")
    with c3:
        lev_def     = pos.get("leverage", 1) if editing else 1
        lev_options = sorted(set(LEVERAGES + ([lev_def] if lev_def not in LEVERAGES else [])))
        lev_idx     = lev_options.index(lev_def)
        leverage    = st.selectbox("Kaldıraç", lev_options, index=lev_idx,
                                   key=f"{px}lev", format_func=lambda x: f"{x}×")
    with c4:
        capital = st.number_input("Sermaye ($)", min_value=1.0,
                                  value=float(pos["capital"]) if editing else 10000.0,
                                  step=100.0, key=f"{px}cap")
    with c5:
        risk_pct = st.number_input("Risk (%)", min_value=0.1, max_value=100.0,
                                   value=float(pos["risk_pct"]) if editing else 2.0,
                                   step=0.1, key=f"{px}rp")

    if leverage > 1:
        notional = capital * leverage
        st.markdown(
            f"<div style='font-size:13px;color:#8b949e;margin:-6px 0 8px'>"
            f"💹 Kaldıraçlı notional: "
            f"<b style='color:#58a6ff'>${notional:,.2f}</b> "
            f"({leverage}× × ${capital:,.0f})</div>",
            unsafe_allow_html=True,
        )

    # ── Entry noktaları ───────────────────────────────────────────────────────
    st.markdown("**📍 Entry Noktaları**")
    ne_def      = len(pos["entries"]) if editing and pos.get("entries") else 1
    num_entries = int(st.number_input("Entry sayısı", min_value=1, max_value=10,
                                      value=ne_def, step=1, key=f"{px}ne"))
    default_ew  = round(100.0 / num_entries, 1)
    ecols       = st.columns(min(num_entries, 4))
    entries            = []
    total_entry_weight = 0.0

    for i in range(num_entries):
        with ecols[i % len(ecols)]:
            ep = float(pos["entries"][i]["price"])  if editing and i < len(pos.get("entries", [])) else 0.0
            ew = float(pos["entries"][i]["weight"]) if editing and i < len(pos.get("entries", [])) else default_ew
            p  = st.number_input(f"Entry {i+1} Fiyat", min_value=0.0, value=ep,
                                 format="%.6f", key=f"{px}ep{i}")
            # ağırlık için max_value yok → 100'ü aşabilir (uyarı verir ama bloklamaz)
            w  = st.number_input(f"Ağırlık {i+1} (%)", min_value=0.0, value=ew,
                                 step=1.0, key=f"{px}ew{i}")
            # USDT karşılığı — weight girildiğinde HEMEN göster (p>0 koşuluna bağlı değil)
            usdt_val = capital * (w / 100.0)
            st.markdown(
                f"<div style='font-size:12px;color:#6e7681;margin-top:-4px;margin-bottom:6px'>"
                f"≈ <b style='color:#8b949e'>${usdt_val:,.2f}</b> "
                f"<span style='color:#484f58'>({w:.1f}% sermaye)</span></div>",
                unsafe_allow_html=True,
            )
            total_entry_weight += w
            if p > 0:
                entries.append({"price": p, "weight": w})

    # ağırlık uyarısı — sadece bilgi, kaydetmeyi bloklamaz
    if num_entries > 1:
        tw_color = "#3fb950" if abs(total_entry_weight - 100.0) <= 1.0 else "#e3b341"
        st.markdown(
            f"<div style='font-size:12px;color:{tw_color};margin-bottom:4px'>"
            f"Entry ağırlıkları toplamı: <b>{total_entry_weight:.1f}%</b>"
            f"{' ✓' if abs(total_entry_weight-100)<1 else ' — 100% önerilir ama zorunlu değil'}"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Stop / TP ─────────────────────────────────────────────────────────────
    sl_col, tp_col = st.columns(2)
    with sl_col:
        sl_def    = float(pos["stop_loss"]) if editing else 0.0
        stop_loss = st.number_input("Stop Loss ✱", min_value=0.0, value=sl_def,
                                    format="%.6f", key=f"{px}sl")
    with tp_col:
        ntp_def = len(pos["take_profits"]) if editing and pos.get("take_profits") else 1
        num_tp  = int(st.number_input("TP sayısı", min_value=1, max_value=5,
                                      value=ntp_def, step=1, key=f"{px}ntp"))

    take_profits = []
    default_tpw  = round(100.0 / num_tp, 1)
    tp_cols      = st.columns(min(num_tp, 5))
    for i in range(num_tp):
        with tp_cols[i % len(tp_cols)]:
            tp_pdef = float(pos["take_profits"][i]["price"])  if editing and i < len(pos.get("take_profits", [])) else 0.0
            tp_wdef = float(pos["take_profits"][i]["weight"]) if editing and i < len(pos.get("take_profits", [])) else default_tpw
            tp_p = st.number_input(f"TP {i+1} Fiyat", min_value=0.0, value=tp_pdef,
                                   format="%.6f", key=f"{px}tpp{i}")
            tp_w = st.number_input(f"TP {i+1} Ağırlık (%)", min_value=0.0, value=tp_wdef,
                                   step=1.0, key=f"{px}tpw{i}")
            if tp_p > 0:
                take_profits.append({"price": tp_p, "weight": tp_w})

    # ── Risk motoru ───────────────────────────────────────────────────────────
    calc      = {}
    avg_entry = 0.0
    rr        = None

    if entries and stop_loss > 0:
        avg_entry = calculate_avg_entry(entries)
        calc      = calculate_position_size(capital, risk_pct, avg_entry, stop_loss)
        rr        = calculate_rr(avg_entry, stop_loss, take_profits, direction)
        rr_str    = f"1:{rr}" if rr else "1:?"
        rrc       = rr_color(rr)

        if calc:
            st.markdown("---")
            st.markdown("**📐 Risk Analizi**")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Avg Entry",      f"${avg_entry:,.4f}")
            m2.metric("Risk Tutarı",     f"${calc['risk_amount']:,.2f}")
            m3.metric("Sermaye Riski",   f"{calc['full_capital_risk_pct']:.2f}%")
            m4.metric("Kaldıraç",        f"{leverage}×")
            m5.markdown(
                f"<div style='padding-top:0.3rem'>"
                f"<div style='font-size:0.72rem;color:#484f58;margin-bottom:3px'>R:R</div>"
                f"<div style='font-size:1.4rem;font-weight:700;color:{rrc}'>{rr_str}</div></div>",
                unsafe_allow_html=True,
            )

            if calc["can_use_full"]:
                st.success(f"✅ Tüm sermaye kullanılabilir — Risk {calc['full_capital_risk_pct']:.2f}% ≤ {risk_pct:.1f}%")
                if sz_key not in st.session_state:
                    st.session_state[sz_key]  = float(capital)
                    st.session_state[ver_key] = 0
            else:
                rec = calc["recommended_size"]
                st.warning(
                    f"⚠️ Tam sermaye riski **{calc['full_capital_risk_pct']:.2f}%** "
                    f"— Önerilen: **${rec:,.2f}**"
                )
                ba, _ = st.columns([1, 3])
                with ba:
                    if st.button("💡 Öneriyi Uygula", key=f"{px}apply"):
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

    position_size = st.number_input(
        "💰 Yatırılacak Tutar ($)",
        min_value=0.0, max_value=float(capital) * 200,
        value=float(st.session_state[sz_key]),
        step=10.0, key=widget_key, on_change=_sync,
    )

    if leverage > 1:
        st.markdown(
            f"<div style='font-size:13px;color:#8b949e;margin-top:-6px'>"
            f"Efektif pozisyon: <b style='color:#58a6ff'>${position_size*leverage:,.2f}</b> "
            f"({leverage}× kaldıraç)</div>",
            unsafe_allow_html=True,
        )

    if calc.get("risk_per_unit") and position_size > 0:
        heat = calculate_position_heat(position_size, capital, calc["risk_per_unit"])
        hc   = "#da3633" if heat > risk_pct * 1.5 else "#e3b341" if heat > risk_pct else "#238636"
        st.markdown(
            f"<span style='font-size:13px;color:#8b949e'>Position Heat: "
            f"<b style='color:{hc}'>{heat:.2f}%</b> sermaye risk altında</span>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Journal bilgileri ─────────────────────────────────────────────────────
    st.markdown("**📓 Journal Bilgileri**")
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
        plan_followed = st.checkbox("Plana uyuldu mu?", value=pf_def, key=f"{px}plan")
    with j3:
        es_def          = int(pos.get("execution_score", 7)) if editing else 7
        execution_score = st.slider("Execution Skoru (0–10)", 0, 10, es_def, key=f"{px}exec")
        mk_def          = pos.get("mistakes", []) if editing else []
        mistakes        = st.multiselect("Hata Etiketleri", MISTAKES, default=mk_def, key=f"{px}mist")

    notes_def = pos.get("notes", "") if editing else ""
    notes     = st.text_area("Notlar", value=notes_def, placeholder="Setup analizi, gerekçe...",
                              key=f"{px}notes", height=75)

    st.markdown("")
    ac1, ac2, *_ = st.columns([1, 1, 3])
    with ac1:
        save_clicked = st.button("💾 Güncelle" if editing else "💾 Kaydet",
                                 type="primary", use_container_width=True, key=f"{px}save")
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
            st.error("Stop loss giriniz."); return

        avg_e  = calculate_avg_entry(entries)
        calc_s = calculate_position_size(capital, risk_pct, avg_e, stop_loss)
        rr_s   = calculate_rr(avg_e, stop_loss, take_profits, direction)
        heat_s = calculate_position_heat(position_size, capital,
                                         calc_s.get("risk_per_unit", 0)) if calc_s else 0

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
