import streamlit as st
from datetime import datetime
from utils.data_manager import save_data
from utils.calculations import (
    calculate_avg_entry, calculate_position_size, calculate_rr,
    calculate_position_heat, rr_color
)

SETUP_TYPES   = ["liquidity", "breakout", "trend", "range", "diğer"]
EMOTIONS      = ["calm", "fomo", "revenge", "anxious", "confident"]
MISTAKES      = ["early exit", "late entry", "no stop", "oversize", "revenge trade", "ignored plan"]
MARKET_CONDS  = ["trend", "range", "news", "volatile", "choppy"]


def render_position_form():
    st.markdown("---")
    st.markdown("#### ➕ Yeni Pozisyon")

    # ── Basic info ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([2, 1, 2, 2])
    with c1:
        symbol = st.text_input("Symbol", placeholder="BTC, ETH...", key="form_symbol").upper().strip()
    with c2:
        direction = st.selectbox("Yön", ["LONG", "SHORT"], key="form_direction")
    with c3:
        capital = st.number_input("Sermaye ($)", min_value=0.0, value=10000.0, step=100.0, key="form_capital")
    with c4:
        risk_pct = st.number_input("Risk (%)", min_value=0.1, max_value=100.0, value=2.0, step=0.1, key="form_risk_pct")

    # ── Entry ─────────────────────────────────────────────────────────────────
    st.markdown("**📍 Entry Noktaları**")
    num_entries = st.number_input("Entry sayısı", min_value=1, max_value=10, value=1, step=1, key="form_num_entries")

    entries = []
    if num_entries == 1:
        ep, ew = st.columns([3, 1])
        with ep:
            price = st.number_input("Entry Fiyat", min_value=0.0, value=0.0, format="%.6f", key="form_e0_price")
        entries = [{"price": price, "weight": 100.0}] if price > 0 else []
    else:
        default_w = round(100.0 / num_entries, 1)
        cols = st.columns(min(int(num_entries), 5))
        for i in range(int(num_entries)):
            col = cols[i % len(cols)]
            with col:
                p = st.number_input(f"Entry {i+1}", min_value=0.0, value=0.0, format="%.6f", key=f"form_e{i}_price")
                w = st.number_input(f"Ağırlık {i+1} (%)", min_value=0.0, max_value=100.0, value=default_w, key=f"form_e{i}_weight")
                if p > 0:
                    entries.append({"price": p, "weight": w})

    # ── Stop / TP ─────────────────────────────────────────────────────────────
    sl_col, tp_col = st.columns(2)
    with sl_col:
        stop_loss = st.number_input("Stop Loss ✱", min_value=0.0, value=0.0, format="%.6f", key="form_stop_loss")
    with tp_col:
        num_tp = st.number_input("TP sayısı", min_value=1, max_value=5, value=1, step=1, key="form_num_tp")

    take_profits = []
    default_tp_w = round(100.0 / num_tp, 1)
    tp_cols = st.columns(min(int(num_tp), 5))
    for i in range(int(num_tp)):
        col = tp_cols[i % len(tp_cols)]
        with col:
            tp_p = st.number_input(f"TP {i+1} Fiyat", min_value=0.0, value=0.0, format="%.6f", key=f"form_tp{i}_price")
            tp_w = st.number_input(f"TP {i+1} Ağırlık (%)", min_value=0.0, max_value=100.0, value=default_tp_w, key=f"form_tp{i}_weight")
            if tp_p > 0:
                take_profits.append({"price": tp_p, "weight": tp_w})

    # ── Risk engine ───────────────────────────────────────────────────────────
    if entries and stop_loss > 0:
        avg_entry = calculate_avg_entry(entries)
        calc = calculate_position_size(capital, risk_pct, avg_entry, stop_loss)
        rr = calculate_rr(avg_entry, stop_loss, take_profits, direction)
        rr_str = f"1:{rr}" if rr else "1:?"
        rrc = rr_color(rr)

        if calc:
            st.markdown("---")
            st.markdown("**📐 Risk Analizi**")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Avg Entry", f"${avg_entry:,.4f}")
            m2.metric("Risk Tutarı", f"${calc['risk_amount']:,.2f}")
            m3.metric("Tam Sermaye Riski", f"{calc['full_capital_risk_pct']:.2f}%")
            m4.markdown(f"<div style='padding-top:0.4rem'><div style='font-size:0.75rem;color:#64748b'>R:R</div>"
                        f"<div style='font-size:1.3rem;font-weight:700;color:{rrc}'>{rr_str}</div></div>",
                        unsafe_allow_html=True)

            if calc["can_use_full"]:
                st.success(f"✅ Tüm sermaye ({capital:,.0f}$) kullanılabilir — Risk: {calc['full_capital_risk_pct']:.2f}% ≤ {risk_pct}%")
                if "form_pos_size_val" not in st.session_state:
                    st.session_state["form_pos_size_val"] = capital
            else:
                st.warning(
                    f"⚠️ Tam sermaye riski **{calc['full_capital_risk_pct']:.2f}%** — "
                    f"Önerilen pozisyon: **${calc['recommended_size']:,.2f}**"
                )
                if st.button("💡 Öneriyi Uygula", key="apply_suggestion"):
                    st.session_state["form_pos_size_val"] = calc["recommended_size"]
                    st.rerun()
    else:
        avg_entry = 0.0
        calc = {}
        rr = None

    pos_size_default = float(st.session_state.get("form_pos_size_val", capital))
    position_size = st.number_input(
        "Pozisyon Büyüklüğü ($)", min_value=0.0,
        value=pos_size_default, step=10.0, key="form_position_size",
    )

    # ── Position heat display ─────────────────────────────────────────────────
    if calc.get("risk_per_unit") and position_size > 0:
        heat = calculate_position_heat(position_size, capital, calc["risk_per_unit"])
        heat_color = "#ef4444" if heat > risk_pct * 1.5 else "#f59e0b" if heat > risk_pct else "#10b981"
        st.markdown(
            f"<span style='font-size:0.8rem;color:#64748b'>Position Heat: "
            f"<b style='color:{heat_color}'>{heat:.2f}%</b> of capital at risk</span>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Journal metadata ──────────────────────────────────────────────────────
    st.markdown("**📓 Journal Bilgileri**")
    j1, j2, j3 = st.columns(3)
    with j1:
        setup_type = st.selectbox("Setup Tipi", SETUP_TYPES, key="form_setup")
        market_cond = st.selectbox("Piyasa Koşulu", MARKET_CONDS, key="form_market")
    with j2:
        emotion = st.selectbox("Psikoloji", EMOTIONS, key="form_emotion")
        plan_followed = st.checkbox("Plana uyuldu mu?", value=True, key="form_plan")
    with j3:
        execution_score = st.slider("Execution Skoru (0–10)", 0, 10, 7, key="form_exec_score")
        mistakes = st.multiselect("Hata Etiketleri", MISTAKES, key="form_mistakes")

    notes = st.text_area("Notlar", placeholder="Setup analizi, gerekçe...", key="form_notes", height=80)

    # ── Save / Cancel ─────────────────────────────────────────────────────────
    st.markdown("")
    ac1, ac2 = st.columns([1, 5])
    with ac1:
        save_clicked = st.button("💾 Kaydet", type="primary", use_container_width=True, key="save_pos_btn")
    with ac2:
        if st.button("İptal", key="cancel_form_btn"):
            st.session_state.show_add_form = False
            _clear_form_state()
            st.rerun()

    if save_clicked:
        if not symbol:
            st.error("Symbol giriniz.")
            return
        if not entries:
            st.error("En az bir entry fiyatı giriniz.")
            return
        if stop_loss <= 0:
            st.error("Stop loss giriniz.")
            return

        avg_entry = calculate_avg_entry(entries)
        calc = calculate_position_size(capital, risk_pct, avg_entry, stop_loss)
        rr = calculate_rr(avg_entry, stop_loss, take_profits, direction)
        heat = calculate_position_heat(position_size, capital, calc.get("risk_per_unit", 0)) if calc else 0

        data = st.session_state.data
        pos_id = data["next_id"]
        data["next_id"] += 1

        position = {
            "id": pos_id,
            "symbol": symbol,
            "direction": direction,
            "capital": capital,
            "risk_pct": risk_pct,
            "entries": entries,
            "avg_entry": avg_entry,
            "stop_loss": stop_loss,
            "take_profits": take_profits,
            "position_size": position_size,
            "risk_calc": calc,
            "rr": rr,
            "heat": heat,
            # journal fields
            "setup_type": setup_type,
            "market_condition": market_cond,
            "emotion": emotion,
            "plan_followed": plan_followed,
            "execution_score": execution_score,
            "mistakes": mistakes,
            "notes": notes,
            "created_at": datetime.now().isoformat(),
        }
        data["active_positions"].append(position)
        save_data(data)
        st.session_state.data = data
        st.session_state.show_add_form = False
        _clear_form_state()
        st.rerun()

    st.markdown("---")


def _clear_form_state():
    remove = [k for k in st.session_state if k.startswith("form_") or k in ("suggested_size", "form_pos_size_val")]
    for k in remove:
        del st.session_state[k]
