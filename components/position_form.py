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

# ─── STATE KEYS ──────────────────────────────────────────────────────────────
_POS_SIZE_KEY = "pos_form__size"   # the ONLY source-of-truth for position size


def render_position_form():
    st.markdown("---")
    st.markdown("#### ➕ Yeni Pozisyon")

    # ── Basic ─────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([2, 1, 2, 2])
    with c1:
        symbol = st.text_input("Symbol", placeholder="BTC, ETH...", key="pf_symbol").upper().strip()
    with c2:
        direction = st.selectbox("Yön", ["LONG", "SHORT"], key="pf_direction")
    with c3:
        capital = st.number_input("Sermaye ($)", min_value=0.0, value=10000.0, step=100.0, key="pf_capital")
    with c4:
        risk_pct = st.number_input("Risk (%)", min_value=0.1, max_value=100.0, value=2.0, step=0.1, key="pf_risk_pct")

    # ── Entries ───────────────────────────────────────────────────────────────
    st.markdown("**📍 Entry Noktaları**")
    num_entries = int(st.number_input("Entry sayısı", min_value=1, max_value=10, value=1, step=1, key="pf_num_entries"))

    entries = []
    default_ew = round(100.0 / num_entries, 1)
    ecols = st.columns(min(num_entries, 5))
    for i in range(num_entries):
        with ecols[i % len(ecols)]:
            p = st.number_input(f"Entry {i+1} Fiyat", min_value=0.0, value=0.0, format="%.6f", key=f"pf_ep{i}")
            w = st.number_input(f"Ağırlık {i+1} (%)", min_value=0.0, max_value=100.0, value=default_ew, key=f"pf_ew{i}")
            if p > 0:
                entries.append({"price": p, "weight": w})

    # ── Stop / TP ─────────────────────────────────────────────────────────────
    sl_col, tp_col = st.columns(2)
    with sl_col:
        stop_loss = st.number_input("Stop Loss ✱", min_value=0.0, value=0.0, format="%.6f", key="pf_sl")
    with tp_col:
        num_tp = int(st.number_input("TP sayısı", min_value=1, max_value=5, value=1, step=1, key="pf_num_tp"))

    take_profits = []
    default_tpw = round(100.0 / num_tp, 1)
    tp_cols = st.columns(min(num_tp, 5))
    for i in range(num_tp):
        with tp_cols[i % len(tp_cols)]:
            tp_p = st.number_input(f"TP {i+1} Fiyat", min_value=0.0, value=0.0, format="%.6f", key=f"pf_tpp{i}")
            tp_w = st.number_input(f"TP {i+1} Ağırlık (%)", min_value=0.0, max_value=100.0, value=default_tpw, key=f"pf_tpw{i}")
            if tp_p > 0:
                take_profits.append({"price": tp_p, "weight": tp_w})

    # ── Risk Engine ───────────────────────────────────────────────────────────
    calc: dict = {}
    avg_entry  = 0.0
    rr         = None

    if entries and stop_loss > 0:
        avg_entry = calculate_avg_entry(entries)
        calc      = calculate_position_size(capital, risk_pct, avg_entry, stop_loss)
        rr        = calculate_rr(avg_entry, stop_loss, take_profits, direction)
        rr_str    = f"1:{rr}" if rr else "1:?"
        rrc       = rr_color(rr)

        if calc:
            st.markdown("---")
            st.markdown("**📐 Risk Analizi**")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Avg Entry",          f"${avg_entry:,.4f}")
            m2.metric("Risk Tutarı",         f"${calc['risk_amount']:,.2f}")
            m3.metric("Tam Sermaye Riski",   f"{calc['full_capital_risk_pct']:.2f}%")
            m4.markdown(
                f"<div style='padding-top:0.35rem'>"
                f"<div style='font-size:0.72rem;color:#64748b;margin-bottom:4px'>R:R</div>"
                f"<div style='font-size:1.3rem;font-weight:700;color:{rrc}'>{rr_str}</div></div>",
                unsafe_allow_html=True,
            )

            if calc["can_use_full"]:
                st.success(
                    f"✅ Tüm sermaye kullanılabilir — "
                    f"Risk: {calc['full_capital_risk_pct']:.2f}% ≤ {risk_pct:.1f}%"
                )
                # Auto-set to full capital when allowed (only if user hasn't overridden)
                if _POS_SIZE_KEY not in st.session_state:
                    st.session_state[_POS_SIZE_KEY] = float(capital)
            else:
                rec = calc["recommended_size"]
                st.warning(
                    f"⚠️ Tam sermaye riski **{calc['full_capital_risk_pct']:.2f}%** "
                    f"— Önerilen pozisyon: **${rec:,.2f}**"
                )
                ba, _ = st.columns([1, 4])
                with ba:
                    # ‼️ FIX: button directly writes to session_state then reruns
                    if st.button("💡 Öneriyi Uygula", key="pf_apply_rec", use_container_width=True):
                        st.session_state[_POS_SIZE_KEY] = float(rec)
                        st.rerun()

    # ── Position size input ───────────────────────────────────────────────────
    # Use session_state value as the live source; fall back to capital
    _current_size = float(st.session_state.get(_POS_SIZE_KEY, capital))

    st.markdown("")
    # ‼️ KEY TECHNIQUE: use on_change callback + manual value injection
    # We render without a key so the widget always reflects session state.
    # To avoid DuplicateWidgetID we give it a stable unique key but set
    # the *value* from session state on every render.
    position_size = st.number_input(
        "Pozisyon Büyüklüğü ($)",
        min_value=0.0,
        max_value=float(capital) * 10,
        value=_current_size,
        step=10.0,
        key="pf_pos_size_widget",
        on_change=_sync_pos_size,
    )
    # Keep session state in sync when user types manually
    st.session_state[_POS_SIZE_KEY] = float(st.session_state.get("pf_pos_size_widget", position_size))

    # Position heat
    if calc.get("risk_per_unit") and position_size > 0:
        heat       = calculate_position_heat(position_size, capital, calc["risk_per_unit"])
        heat_color = "#ef4444" if heat > risk_pct * 1.5 else "#f59e0b" if heat > risk_pct else "#10b981"
        st.markdown(
            f"<span style='font-size:0.8rem;color:#64748b'>Position Heat: "
            f"<b style='color:{heat_color}'>{heat:.2f}%</b> sermaye risk altında</span>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Journal metadata ──────────────────────────────────────────────────────
    st.markdown("**📓 Journal Bilgileri**")
    j1, j2, j3 = st.columns(3)
    with j1:
        setup_type  = st.selectbox("Setup Tipi",     SETUP_TYPES,  key="pf_setup")
        market_cond = st.selectbox("Piyasa Koşulu",  MARKET_CONDS, key="pf_market")
    with j2:
        emotion      = st.selectbox("Psikoloji", EMOTIONS, key="pf_emotion")
        plan_followed = st.checkbox("Plana uyuldu mu?", value=True, key="pf_plan")
    with j3:
        execution_score = st.slider("Execution Skoru (0–10)", 0, 10, 7, key="pf_exec_score")
        mistakes        = st.multiselect("Hata Etiketleri", MISTAKES, key="pf_mistakes")

    notes = st.text_area("Notlar", placeholder="Setup analizi, gerekçe...", key="pf_notes", height=80)

    # ── Save / Cancel ─────────────────────────────────────────────────────────
    st.markdown("")
    ac1, ac2 = st.columns([1, 5])
    with ac1:
        save_clicked = st.button("💾 Kaydet", type="primary", use_container_width=True, key="pf_save")
    with ac2:
        if st.button("İptal", key="pf_cancel"):
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

        avg_entry_save = calculate_avg_entry(entries)
        calc_save      = calculate_position_size(capital, risk_pct, avg_entry_save, stop_loss)
        rr_save        = calculate_rr(avg_entry_save, stop_loss, take_profits, direction)
        heat_save      = calculate_position_heat(position_size, capital, calc_save.get("risk_per_unit", 0)) if calc_save else 0

        data   = st.session_state.data
        pos_id = data["next_id"]
        data["next_id"] += 1

        position = {
            "id":              pos_id,
            "symbol":          symbol,
            "direction":       direction,
            "capital":         capital,
            "risk_pct":        risk_pct,
            "entries":         entries,
            "avg_entry":       avg_entry_save,
            "stop_loss":       stop_loss,
            "take_profits":    take_profits,
            "position_size":   position_size,
            "risk_calc":       calc_save,
            "rr":              rr_save,
            "heat":            heat_save,
            "setup_type":      setup_type,
            "market_condition": market_cond,
            "emotion":         emotion,
            "plan_followed":   plan_followed,
            "execution_score": execution_score,
            "mistakes":        mistakes,
            "notes":           notes,
            "created_at":      datetime.now().isoformat(),
        }
        data["active_positions"].append(position)
        save_data(data)
        st.session_state.data = data
        _clear_form_state()
        st.rerun()

    st.markdown("---")


def _sync_pos_size():
    """on_change callback: keep session state in sync when user edits the widget."""
    st.session_state[_POS_SIZE_KEY] = float(st.session_state.get("pf_pos_size_widget", 0.0))


def _clear_form_state():
    keys = [k for k in list(st.session_state.keys())
            if k.startswith("pf_") or k == _POS_SIZE_KEY]
    for k in keys:
        del st.session_state[k]
