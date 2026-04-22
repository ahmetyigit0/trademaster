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

# ── Session state key for position size ───────────────────────────────────────
_SZ = "pf__pos_size"   # single source-of-truth; never collides with widget keys


def render_position_form():
    _render_new_form()


def render_edit_form(pos_id: int):
    """Called from active_positions when user clicks Edit."""
    _render_new_form(edit_id=pos_id)


# ─────────────────────────────────────────────────────────────────────────────

def _render_new_form(edit_id: int | None = None):
    editing = edit_id is not None
    prefix  = f"pf_{edit_id}_" if editing else "pf_"

    if editing:
        pos = next((p for p in st.session_state.data["active_positions"] if p["id"] == edit_id), None)
        if pos is None:
            st.error("Pozisyon bulunamadı.")
            return
        st.markdown(f"#### ✏️ Pozisyonu Düzenle — #{edit_id} {pos['symbol']}")
    else:
        st.markdown("---")
        st.markdown("#### ➕ Yeni Pozisyon")

    # ── Basic ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([2, 1, 2, 2])
    with c1:
        _sym_default = pos["symbol"] if editing else ""
        symbol = st.text_input("Symbol", value=_sym_default, placeholder="BTC, ETH...", key=f"{prefix}symbol").upper().strip()
    with c2:
        _dir_idx = 0 if (not editing or pos["direction"] == "LONG") else 1
        direction = st.selectbox("Yön", ["LONG", "SHORT"], index=_dir_idx, key=f"{prefix}direction")
    with c3:
        _cap = float(pos["capital"]) if editing else 10000.0
        capital = st.number_input("Sermaye ($)", min_value=0.0, value=_cap, step=100.0, key=f"{prefix}capital")
    with c4:
        _rp = float(pos["risk_pct"]) if editing else 2.0
        risk_pct = st.number_input("Risk (%)", min_value=0.1, max_value=100.0, value=_rp, step=0.1, key=f"{prefix}risk_pct")

    # ── Entries ────────────────────────────────────────────────────────────
    st.markdown("**📍 Entry Noktaları**")
    _ne_default = len(pos["entries"]) if editing and pos.get("entries") else 1
    num_entries = int(st.number_input("Entry sayısı", min_value=1, max_value=10,
                                      value=_ne_default, step=1, key=f"{prefix}num_entries"))

    entries = []
    default_ew = round(100.0 / num_entries, 1)
    ecols = st.columns(min(num_entries, 5))
    for i in range(num_entries):
        with ecols[i % len(ecols)]:
            _ep = float(pos["entries"][i]["price"])   if editing and i < len(pos.get("entries", [])) else 0.0
            _ew = float(pos["entries"][i]["weight"])  if editing and i < len(pos.get("entries", [])) else default_ew
            p = st.number_input(f"Entry {i+1} Fiyat", min_value=0.0, value=_ep, format="%.6f", key=f"{prefix}ep{i}")
            w = st.number_input(f"Ağırlık {i+1} (%)", min_value=0.0, max_value=100.0, value=_ew, key=f"{prefix}ew{i}")
            if p > 0:
                entries.append({"price": p, "weight": w})

    # ── Stop / TP ──────────────────────────────────────────────────────────
    sl_col, tp_col = st.columns(2)
    with sl_col:
        _sl = float(pos["stop_loss"]) if editing else 0.0
        stop_loss = st.number_input("Stop Loss ✱", min_value=0.0, value=_sl, format="%.6f", key=f"{prefix}sl")
    with tp_col:
        _ntp = len(pos["take_profits"]) if editing and pos.get("take_profits") else 1
        num_tp = int(st.number_input("TP sayısı", min_value=1, max_value=5, value=_ntp, step=1, key=f"{prefix}num_tp"))

    take_profits = []
    default_tpw = round(100.0 / num_tp, 1)
    tp_cols = st.columns(min(num_tp, 5))
    for i in range(num_tp):
        with tp_cols[i % len(tp_cols)]:
            _tp_p = float(pos["take_profits"][i]["price"])  if editing and i < len(pos.get("take_profits", [])) else 0.0
            _tp_w = float(pos["take_profits"][i]["weight"]) if editing and i < len(pos.get("take_profits", [])) else default_tpw
            tp_p = st.number_input(f"TP {i+1} Fiyat", min_value=0.0, value=_tp_p, format="%.6f", key=f"{prefix}tpp{i}")
            tp_w = st.number_input(f"TP {i+1} Ağırlık (%)", min_value=0.0, max_value=100.0, value=_tp_w, key=f"{prefix}tpw{i}")
            if tp_p > 0:
                take_profits.append({"price": tp_p, "weight": tp_w})

    # ── Risk Engine ────────────────────────────────────────────────────────
    calc: dict = {}
    avg_entry  = 0.0
    rr         = None

    sz_key = f"{_SZ}_{prefix}"   # unique per form instance

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
            m1.metric("Avg Entry",         f"${avg_entry:,.4f}")
            m2.metric("Risk Tutarı",        f"${calc['risk_amount']:,.2f}")
            m3.metric("Tam Sermaye Riski",  f"{calc['full_capital_risk_pct']:.2f}%")
            m4.markdown(
                f"<div style='padding-top:0.35rem'>"
                f"<div style='font-size:0.7rem;color:#484f58;margin-bottom:4px'>R:R</div>"
                f"<div style='font-size:1.3rem;font-weight:700;color:{rrc}'>{rr_str}</div></div>",
                unsafe_allow_html=True,
            )

            if calc["can_use_full"]:
                st.success(
                    f"✅ Tüm sermaye kullanılabilir — "
                    f"Risk: {calc['full_capital_risk_pct']:.2f}% ≤ {risk_pct:.1f}%"
                )
                # Only set default once if user hasn't interacted
                if sz_key not in st.session_state:
                    st.session_state[sz_key] = float(capital)
            else:
                rec = calc["recommended_size"]
                st.warning(
                    f"⚠️ Tam sermaye riski **{calc['full_capital_risk_pct']:.2f}%** "
                    f"— Önerilen: **${rec:,.2f}**"
                )
                # ‼️ THE FIX: button sets state key, then we IMMEDIATELY use
                # that value below — no rerun needed because the number_input
                # below reads directly from session_state[sz_key] every render.
                if st.button("💡 Öneriyi Uygula", key=f"{prefix}apply_rec"):
                    st.session_state[sz_key] = float(rec)
                    st.rerun()

    # ── Position size ──────────────────────────────────────────────────────
    # Determine default: editing → saved value; new → capital or suggestion
    if sz_key not in st.session_state:
        if editing:
            st.session_state[sz_key] = float(pos.get("position_size", capital))
        else:
            st.session_state[sz_key] = float(capital)

    # We do NOT use key= on this widget intentionally — instead we pass
    # value= from session_state every render so "Öneriyi Uygula" always works.
    # on_change writes back so manual edits are also preserved.
    _widget_key = f"{prefix}ps_widget"

    def _on_ps_change():
        st.session_state[sz_key] = float(st.session_state.get(_widget_key, 0.0))

    position_size = st.number_input(
        "💰 Yatırılacak Tutar ($)",
        min_value=0.0,
        max_value=float(capital) * 20,
        value=float(st.session_state[sz_key]),
        step=10.0,
        key=_widget_key,
        on_change=_on_ps_change,
    )

    # Position heat indicator
    if calc.get("risk_per_unit") and position_size > 0:
        heat       = calculate_position_heat(position_size, capital, calc["risk_per_unit"])
        heat_color = "#da3633" if heat > risk_pct * 1.5 else "#e3b341" if heat > risk_pct else "#238636"
        st.markdown(
            f"<span style='font-size:0.78rem;color:#484f58'>Position Heat: "
            f"<b style='color:{heat_color}'>{heat:.2f}%</b> sermaye risk altında</span>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Journal ────────────────────────────────────────────────────────────
    st.markdown("**📓 Journal Bilgileri**")
    j1, j2, j3 = st.columns(3)
    with j1:
        _si = SETUP_TYPES.index(pos["setup_type"]) if editing and pos.get("setup_type") in SETUP_TYPES else 0
        setup_type   = st.selectbox("Setup Tipi",    SETUP_TYPES,  index=_si, key=f"{prefix}setup")
        _mi = MARKET_CONDS.index(pos["market_condition"]) if editing and pos.get("market_condition") in MARKET_CONDS else 0
        market_cond  = st.selectbox("Piyasa Koşulu", MARKET_CONDS, index=_mi, key=f"{prefix}market")
    with j2:
        _ei = EMOTIONS.index(pos["emotion"]) if editing and pos.get("emotion") in EMOTIONS else 0
        emotion       = st.selectbox("Psikoloji", EMOTIONS, index=_ei, key=f"{prefix}emotion")
        _pf = pos.get("plan_followed", True) if editing else True
        plan_followed = st.checkbox("Plana uyuldu mu?", value=_pf, key=f"{prefix}plan")
    with j3:
        _es = int(pos.get("execution_score", 7)) if editing else 7
        execution_score = st.slider("Execution Skoru (0–10)", 0, 10, _es, key=f"{prefix}exec_score")
        _mk = pos.get("mistakes", []) if editing else []
        mistakes        = st.multiselect("Hata Etiketleri", MISTAKES, default=_mk, key=f"{prefix}mistakes")

    _notes = pos.get("notes", "") if editing else ""
    notes = st.text_area("Notlar", value=_notes, placeholder="Setup analizi, gerekçe...",
                          key=f"{prefix}notes", height=75)

    # ── Buttons ────────────────────────────────────────────────────────────
    st.markdown("")
    ac1, ac2, *_ = st.columns([1, 1, 3])
    with ac1:
        btn_label    = "💾 Güncelle" if editing else "💾 Kaydet"
        save_clicked = st.button(btn_label, type="primary", use_container_width=True, key=f"{prefix}save")
    with ac2:
        cancel_label = "İptal"
        if st.button(cancel_label, key=f"{prefix}cancel", use_container_width=True):
            if editing:
                st.session_state[f"edit_mode_{edit_id}"] = False
            _clear_form_state(prefix, sz_key)
            st.rerun()

    if save_clicked:
        # Validation
        if not symbol:
            st.error("Symbol giriniz.")
            return
        if not entries:
            st.error("En az bir entry fiyatı giriniz.")
            return
        if stop_loss <= 0:
            st.error("Stop loss giriniz.")
            return

        avg_e  = calculate_avg_entry(entries)
        calc_s = calculate_position_size(capital, risk_pct, avg_e, stop_loss)
        rr_s   = calculate_rr(avg_e, stop_loss, take_profits, direction)
        heat_s = calculate_position_heat(position_size, capital,
                                         calc_s.get("risk_per_unit", 0)) if calc_s else 0

        data = st.session_state.data

        record = {
            "symbol":          symbol,
            "direction":       direction,
            "capital":         capital,
            "risk_pct":        risk_pct,
            "entries":         entries,
            "avg_entry":       avg_e,
            "stop_loss":       stop_loss,
            "take_profits":    take_profits,
            "position_size":   position_size,
            "risk_calc":       calc_s,
            "rr":              rr_s,
            "heat":            heat_s,
            "setup_type":      setup_type,
            "market_condition": market_cond,
            "emotion":         emotion,
            "plan_followed":   plan_followed,
            "execution_score": execution_score,
            "mistakes":        mistakes,
            "notes":           notes,
        }

        if editing:
            # Update existing position in-place
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
        _clear_form_state(prefix, sz_key)
        st.rerun()

    if not editing:
        st.markdown("---")


def _clear_form_state(prefix: str, sz_key: str):
    remove = [k for k in list(st.session_state.keys())
              if k.startswith(prefix) or k == sz_key]
    for k in remove:
        del st.session_state[k]
