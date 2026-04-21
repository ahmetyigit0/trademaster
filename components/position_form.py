import streamlit as st
from datetime import datetime
from utils.data_manager import save_data
from utils.calculations import calculate_avg_entry, calculate_position_size


def render_position_form():
    st.markdown("---")
    st.markdown("#### ➕ Yeni Pozisyon")

    with st.container():
        col1, col2, col3, col4 = st.columns([2, 1, 2, 2])
        with col1:
            symbol = st.text_input("Symbol", placeholder="BTC, ETH, AAPL...", key="form_symbol").upper().strip()
        with col2:
            direction = st.selectbox("Yön", ["LONG", "SHORT"], key="form_direction")
        with col3:
            capital = st.number_input("Toplam Sermaye ($)", min_value=0.0, value=10000.0, step=100.0, key="form_capital")
        with col4:
            risk_pct = st.number_input("Risk (%)", min_value=0.1, max_value=100.0, value=2.0, step=0.1, key="form_risk_pct")

        st.markdown("##### Entry Noktaları")
        partial_entry = st.checkbox("Parçalı giriş (3 entry)", key="form_partial")

        entries = []
        if partial_entry:
            ecol1, ecol2, ecol3 = st.columns(3)
            weights = [30.0, 40.0, 30.0]
            for i, (col, default_w) in enumerate(zip([ecol1, ecol2, ecol3], weights)):
                with col:
                    price = st.number_input(f"Entry {i+1} Fiyat", min_value=0.0, value=0.0, format="%.4f", key=f"form_entry_price_{i}")
                    weight = st.number_input(f"Ağırlık {i+1} (%)", min_value=0.0, max_value=100.0, value=default_w, key=f"form_entry_weight_{i}")
                    if price > 0:
                        entries.append({"price": price, "weight": weight})
        else:
            single_price = st.number_input("Entry Fiyat", min_value=0.0, value=0.0, format="%.4f", key="form_single_entry")
            if single_price > 0:
                entries.append({"price": single_price, "weight": 100.0})

        col_sl, col_tp_partial = st.columns([1, 1])
        with col_sl:
            stop_loss = st.number_input("Stop Loss ✱", min_value=0.0, value=0.0, format="%.4f", key="form_stop_loss")
        with col_tp_partial:
            partial_tp = st.checkbox("Parçalı TP", key="form_partial_tp")

        take_profits = []
        if partial_tp:
            tp_cols = st.columns(3)
            tp_defaults = [(33.3, "TP 1"), (33.3, "TP 2"), (33.4, "TP 3")]
            for i, (tc, (dw, label)) in enumerate(zip(tp_cols, tp_defaults)):
                with tc:
                    tp_price = st.number_input(f"{label} Fiyat", min_value=0.0, value=0.0, format="%.4f", key=f"form_tp_price_{i}")
                    tp_weight = st.number_input(f"{label} Ağırlık (%)", min_value=0.0, max_value=100.0, value=dw, key=f"form_tp_weight_{i}")
                    if tp_price > 0:
                        take_profits.append({"price": tp_price, "weight": tp_weight})
        else:
            tp_single = st.number_input("Take Profit Fiyat", min_value=0.0, value=0.0, format="%.4f", key="form_tp_single")
            if tp_single > 0:
                take_profits.append({"price": tp_single, "weight": 100.0})

        # ── Risk calculation preview ──────────────────────────────────────────
        if entries and stop_loss > 0:
            avg_entry = calculate_avg_entry(entries)
            calc = calculate_position_size(capital, risk_pct, avg_entry, stop_loss)
            if calc:
                st.markdown("---")
                st.markdown("**📐 Risk Analizi**")
                rcol1, rcol2, rcol3 = st.columns(3)
                with rcol1:
                    st.metric("Ort. Entry", f"${avg_entry:,.4f}")
                with rcol2:
                    st.metric("Risk Tutarı", f"${calc['risk_amount']:,.2f}")
                with rcol3:
                    st.metric("Tam Sermaye Riski", f"{calc['full_capital_risk_pct']:.2f}%")

                if calc["can_use_full"]:
                    st.success(f"✅ Tüm sermaye ile girilebilir. Risk: {calc['full_capital_risk_pct']:.2f}% ≤ {risk_pct}%")
                    st.session_state["suggested_size"] = capital
                else:
                    st.warning(f"⚠️ Önerilen pozisyon büyüklüğü: **${calc['recommended_size']:,.2f}** (Tam sermaye riski: {calc['full_capital_risk_pct']:.2f}%)")
                    st.session_state["suggested_size"] = calc["recommended_size"]

                    if st.button("💡 Öneriyi Uygula", key="apply_suggestion"):
                        st.session_state["form_position_size_override"] = calc["recommended_size"]

        pos_size_override = st.session_state.get("form_position_size_override", None)
        pos_size_default = pos_size_override if pos_size_override else (
            st.session_state.get("suggested_size", capital)
        )
        position_size = st.number_input(
            "Pozisyon Büyüklüğü ($)",
            min_value=0.0,
            value=float(pos_size_default),
            step=10.0,
            key="form_position_size",
        )

        notes = st.text_area("Notlar (opsiyonel)", placeholder="Analiz, setup açıklaması...", key="form_notes", height=80)

        st.markdown("")
        acol1, acol2 = st.columns([1, 4])
        with acol1:
            save_clicked = st.button("💾 Kaydet", type="primary", use_container_width=True, key="save_position_btn")
        with acol2:
            if st.button("İptal", use_container_width=False, key="cancel_form_btn"):
                st.session_state.show_add_form = False
                _clear_form_state()
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

            avg_entry = calculate_avg_entry(entries)
            calc = calculate_position_size(capital, risk_pct, avg_entry, stop_loss) if entries else {}
            from utils.calculations import calculate_rr
            rr = calculate_rr(avg_entry, stop_loss, take_profits)

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
                "partial_entry": partial_entry,
                "avg_entry": avg_entry,
                "stop_loss": stop_loss,
                "take_profits": take_profits,
                "position_size": position_size,
                "risk_calc": calc,
                "rr": rr,
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
    keys_to_clear = [k for k in st.session_state.keys() if k.startswith("form_") or k in ("suggested_size",)]
    for k in keys_to_clear:
        del st.session_state[k]
