import streamlit as st
from utils.data_manager import save_data
from utils.calculations import format_pnl
from datetime import datetime


def _position_title(pos: dict) -> str:
    rr = pos.get("rr")
    rr_str = f"1:{rr}" if rr else "1:?"
    return f"#{pos['id']} · {pos['symbol']} · {pos['direction']} · {rr_str}"


def render_active_positions(data: dict):
    positions = data.get("active_positions", [])

    if not positions:
        st.markdown("""
        <div style="text-align:center;padding:2.5rem;color:#2d3a4e;border:1px dashed #1e2530;border-radius:12px">
            <div style="font-size:2rem;margin-bottom:0.5rem">📭</div>
            <div style="font-family:'Space Mono',monospace;font-size:0.8rem">Aktif pozisyon bulunmuyor</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Iterate over a stable copy of IDs to avoid mutation-during-iteration bugs
    position_ids = [p["id"] for p in positions]

    for pos_id in position_ids:
        # Re-fetch position from live data each iteration
        pos_list = st.session_state.data.get("active_positions", [])
        pos = next((p for p in pos_list if p["id"] == pos_id), None)
        if pos is None:
            continue

        _render_single_active(pos, data)


def _render_single_active(pos: dict, data: dict):
    direction = pos.get("direction", "LONG")
    badge_cls = "badge-long" if direction == "LONG" else "badge-short"
    card_cls  = "card-long"  if direction == "LONG" else "card-short"
    rr        = pos.get("rr")
    rr_str    = f"1:{rr}" if rr else "1:?"
    title     = f"#{pos['id']} · {pos['symbol']} · {direction} · {rr_str}"

    with st.expander(title, expanded=False):
        # ── Detail grid ──
        entries = pos.get("entries", [])
        avg_entry = pos.get("avg_entry", 0)
        stop_loss = pos.get("stop_loss", 0)
        take_profits = pos.get("take_profits", [])
        pos_size = pos.get("position_size", 0)
        calc = pos.get("risk_calc", {})

        entry_display = " / ".join(
            f"${e['price']:,.4f} ({e['weight']:.0f}%)" for e in entries
        ) if entries else "—"

        tp_display = " / ".join(
            f"${t['price']:,.4f} ({t['weight']:.0f}%)" for t in take_profits
        ) if take_profits else "—"

        created = pos.get("created_at", "")
        if created:
            try:
                created = datetime.fromisoformat(created).strftime("%d %b %Y %H:%M")
            except ValueError:
                pass

        st.markdown(f"""
        <div class="detail-grid">
            <div class="detail-item">
                <div class="detail-label">Yön</div>
                <div class="detail-value"><span class="badge {badge_cls}">{direction}</span></div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Avg Entry</div>
                <div class="detail-value">${avg_entry:,.4f}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Stop Loss</div>
                <div class="detail-value" style="color:#ef4444">${stop_loss:,.4f}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Take Profit</div>
                <div class="detail-value" style="color:#10b981;font-size:0.75rem">{tp_display}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Pozisyon Büyüklüğü</div>
                <div class="detail-value">${pos_size:,.2f}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Risk/Ödül</div>
                <div class="detail-value">{rr_str}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Risk Tutarı</div>
                <div class="detail-value">${calc.get('risk_amount', 0):,.2f}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Tarih</div>
                <div class="detail-value" style="font-size:0.75rem">{created}</div>
            </div>
        </div>
        <div class="detail-item" style="margin-top:0.6rem">
            <div class="detail-label">Entry Noktaları</div>
            <div class="detail-value" style="font-size:0.78rem">{entry_display}</div>
        </div>
        """, unsafe_allow_html=True)

        if pos.get("notes"):
            st.markdown(f"""
            <div class="detail-item" style="margin-top:0.5rem">
                <div class="detail-label">Notlar</div>
                <div class="detail-value" style="font-size:0.8rem;color:#94a3b8">{pos['notes']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        # ── Close modal state ─────────────────────────────────────────────
        close_key = f"show_close_{pos['id']}"
        if close_key not in st.session_state:
            st.session_state[close_key] = False

        btn_col1, btn_col2, btn_col3 = st.columns([2, 2, 4])

        with btn_col1:
            if st.button("✅ Pozisyonu Kapat", key=f"close_btn_{pos['id']}", use_container_width=True):
                st.session_state[close_key] = True

        with btn_col2:
            if st.button("🗑️ Sil", key=f"delete_btn_{pos['id']}", use_container_width=True):
                _delete_position(pos["id"])
                st.rerun()

        # ── Close form ────────────────────────────────────────────────────
        if st.session_state.get(close_key, False):
            st.markdown("---")
            st.markdown("**Pozisyonu Kapat**")
            pnl_val = st.number_input(
                "PnL ($)",
                value=0.0,
                step=1.0,
                key=f"close_pnl_{pos['id']}",
                help="Pozitif = kâr, negatif = zarar",
            )
            comment = st.text_area(
                "Yorum",
                placeholder="İşlem yorumu, öğrenilen ders...",
                key=f"close_comment_{pos['id']}",
                height=70,
            )
            cc1, cc2 = st.columns(2)
            with cc1:
                if st.button("Onayla ve Kapat", type="primary", key=f"confirm_close_{pos['id']}", use_container_width=True):
                    _close_position(pos["id"], pnl_val, comment)
                    st.session_state[close_key] = False
                    st.rerun()
            with cc2:
                if st.button("İptal", key=f"cancel_close_{pos['id']}", use_container_width=True):
                    st.session_state[close_key] = False
                    st.rerun()


def _delete_position(pos_id: int):
    data = st.session_state.data
    data["active_positions"] = [p for p in data["active_positions"] if p["id"] != pos_id]
    save_data(data)
    st.session_state.data = data


def _close_position(pos_id: int, pnl: float, comment: str):
    data = st.session_state.data
    pos = next((p for p in data["active_positions"] if p["id"] == pos_id), None)
    if pos is None:
        return

    rr = pos.get("rr")
    closed = {
        **pos,
        "pnl": pnl,
        "comment": comment,
        "closed_at": datetime.now().isoformat(),
        "result": "WIN" if pnl > 0 else "LOSS",
        "rr_display": f"1:{rr}" if rr else "1:?",
    }
    data["active_positions"] = [p for p in data["active_positions"] if p["id"] != pos_id]
    data["closed_trades"].append(closed)
    save_data(data)
    st.session_state.data = data
