import streamlit as st
from utils.data_manager import save_data
from utils.calculations import format_pnl, rr_color, calculate_r_multiple
from datetime import datetime

EMOTIONS  = ["calm", "fomo", "revenge", "anxious", "confident"]
MISTAKES  = ["early exit", "late entry", "no stop", "oversize", "revenge trade", "ignored plan"]


def render_active_positions(data: dict):
    positions = data.get("active_positions", [])
    if not positions:
        st.markdown("""
        <div style="text-align:center;padding:2.5rem;color:#2d3a4e;border:1px dashed #1e2530;border-radius:12px">
            <div style="font-size:2rem;margin-bottom:0.5rem">📭</div>
            <div style="font-family:'Space Mono',monospace;font-size:0.8rem">Aktif pozisyon bulunmuyor</div>
        </div>""", unsafe_allow_html=True)
        return

    for pos_id in [p["id"] for p in positions]:
        pos = next((p for p in st.session_state.data.get("active_positions", []) if p["id"] == pos_id), None)
        if pos:
            _render_single(pos)


def _render_single(pos: dict):
    direction = pos.get("direction", "LONG")
    rr        = pos.get("rr")
    rr_str    = f"1:{rr}" if rr else "1:?"
    rc        = rr_color(rr)
    title     = f"#{pos['id']} · {pos['symbol']} · {direction} · {rr_str}"

    with st.expander(title, expanded=False):
        entries      = pos.get("entries", [])
        avg_entry    = pos.get("avg_entry", 0)
        stop_loss    = pos.get("stop_loss", 0)
        take_profits = pos.get("take_profits", [])
        pos_size     = pos.get("position_size", 0)
        calc         = pos.get("risk_calc", {})
        heat         = pos.get("heat", 0)
        dir_color    = "#10b981" if direction == "LONG" else "#ef4444"
        dir_bg       = "#064e3b" if direction == "LONG" else "#450a0a"

        entry_str = " / ".join(f"${e['price']:,.4f} ({e['weight']:.0f}%)" for e in entries) or "—"
        tp_str    = " / ".join(f"${t['price']:,.4f} ({t['weight']:.0f}%)" for t in take_profits) or "—"
        created   = pos.get("created_at", "")
        try:
            created = datetime.fromisoformat(created).strftime("%d %b %Y %H:%M")
        except Exception:
            pass

        st.markdown(f"""
        <div class="detail-grid">
          <div class="detail-item">
            <div class="detail-label">Yön</div>
            <div class="detail-value">
              <span style="background:{dir_bg};color:{dir_color};padding:2px 8px;border-radius:4px;font-size:0.72rem;font-weight:700">{direction}</span>
            </div>
          </div>
          <div class="detail-item">
            <div class="detail-label">R:R</div>
            <div class="detail-value" style="color:{rc}">{rr_str}</div>
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
            <div class="detail-label">Pozisyon ($)</div>
            <div class="detail-value">${pos_size:,.2f}</div>
          </div>
          <div class="detail-item">
            <div class="detail-label">Risk $</div>
            <div class="detail-value">${calc.get('risk_amount', 0):,.2f}</div>
          </div>
          <div class="detail-item">
            <div class="detail-label">Position Heat</div>
            <div class="detail-value" style="color:{'#ef4444' if heat > 3 else '#f59e0b' if heat > 2 else '#10b981'}">{heat:.2f}%</div>
          </div>
          <div class="detail-item">
            <div class="detail-label">Tarih</div>
            <div class="detail-value" style="font-size:0.72rem">{created}</div>
          </div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;margin-top:0.5rem">
          <div class="detail-item">
            <div class="detail-label">Entries</div>
            <div class="detail-value" style="font-size:0.72rem">{entry_str}</div>
          </div>
          <div class="detail-item">
            <div class="detail-label">Take Profits</div>
            <div class="detail-value" style="font-size:0.72rem;color:#10b981">{tp_str}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Journal meta
        meta_items = []
        if pos.get("setup_type"):
            meta_items.append(f"<span style='background:#1e293b;color:#94a3b8;padding:2px 7px;border-radius:4px;font-size:0.68rem;margin-right:4px'>⚙️ {pos['setup_type']}</span>")
        if pos.get("market_condition"):
            meta_items.append(f"<span style='background:#1e293b;color:#94a3b8;padding:2px 7px;border-radius:4px;font-size:0.68rem;margin-right:4px'>📊 {pos['market_condition']}</span>")
        if pos.get("emotion"):
            meta_items.append(f"<span style='background:#1e293b;color:#94a3b8;padding:2px 7px;border-radius:4px;font-size:0.68rem;margin-right:4px'>🧠 {pos['emotion']}</span>")
        if pos.get("execution_score") is not None:
            meta_items.append(f"<span style='background:#1e293b;color:#94a3b8;padding:2px 7px;border-radius:4px;font-size:0.68rem;margin-right:4px'>⭐ exec:{pos['execution_score']}/10</span>")
        if meta_items:
            st.markdown("<div style='margin-top:0.5rem'>" + "".join(meta_items) + "</div>", unsafe_allow_html=True)

        if pos.get("notes"):
            st.markdown(f"""<div class="detail-item" style="margin-top:0.5rem">
                <div class="detail-label">Notlar</div>
                <div class="detail-value" style="font-size:0.8rem;color:#94a3b8">{pos['notes']}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # ── Buttons ──
        close_key = f"show_close_{pos['id']}"
        st.session_state.setdefault(close_key, False)

        b1, b2, _ = st.columns([2, 2, 4])
        with b1:
            if st.button("✅ Pozisyonu Kapat", key=f"close_btn_{pos['id']}", use_container_width=True):
                st.session_state[close_key] = True
        with b2:
            if st.button("🗑️ Sil", key=f"del_btn_{pos['id']}", use_container_width=True):
                _delete_position(pos["id"])
                st.rerun()

        # ── Close form ────────────────────────────────────────────────────
        if st.session_state.get(close_key):
            st.markdown("---")
            st.markdown("**Pozisyonu Kapat**")
            cf1, cf2 = st.columns(2)
            with cf1:
                pnl_val = st.number_input("PnL ($)", value=0.0, step=1.0, key=f"cpnl_{pos['id']}")
                exec_score = st.slider("Execution Skoru", 0, 10, pos.get("execution_score", 7), key=f"cexec_{pos['id']}")
            with cf2:
                emotion = st.selectbox("Psikoloji", EMOTIONS,
                                       index=EMOTIONS.index(pos.get("emotion", "calm")) if pos.get("emotion") in EMOTIONS else 0,
                                       key=f"cemo_{pos['id']}")
                mistakes = st.multiselect("Hata Etiketleri", MISTAKES,
                                          default=pos.get("mistakes", []),
                                          key=f"cmist_{pos['id']}")
            comment = st.text_area("Yorum", placeholder="Öğrenilen ders...", key=f"ccomm_{pos['id']}", height=60)

            ok1, ok2 = st.columns(2)
            with ok1:
                if st.button("Onayla ve Kapat", type="primary", key=f"conf_close_{pos['id']}", use_container_width=True):
                    _close_position(pos["id"], pnl_val, comment, exec_score, emotion, mistakes)
                    st.session_state[close_key] = False
                    st.rerun()
            with ok2:
                if st.button("İptal", key=f"cancel_close_{pos['id']}", use_container_width=True):
                    st.session_state[close_key] = False
                    st.rerun()


def _delete_position(pos_id):
    data = st.session_state.data
    data["active_positions"] = [p for p in data["active_positions"] if p["id"] != pos_id]
    save_data(data)
    st.session_state.data = data


def _close_position(pos_id, pnl, comment, exec_score, emotion, mistakes):
    data = st.session_state.data
    pos = next((p for p in data["active_positions"] if p["id"] == pos_id), None)
    if not pos:
        return
    rr = pos.get("rr")
    risk_amount = pos.get("risk_calc", {}).get("risk_amount", 0)
    r_multiple = calculate_r_multiple(pnl, risk_amount)
    closed = {
        **pos,
        "pnl": pnl,
        "comment": comment,
        "execution_score": exec_score,
        "emotion": emotion,
        "mistakes": mistakes,
        "r_multiple": r_multiple,
        "closed_at": datetime.now().isoformat(),
        "result": "WIN" if pnl > 0 else "LOSS",
        "rr_display": f"1:{rr}" if rr else "1:?",
    }
    data["active_positions"] = [p for p in data["active_positions"] if p["id"] != pos_id]
    data["closed_trades"].append(closed)
    save_data(data)
    st.session_state.data = data
