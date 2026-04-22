import streamlit as st
from utils.data_manager import save_data
from utils.calculations import format_pnl, rr_color, calculate_r_multiple
from datetime import datetime

EMOTIONS = ["calm", "fomo", "revenge", "anxious", "confident"]
MISTAKES = ["early exit", "late entry", "no stop", "oversize", "revenge trade", "ignored plan"]


def render_active_positions(data: dict):
    positions = data.get("active_positions", [])
    if not positions:
        st.markdown("""
        <div style="text-align:center;padding:2.5rem;color:#30363d;border:1px dashed #21262d;border-radius:10px">
            <div style="font-size:2rem;margin-bottom:0.5rem">📭</div>
            <div style="font-family:'Space Mono',monospace;font-size:0.78rem;color:#484f58">Aktif pozisyon yok</div>
        </div>""", unsafe_allow_html=True)
        return

    for pos_id in [p["id"] for p in positions]:
        pos = next((p for p in st.session_state.data.get("active_positions", []) if p["id"] == pos_id), None)
        if pos:
            _render_single(pos)


def _render_single(pos: dict):
    from components.position_form import render_edit_form

    pid       = pos["id"]
    direction = pos.get("direction", "LONG")
    rr        = pos.get("rr")
    rr_str    = f"1:{rr}" if rr else "1:?"
    rc        = rr_color(rr)
    title     = f"#{pid} · {pos['symbol']} · {direction} · RR {rr_str}"

    edit_key  = f"edit_mode_{pid}"
    st.session_state.setdefault(edit_key, False)

    with st.expander(title, expanded=False):

        # ── Edit mode toggle ─────────────────────────────────────────────
        if st.session_state.get(edit_key):
            render_edit_form(pid)
            return   # form renders in place, rest hidden

        # ── Detail view ──────────────────────────────────────────────────
        entries      = pos.get("entries", [])
        avg_entry    = pos.get("avg_entry", 0)
        stop_loss    = pos.get("stop_loss", 0)
        take_profits = pos.get("take_profits", [])
        pos_size     = pos.get("position_size", 0)
        calc         = pos.get("risk_calc", {})
        heat         = pos.get("heat", 0)

        dir_color = "#3fb950" if direction == "LONG" else "#ff7b72"
        dir_bg    = "#0a2e1a" if direction == "LONG" else "#2d0f0f"
        rrc_css   = rc

        entry_str = " / ".join(f"${e['price']:,.4f} ({e['weight']:.0f}%)" for e in entries) or "—"
        tp_str    = " / ".join(f"${t['price']:,.4f} ({t['weight']:.0f}%)" for t in take_profits) or "—"
        created   = pos.get("created_at", "")
        try:
            created = datetime.fromisoformat(created).strftime("%d %b %Y %H:%M")
        except Exception:
            pass

        heat_color = "#da3633" if heat > 3 else "#e3b341" if heat > 2 else "#3fb950"

        st.markdown(f"""
        <div class="detail-grid">
          <div class="detail-item">
            <div class="detail-label">Yön</div>
            <div class="detail-value">
              <span style="background:{dir_bg};color:{dir_color};padding:2px 10px;border-radius:5px;font-size:0.7rem;font-weight:700;letter-spacing:0.05em">{direction}</span>
            </div>
          </div>
          <div class="detail-item">
            <div class="detail-label">R:R</div>
            <div class="detail-value" style="color:{rrc_css}">{rr_str}</div>
          </div>
          <div class="detail-item">
            <div class="detail-label">Avg Entry</div>
            <div class="detail-value">${avg_entry:,.4f}</div>
          </div>
          <div class="detail-item">
            <div class="detail-label">Stop Loss</div>
            <div class="detail-value" style="color:#ff7b72">${stop_loss:,.4f}</div>
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
            <div class="detail-label">Heat</div>
            <div class="detail-value" style="color:{heat_color}">{heat:.2f}%</div>
          </div>
          <div class="detail-item">
            <div class="detail-label">Tarih</div>
            <div class="detail-value" style="font-size:0.7rem">{created}</div>
          </div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;margin-top:0.5rem">
          <div class="detail-item">
            <div class="detail-label">Entries</div>
            <div class="detail-value" style="font-size:0.7rem">{entry_str}</div>
          </div>
          <div class="detail-item">
            <div class="detail-label">Take Profits</div>
            <div class="detail-value" style="font-size:0.7rem;color:#3fb950">{tp_str}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Meta badges
        meta = []
        for key, icon in [("setup_type", "⚙️"), ("market_condition", "📊"),
                           ("emotion", "🧠")]:
            if pos.get(key):
                meta.append(f"<span style='background:#161b22;border:1px solid #21262d;color:#8b949e;"
                             f"padding:2px 8px;border-radius:5px;font-size:0.67rem;margin-right:4px'>"
                             f"{icon} {pos[key]}</span>")
        if pos.get("execution_score") is not None:
            meta.append(f"<span style='background:#161b22;border:1px solid #21262d;color:#8b949e;"
                        f"padding:2px 8px;border-radius:5px;font-size:0.67rem;margin-right:4px'>"
                        f"⭐ exec:{pos['execution_score']}/10</span>")
        if meta:
            st.markdown("<div style='margin-top:0.5rem'>" + "".join(meta) + "</div>", unsafe_allow_html=True)

        if pos.get("notes"):
            st.markdown(f"""<div class="detail-item" style="margin-top:0.5rem">
                <div class="detail-label">Notlar</div>
                <div class="detail-value" style="font-size:0.78rem;color:#8b949e">{pos['notes']}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # ── Action buttons ───────────────────────────────────────────────
        close_key = f"show_close_{pid}"
        st.session_state.setdefault(close_key, False)

        b1, b2, b3, _ = st.columns([2, 2, 2, 2])
        with b1:
            if st.button("✅ Kapat", key=f"closebtn_{pid}", use_container_width=True):
                st.session_state[close_key] = True
                st.session_state[edit_key]  = False
        with b2:
            if st.button("✏️ Düzenle", key=f"editbtn_{pid}", use_container_width=True):
                st.session_state[edit_key]  = True
                st.session_state[close_key] = False
                st.rerun()
        with b3:
            if st.button("🗑️ Sil", key=f"delbtn_{pid}", use_container_width=True):
                _delete_position(pid)
                st.rerun()

        # ── Close form ────────────────────────────────────────────────────
        if st.session_state.get(close_key):
            st.markdown("---")
            st.markdown("**Pozisyonu Kapat**")
            cf1, cf2 = st.columns(2)
            with cf1:
                pnl_val    = st.number_input("PnL ($)", value=0.0, step=1.0, key=f"cpnl_{pid}")
                exec_score = st.slider("Execution Skoru", 0, 10,
                                       int(pos.get("execution_score") or 7),
                                       key=f"cexec_{pid}")
            with cf2:
                _ei = EMOTIONS.index(pos.get("emotion", "calm")) if pos.get("emotion") in EMOTIONS else 0
                emotion  = st.selectbox("Psikoloji", EMOTIONS, index=_ei, key=f"cemo_{pid}")
                mistakes = st.multiselect("Hata Etiketleri", MISTAKES,
                                          default=pos.get("mistakes", []),
                                          key=f"cmist_{pid}")
            comment = st.text_area("Yorum", placeholder="Öğrenilen ders...",
                                   key=f"ccomm_{pid}", height=60)
            ok1, ok2 = st.columns(2)
            with ok1:
                if st.button("Onayla ✅", type="primary", key=f"confclose_{pid}", use_container_width=True):
                    _close_position(pid, pnl_val, comment, exec_score, emotion, mistakes)
                    st.session_state[close_key] = False
                    st.rerun()
            with ok2:
                if st.button("İptal", key=f"canclose_{pid}", use_container_width=True):
                    st.session_state[close_key] = False
                    st.rerun()


def _delete_position(pos_id: int):
    data = st.session_state.data
    data["active_positions"] = [p for p in data["active_positions"] if p["id"] != pos_id]
    save_data(data)
    st.session_state.data = data


def _close_position(pos_id, pnl, comment, exec_score, emotion, mistakes):
    data = st.session_state.data
    pos  = next((p for p in data["active_positions"] if p["id"] == pos_id), None)
    if not pos:
        return
    r_mult  = calculate_r_multiple(pnl, pos.get("risk_calc", {}).get("risk_amount", 0))
    rr      = pos.get("rr")
    closed  = {
        **pos,
        "pnl":             pnl,
        "comment":         comment,
        "execution_score": exec_score,
        "emotion":         emotion,
        "mistakes":        mistakes,
        "r_multiple":      r_mult,
        "closed_at":       datetime.now().isoformat(),
        "result":          "WIN" if pnl > 0 else "LOSS",
        "rr_display":      f"1:{rr}" if rr else "1:?",
    }
    data["active_positions"] = [p for p in data["active_positions"] if p["id"] != pos_id]
    data["closed_trades"].append(closed)
    save_data(data)
    st.session_state.data = data
