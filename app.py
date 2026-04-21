import streamlit as st
import json
import os
from datetime import datetime
from utils import load_data, save_data, calculate_avg_entry, calculate_position_size, calculate_rr
from components import render_new_position_form, render_active_position, render_closed_position

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trading Journal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── INJECT CSS ─────────────────────────────────────────────────────────────────
with open(os.path.join(os.path.dirname(__file__), "style.css")) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ─── SESSION STATE INIT ─────────────────────────────────────────────────────────
if "data" not in st.session_state:
    st.session_state.data = load_data()

if "show_new_form" not in st.session_state:
    st.session_state.show_new_form = False

if "close_modal" not in st.session_state:
    st.session_state.close_modal = None  # holds position id to close

if "expanded_positions" not in st.session_state:
    st.session_state.expanded_positions = set()

if "expanded_closed" not in st.session_state:
    st.session_state.expanded_closed = set()

# ─── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="header-logo">⬡ TRADE<span>VAULT</span></div>
    <div class="header-sub">Professional Trading Journal & Position Calculator</div>
</div>
""", unsafe_allow_html=True)

# ─── STATS BAR ──────────────────────────────────────────────────────────────────
data = st.session_state.data
closed = data.get("closed_trades", [])
active = data.get("active_positions", [])

total_pnl = sum(t.get("pnl", 0) for t in closed)
wins = [t for t in closed if t.get("pnl", 0) > 0]
losses = [t for t in closed if t.get("pnl", 0) <= 0]
win_rate = (len(wins) / len(closed) * 100) if closed else 0

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f"""<div class="stat-card">
        <div class="stat-label">TOTAL P&L</div>
        <div class="stat-value {'positive' if total_pnl >= 0 else 'negative'}">${total_pnl:+.2f}</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="stat-card">
        <div class="stat-label">WIN RATE</div>
        <div class="stat-value">{win_rate:.1f}%</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="stat-card">
        <div class="stat-label">TOTAL TRADES</div>
        <div class="stat-value">{len(closed)}</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="stat-card">
        <div class="stat-label">WINS / LOSSES</div>
        <div class="stat-value">{len(wins)} / {len(losses)}</div>
    </div>""", unsafe_allow_html=True)
with col5:
    st.markdown(f"""<div class="stat-card">
        <div class="stat-label">ACTIVE</div>
        <div class="stat-value accent">{len(active)}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ─── NEW POSITION BUTTON ────────────────────────────────────────────────────────
col_btn, _ = st.columns([1, 4])
with col_btn:
    if st.button("＋  Yeni Pozisyon Ekle", key="open_new_form", use_container_width=True):
        st.session_state.show_new_form = not st.session_state.show_new_form

# ─── NEW POSITION FORM ──────────────────────────────────────────────────────────
if st.session_state.show_new_form:
    render_new_position_form()

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ─── CLOSE POSITION MODAL ──────────────────────────────────────────────────────
if st.session_state.close_modal is not None:
    pos_id = st.session_state.close_modal
    positions = st.session_state.data.get("active_positions", [])
    pos = next((p for p in positions if p["id"] == pos_id), None)
    if pos:
        st.markdown(f"""<div class="modal-overlay">
            <div class="modal-title">🔒 Pozisyonu Kapat — {pos['symbol']} {pos['direction']}</div>
        </div>""", unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="close-modal-box">', unsafe_allow_html=True)
            st.markdown(f"### 🔒 Pozisyonu Kapat — `{pos['symbol']} {pos['direction']}`")
            close_col1, close_col2 = st.columns(2)
            with close_col1:
                pnl_val = st.number_input("PnL ($)", value=0.0, step=0.01, key=f"close_pnl_{pos_id}")
            with close_col2:
                comment = st.text_input("Yorum / Not", key=f"close_comment_{pos_id}")

            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("✅ Kapat ve Kaydet", key=f"confirm_close_{pos_id}", use_container_width=True):
                    # move to closed
                    rr = calculate_rr(pos, pnl_val)
                    closed_trade = {**pos, "pnl": pnl_val, "comment": comment,
                                    "rr": rr, "closed_at": datetime.now().strftime("%Y-%m-%d %H:%M")}
                    st.session_state.data["closed_trades"].append(closed_trade)
                    st.session_state.data["active_positions"] = [
                        p for p in st.session_state.data["active_positions"] if p["id"] != pos_id
                    ]
                    save_data(st.session_state.data)
                    st.session_state.close_modal = None
                    st.rerun()
            with btn_col2:
                if st.button("✖ İptal", key=f"cancel_close_{pos_id}", use_container_width=True):
                    st.session_state.close_modal = None
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

# ─── ACTIVE POSITIONS ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header"><span class="section-dot active-dot"></span>AKTİF POZİSYONLAR</div>', unsafe_allow_html=True)

active_positions = st.session_state.data.get("active_positions", [])

if not active_positions:
    st.markdown('<div class="empty-state">Henüz aktif pozisyon yok. Yeni pozisyon ekleyerek başlayın.</div>', unsafe_allow_html=True)
else:
    for i, pos in enumerate(active_positions):
        render_active_position(pos, i)

st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

# ─── CLOSED TRADES ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header"><span class="section-dot closed-dot"></span>KAPALI İŞLEMLER</div>', unsafe_allow_html=True)

# Filters
filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 4])
with filter_col1:
    dir_filter = st.selectbox("Yön", ["Tümü", "LONG", "SHORT"], key="dir_filter")
with filter_col2:
    result_filter = st.selectbox("Sonuç", ["Tümü", "WIN", "LOSS"], key="result_filter")

closed_trades = st.session_state.data.get("closed_trades", [])

# Apply filters
filtered = closed_trades
if dir_filter != "Tümü":
    filtered = [t for t in filtered if t.get("direction") == dir_filter]
if result_filter == "WIN":
    filtered = [t for t in filtered if t.get("pnl", 0) > 0]
elif result_filter == "LOSS":
    filtered = [t for t in filtered if t.get("pnl", 0) <= 0]

if not filtered:
    st.markdown('<div class="empty-state">Filtreyle eşleşen kapalı işlem bulunamadı.</div>', unsafe_allow_html=True)
else:
    for i, trade in enumerate(reversed(filtered)):
        render_closed_position(trade, i)
