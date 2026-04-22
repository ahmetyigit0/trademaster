import streamlit as st
from utils.data_manager import load_data, save_data
from components.position_form import render_position_form
from components.active_positions import render_active_positions
from components.closed_trades import render_closed_trades
from components.stats_bar import render_stats_bar
from components.analytics_dashboard import render_analytics

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TradeVault — Trading Journal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0c10 !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 0%, #0f1520 0%, #0a0c10 60%) !important;
}
#MainMenu, footer, header, [data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
.block-container { padding: 1.5rem 2rem 4rem !important; max-width: 1400px; }

/* Header */
.tv-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1.2rem 0 1.5rem; border-bottom: 1px solid #1e2530; margin-bottom: 1.5rem;
}
.tv-logo { font-family: 'Space Mono', monospace; font-size: 1.5rem; font-weight: 700; letter-spacing: -0.02em; color: #fff; }
.tv-logo span { color: #3b82f6; }
.tv-tagline { font-size: 0.72rem; color: #4a5568; letter-spacing: 0.12em; text-transform: uppercase; margin-top: 0.15rem; }

/* Section titles */
.section-title {
    font-family: 'Space Mono', monospace; font-size: 0.68rem;
    letter-spacing: 0.2em; text-transform: uppercase; color: #4a5568;
    margin: 1.8rem 0 0.8rem; display: flex; align-items: center; gap: 0.6rem;
}
.section-title::after { content: ''; flex: 1; height: 1px; background: #1e2530; }

/* Cards */
.card { background: #0f1520; border: 1px solid #1e2530; border-radius: 12px; padding: 1.25rem; margin-bottom: 0.75rem; transition: border-color 0.2s; }
.card:hover { border-color: #2d3a4e; }
.card-long  { border-left: 3px solid #10b981; }
.card-short { border-left: 3px solid #ef4444; }

/* Detail grid */
.detail-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 0.6rem; margin-top: 1rem; }
.detail-item { background: #131b28; border-radius: 8px; padding: 0.6rem 0.75rem; }
.detail-label { font-size: 0.63rem; color: #4a5568; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.25rem; }
.detail-value { font-family: 'Space Mono', monospace; font-size: 0.85rem; color: #cbd5e1; font-weight: 700; }

/* Stats bar */
.stats-container { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 0.75rem; margin-bottom: 1.5rem; }
.stat-card { background: #0f1520; border: 1px solid #1e2530; border-radius: 10px; padding: 0.9rem 1rem; }
.stat-label { font-size: 0.63rem; color: #4a5568; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.4rem; }
.stat-value { font-family: 'Space Mono', monospace; font-size: 1.1rem; font-weight: 700; color: #fff; }

/* Inputs */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background: #131b28 !important; border: 1px solid #1e2530 !important;
    border-radius: 8px !important; color: #e2e8f0 !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {
    border-color: #3b82f6 !important; box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important;
}

/* Buttons */
.stButton button { border-radius: 8px !important; font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important; transition: all 0.15s !important; }
[data-testid="baseButton-primary"] { background: linear-gradient(135deg, #2563eb, #1d4ed8) !important; border: none !important; color: #fff !important; }

/* Expander */
[data-testid="stExpander"] { background: #0f1520 !important; border: 1px solid #1e2530 !important; border-radius: 12px !important; margin-bottom: 0.75rem !important; }
[data-testid="stExpander"] summary { padding: 0.9rem 1.1rem !important; font-family: 'Space Mono', monospace !important; font-size: 0.8rem !important; color: #e2e8f0 !important; }

/* Alerts */
[data-testid="stInfo"]    { background: #0c1a2e !important; border-color: #1d4ed8 !important; color: #93c5fd !important; border-radius: 8px !important; }
[data-testid="stSuccess"] { background: #052e16 !important; border-color: #15803d !important; color: #86efac !important; border-radius: 8px !important; }
[data-testid="stWarning"] { background: #1c1007 !important; border-color: #b45309 !important; color: #fcd34d !important; border-radius: 8px !important; }
[data-testid="stError"]   { background: #1c0505 !important; border-color: #b91c1c !important; color: #fca5a5 !important; border-radius: 8px !important; }

/* Tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #0f1520 !important; border-radius: 10px !important;
    gap: 0.25rem !important; padding: 0.25rem !important; border: 1px solid #1e2530 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] { background: transparent !important; color: #64748b !important; border-radius: 8px !important; font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important; }
[data-testid="stTabs"] [aria-selected="true"] { background: #1e293b !important; color: #e2e8f0 !important; }

hr { border-color: #1e2530 !important; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0a0c10; }
::-webkit-scrollbar-thumb { background: #1e2530; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2d3a4e; }

@media (max-width: 768px) {
    .block-container { padding: 1rem 1rem 3rem !important; }
    .detail-grid { grid-template-columns: repeat(2, 1fr); }
    .stats-container { grid-template-columns: repeat(2, 1fr); }
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="tv-header">
    <div>
        <div class="tv-logo">Trade<span>Vault</span></div>
        <div class="tv-tagline">Professional Trading Journal & Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── State ─────────────────────────────────────────────────────────────────────
if "data" not in st.session_state:
    st.session_state.data = load_data()
if "show_add_form" not in st.session_state:
    st.session_state.show_add_form = False

# ── Stats bar (always visible) ────────────────────────────────────────────────
render_stats_bar(st.session_state.data)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 Aktif Pozisyonlar", "📁 Kapalı İşlemler", "📊 Analitik", "➕ Yeni Pozisyon"])

with tab1:
    render_active_positions(st.session_state.data)

with tab2:
    render_closed_trades(st.session_state.data)

with tab3:
    render_analytics(st.session_state.data)

with tab4:
    render_position_form()
