import streamlit as st
from utils.data_manager import load_data, save_data
from components.position_form import render_position_form
from components.active_positions import render_active_positions
from components.closed_trades import render_closed_trades
from components.stats_bar import render_stats_bar
from components.analytics_dashboard import render_analytics
from components.backup import render_backup

st.set_page_config(
    page_title="TradeVault",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main, .stApp {
    background: #0d1117 !important;
    color: #c9d1d9 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Hide chrome ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; visibility: hidden !important; }

.block-container { padding: 1.25rem 1.5rem 4rem !important; max-width: 1400px !important; }

/* ── Header ── */
.tv-header { display:flex; align-items:center; justify-content:space-between;
  padding:1rem 0 1.25rem; border-bottom:1px solid #21262d; margin-bottom:1.25rem; }
.tv-logo { font-family:'Space Mono',monospace; font-size:1.4rem; font-weight:700; color:#f0f6fc; letter-spacing:-0.02em; }
.tv-logo span { color:#58a6ff; }
.tv-tagline { font-size:0.68rem; color:#484f58; letter-spacing:0.12em; text-transform:uppercase; margin-top:0.15rem; }

/* ── Section titles ── */
.section-title { font-family:'Space Mono',monospace; font-size:0.65rem; letter-spacing:0.18em;
  text-transform:uppercase; color:#484f58; margin:1.6rem 0 0.7rem;
  display:flex; align-items:center; gap:0.6rem; }
.section-title::after { content:''; flex:1; height:1px; background:#21262d; }

/* ── Cards ── */
.card { background:#161b22; border:1px solid #21262d; border-radius:10px;
  padding:1.1rem; margin-bottom:0.6rem; transition:border-color 0.2s; }
.card:hover { border-color:#30363d; }

/* ── Detail grid ── */
.detail-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(130px,1fr));
  gap:0.5rem; margin-top:0.75rem; }
.detail-item { background:#0d1117; border:1px solid #21262d; border-radius:7px; padding:0.55rem 0.7rem; }
.detail-label { font-size:0.6rem; color:#484f58; text-transform:uppercase;
  letter-spacing:0.1em; margin-bottom:0.2rem; }
.detail-value { font-family:'Space Mono',monospace; font-size:0.82rem; color:#c9d1d9; font-weight:700; }

/* ── Stats ── */
.stats-container { display:grid; grid-template-columns:repeat(auto-fill,minmax(145px,1fr));
  gap:0.6rem; margin-bottom:1.25rem; }
.stat-card { background:#161b22; border:1px solid #21262d; border-radius:9px; padding:0.85rem 0.95rem; }
.stat-label { font-size:0.6rem; color:#484f58; text-transform:uppercase;
  letter-spacing:0.1em; margin-bottom:0.35rem; }
.stat-value { font-family:'Space Mono',monospace; font-size:1.05rem; font-weight:700; color:#f0f6fc; }

/* ══════════════════════════════════════════════════════
   FORM CONTROLS — dark everywhere (mobile fix)
══════════════════════════════════════════════════════ */

/* Text & number inputs */
input[type="text"],
input[type="number"],
input[type="email"],
input[type="password"],
input[type="search"],
textarea,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 7px !important;
    color: #c9d1d9 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    -webkit-text-fill-color: #c9d1d9 !important;
}

/* Autofill override */
input:-webkit-autofill,
input:-webkit-autofill:hover,
input:-webkit-autofill:focus {
    -webkit-box-shadow: 0 0 0 100px #0d1117 inset !important;
    -webkit-text-fill-color: #c9d1d9 !important;
    border-color: #30363d !important;
}

/* Focus ring */
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 2px rgba(88,166,255,0.15) !important;
    outline: none !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div,
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 7px !important;
    color: #c9d1d9 !important;
}
[data-testid="stSelectbox"] svg { color: #58a6ff !important; fill: #58a6ff !important; }

/* Selectbox dropdown list */
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="menu"],
[data-baseweb="popover"] ul {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}
[data-baseweb="popover"] li,
[data-baseweb="menu"] li {
    background: #161b22 !important;
    color: #c9d1d9 !important;
}
[data-baseweb="popover"] li:hover,
[data-baseweb="menu"] li:hover {
    background: #21262d !important;
    color: #f0f6fc !important;
}

/* Multiselect */
[data-testid="stMultiSelect"] > div > div {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 7px !important;
    color: #c9d1d9 !important;
}
[data-testid="stMultiSelect"] [data-baseweb="tag"] {
    background: #21262d !important;
    color: #c9d1d9 !important;
    border-radius: 5px !important;
}

/* Slider */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #58a6ff !important;
    border-color: #58a6ff !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [class*="Track"] {
    background: #21262d !important;
}

/* Checkbox */
[data-testid="stCheckbox"] label {
    color: #c9d1d9 !important;
}
[data-testid="stCheckbox"] [data-baseweb="checkbox"] div {
    background: #0d1117 !important;
    border-color: #30363d !important;
    border-radius: 4px !important;
}
[data-testid="stCheckbox"] input:checked + div {
    background: #238636 !important;
    border-color: #238636 !important;
}

/* Number input +/- steppers */
[data-testid="stNumberInput"] button {
    background: #161b22 !important;
    border-color: #30363d !important;
    color: #8b949e !important;
}
[data-testid="stNumberInput"] button:hover {
    background: #21262d !important;
    color: #c9d1d9 !important;
}

/* File uploader */
[data-testid="stFileUploader"] > div {
    background: #161b22 !important;
    border: 1px dashed #30363d !important;
    border-radius: 8px !important;
    color: #8b949e !important;
}
[data-testid="stFileUploader"] button {
    background: #21262d !important;
    color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
}

/* ── Buttons ── */
.stButton button {
    border-radius: 7px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    transition: all 0.15s ease !important;
    border: 1px solid #30363d !important;
    background: #21262d !important;
    color: #c9d1d9 !important;
}
.stButton button:hover {
    background: #30363d !important;
    border-color: #58a6ff !important;
    color: #f0f6fc !important;
    transform: translateY(-1px) !important;
}
/* Primary button */
[data-testid="baseButton-primary"],
[kind="primary"] {
    background: linear-gradient(135deg,#1f6feb,#1158c7) !important;
    border-color: #1f6feb !important;
    color: #ffffff !important;
}
[data-testid="baseButton-primary"]:hover,
[kind="primary"]:hover {
    background: linear-gradient(135deg,#388bfd,#1f6feb) !important;
    border-color: #388bfd !important;
    color: #fff !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 10px !important;
    margin-bottom: 0.6rem !important;
}
[data-testid="stExpander"] summary {
    padding: 0.85rem 1rem !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    color: #c9d1d9 !important;
    background: transparent !important;
}
[data-testid="stExpander"] summary:hover { color: #f0f6fc !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #161b22 !important;
    border-radius: 9px !important;
    gap: 0.2rem !important;
    padding: 0.2rem !important;
    border: 1px solid #21262d !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: #484f58 !important;
    border-radius: 7px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 0.45rem 0.9rem !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: #21262d !important;
    color: #f0f6fc !important;
}

/* ── Alerts ── */
[data-testid="stInfo"]    { background:#0d2238 !important; border-color:#1f6feb !important; color:#79c0ff !important; border-radius:8px !important; }
[data-testid="stSuccess"] { background:#0a2e1a !important; border-color:#238636 !important; color:#56d364 !important; border-radius:8px !important; }
[data-testid="stWarning"] { background:#2b1d0a !important; border-color:#9e6a03 !important; color:#e3b341 !important; border-radius:8px !important; }
[data-testid="stError"]   { background:#2d0f0f !important; border-color:#da3633 !important; color:#ff7b72 !important; border-radius:8px !important; }

/* ── Download button ── */
[data-testid="stDownloadButton"] button {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    border-radius: 7px !important;
}
[data-testid="stDownloadButton"] button:hover {
    background: #30363d !important;
    border-color: #58a6ff !important;
    color: #f0f6fc !important;
}

hr { border-color: #21262d !important; margin: 1rem 0 !important; }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #21262d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #30363d; }

/* ── Mobile ── */
@media (max-width: 768px) {
    .block-container { padding: 0.75rem 0.75rem 3rem !important; }
    .detail-grid { grid-template-columns: repeat(2,1fr) !important; }
    .stats-container { grid-template-columns: repeat(2,1fr) !important; }
    [data-testid="stTabs"] [data-baseweb="tab"] { font-size:0.78rem !important; padding:0.4rem 0.6rem !important; }
    .tv-logo { font-size:1.1rem; }
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

# ── State init ────────────────────────────────────────────────────────────────
if "data" not in st.session_state:
    st.session_state.data = load_data()

# ── Stats bar ─────────────────────────────────────────────────────────────────
render_stats_bar(st.session_state.data)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Aktif",
    "📁 Kapalı",
    "📊 Analitik",
    "➕ Yeni Pozisyon",
    "💾 Yedek / Arşiv",
])

with tab1:
    render_active_positions(st.session_state.data)
with tab2:
    render_closed_trades(st.session_state.data)
with tab3:
    render_analytics(st.session_state.data)
with tab4:
    render_position_form()
with tab5:
    render_backup(st.session_state.data)
