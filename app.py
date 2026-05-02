import streamlit as st
from utils.data_manager import load_data, save_data
from components.position_form import render_position_form
from components.active_positions import render_active_positions
from components.closed_trades import render_closed_trades
from components.stats_bar import render_stats_bar
from components.analytics_dashboard import render_analytics
from components.backup import render_backup
from components.pnl_chart import render_pnl_chart
from components.trade_rules import render_trade_rules
from components.arge_lab import render_arge_lab
from components.sniper import render_sniper

st.set_page_config(
    page_title="TradeVault",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

*, *::before, *::after { box-sizing: border-box; }

/* ══ BASE — her şey büyük ve net ══ */
html { font-size: 16px !important; }

body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main, .stApp {
    background: #0d1117 !important;
    color: #e6edf3 !important;              /* daha parlak beyaz */
    font-family: 'DM Sans', sans-serif !important;
    font-size: 16px !important;
    line-height: 1.6 !important;
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display:none !important; visibility:hidden !important; }

.block-container { padding:1.25rem 1.5rem 4rem !important; max-width:1400px !important; }

/* ══ TÜM METİNLER — büyük ve okunabilir ══ */
p, li, td, th { font-size:16px !important; color:#e6edf3 !important; }
span          { font-size:inherit; color:inherit; }

/* Widget label'ları — belirgin gri yerine açık gri */
label,
[data-testid="stWidgetLabel"] > div,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {
    font-size: 15px !important;
    font-weight: 600 !important;
    color: #b1bac4 !important;       /* eskiden #8b949e, şimdi daha açık */
    margin-bottom: 5px !important;
    letter-spacing: 0.01em !important;
}

/* Metric */
[data-testid="stMetricLabel"] > div  { font-size:14px !important; color:#b1bac4 !important; font-weight:600 !important; }
[data-testid="stMetricValue"] > div  { font-size:24px !important; font-weight:700 !important; color:#f0f6fc !important; }
[data-testid="stCaptionContainer"] p { font-size:14px !important; color:#8b949e !important; }

/* Headings */
h1 { font-size:2rem   !important; color:#f0f6fc !important; }
h2 { font-size:1.6rem !important; color:#f0f6fc !important; }
h3 { font-size:1.3rem !important; color:#f0f6fc !important; }
h4 { font-size:1.15rem !important; color:#f0f6fc !important; }
h5 { font-size:1.05rem !important; color:#e6edf3 !important; }
strong, b { color:#f0f6fc !important; }

/* ── Header ── */
.tv-header { display:flex; align-items:center; justify-content:space-between;
  padding:1rem 0 1.25rem; border-bottom:1px solid #21262d; margin-bottom:1.25rem; }
.tv-logo { font-family:'Space Mono',monospace; font-size:1.55rem; font-weight:700;
  color:#f0f6fc; letter-spacing:-0.02em; }
.tv-logo span { color:#58a6ff; }
.tv-tagline { font-size:0.78rem; color:#6e7681; letter-spacing:0.12em;
  text-transform:uppercase; margin-top:0.15rem; }

/* ── Yeni Pozisyon paneli — mavi çerçeve, sıfır şerit ── */
.new-pos-panel {
    background: #161b22;
    border: 2px solid #1f6feb;
    border-radius: 16px;
    padding: 1.4rem 1.4rem 0.8rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 0 28px rgba(31,111,235,0.14);
}
/* Streamlit markdown h4 öncesindeki otomatik divider'ı gizle */
.new-pos-panel hr,
.new-pos-panel [data-testid="stMarkdownContainer"] hr { display:none !important; }

/* ── Section titles ── */
.section-title { font-family:'Space Mono',monospace; font-size:0.74rem; letter-spacing:0.18em;
  text-transform:uppercase; color:#6e7681; margin:1.6rem 0 0.8rem;
  display:flex; align-items:center; gap:0.6rem; }
.section-title::after { content:''; flex:1; height:1px; background:#21262d; }

/* ── Cards ── */
.card { background:#161b22; border:1px solid #21262d; border-radius:16px;
  padding:1.2rem; margin-bottom:0.65rem; transition:border-color 0.2s; }
.card:hover { border-color:#58a6ff; }

/* ── Detail grid ── */
.detail-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(145px,1fr));
  gap:0.55rem; margin-top:0.8rem; }
.detail-item { background:#0d1117; border:1px solid #21262d;
  border-radius:8px; padding:0.65rem 0.8rem; }
.detail-label { font-size:0.68rem; color:#8b949e; text-transform:uppercase;
  letter-spacing:0.1em; margin-bottom:0.25rem; }
.detail-value { font-family:'Space Mono',monospace; font-size:0.95rem;
  color:#e6edf3; font-weight:700; }

/* ── Stats bar ── */
.stats-container { display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr));
  gap:0.65rem; margin-bottom:1.3rem; }
.stat-card { background:#161b22; border:1px solid #21262d;
  border-radius:16px; padding:0.9rem 1rem; transition:border-color 0.2s; }
.stat-card:hover { border-color:#58a6ff; }
.stat-label { font-size:0.72rem; color:#8b949e; text-transform:uppercase;
  letter-spacing:0.1em; margin-bottom:0.35rem; font-weight:600; }
.stat-value { font-family:'Space Mono',monospace; font-size:1.2rem;
  font-weight:700; color:#f0f6fc; }

/* ══ FORM CONTROLS ══ */
input[type="text"], input[type="number"], input[type="email"],
input[type="password"], input[type="search"], textarea,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea {
    background: #161b22 !important;
    border: 1.5px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 16px !important;
    font-weight: 500 !important;
    -webkit-text-fill-color: #e6edf3 !important;
    padding: 0.45rem 0.7rem !important;
}
input:-webkit-autofill, input:-webkit-autofill:hover, input:-webkit-autofill:focus {
    -webkit-box-shadow: 0 0 0 100px #161b22 inset !important;
    -webkit-text-fill-color: #e6edf3 !important;
    border-color: #30363d !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,0.18) !important;
    outline: none !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div,
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: #161b22 !important;
    border: 1.5px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
    font-size: 16px !important;
    font-weight: 500 !important;
}
[data-testid="stSelectbox"] svg { color:#58a6ff !important; fill:#58a6ff !important; }

[data-baseweb="popover"] [role="listbox"],
[data-baseweb="menu"],
[data-baseweb="popover"] ul {
    background:#1c2128 !important; border:1.5px solid #30363d !important; border-radius:10px !important;
}
[data-baseweb="popover"] li,
[data-baseweb="menu"] li {
    background:#1c2128 !important; color:#e6edf3 !important; font-size:16px !important;
}
[data-baseweb="popover"] li:hover,
[data-baseweb="menu"] li:hover { background:#21262d !important; color:#f0f6fc !important; }

/* Multiselect */
[data-testid="stMultiSelect"] > div > div {
    background:#161b22 !important; border:1.5px solid #30363d !important;
    border-radius:8px !important; color:#e6edf3 !important; font-size:16px !important;
}
[data-testid="stMultiSelect"] [data-baseweb="tag"] {
    background:#21262d !important; color:#e6edf3 !important; border-radius:6px !important;
    font-size:14px !important; padding:2px 8px !important;
}

/* Slider */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background:#58a6ff !important; border-color:#58a6ff !important; width:20px !important; height:20px !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [class*="Track"] { background:#21262d !important; height:6px !important; }
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] { font-size:14px !important; color:#8b949e !important; }

/* Checkbox */
[data-testid="stCheckbox"] { margin-bottom:4px !important; }
[data-testid="stCheckbox"] label {
    color:#e6edf3 !important; font-size:16px !important; font-weight:500 !important;
    gap:10px !important; align-items:center !important;
}
[data-testid="stCheckbox"] [data-baseweb="checkbox"] > div:first-child {
    background:#161b22 !important; border:1.5px solid #30363d !important;
    border-radius:5px !important; width:20px !important; height:20px !important;
    box-shadow:none !important;
}
[data-testid="stCheckbox"] [data-baseweb="checkbox"][data-checked="true"] > div:first-child {
    background:#1f6feb !important; border-color:#1f6feb !important;
}
[data-testid="stCheckbox"] [data-baseweb="checkbox"] svg {
    color:#ffffff !important; fill:#ffffff !important;
}

/* Number stepper */
[data-testid="stNumberInput"] button {
    background:#1c2128 !important; border-color:#30363d !important;
    color:#b1bac4 !important; font-size:18px !important;
}
[data-testid="stNumberInput"] button:hover { background:#21262d !important; color:#e6edf3 !important; }

/* File uploader */
[data-testid="stFileUploader"] > div {
    background:#161b22 !important; border:1.5px dashed #30363d !important;
    border-radius:10px !important; color:#8b949e !important;
}
[data-testid="stFileUploader"] button {
    background:#21262d !important; color:#e6edf3 !important;
    border:1.5px solid #30363d !important; border-radius:7px !important; font-size:15px !important;
}

/* ── Buttons ── */
.stButton button {
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    transition: all 0.15s ease !important;
    border: 1.5px solid #30363d !important;
    background: #21262d !important;
    color: #e6edf3 !important;
    padding: 0.5rem 1rem !important;
    letter-spacing: 0.01em !important;
}
.stButton button:hover {
    background: #30363d !important;
    border-color: #58a6ff !important;
    color: #f0f6fc !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 2px 8px rgba(88,166,255,0.15) !important;
}
[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg,#1f6feb,#1158c7) !important;
    border-color: #1f6feb !important;
    color: #ffffff !important;
    font-weight: 700 !important;
}
[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(135deg,#388bfd,#1f6feb) !important;
    border-color: #388bfd !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: #161b22 !important;
    border: 1.5px solid #21262d !important;
    border-radius: 12px !important;
    margin-bottom: 0.65rem !important;
}

/* ── Form Input Alanları — büyük ve okunabilir ── */
div[data-testid="stNumberInput"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stTextArea"] label,
div[data-testid="stSlider"] label,
div[data-testid="stDateInput"] label,
div[data-testid="stTimeInput"] label {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #8b949e !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    font-size: 15px !important;
    background: #0d1117 !important;
    border: 1.5px solid #21262d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stTextInput"] input:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,0.12) !important;
}
[data-testid="stExpander"] summary {
    padding: 0.95rem 1.1rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    color: #e6edf3 !important;
    background: transparent !important;
}
[data-testid="stExpander"] summary:hover { color: #f0f6fc !important; }
[data-testid="stExpander"] [data-testid="stExpanderDetails"] { padding: 0 1.1rem 1rem !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #161b22 !important;
    border-radius: 10px !important;
    gap: 0.2rem !important;
    padding: 0.25rem !important;
    border: 1.5px solid #21262d !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: #8b949e !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 0.55rem 1.1rem !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: #21262d !important;
    color: #f0f6fc !important;
    box-shadow: inset 0 0 0 1px #30363d !important;
}

/* ── Alerts ── */
[data-testid="stInfo"]    { background:#0d2238 !important; border-color:#1f6feb !important; color:#79c0ff !important; border-radius:10px !important; font-size:15px !important; }
[data-testid="stSuccess"] { background:#0a2e1a !important; border-color:#238636 !important; color:#56d364 !important; border-radius:10px !important; font-size:15px !important; }
[data-testid="stWarning"] { background:#2b1d0a !important; border-color:#9e6a03 !important; color:#e3b341 !important; border-radius:10px !important; font-size:15px !important; }
[data-testid="stError"]   { background:#2d0f0f !important; border-color:#da3633 !important; color:#ff7b72 !important; border-radius:10px !important; font-size:15px !important; }

/* Alert metni */
[data-testid="stInfo"] p,
[data-testid="stSuccess"] p,
[data-testid="stWarning"] p,
[data-testid="stError"] p { font-size:15px !important; font-weight:500 !important; }

/* Download */
[data-testid="stDownloadButton"] button {
    background:#21262d !important; border:1.5px solid #30363d !important;
    color:#e6edf3 !important; border-radius:8px !important; font-size:15px !important; font-weight:600 !important;
}
[data-testid="stDownloadButton"] button:hover {
    background:#30363d !important; border-color:#58a6ff !important; color:#f0f6fc !important;
}

hr { border-color:#21262d !important; margin:1rem 0 !important; }

::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:#0d1117; }
::-webkit-scrollbar-thumb { background:#30363d; border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:#484f58; }

@media (max-width:768px) {
    html { font-size:15px !important; }
    .block-container { padding:0.75rem 0.75rem 3rem !important; }
    .detail-grid  { grid-template-columns:repeat(2,1fr) !important; }
    .stats-container { grid-template-columns:repeat(2,1fr) !important; }
    [data-testid="stTabs"] [data-baseweb="tab"] { font-size:12px !important; padding:0.4rem 0.5rem !important; }

    /* Kapalı işlemler — mobilde kart görünümü */
    .ct-table-header { display:none !important; }
    .ct-row {
        display:flex !important;
        flex-direction:column !important;
        background:#161b22;
        border:1px solid #21262d;
        border-radius:10px;
        padding:10px 12px;
        margin-bottom:8px;
    }

    /* Trade yasaları — mobilde tam genişlik */
    div[data-testid="column"] { min-width:0 !important; }

    /* Form — mobilde tek kolon */
    .pos-form-row { flex-direction:column !important; }
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

# ── Şifre Koruması ────────────────────────────────────────────────────────────
_PASSWORD = "efe"

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("<div style='height:15vh'></div>", unsafe_allow_html=True)
    lc, cc, rc = st.columns([1, 1.2, 1])
    with cc:
        st.markdown(
            f"<div style='background:#161b22;border:1px solid #21262d;"
            f"border-radius:16px;padding:2rem 2rem 1.5rem;text-align:center'>"
            f"<div style='font-size:2rem;margin-bottom:8px'>📊</div>"
            f"<div style='font-family:\"Space Mono\",monospace;font-size:1.1rem;"
            f"font-weight:700;color:#f0f6fc;margin-bottom:4px'>TRADEVAULT</div>"
            f"<div style='font-size:13px;color:#6e7681;margin-bottom:1.5rem'>"
            f"Professional Trading Journal</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        pwd = st.text_input("Şifre", type="password", key="pwd_input",
                            placeholder="••••••••", label_visibility="collapsed")
        if st.button("Giriş Yap", type="primary", use_container_width=True, key="login_btn"):
            if pwd == _PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Hatalı şifre")
    st.stop()

if "data" not in st.session_state:
    st.session_state.data = load_data()
if "show_new_pos" not in st.session_state:
    st.session_state.show_new_pos = False

# ── API keys: Streamlit secrets → Render env vars → manuel giriş ────────────
if "bot_api_key" not in st.session_state:
    import os
    # 1) Streamlit Cloud secrets
    try:
        # Streamlit Cloud secrets (okx bölümü)
        st.session_state["bot_api_key"]    = st.secrets["okx"]["api_key"]
        st.session_state["bot_api_secret"] = st.secrets["okx"]["api_secret"]
        st.session_state["bot_passphrase"] = st.secrets["okx"]["passphrase"]
    except Exception:
        # Env vars (Render / Railway)
        key    = os.environ.get("OKX_API_KEY",    "")
        secret = os.environ.get("OKX_API_SECRET", "")
        ppass  = os.environ.get("OKX_PASSPHRASE", "")
        if key and secret:
            st.session_state["bot_api_key"]    = key
            st.session_state["bot_api_secret"] = secret
            st.session_state["bot_passphrase"] = ppass
        # 3) Hiçbiri yoksa kullanıcı TradeBot sekmesinde manuel girer

render_stats_bar(st.session_state.data)

btn_col, _ = st.columns([1, 5])
with btn_col:
    btn_label = "✕ Formu Kapat" if st.session_state.show_new_pos else "＋ Yeni Pozisyon Ekle"
    if st.button(btn_label, type="primary", use_container_width=True, key="main_add_btn"):
        st.session_state.show_new_pos = not st.session_state.show_new_pos
        st.rerun()

if st.session_state.show_new_pos:
    st.markdown('<div class="new-pos-panel">', unsafe_allow_html=True)
    render_position_form()
    st.markdown('</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📈 Aktif Pozisyonlar",
    "📁 Kapalı İşlemler",
    "📉 PnL Grafik",
    "📊 Analitik",
    "🧪 AR-GE LAB",
    "🎯 Sniper",
    "⚖️ Trade Yasaları",
    "💾 Yedek / Arşiv",
])

with tab1: render_active_positions(st.session_state.data)
with tab2: render_closed_trades(st.session_state.data)
with tab3: render_pnl_chart(st.session_state.data)
with tab4: render_analytics(st.session_state.data)
with tab5: render_arge_lab(st.session_state.data)
with tab6: render_sniper()
with tab7: render_trade_rules()
with tab8: render_backup(st.session_state.data)
