import streamlit as st
import pandas as pd
import yfinance as yf
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- KONFİGÜRASYON ---
DATA_FILE = "orion_v4_data.json"
st.set_page_config(page_title="Orion Terminal", layout="wide")

# --- VERİ YÖNETİMİ ---
def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f: return json.load(f)
        except: return []
    return []

def save_data(data):
    with open(DATA_FILE, "w") as f: json.dump(data, f, indent=4)

# --- CSS (ULTRA MODERN DARK UI) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
    * { font-family: 'Plus Jakarta Sans', sans-serif; }
    [data-testid="stAppViewContainer"] { background: #05070a; color: #e6edf3; }
    
    /* Premium Glass Cards */
    .glass-card {
        background: rgba(23, 28, 36, 0.6);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px; padding: 20px; margin-bottom: 15px;
    }
    
    .metric-title { color: #8b949e; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 26px; font-weight: 800; color: #fff; margin-top: 5px; }
    
    /* Buton Grupları */
    .stButton>button { border-radius: 10px; transition: 0.3s; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.5); }
</style>
""", unsafe_allow_html=True)

# --- İLK KURULUM ---
if not os.path.exists(DATA_FILE):
    initial = [
        {"symbol": "NVDA", "name": "Nvidia", "amount": 2.0, "cost": 115.0},
        {"symbol": "BTC-USD", "name": "Bitcoin", "amount": 0.02, "cost": 62000.0},
        {"symbol": "THYAO.IS", "name": "THY", "amount": 50.0, "cost": 290.0}
    ]
    save_data(initial)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='color:#3b82f6;'>🛡️ KONSEY</h2>", unsafe_allow_html=True)
    with st.expander("➕ Varlık Ekle", expanded=False):
        q = st.text_input("Arama")
        if q:
            res = yf.Search(q, max_results=3).quotes
            if res:
                opt = {f"{r.get('shortname','')} ({r['symbol']})": r['symbol'] for r in res}
                sel = st.selectbox("Seç", list(opt.keys()))
                with st.form("add_form"):
                    a, c = st.number_input("Adet"), st.number_input("Maliyet")
                    if st.form_submit_button("Sisteme İşle"):
                        p = load_data(); p.append({"symbol": opt[sel], "name": sel.split(' (')[0], "amount": a, "cost": c})
                        save_data(p); st.rerun()
    usd_try = st.number_input("💵 USD/TRY", value=33.20)

# --- ANALİZ & HESAPLAMA ---
portfolio = load_data()
processed = []
if portfolio:
    with st.spinner('Analizörler çalışıyor...'):
        for i, item in enumerate(portfolio):
            try:
                t = yf.Ticker(item['symbol'])
                hist = t.history(period="1d")
                curr_p = hist['Close'].iloc[-1]
                item.update({"price": curr_p, "total": curr_p * item['amount'], "pnl": (curr_p - item['cost']) * item['amount']})
                item['pnl_perc'] = ((curr_p - item['cost']) / item['cost'] * 100) if item['cost'] > 0 else 0
                item['id'] = i
                processed.append(item)
            except: continue

df = pd.DataFrame(processed)

# --- ANA EKRAN ---
st.markdown("<h1 style='color: #fff; font-weight:800; letter-spacing:-1px;'>ORION <span style='color:#3b82f6;'>TERMINAL</span></h1>", unsafe_allow_html=True)

# 1. TOP METRICS
if not df.empty:
    t_val = df['total'].sum()
    t_pnl = df['pnl'].sum()
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="glass-card"><div class="metric-title">Toplam Portföy</div><div class="metric-value">{t_val:,.2f} $</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="glass-card"><div class="metric-title">Net Kâr/Zarar</div><div class="metric-value" style="color:#10b981;">{t_pnl:+,.2f} $</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="glass-card"><div class="metric-title">Piyasa Durumu</div><div class="metric-value" style="color:#3b82f6; font-size:20px;">BOĞA DOMİNASYONU</div></div>', unsafe_allow_html=True)

    # 2. PERFORMANS GRAFİĞİ (Aylık Durum)
    st.markdown("### 📈 Aylık Performans Projeksiyonu")
    # Örnek bir aylık büyüme grafiği simüle ediyoruz (Gerçek veri için tarihli kayıt gerekir)
    chart_data = pd.DataFrame({
        'Gün': [f"G-{i}" for i in range(30, 0, -1)],
        'Değer': [t_val * (1 + (i*0.002)) for i in range(30)]
    })
    fig_perf = px.line(chart_data, x='Gün', y='Değer', template="plotly_dark")
    fig_perf.update_traces(line_color='#3b82f6', line_width=4, fill='tozeroy')
    fig_perf.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=0,b=0), height=200, xaxis_visible=False)
    st.plotly_chart(fig_perf, use_container_width=True)

    # 3. VARLIK LİSTESİ (Butonlar Yan Yana)
    st.markdown("### 📋 Varlık Detayları & Yönetim")
    for i, row in df.iterrows():
        with st.container():
            # Ana Bilgi Satırı
            col_main, col_btns = st.columns([4, 1])
            
            with col_main:
                color = "#10b981" if row['pnl'] >= 0 else "#ef4444"
                st.markdown(f"""
                <div class="glass-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span style="font-size:18px; font-weight:800;">{row['name']}</span> 
                            <span style="color:#8b949e; font-size:12px;">{row['symbol']}</span><br>
                            <span style="color:#3b82f6; font-size:13px; font-weight:600;">Anlık: {row['price']:,.2f} $</span>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-size:18px; font-weight:800;">{row['total']:,.2f} $</div>
                            <div style="color:{color}; font-size:14px; font-weight:700;">%{row['pnl_perc']:+.2f}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Düzenle ve Sil Butonları Yan Yana
            with col_btns:
                st.write("") # Boşluk
                b1, b2 = st.columns(2)
                if b1.button("📝", key=f"edit_{i}"): st.session_state[f"ed_{i}"] = True
                if b2.button("🗑️", key=f"del_{i}"): st.session_state[f"cf_{i}"] = True

            # DÜZENLEME FORMU
            if st.session_state.get(f"ed_{i}", False):
                with st.form(f"f_ed_{i}"):
                    n_a = st.number_input("Adet", value=row['amount'])
                    n_c = st.number_input("Maliyet", value=row['cost'])
                    c_ev, c_ipt = st.columns(2)
                    if c_ev.form_submit_button("✅"):
                        portfolio[i].update({"amount": n_a, "cost": n_c}); save_data(portfolio)
                        st.session_state[f"ed_{i}"] = False; st.rerun()
                    if c_ipt.form_submit_button("❌"):
                        st.session_state[f"ed_{i}"] = False; st.rerun()

            # SİLME ONAYI
            if st.session_state.get(f"cf_{i}", False):
                st.error(f"Emin misin?")
                o1, o2 = st.columns(2)
                if o1.button("Evet, SİL", key=f"y_{i}"):
                    portfolio.pop(i); save_data(portfolio)
                    del st.session_state[f"cf_{i}"]; st.rerun()
                if o2.button("İptal", key=f"n_{i}"):
                    del st.session_state[f"cf_{i}"]; st.rerun()

    # 4. DAĞILIM
    st.markdown("### 📊 Portföy Kompozisyonu")
    fig_pie = px.pie(df, values='total', names='name', hole=0.8, color_discrete_sequence=px.colors.qualitative.Prism)
    fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white", showlegend=False, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig_pie, use_container_width=True)

else:
    st.info("Konsey henüz toplanmadı. Sidebar'dan varlık ekle.")
