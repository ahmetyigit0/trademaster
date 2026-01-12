import streamlit as st
import pandas as pd
import yfinance as yf
import json
import os
import plotly.express as px
from datetime import datetime

# --- KONFİGÜRASYON ---
DATA_FILE = "orion_portfolio.json"
st.set_page_config(page_title="TradeMaster Orion", layout="wide")

# --- VERİ YÖNETİMİ ---
def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                return json.load(f)
        except: return []
    return []

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

# --- İLK KURULUM (İstediğin Varlıkları Başlangıçta Ekleme) ---
if not os.path.exists(DATA_FILE):
    initial_assets = [
        {"symbol": "NVDA", "name": "Nvidia", "amount": 1.0, "cost": 120.0},
        {"symbol": "GOOGL", "name": "Google", "amount": 1.0, "cost": 140.0},
        {"symbol": "MSFT", "name": "Microsoft", "amount": 1.0, "cost": 400.0},
        {"symbol": "AMZN", "name": "Amazon", "amount": 1.0, "cost": 175.0},
        {"symbol": "THYAO.IS", "name": "THY", "amount": 10.0, "cost": 280.0},
        {"symbol": "FROTO.IS", "name": "Ford Otosan", "amount": 5.0, "cost": 900.0},
        {"symbol": "TUPRS.IS", "name": "Tüpraş", "amount": 20.0, "cost": 160.0},
        {"symbol": "GC=F", "name": "Altın", "amount": 1.0, "cost": 2300.0},
        {"symbol": "BTC-USD", "name": "Bitcoin", "amount": 0.01, "cost": 60000.0},
    ]
    save_data(initial_assets)

# --- CSS (MODERN ORION STYLE) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;700&display=swap');
    html, body, [data-testid="stAppViewContainer"] { background-color: #05070a; font-family: 'Plus Jakarta Sans', sans-serif; }
    .asset-card { 
        background: #0d1117; padding: 15px; border-radius: 15px; 
        border: 1px solid #1f2937; margin-bottom: 10px;
    }
    .metric-v { font-size: 20px; font-weight: 800; color: #fff; }
    .stButton>button { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR (YENİ VARLIK EKLEME) ---
with st.sidebar:
    st.title("🛡️ Konsey Paneli")
    with st.expander("➕ Yeni Varlık Ara & Ekle"):
        query = st.text_input("Arama (Örn: Apple, SASA, Ethereum)")
        if query:
            res = yf.Search(query, max_results=3).quotes
            if res:
                opt = {f"{r.get('shortname','')} ({r['symbol']})": r['symbol'] for r in res}
                sel = st.selectbox("Seç:", list(opt.keys()))
                with st.form("add_form"):
                    amt = st.number_input("Adet", min_value=0.0)
                    cst = st.number_input("Maliyet", min_value=0.0)
                    if st.form_submit_button("Sisteme Ekle"):
                        p = load_data()
                        p.append({"symbol": opt[sel], "name": sel.split(' (')[0], "amount": amt, "cost": cst})
                        save_data(p)
                        st.rerun()

# --- ANA EKRAN ---
st.markdown("<h1 style='text-align: center; color: #10b981;'>ORION <span style='color:white;'>DASHBOARD</span></h1>", unsafe_allow_html=True)

portfolio = load_data()

if portfolio:
    processed = []
    with st.spinner('Piyasa verileri senkronize ediliyor...'):
        for item in portfolio:
            try:
                price = yf.Ticker(item['symbol']).history(period="1d")['Close'].iloc[-1]
                item['current_price'] = price
                item['total_val'] = price * item['amount']
                item['pnl'] = (price - item['cost']) * item['amount']
                item['pnl_perc'] = ((price - item['cost']) / item['cost'] * 100) if item['cost'] > 0 else 0
                processed.append(item)
            except: continue

    df = pd.DataFrame(processed)
    
    # --- ÜST ÖZET ---
    t_val = df['total_val'].sum()
    t_pnl = df['pnl'].sum()
    c1, c2 = st.columns(2)
    c1.metric("Toplam Varlık", f"{t_val:,.2f} $", f"{t_pnl:,.2f} $")
    
    # --- DAĞILIM ---
    fig = px.pie(df, values='total_val', names='name', hole=0.7, title="Portföy Dağılımı")
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- VARLIK LİSTESİ ---
    st.subheader("📋 Varlık Yönetimi")
    
    for i, row in df.iterrows():
        with st.container():
            col_info, col_edit, col_del = st.columns([4, 0.5, 0.5])
            
            with col_info:
                color = "#10b981" if row['pnl'] >= 0 else "#ef4444"
                st.markdown(f"""
                <div class="asset-card">
                    <div style="display:flex; justify-content:space-between;">
                        <div><b>{row['name']}</b> ({row['symbol']}) <br> <small>{row['amount']} Adet @ {row['cost']:.2f}</small></div>
                        <div style="text-align:right;">
                            <div class="metric-v">{row['total_val']:,.2f} $</div>
                            <div style="color:{color}; font-weight:bold;">%{row['pnl_perc']:+.2f}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # DÜZENLEME BUTONU (📝)
            with col_edit:
                if st.button("📝", key=f"edit_{i}"):
                    st.session_state[f"edit_mode_{i}"] = True

            # SİLME BUTONU (🗑️)
            with col_del:
                if st.button("🗑️", key=f"del_{i}"):
                    st.session_state[f"confirm_del_{i}"] = True

            # DÜZENLEME FORMU (Popup gibi açılır)
            if st.session_state.get(f"edit_mode_{i}", False):
                with st.form(f"form_edit_{i}"):
                    new_amt = st.number_input("Yeni Adet", value=row['amount'])
                    new_cst = st.number_input("Yeni Maliyet", value=row['cost'])
                    if st.form_submit_button("Güncelle"):
                        portfolio[i]['amount'] = new_amt
                        portfolio[i]['cost'] = new_cst
                        save_data(portfolio)
                        st.session_state[f"edit_mode_{i}"] = False
                        st.rerun()
                    if st.form_submit_button("İptal"):
                        st.session_state[f"edit_mode_{i}"] = False
                        st.rerun()

            # SİLME ONAYI (Emin misiniz?)
            if st.session_state.get(f"confirm_del_{i}", False):
                st.warning(f"{row['name']} silinecek. Emin misin?")
                col_evet, col_hayir = st.columns(2)
                if col_evet.button("✅ Evet, Sil", key=f"yes_{i}"):
                    portfolio.pop(i)
                    save_data(portfolio)
                    del st.session_state[f"confirm_del_{i}"]
                    st.rerun()
                if col_hayir.button("❌ Hayır", key=f"no_{i}"):
                    del st.session_state[f"confirm_del_{i}"]
                    st.rerun()

st.caption(f"TradeMaster Orion v3.0 • Son Güncelleme: {datetime.now().strftime('%H:%M:%S')}")
