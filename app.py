import streamlit as st
import pandas as pd
import yfinance as yf
import json
import os
import plotly.express as px
from datetime import datetime

# --- KONFİGÜRASYON ---
DATA_FILE = "orion_v7_final.json"
HEDEF_USD = 666000
st.set_page_config(page_title="Orion V7 Terminal", layout="wide")

# --- VERİ YÖNETİMİ ---
def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f: return json.load(f)
        except: return []
    return []

def save_data(data):
    with open(DATA_FILE, "w") as f: json.dump(data, f, indent=4)

# --- CSS (GRID & PREMIUM PROGRESS BAR) ---
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
    * {{ font-family: 'Plus Jakarta Sans', sans-serif; }}
    [data-testid="stAppViewContainer"] {{ background: #05070a; }}
    
    .target-box {{
        background: linear-gradient(90deg, #1e293b, #0f172a);
        padding: 20px; border-radius: 20px; border: 1px solid #3b82f6;
        text-align: center; margin-bottom: 25px;
    }}
    .glass-card {{
        background: rgba(23, 28, 36, 0.9);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 18px; padding: 15px; margin-bottom: 10px;
    }}
    .symbol-badge {{ background: #1e293b; color: #3b82f6; padding: 3px 8px; border-radius: 6px; font-weight: 800; font-size: 11px; }}
</style>
""", unsafe_allow_html=True)

# --- ANALİZ MOTORU ---
def get_category(symbol):
    if "-USD" in symbol: return "Kripto"
    if ".IS" in symbol: return "Hisse"
    if "=F" in symbol: return "Emtia"
    return "Diğer"

# --- TEST VERİSİ ---
if not os.path.exists(DATA_FILE):
    initial_data = [
        {"symbol": "NVDA", "name": "Nvidia", "amount": 10.0, "cost": 105.0, "date": "2024-06-01"},
        {"symbol": "BTC-USD", "name": "Bitcoin", "amount": 0.5, "cost": 55000.0, "date": "2023-12-15"},
        {"symbol": "THYAO.IS", "name": "THY", "amount": 500.0, "cost": 240.0, "date": "2024-02-10"},
        {"symbol": "GC=F", "name": "Altın", "amount": 5.0, "cost": 2100.0, "date": "2023-05-10"}
    ]
    save_data(initial_data)

portfolio = load_data()
usd_try = 33.50 # Manuel kur veya API eklenebilir

if portfolio:
    processed = []
    with st.spinner('Konsey verileri işliyor...'):
        for item in portfolio:
            try:
                t = yf.Ticker(item['symbol'])
                curr_p = t.history(period="1d")['Close'].iloc[-1]
                item['curr_p'] = curr_p
                item['val_usd'] = curr_p if ".IS" not in item['symbol'] else curr_p / usd_try
                item['total_usd'] = item['val_usd'] * item['amount']
                item['cat'] = get_category(item['symbol'])
                processed.append(item)
            except: continue

    df = pd.DataFrame(processed)
    total_port_usd = df['total_usd'].sum()
    total_port_try = total_port_usd * usd_try
    progress_perc = (total_port_usd / HEDEF_USD) * 100

    # --- ÜST PANEL (HEDEF & METRİKLER) ---
    st.markdown(f"""
    <div class="target-box">
        <small style="color:#8b949e;">🎯 HEDEF: {HEDEF_USD:,.0f} USD</small>
        <div style="font-size:32px; font-weight:800; color:#fff;">${total_port_usd:,.2f} / {total_port_try:,.0f} ₺</div>
        <div style="background:#1e293b; border-radius:10px; height:12px; margin:15px 0; overflow:hidden;">
            <div style="background:#3b82f6; width:{min(progress_perc, 100)}%; height:100%;"></div>
        </div>
        <small style="color:#3b82f6; font-weight:bold;">HEDEFE %{progress_perc:.2f} ULAŞILDI</small>
    </div>
    """, unsafe_allow_html=True)

    # --- KATEGORİ DAĞILIMI (GÖRSELDEKİ ANALİZLERE BENZER) ---
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.markdown("### 📊 Varlık Dağılımı")
        cat_df = df.groupby('cat')['total_usd'].sum().reset_index()
        fig = px.pie(cat_df, values='total_usd', names='cat', hole=0.7, template="plotly_dark")
        fig.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', height=250, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.markdown("### 🧬 Kategorik Bilgi")
        for _, row in cat_df.iterrows():
            perc = (row['total_usd'] / total_port_usd) * 100
            st.write(f"**{row['cat']}:** ${row['total_usd']:,.2f} (%{perc:.1f})")

    st.divider()

    # --- GRID YAPISI (YAN YANA 3 KART) ---
    st.markdown("### 📋 Portföy Detayları")
    # Mobil ve web uyumu için döngüsel kolon yapısı
    for i in range(0, len(processed), 3):
        cols = st.columns(3)
        chunk = processed[i:i+3]
        for idx, item in enumerate(chunk):
            with cols[idx]:
                st.markdown(f"""
                <div class="glass-card">
                    <div style="display:flex; justify-content:space-between; align-items:start;">
                        <span class="symbol-badge">{item['symbol']}</span>
                        <small style="color:#8b949e;">{item['date']}</small>
                    </div>
                    <div style="font-size:22px; font-weight:800; color:#fff; margin:10px 0;">${item['curr_p']:,.2f}</div>
                    <div style="color:#3b82f6; font-weight:600; font-size:14px;">Bakiye: ${item['total_usd']:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Aksiyon Butonları
                b_ed, b_de, b_dt = st.columns(3)
                if b_ed.button("📝", key=f"e_{item['symbol']}"): st.info("Düzenleme Modu")
                if b_de.button("🗑️", key=f"d_{item['symbol']}"): 
                    portfolio.pop(next(i for i, x in enumerate(portfolio) if x['symbol'] == item['symbol']))
                    save_data(portfolio); st.rerun()
                if b_dt.button("🔍", key=f"t_{item['symbol']}"): st.write(f"**Analiz:** RSI Nötr. {item['cat']} trendi izleniyor.")

else:
    st.info("Kanka Konsey boş. Sidebar'dan varlık ekleyerek başla!")

with st.sidebar:
    st.title("🛡️ Konsey Paneli")
    with st.form("add_new"):
        s = st.text_input("Sembol (AAPL, BTC-USD, THYAO.IS)").upper()
        a = st.number_input("Adet", min_value=0.0)
        c = st.number_input("Maliyet", min_value=0.0)
        if st.form_submit_button("Ekle"):
            p = load_data(); p.append({"symbol": s, "name": s, "amount": a, "cost": c, "date": str(datetime.now().date())})
            save_data(p); st.rerun()
