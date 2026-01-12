import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os

# -------------------- KONFİGÜRASYON --------------------
st.set_page_config(page_title="TradeMaster Pro", page_icon="📈", layout="wide")

DATA_FILE = "portfolio_data.json"
HISTORY_FILE = "portfolio_history.csv"

# -------------------- STİLLER --------------------
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #020617; }
    .card {
        background: #0f172a;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #1e293b;
        margin-bottom: 15px;
        transition: transform 0.2s;
    }
    .card:hover { transform: translateY(-3px); border-color: #3b82f6; }
    .asset-name { font-size: 20px; font-weight: 700; color: #f8fafc; }
    .pnl-pos { color: #22c55e; font-weight: bold; }
    .pnl-neg { color: #ef4444; font-weight: bold; }
    .metric-box { text-align: center; padding: 10px; background: #1e293b; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# -------------------- YARDIMCI FONKSİYONLAR --------------------
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {"portfolio": [], "settings": {"target": 5000000, "usdtry": 32.5}}

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

def update_history(total_val):
    today = datetime.now().strftime("%Y-%m-%d")
    new_entry = pd.DataFrame({"tarih": [today], "deger": [total_val]})
    if os.path.exists(HISTORY_FILE):
        hist_df = pd.read_csv(HISTORY_FILE)
        if hist_df.empty or hist_df.iloc[-1]["tarih"] != today:
            pd.concat([hist_df, new_entry]).to_csv(HISTORY_FILE, index=False)
    else:
        new_entry.to_csv(HISTORY_FILE, index=False)

# -------------------- VERİ HAZIRLIĞI --------------------
data = load_data()
df = pd.DataFrame(data["portfolio"])

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.title("🚀 TradeMaster")
    usdtry = st.number_input("USD / TRY", value=data["settings"].get("usdtry", 32.5), step=0.1)
    target = st.number_input("Hedef Portföy (₺)", value=data["settings"].get("target", 5000000))
    
    st.divider()
    st.subheader("➕ Varlık Ekle")
    with st.form("add_form", clear_on_submit=True):
        name = st.text_input("Sembol (BTC, THY vb.)")
        cat = st.selectbox("Kategori", ["Kripto", "Hisse", "Emtia", "Döviz"])
        amt = st.number_input("Miktar", min_value=0.0, format="%.4f")
        cst = st.number_input("Maliyet", min_value=0.0)
        prc = st.number_input("Güncel Fiyat", min_value=0.0)
        curr = st.radio("Birim", ["USD", "TRY"], horizontal=True)
        if st.form_submit_button("Ekle") and name:
            data["portfolio"].append({"asset": name, "category": cat, "amount": amt, "cost": cst, "price": prc, "currency": curr})
            save_data(data)
            st.rerun()

# -------------------- HESAPLAMALAR --------------------
if not df.empty:
    df['multiplier'] = df['currency'].apply(lambda x: usdtry if x == "USD" else 1)
    df['current_val_try'] = df['amount'] * df['price'] * df['multiplier']
    df['total_cost_try'] = df['amount'] * df['cost'] * df['multiplier']
    df['pnl_try'] = df['current_val_try'] - df['total_cost_try']
    df['pnl_perc'] = (df['pnl_try'] / df['total_cost_try']) * 100
    
    total_val = df['current_val_try'].sum()
    total_pnl = df['pnl_try'].sum()
    update_history(total_val)
else:
    total_val, total_pnl = 0, 0

# -------------------- ANA PANEL --------------------
st.title("📊 Portföy Dashboard")

# Metrikler
c1, c2, c3, c4 = st.columns(4)
c1.metric("Toplam Varlık", f"{total_val:,.0f} ₺")
c2.metric("Toplam Kâr/Zarar", f"{total_pnl:,.0f} ₺", f"{ (total_pnl/(total_val-total_pnl)*100) if total_val != total_pnl else 0:.1f}%")
c3.metric("Hedef İlerleme", f"%{(total_val/target*100):.1f}")
c4.metric("Dolar Kuru", f"{usdtry} ₺")

st.progress(min(total_val/target, 1.0))

# Grafikler
if not df.empty:
    g1, g2 = st.columns(2)
    with g1:
        fig_pie = px.pie(df, values='current_val_try', names='category', hole=0.4, title="Varlık Dağılımı")
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with g2:
        if os.path.exists(HISTORY_FILE):
            hist_df = pd.read_csv(HISTORY_FILE)
            fig_line = px.line(hist_df, x="tarih", y="deger", title="Zaman İçinde Büyüme")
            fig_line.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white")
            st.plotly_chart(fig_line, use_container_width=True)

# Varlık Listesi
st.subheader("📋 Varlık Detayları")
for i, row in df.iterrows():
    pnl_color = "pnl-pos" if row['pnl_try'] >= 0 else "pnl-neg"
    with st.container():
        st.markdown(f"""
        <div class="card">
            <div style="display:flex; justify-content:space-between;">
                <div>
                    <span class="asset-name">{row['asset']}</span> 
                    <span style="color:#64748b; margin-left:10px;">{row['category']}</span>
                </div>
                <div class="{pnl_color}">{row['pnl_try']:,.0f} ₺ ({row['pnl_perc']:+.1f}%)</div>
            </div>
            <div style="display: flex; gap: 40px; margin-top:15px;">
                <div class="metric-box"><div style="font-size:12px; color:#94a3b8">Adet</div><div>{row['amount']}</div></div>
                <div class="metric-box"><div style="font-size:12px; color:#94a3b8">Maliyet</div><div>{row['cost']:,.2f}</div></div>
                <div class="metric-box"><div style="font-size:12px; color:#94a3b8">Fiyat</div><div>{row['price']:,.2f}</div></div>
                <div class="metric-box"><div style="font-size:12px; color:#94a3b8">Değer (₺)</div><div>{row['current_val_try']:,.0f}</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button(f"🗑️ {row['asset']} Sil", key=f"del_{i}"):
            data["portfolio"].pop(i)
            save_data(data)
            st.rerun()

st.caption(f"Veriler yerel 'portfolio_data.json' dosyasına kaydedilmektedir. • {datetime.now().strftime('%H:%M:%S')}")
