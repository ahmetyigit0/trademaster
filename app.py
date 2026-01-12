import streamlit as st
import pandas as pd
import plotly.express as px
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
    }
    .asset-name { font-size: 20px; font-weight: 700; color: #f8fafc; }
    .pnl-pos { color: #22c55e; font-weight: bold; }
    .pnl-neg { color: #ef4444; font-weight: bold; }
    .metric-box { text-align: center; padding: 10px; background: #1e293b; border-radius: 10px; flex: 1; }
</style>
""", unsafe_allow_html=True)

# -------------------- VERİ YÖNETİMİ --------------------
def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                data = json.load(f)
                # Eksik anahtarları kontrol et (Hata almamak için kritik)
                for item in data.get("portfolio", []):
                    if "cost" not in item:
                        item["cost"] = item.get("cost_basis", item.get("price", 0))
                return data
        except:
            pass
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

# -------------------- BAŞLATMA --------------------
data = load_data()
df = pd.DataFrame(data["portfolio"])

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.title("🚀 TradeMaster")
    usdtry = st.number_input("USD / TRY", value=float(data["settings"].get("usdtry", 32.5)), step=0.1)
    target = st.number_input("Hedef Portföy (₺)", value=int(data["settings"].get("target", 5000000)))
    
    st.divider()
    st.subheader("➕ Varlık Ekle")
    with st.form("add_form", clear_on_submit=True):
        name = st.text_input("Sembol (BTC, THY vb.)")
        cat = st.selectbox("Kategori", ["Kripto", "Hisse", "Emtia", "Döviz"])
        amt = st.number_input("Miktar", min_value=0.0, format="%.4f")
        cst = st.number_input("Maliyet (Birim Alış)", min_value=0.0)
        prc = st.number_input("Güncel Fiyat", min_value=0.0)
        curr = st.radio("Birim", ["USD", "TRY"], horizontal=True)
        if st.form_submit_button("Ekle") and name:
            data["portfolio"].append({
                "asset": name, "category": cat, "amount": amt, 
                "cost": cst, "price": prc, "currency": curr
            })
            save_data(data)
            st.rerun()

# -------------------- HESAPLAMALAR --------------------
if not df.empty:
    # Sütun isimlerinin varlığından emin olalım
    if 'cost' not in df.columns:
        df['cost'] = 0
        
    df['multiplier'] = df['currency'].apply(lambda x: usdtry if x == "USD" else 1)
    df['current_val_try'] = df['amount'] * df['price'] * df['multiplier']
    df['total_cost_try'] = df['amount'] * df['cost'] * df['multiplier']
    df['pnl_try'] = df['current_val_try'] - df['total_cost_try']
    
    # Sıfıra bölünme hatasını engelle
    df['pnl_perc'] = df.apply(lambda r: (r['pnl_try'] / r['total_cost_try'] * 100) if r['total_cost_try'] > 0 else 0, axis=1)
    
    total_val = df['current_val_try'].sum()
    total_pnl = df['pnl_try'].sum()
    update_history(total_val)
else:
    total_val, total_pnl = 0, 0

# -------------------- ANA PANEL --------------------
st.title("📊 Portföy Dashboard")

c1, c2, c3 = st.columns(3)
c1.metric("Toplam Varlık", f"{total_val:,.0f} ₺")
pnl_perc_total = (total_pnl / (total_val - total_pnl) * 100) if (total_val - total_pnl) > 0 else 0
c2.metric("Toplam Kâr/Zarar", f"{total_pnl:,.0f} ₺", f"{pnl_perc_total:.1f}%")
c3.metric("Hedef İlerleme", f"%{(total_val/target*100):.1f}")

st.progress(min(total_val/target, 1.0))

# Grafikler
if not df.empty:
    g1, g2 = st.columns(2)
    with g1:
        fig_pie = px.pie(df, values='current_val_try', names='category', hole=0.4, title="Dağılım")
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white", showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)
    with g2:
        if os.path.exists(HISTORY_FILE):
            hist_df = pd.read_csv(HISTORY_FILE)
            fig_line = px.line(hist_df, x="tarih", y="deger", title="Performans")
            fig_line.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white")
            st.plotly_chart(fig_line, use_container_width=True)

# Kartlar
st.subheader("📋 Detaylar")
for i, row in df.iterrows():
    pnl_class = "pnl-pos" if row['pnl_try'] >= 0 else "pnl-neg"
    st.markdown(f"""
    <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span class="asset-name">{row['asset']}</span> 
                <span style="color:#64748b; font-size:12px;">{row['category']}</span>
            </div>
            <div class="{pnl_class}">{row['pnl_try']:,.0f} ₺ ({row['pnl_perc']:+.1f}%)</div>
        </div>
        <div style="display: flex; gap: 15px; margin-top:15px; flex-wrap: wrap;">
            <div class="metric-box"><div style="font-size:11px; color:#94a3b8">Miktar</div><div>{row['amount']}</div></div>
            <div class="metric-box"><div style="font-size:11px; color:#94a3b8">Maliyet</div><div>{row['cost']:,.2f}</div></div>
            <div class="metric-box"><div style="font-size:11px; color:#94a3b8">Fiyat</div><div>{row['price']:,.2f}</div></div>
            <div class="metric-box"><div style="font-size:11px; color:#94a3b8">Değer</div><div>{row['current_val_try']:,.0f} ₺</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button(f"🗑️ {row['asset']} Sil", key=f"del_{i}"):
        data["portfolio"].pop(i)
        save_data(data)
        st.rerun()
