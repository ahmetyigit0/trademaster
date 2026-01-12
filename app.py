import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# Sayfa Ayarları
st.set_page_config(page_title="TradeMaster Pro", layout="wide", page_icon="📈")

# -------------------- STYLE (Geliştirilmiş CSS) --------------------
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
    .asset-name { font-size: 22px; font-weight: 700; color: #f8fafc; }
    .category-tag { font-size: 12px; color: #94a3b8; background: #1e293b; padding: 2px 8px; border-radius: 10px; }
    .metric-label { font-size: 13px; color: #64748b; }
    .metric-value { font-size: 17px; font-weight: 600; color: #f1f5f9; }
    .green-text { color: #22c55e; font-weight: bold; }
    .red-text { color: #ef4444; font-weight: bold; }
    .stProgress > div > div > div > div { background-color: #3b82f6; }
</style>
""", unsafe_allow_html=True)

# -------------------- SESSION STATE (Veri Saklama) --------------------
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = [
        {"asset": "BTC", "category": "Kripto", "amount": 0.15, "cost": 60000, "price": 91700, "currency": "USD"},
        {"asset": "THETA", "category": "Kripto", "amount": 56400, "cost": 1.2, "price": 1.34, "currency": "USD"},
        {"asset": "AAPL", "category": "Hisse", "amount": 15, "cost": 190, "price": 259, "currency": "USD"},
        {"asset": "Altın", "category": "Emtia", "amount": 100, "cost": 2500, "price": 3000, "currency": "TRY"},
    ]

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.title("⚙️ Kontrol Paneli")
    usdtry = st.number_input("USD / TRY Kuru", value=32.5, step=0.1)
    target_try = st.number_input("🎯 Hedef Portföy (₺)", value=5_000_000, step=100000)
    
    st.divider()
    st.subheader("➕ Yeni Varlık Ekle")
    with st.form("add_form", clear_on_submit=True):
        name = st.text_input("Varlık Sembolü (Örn: ETH)")
        cat = st.selectbox("Kategori", ["Kripto", "Hisse", "Emtia", "Döviz"])
        amt = st.number_input("Adet/Miktar", min_value=0.0, format="%.4f")
        cst = st.number_input("Alış Fiyatı (Maliyet)", min_value=0.0)
        prc = st.number_input("Güncel Fiyat", min_value=0.0)
        curr = st.radio("Birim", ["USD", "TRY"], horizontal=True)
        
        if st.form_submit_button("Portföye Ekle"):
            if name:
                new_item = {"asset": name, "category": cat, "amount": amt, "cost": cst, "price": prc, "currency": curr}
                st.session_state.portfolio.append(new_item)
                st.rerun()

# -------------------- HESAPLAMALAR --------------------
df = pd.DataFrame(st.session_state.portfolio)

def process_df(row):
    multiplier = usdtry if row['currency'] == "USD" else 1
    current_val = row['amount'] * row['price'] * multiplier
    cost_val = row['amount'] * row['cost'] * multiplier
    return pd.Series([current_val, cost_val, current_val - cost_val])

df[['total_val', 'total_cost', 'pnl']] = df.apply(process_df, axis=1)
total_portfolio_val = df['total_val'].sum()
df['weight'] = (df['total_val'] / total_portfolio_val) * 100

# -------------------- ANA EKRAN --------------------
st.title("🧠 TradeMaster")
st.caption(f"Son Güncelleme: {datetime.now().strftime('%d.%m.%Y %H:%M')}")

# Metrikler
m1, m2, m3, m4 = st.columns(4)
m1.metric("💰 Toplam Değer", f"{total_portfolio_val:,.0f} ₺")
m2.metric("📈 Toplam Kâr/Zarar", f"{df['pnl'].sum():,.0f} ₺", delta=f"{(df['pnl'].sum()/df['total_cost'].sum())*100:.1f}%")
m3.metric("🎯 Hedef", f"{target_try:,.0f} ₺")
progress = min(total_portfolio_val / target_try, 1.0)
m4.metric("📊 Hedef Oranı", f"%{progress*100:.1f}")
st.progress(progress)

st.divider()

# Grafikler
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📂 Varlık Dağılımı")
    fig_pie = px.pie(df, values='total_val', names='category', hole=0.5,
                     color_discrete_sequence=px.colors.sequential.RdBu)
    fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.subheader("💰 Kâr / Zarar Durumu (₺)")
    fig_bar = px.bar(df, x='asset', y='pnl', color='pnl', 
                     color_continuous_scale=['#ef4444', '#22c55e'])
    fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
    st.plotly_chart(fig_bar, use_container_width=True)

# Portföy Detay Kartları
st.subheader("📋 Varlık Detayları")

for index, row in df.sort_values("total_val", ascending=False).iterrows():
    pnl_style = "green-text" if row['pnl'] >= 0 else "red-text"
    
    st.markdown(f"""
    <div class="card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span class="asset-name">{row['asset']}</span>
                <span class="category-tag">{row['category']}</span>
            </div>
            <div class="{pnl_style}" style="font-size: 20px;">
                {'+' if row['pnl'] > 0 else ''}{row['pnl']:,.2f} ₺
            </div>
        </div>
        <hr style="border: 0.5px solid #1e293b; margin: 15px 0;">
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
            <div><div class="metric-label">Miktar</div><div class="metric-value">{row['amount']}</div></div>
            <div><div class="metric-label">Fiyat</div><div class="metric-value">{row['price']:,.2f} {row['currency']}</div></div>
            <div><div class="metric-label">Toplam Değer</div><div class="metric-value">{row['total_val']:,.0f} ₺</div></div>
            <div><div class="metric-label">Portföy Payı</div><div class="metric-value">%{row['weight']:.1f}</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Silme butonu için küçük bir Streamlit kolonu (HTML içine buton gömülemediği için hemen altına)
    if st.button(f"🗑️ {row['asset']} Sil", key=f"del_{index}"):
        st.session_state.portfolio.pop(index)
        st.rerun()

