import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os

# -------------------- SAYFA YAPILANDIRMA --------------------
st.set_page_config(
    page_title="TradeMaster Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- STYLE --------------------
st.markdown("""
<style>
body { background:#020617; }

.card {
    background:#020617;
    padding:20px;
    border-radius:18px;
    border:1px solid #1e293b;
    box-shadow:0 0 20px rgba(0,0,0,0.5);
    margin-bottom:16px;
    transition: all 0.3s ease;
}
.card:hover {
    transform: translateY(-2px);
    border-color: #3b82f6;
}

.asset { font-size:20px; font-weight:600; color:white; }
.cat { font-size:13px; color:#94a3b8; }
.label { font-size:12px; color:#64748b; }
.val { font-size:16px; color:#e5e7eb; }

.green { color:#22c55e; font-weight:600; }
.red { color:#ef4444; font-weight:600; }
.blue { color:#3b82f6; font-weight:600; }

.progress-wrap {
    background:#020617;
    border-radius:10px;
    height:8px;
    overflow:hidden;
    margin-top:10px;
}
.progress-bar {
    height:8px;
    background:#22c55e;
}

.metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #334155;
}

.header-gradient {
    background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sidebar-header {
    font-size: 14px;
    color: #94a3b8;
    margin-top: 20px;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px;
    font-weight: 600;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
}
</style>
""", unsafe_allow_html=True)

# -------------------- VERİ YÖNETİMİ --------------------
def load_data():
    """Örnek veri veya kayıtlı veriyi yükle"""
    if os.path.exists("portfolio_data.json"):
        with open("portfolio_data.json", "r") as f:
            return json.load(f)
    else:
        return {
            "portfolio": [
                {"asset": "BTC", "category": "Kripto", "amount": 0.15, "price": 91700, "currency": "USD", "cost_basis": 85000},
                {"asset": "THETA", "category": "Kripto", "amount": 56400, "price": 0.34, "currency": "USD", "cost_basis": 0.30},
                {"asset": "AAPL", "category": "Hisse", "amount": 15, "price": 259, "currency": "USD", "cost_basis": 240},
                {"asset": "Altın (gr)", "category": "Altın", "amount": 100, "price": 3000, "currency": "TRY", "cost_basis": 2800},
                {"asset": "BIST30", "category": "ETF", "amount": 500, "price": 120, "currency": "TRY", "cost_basis": 110},
                {"asset": "ETH", "category": "Kripto", "amount": 0.5, "price": 3500, "currency": "USD", "cost_basis": 3200},
            ],
            "settings": {
                "target": 5000000,
                "usdtry": 32.0
            }
        }

def save_data(data):
    """Veriyi kaydet"""
    with open("portfolio_data.json", "w") as f:
        json.dump(data, f, indent=2)

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown('<h1 class="header-gradient">⚙️ Ayarlar</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-header">Döviz Kurları</div>', unsafe_allow_html=True)
    usdtry = st.number_input("USD / TRY", value=32.0, step=0.1, key="usdtry")
    eurtry = st.number_input("EUR / TRY", value=35.0, step=0.1, key="eurtry")
    
    st.markdown('<div class="sidebar-header">Portföy Hedefleri</div>', unsafe_allow_html=True)
    target = st.number_input("🎯 Hedef (TRY)", value=5_000_000, step=100_000, format="%d")
    
    st.markdown('<div class="sidebar-header">Risk Profili</div>', unsafe_allow_html=True)
    risk_profile = st.select_slider(
        "Risk Seviyesi",
        options=["Çok Düşük", "Düşük", "Orta", "Yüksek", "Çok Yüksek"],
        value="Orta"
    )
    
    st.markdown('<div class="sidebar-header">Portföy İşlemleri</div>', unsafe_allow_html=True)
    
    with st.expander("➕ Yeni Varlık Ekle"):
        with st.form("add_asset_form"):
            col1, col2 = st.columns(2)
            with col1:
                new_asset = st.text_input("Varlık")
                new_category = st.selectbox("Kategori", ["Kripto", "Hisse", "Altın", "ETF", "Döviz", "Diğer"])
                new_amount = st.number_input("Miktar", value=1.0, step=0.01)
            with col2:
                new_price = st.number_input("Fiyat", value=100.0, step=0.01)
                new_currency = st.selectbox("Para Birimi", ["TRY", "USD", "EUR"])
                new_cost = st.number_input("Maliyet Fiyatı", value=95.0, step=0.01)
            
            if st.form_submit_button("Ekle"):
                data = load_data()
                data["portfolio"].append({
                    "asset": new_asset,
                    "category": new_category,
                    "amount": new_amount,
                    "price": new_price,
                    "currency": new_currency,
                    "cost_basis": new_cost
                })
                save_data(data)
                st.success("Varlık eklendi!")
                st.rerun()

# -------------------- VERİ İŞLEME --------------------
data = load_data()
df = pd.DataFrame(data["portfolio"])

# Döviz çevrim faktörü
def get_conversion_rate(currency):
    rates = {"USD": usdtry, "EUR": eurtry, "TRY": 1.0}
    return rates.get(currency, 1.0)

df["conversion_rate"] = df["currency"].apply(get_conversion_rate)
df["value_try"] = df["amount"] * df["price"] * df["conversion_rate"]
df["cost_try"] = df["amount"] * df["cost_basis"] * df["conversion_rate"]
df["pnl_try"] = df["value_try"] - df["cost_try"]
df["pnl_percent"] = (df["pnl_try"] / df["cost_try"]) * 100

total = df["value_try"].sum()
total_cost = df["cost_try"].sum()
total_pnl = total - total_cost
total_pnl_percent = (total_pnl / total_cost) * 100 if total_cost > 0 else 0

df["weight"] = (df["value_try"] / total) * 100

# -------------------- HEADER --------------------
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.markdown('<h1 class="header-gradient">🧠 TradeMaster Pro</h1>', unsafe_allow_html=True)
    st.caption("Profesyonel Portföy Yönetim Platformu")
with col2:
    st.metric("💰 Toplam", f"{total:,.0f} ₺", f"{total_pnl_percent:+.1f}%")
with col3:
    st.metric("🎯 Hedef", f"{target:,.0f} ₺", f"{(total/target*100):.1f}%")

# İlerleme çubuğu
progress = min(total / target, 1.0)
st.progress(progress, text=f"Hedefin %{progress*100:.1f}'i tamamlandı")

# -------------------- METRİKLER --------------------
st.markdown("## 📊 Performans Özeti")
cols = st.columns(4)

with cols[0]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Toplam Kar/Zarar</div>
        <div class="val" style="font-size:24px;{'color:#22c55e' if total_pnl >= 0 else 'color:#ef4444'}">
            {total_pnl:+,.0f} ₺
        </div>
        <div class="cat" style="{'color:#22c55e' if total_pnl_percent >= 0 else 'color:#ef4444'}">
            {total_pnl_percent:+.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

with cols[1]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Portföy Çeşitliliği</div>
        <div class="val" style="font-size:24px;color:#3b82f6">
            {df['category'].nunique()}
        </div>
        <div class="cat">farklı kategori</div>
    </div>
    """, unsafe_allow_html=True)

with cols[2]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Varlık Sayısı</div>
        <div class="val" style="font-size:24px;color:#8b5cf6">
            {len(df)}
        </div>
        <div class="cat">toplam varlık</div>
    </div>
    """, unsafe_allow_html=True)

with cols[3]:
    avg_return = df["pnl_percent"].mean()
    color = "#22c55e" if avg_return >= 0 else "#ef4444"
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Ortalama Getiri</div>
        <div class="val" style="font-size:24px;color:{color}">
            {avg_return:+.1f}%
        </div>
        <div class="cat">varlık başına</div>
    </div>
    """, unsafe_allow_html=True)

# -------------------- GRAFİKLER --------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Dağılım")
    fig1 = px.pie(df, values="value_try", names="category", hole=0.5,
                 color_discrete_sequence=px.colors.sequential.Plasma)
    fig1.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown("### 📈 Performans")
    fig2 = go.Figure(data=[
        go.Bar(name='Maliyet', x=df['asset'], y=df['cost_try'], marker_color='#64748b'),
        go.Bar(name='Güncel Değer', x=df['asset'], y=df['value_try'], marker_color='#3b82f6')
    ])
    fig2.update_layout(barmode='group', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig2, use_container_width=True)

# -------------------- PORTFÖY DETAYI --------------------
st.markdown("## 📋 Portföy Detayı")

# Filtreleme seçenekleri
col1, col2, col3 = st.columns(3)
with col1:
    sort_by = st.selectbox("Sırala", ["Değere Göre", "Kara Göre", "A-Z"])
with col2:
    filter_category = st.multiselect("Kategori Filtresi", options=df['category'].unique(), default=df['category'].unique())
with col3:
    show_only_profitable = st.checkbox("Sadece Karlılar")

# Filtreleme
filtered_df = df[df['category'].isin(filter_category)]
if show_only_profitable:
    filtered_df = filtered_df[filtered_df['pnl_try'] > 0]

# Sıralama
if sort_by == "Değere Göre":
    filtered_df = filtered_df.sort_values("value_try", ascending=False)
elif sort_by == "Kara Göre":
    filtered_df = filtered_df.sort_values("pnl_try", ascending=False)
else:
    filtered_df = filtered_df.sort_values("asset")

# Kartları göster
for _, r in filtered_df.iterrows():
    pnl_class = "green" if r["pnl_try"] >= 0 else "red"
    pnl_sign = "+" if r["pnl_try"] >= 0 else ""
    
    st.markdown(f"""
    <div class="card">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div style="flex:1;">
                <div class="asset">{r['asset']} <span class="cat">({r['currency']})</span></div>
                <div class="cat">{r['category']}</div>
            </div>
            <div style="flex:1;text-align:center;">
                <div class="label">Kar/Zarar</div>
                <div class="{pnl_class}" style="font-size:18px;">
                    {pnl_sign}{r['pnl_try']:,.0f} ₺ <span style="font-size:14px;">({r['pnl_percent']:+.1f}%)</span>
                </div>
            </div>
            <div style="flex:1;text-align:right;">
                <div class="label">Güncel Değer</div>
                <div class="val" style="font-size:18px;">{r['value_try']:,.0f} ₺</div>
            </div>
        </div>

        <div style="display:flex;justify-content:space-between;margin-top:16px;">
            <div><div class="label">Adet</div><div class="val">{r['amount']:,.2f}</div></div>
            <div><div class="label">Fiyat</div><div class="val">{r['price']:,.2f} {r['currency']}</div></div>
            <div><div class="label">Maliyet</div><div class="val">{r['cost_basis']:,.2f} {r['currency']}</div></div>
            <div><div class="label">Döviz Kuru</div><div class="val">1 {r['currency']} = {r['conversion_rate']:,.2f} ₺</div></div>
        </div>

        <div class="progress-wrap">
            <div class="progress-bar" style="width:{r['weight']:.1f}%"></div>
        </div>
        <div style="display:flex;justify-content:space-between;margin-top:8px;">
            <div class="label">Portföy Ağırlığı: {r['weight']:.1f}%</div>
            <div class="label">Maliyet: {r['cost_try']:,.0f} ₺</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -------------------- ÖNERİLER --------------------
st.markdown("## 💡 Öneriler")

# Basit öneri algoritması
recommendations = []
if df['weight'].max() > 40:
    max_asset = df.loc[df['weight'].idxmax(), 'asset']
    recommendations.append(f"**{max_asset}** varlığı portföyün %{df['weight'].max():.1f}'ini oluşturuyor. Çeşitlendirmeyi artırmayı düşünebilirsiniz.")

if df[df['pnl_percent'] < -10].shape[0] > 0:
    loss_assets = df[df['pnl_percent'] < -10]['asset'].tolist()
    recommendations.append(f"**{', '.join(loss_assets)}** varlıkları %10'dan fazla zararda. Stop-loss seviyelerinizi gözden geçirin.")

if total_pnl_percent > 20:
    recommendations.append("Portföyünüz %20'den fazla kar etti. Kâr almayı düşünebilirsiniz.")

if len(recommendations) == 0:
    recommendations.append("Portföy dengeli görünüyor. Mevcut stratejinize devam edebilirsiniz.")

for i, rec in enumerate(recommendations, 1):
    st.info(f"{i}. {rec}")

# -------------------- FOOTER --------------------
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.caption(f"📅 Son güncelleme: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
with col2:
    st.caption(f"⚖️ Risk profili: {risk_profile}")

# Veriyi otomatik kaydet
data_to_save = {
    "portfolio": df.drop(columns=['conversion_rate', 'value_try', 'cost_try', 'pnl_try', 'pnl_percent', 'weight']).to_dict('records'),
    "settings": {
        "target": target,
        "usdtry": usdtry,
        "eurtry": eurtry,
        "risk_profile": risk_profile
    }
}
save_data(data_to_save)