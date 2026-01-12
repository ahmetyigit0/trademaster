import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="TradeMaster • Portföy Yönetimi",
    layout="wide"
)

# --------------------------------------------------
# STYLE
# --------------------------------------------------
st.markdown("""
<style>
.card {
    background:#020617;
    padding:20px;
    border-radius:18px;
    border-left:6px solid #22c55e;
    box-shadow:0 0 25px rgba(0,0,0,0.5);
}
.card h2 {margin:0;color:white;}
.card h4 {margin:0;color:#94a3b8;}
.subtle {color:#64748b;font-size:14px;}
.signal {
    background:linear-gradient(135deg,#020617,#020617);
    padding:25px;
    border-radius:22px;
    border:1px solid #1e293b;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("🧠 TradeMaster")
st.caption("Kripto • Hisse • Altın • Nakit | Akıllı Portföy Takibi")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("⚙️ Ayarlar")
usdtry = st.sidebar.number_input("USD / TRY", value=32.0, step=0.1)
target = st.sidebar.number_input("🎯 Portföy Hedefi (TRY)", value=5_000_000, step=250_000)

# --------------------------------------------------
# DATA
# --------------------------------------------------
df = pd.read_csv("portfolio.csv")

def value_try(row):
    if row["currency"] == "USD":
        return row["amount"] * row["price"] * usdtry
    return row["amount"] * row["price"]

df["value_try"] = df.apply(value_try, axis=1)
total_value = df["value_try"].sum()

# --------------------------------------------------
# METRIC CARDS
# --------------------------------------------------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="card">
        <h4>💰 Toplam Portföy</h4>
        <h2>{total_value:,.0f} ₺</h2>
        <span class="subtle">Güncel Değer</span>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="card" style="border-left-color:#38bdf8">
        <h4>🎯 Hedef</h4>
        <h2>{target:,.0f} ₺</h2>
        <span class="subtle">Uzun Vade</span>
    </div>
    """, unsafe_allow_html=True)

with c3:
    remaining = target - total_value
    color = "#22c55e" if remaining <= 0 else "#facc15"
    st.markdown(f"""
    <div class="card" style="border-left-color:{color}">
        <h4>📉 Kalan</h4>
        <h2>{remaining:,.0f} ₺</h2>
        <span class="subtle">Hedefe Mesafe</span>
    </div>
    """, unsafe_allow_html=True)

st.progress(min(total_value / target, 1.0))

# --------------------------------------------------
# DISTRIBUTION
# --------------------------------------------------
st.subheader("📊 Varlık Dağılımı")

fig = px.pie(
    df,
    values="value_try",
    names="category",
    hole=0.6,
    color_discrete_sequence=px.colors.sequential.Teal
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# ADVISOR PANEL
# --------------------------------------------------
st.subheader("🧠 Yapay Danışman")

signal = "BİRİKTİR"
confidence = 63

st.markdown(f"""
<div class="signal">
    <h2 style="color:#22c55e">📌 {signal}</h2>
    <p style="color:#94a3b8">
    BTC ana trend yukarı. Altcoinler henüz tam kopmadı.
    Theta güçlü destekten dönmüş görünüyor.
    Swing için kademeli satış, uzun vade için tutma mantıklı.
    </p>
    <progress value="{confidence}" max="100" style="width:100%"></progress>
    <small class="subtle">{confidence}% güven</small>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TABLE
# --------------------------------------------------
st.subheader("📋 Portföy Detayı")
st.dataframe(
    df.sort_values("value_try", ascending=False),
    use_container_width=True
)

# --------------------------------------------------
# CATEGORY DETAIL
# --------------------------------------------------
st.subheader("🔍 Kategori Analizi")
selected = st.selectbox("Kategori Seç", df["category"].unique())
filtered = df[df["category"] == selected]

st.bar_chart(filtered.set_index("asset")["value_try"])

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.caption(f"Son güncelleme: {datetime.now().strftime('%d.%m.%Y %H:%M')}")