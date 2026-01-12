import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from data import portfolio, USDTRY
from utils import value_in_try
from datetime import datetime

st.set_page_config(layout="wide", page_title="Portföy Yönetimi")

st.title("📊 Portföy Yönetimi Dashboard")
st.caption("Kişisel yatırım kokpiti")

# ---------- DATA ----------
df = pd.DataFrame(portfolio)
df["value_try"] = df.apply(lambda r: value_in_try(r, USDTRY), axis=1)

total_value = df["value_try"].sum()

# ---------- SIDEBAR ----------
st.sidebar.header("⚙️ Ayarlar")

target = st.sidebar.number_input(
    "🎯 Portföy Hedefi (TRY)",
    value=5_000_000,
    step=250_000
)

usdtry_input = st.sidebar.number_input(
    "💱 USD/TRY",
    value=USDTRY,
    step=0.1
)

# ---------- METRICS ----------
c1, c2, c3 = st.columns(3)

c1.metric("💰 Toplam Portföy", f"{total_value:,.0f} ₺")
c2.metric("🎯 Hedef", f"{target:,.0f} ₺")
c3.metric("📉 Kalan", f"{target - total_value:,.0f} ₺")

st.progress(min(total_value / target, 1.0))

# ---------- DISTRIBUTION ----------
st.subheader("📊 Kategori Dağılımı")

cat = df.groupby("category")["value_try"].sum()

fig, ax = plt.subplots()
ax.pie(cat.values, labels=cat.index, autopct="%1.1f%%")
st.pyplot(fig)

# ---------- ASSET TABLE ----------
st.subheader("📋 Varlık Detayı")

st.dataframe(
    df[["category", "asset", "amount", "price", "currency", "value_try"]]
    .sort_values("value_try", ascending=False),
    use_container_width=True
)

# ---------- CATEGORY DETAILS ----------
st.subheader("🔍 Kategori Bazlı Detay")

selected_cat = st.selectbox(
    "Kategori seç",
    df["category"].unique()
)

filtered = df[df["category"] == selected_cat]

st.bar_chart(
    filtered.set_index("asset")["value_try"]
)

st.caption(f"Son güncelleme: {datetime.now().strftime('%d.%m.%Y %H:%M')}")