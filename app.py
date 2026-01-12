import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="Portföy Yönetimi", layout="wide")

st.title("📊 Kişisel Portföy Yönetimi")
st.caption("Kripto • Hisse • Altın • Fon")

# ---------- AYARLAR ----------
st.sidebar.header("⚙️ Ayarlar")
usdtry = st.sidebar.number_input("USD / TRY", value=32.0, step=0.1)
target = st.sidebar.number_input("🎯 Portföy Hedefi (TRY)", value=5_000_000, step=250_000)

# ---------- DATA ----------
df = pd.read_csv("portfolio.csv")

def value_try(row):
    if row["currency"] == "USD":
        return row["amount"] * row["price"] * usdtry
    return row["amount"] * row["price"]

df["value_try"] = df.apply(value_try, axis=1)

total_value = df["value_try"].sum()

# ---------- METRICS ----------
c1, c2, c3 = st.columns(3)
c1.metric("💰 Toplam Portföy", f"{total_value:,.0f} ₺")
c2.metric("🎯 Hedef", f"{target:,.0f} ₺")
c3.metric("📉 Kalan", f"{target-total_value:,.0f} ₺")

st.progress(min(total_value / target, 1.0))

# ---------- DAĞILIM ----------
st.subheader("📊 Kategori Dağılımı")
cat = df.groupby("category")["value_try"].sum()

fig, ax = plt.subplots()
ax.pie(cat.values, labels=cat.index, autopct="%1.1f%%")
st.pyplot(fig)

# ---------- TABLO ----------
st.subheader("📋 Varlık Detayı")
st.dataframe(
    df.sort_values("value_try", ascending=False),
    use_container_width=True
)

# ---------- KATEGORİ DETAY ----------
st.subheader("🔍 Kategori Bazlı İnceleme")
selected = st.selectbox("Kategori Seç", df["category"].unique())
filtered = df[df["category"] == selected]

st.bar_chart(filtered.set_index("asset")["value_try"])

st.caption(f"Son güncelleme: {datetime.now().strftime('%d.%m.%Y %H:%M')}")