import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(layout="wide", page_title="Portföy Dashboard")

st.title("📊 Yatırım Portföyü Dashboard")
st.caption("Demo veriler – yapı gerçek portföy mantığında")

# ======================================================
# 🔹 DEMO PORTFÖY VERİLERİ
# ======================================================

portfolio = [
    # --- KRIPTO ---
    {"category": "Kripto", "asset": "THETA", "amount": 56400, "price": 0.45, "currency": "USD"},
    {"category": "Kripto", "asset": "BTC", "amount": 0.15, "price": 95000, "currency": "USD"},
    {"category": "Kripto", "asset": "ETH", "amount": 2.3, "price": 3800, "currency": "USD"},

    # --- HISSE ---
    {"category": "Hisse", "asset": "AAPL", "amount": 25, "price": 195, "currency": "USD"},
    {"category": "Hisse", "asset": "MSFT", "amount": 15, "price": 420, "currency": "USD"},
    {"category": "Hisse", "asset": "TSLA", "amount": 10, "price": 260, "currency": "USD"},

    # --- ALTIN ---
    {"category": "Altın", "asset": "Gram Altın", "amount": 120, "price": 2500, "currency": "TRY"},

    # --- GÜMÜŞ ---
    {"category": "Gümüş", "asset": "Gram Gümüş", "amount": 300, "price": 30, "currency": "TRY"},

    # --- FON ---
    {"category": "Fon", "asset": "BIST 30 Fon", "amount": 1, "price": 250000, "currency": "TRY"},
    {"category": "Fon", "asset": "ABD Teknoloji Fon", "amount": 1, "price": 180000, "currency": "TRY"},
]

USDTRY = 32.0  # demo kur

df = pd.DataFrame(portfolio)

# ======================================================
# 🔹 HESAPLAMALAR
# ======================================================

def to_try(row):
    if row["currency"] == "USD":
        return row["amount"] * row["price"] * USDTRY
    return row["amount"] * row["price"]

df["value_try"] = df.apply(to_try, axis=1)

# ======================================================
# 🔹 SIDEBAR
# ======================================================

st.sidebar.header("⚙️ Ayarlar")

target = st.sidebar.number_input(
    "🎯 Portföy Hedefi (TRY)",
    value=5_000_000,
    step=250_000
)

# ======================================================
# 🔹 ÖZET METRİKLER
# ======================================================

total_value = df["value_try"].sum()

c1, c2, c3 = st.columns(3)
c1.metric("💰 Toplam Portföy", f"{total_value:,.0f} ₺")
c2.metric("🎯 Hedef", f"{target:,.0f} ₺")
c3.metric("📈 Hedefe Kalan", f"{target - total_value:,.0f} ₺")

st.progress(min(total_value / target, 1.0))

# ======================================================
# 🔹 KATEGORİ DAĞILIMI
# ======================================================

st.subheader("📊 Kategori Dağılımı")

cat = df.groupby("category")["value_try"].sum()

fig, ax = plt.subplots()
ax.pie(cat.values, labels=cat.index, autopct="%1.1f%%")
st.pyplot(fig)

# ======================================================
# 🔹 VARLIK TABLOSU
# ======================================================

st.subheader("📋 Varlık Detayı")

st.dataframe(
    df[["category", "asset", "amount", "price", "currency", "value_try"]]
    .sort_values("value_try", ascending=False),
    use_container_width=True
)

# ======================================================
# 🔹 KRIPTO ÖZEL
# ======================================================

st.subheader("🪙 Kripto Özel")

crypto = df[df["category"] == "Kripto"]

st.bar_chart(
    crypto.set_index("asset")["value_try"]
)

# ======================================================
# 🔹 HISSE ÖZEL
# ======================================================

st.subheader("📈 Hisse Özel")

stocks = df[df["category"] == "Hisse"]

st.bar_chart(
    stocks.set_index("asset")["value_try"]
)

# ======================================================
# 🔹 OTOMATİK YORUM
# ======================================================

st.subheader("🧠 Portföy Yorumu")

if cat["Kripto"] / total_value > 0.4:
    st.warning("Kripto ağırlığı yüksek. Volatilite riski var.")
else:
    st.success("Kripto ağırlığı dengeli.")

if cat.get("Altın", 0) + cat.get("Gümüş", 0) > total_value * 0.2:
    st.info("Kıymetli metaller portföyü dengeliyor.")

st.caption(f"Demo Dashboard – {datetime.now().strftime('%d.%m.%Y %H:%M')}")