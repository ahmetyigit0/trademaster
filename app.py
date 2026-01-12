import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ----------------- SAYFA AYAR -----------------
st.set_page_config(
    page_title="Portföy Yönetimi",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="metric-container"] {
    background-color: #020617;
    border: 1px solid #1e293b;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Kişisel Portföy Yönetimi")
st.caption("Kripto • Hisse • Altın • Nakit")

# ----------------- SIDEBAR -----------------
st.sidebar.header("⚙️ Ayarlar")
usdtry = st.sidebar.number_input("USD / TRY", value=32.0, step=0.1)
target = st.sidebar.number_input("🎯 Portföy Hedefi (₺)", value=5_000_000, step=250_000)

# ----------------- DATA -----------------
df = pd.read_csv("portfolio.csv")

def value_try(row):
    if row["currency"] == "USD":
        return row["amount"] * row["price"] * usdtry
    return row["amount"] * row["price"]

df["value_try"] = df.apply(value_try, axis=1)
df["cost_try"] = df["value_try"] * 0.85  # örnek maliyet varsayımı
df["pnl_try"] = df["value_try"] - df["cost_try"]

total_value = df["value_try"].sum()
df["weight"] = (df["value_try"] / total_value) * 100

# ----------------- METRICS -----------------
c1, c2, c3, c4 = st.columns(4)

c1.metric("💰 Toplam Portföy", f"{total_value:,.0f} ₺")
c2.metric("🎯 Hedef", f"{target:,.0f} ₺")
c3.metric("📉 Kalan", f"{max(target-total_value,0):,.0f} ₺")
c4.metric("📈 İlerleme", f"%{(total_value/target)*100:.1f}")

st.progress(min(total_value / target, 1.0))

# ----------------- DAĞILIM -----------------
st.subheader("📊 Kategori Dağılımı")
cat = df.groupby("category")["value_try"].sum()

fig, ax = plt.subplots()
ax.pie(cat.values, labels=cat.index, autopct="%1.1f%%", startangle=90)
ax.axis("equal")
st.pyplot(fig)

# ----------------- PORTFÖY TABLO -----------------
st.subheader("📋 Portföy Detayı")

styled = df.copy()
styled["Değer (₺)"] = styled["value_try"].map(lambda x: f"{x:,.0f}")
styled["K/Z (₺)"] = styled["pnl_try"].map(lambda x: f"{x:,.0f}")
styled["Ağırlık %"] = styled["weight"].map(lambda x: f"{x:.1f}%")

table = styled[[
    "asset",
    "category",
    "amount",
    "price",
    "Değer (₺)",
    "K/Z (₺)",
    "Ağırlık %"
]].sort_values("Ağırlık %", ascending=False)

def pnl_color(val):
    try:
        val = float(val.replace(",", ""))
        return "color:#22c55e;font-weight:bold" if val >= 0 else "color:#ef4444;font-weight:bold"
    except:
        return ""

st.dataframe(
    table.style
        .applymap(pnl_color, subset=["K/Z (₺)"])
        .set_properties(**{
            "background-color": "#020617",
            "color": "#e5e7eb",
            "border-color": "#1e293b"
        }),
    use_container_width=True,
    height=420
)

# ----------------- KATEGORİ DETAY -----------------
st.subheader("🔍 Kategori Bazlı İnceleme")
selected = st.selectbox("Kategori Seç", df["category"].unique())

filtered = df[df["category"] == selected]
st.bar_chart(filtered.set_index("asset")["value_try"])

# ----------------- FOOTER -----------------
st.caption(f"🕒 Son güncelleme: {datetime.now().strftime('%d.%m.%Y %H:%M')}")