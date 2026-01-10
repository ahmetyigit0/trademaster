import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Finansal Net Worth Dashboard",
    layout="wide"
)

st.title("📊 Finansal Durum Dashboard")
st.caption("Kaynak: Kişisel Excel → Otomatik CSV")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data.csv")
df = df.sort_values("period").reset_index(drop=True)

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Ayarlar")

target = st.sidebar.number_input(
    "🎯 Net Varlık Hedefi (TRY)",
    value=3_000_000,
    step=100_000
)

show_raw = st.sidebar.checkbox("Ham veriyi göster", False)

# ---------------- CORE METRICS ----------------
current = df.iloc[-1]["net_worth_try"]
previous = df.iloc[-2]["net_worth_try"] if len(df) > 1 else current
delta = current - previous
delta_pct = (delta / previous * 100) if previous != 0 else 0

c1, c2, c3, c4 = st.columns(4)

c1.metric("💰 Güncel Net", f"{current:,.0f} ₺")
c2.metric("📅 Önceki Ay", f"{previous:,.0f} ₺")
c3.metric(
    "📈 Aylık Değişim",
    f"{delta:,.0f} ₺",
    f"{delta_pct:.2f}%"
)
c4.metric(
    "🎯 Hedefe Kalan",
    f"{target - current:,.0f} ₺"
)

st.progress(min(current / target, 1.0))

# ---------------- TIME SERIES ----------------
st.subheader("📈 Ay Ay Net Varlık")

st.line_chart(
    df.set_index("period")["net_worth_try"],
    height=320
)

# ---------------- MONTHLY CHANGE ----------------
df["Prev"] = df["net_worth_try"].shift(1)
df["Delta"] = df["net_worth_try"] - df["Prev"]
df["Delta_%"] = (df["Delta"] / df["Prev"]) * 100

df["Durum"] = df["Delta"].apply(
    lambda x: "🟢 Artış" if x > 0 else ("🔴 Düşüş" if x < 0 else "—")
)

st.subheader("🟢🔴 Ay Ay Değişim Tablosu")

st.dataframe(
    df[["period", "net_worth_try", "Delta", "Delta_%", "Durum"]],
    use_container_width=True,
    height=320
)

# ---------------- BEST / WORST ----------------
best = df.loc[df["Delta"].idxmax()]
worst = df.loc[df["Delta"].idxmin()]

c5, c6 = st.columns(2)

c5.success(
    f"🏆 En İyi Ay: {best['period']}\n\n"
    f"+{best['Delta']:,.0f} ₺ (%{best['Delta_%']:.1f})"
)

c6.error(
    f"💀 En Kötü Ay: {worst['period']}\n\n"
    f"{worst['Delta']:,.0f} ₺ (%{worst['Delta_%']:.1f})"
)

# ---------------- DRAWDOWN ----------------
st.subheader("📉 Drawdown (Zirveden Düşüş)")

df["Peak"] = df["net_worth_try"].cummax()
df["Drawdown_%"] = (df["net_worth_try"] - df["Peak"]) / df["Peak"] * 100

fig, ax = plt.subplots()
ax.fill_between(
    df["period"],
    df["Drawdown_%"],
    color="red",
    alpha=0.4
)
ax.set_ylabel("Drawdown %")
ax.set_xlabel("Dönem")
plt.xticks(rotation=45)
st.pyplot(fig)

# ---------------- AUTO COMMENT ----------------
st.subheader("🧠 Otomatik Yorum")

if delta > 0:
    st.success(
        f"{df.iloc[-1]['period']} ayında net varlık artışı var. "
        f"Aylık +{delta:,.0f} ₺ (%{delta_pct:.2f}). "
        "Mevcut trend pozitif."
    )
elif delta < 0:
    st.warning(
        f"{df.iloc[-1]['period']} ayında net varlık düşüşü var. "
        f"Aylık {delta:,.0f} ₺ (%{delta_pct:.2f}). "
        "Risk yönetimi ve nakit dengesi gözden geçirilmeli."
    )
else:
    st.info("Bu ay net varlık değişimi yok.")

# ---------------- RAW DATA ----------------
if show_raw:
    st.subheader("🧾 Ham CSV Verisi")
    st.dataframe(df, use_container_width=True)

st.caption(f"Son güncelleme: {datetime.now().strftime('%d.%m.%Y %H:%M')}")