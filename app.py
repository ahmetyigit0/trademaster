import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="TradeMaster", layout="wide")

# --------------------------------------------------
# PREMIUM DARK STYLE
# --------------------------------------------------
st.markdown("""
<style>
body { background-color:#020617; }

.card {
    background:#020617;
    padding:18px 20px;
    border-radius:18px;
    border:1px solid #1e293b;
    box-shadow:0 0 25px rgba(0,0,0,0.45);
    margin-bottom:14px;
}

.asset { font-size:20px;font-weight:600;color:white; }
.cat { font-size:13px;color:#94a3b8; }
.label { font-size:12px;color:#64748b; }
.val { font-size:16px;color:#e5e7eb; }

.green { color:#22c55e;font-weight:600; }
.red { color:#ef4444;font-weight:600; }

.progress-wrap {
    background:#020617;
    border-radius:10px;
    overflow:hidden;
    height:8px;
    margin-top:8px;
}
.progress-bar { height:8px; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# CATEGORY META
# --------------------------------------------------
CATEGORY_META = {
    "Crypto": {"color": "#22c55e", "icon": "🪙"},
    "Stock": {"color": "#38bdf8", "icon": "📈"},
    "Gold": {"color": "#facc15", "icon": "🥇"},
    "Cash": {"color": "#94a3b8", "icon": "💵"},
    "Fund": {"color": "#a78bfa", "icon": "📊"},
}

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("🧠 TradeMaster")
st.caption("Kişisel Portföy Yönetimi • Premium Görünüm")

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

# varsayımsal maliyet (istersen CSV’ye eklersin)
df["cost_try"] = df["value_try"] * 0.85
df["pnl_try"] = df["value_try"] - df["cost_try"]
df["pnl_pct"] = (df["pnl_try"] / df["cost_try"]) * 100

total_value = df["value_try"].sum()
df["weight"] = (df["value_try"] / total_value) * 100

# --------------------------------------------------
# TOP METRICS
# --------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("💰 Toplam Portföy", f"{total_value:,.0f} ₺")
c2.metric("🎯 Hedef", f"{target:,.0f} ₺")
c3.metric("📉 Kalan", f"{max(target-total_value,0):,.0f} ₺")

st.progress(min(total_value / target, 1.0))

# --------------------------------------------------
# DISTRIBUTION
# --------------------------------------------------
st.subheader("📊 Dağılım")

fig = px.pie(
    df,
    values="value_try",
    names="category",
    hole=0.65,
    color_discrete_sequence=px.colors.sequential.Teal
)
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# FILTER
# --------------------------------------------------
st.subheader("📋 Portföy Detayı")
filter_cat = st.selectbox(
    "Kategori Filtresi",
    ["Tümü"] + sorted(df["category"].unique().tolist())
)

view_df = df if filter_cat == "Tümü" else df[df["category"] == filter_cat]

# --------------------------------------------------
# PORTFÖY CARDS
# --------------------------------------------------
for _, r in view_df.sort_values("value_try", ascending=False).iterrows():

    meta = CATEGORY_META.get(r["category"], {"color":"#22c55e","icon":"💼"})
    pnl_class = "green" if r["pnl_try"] >= 0 else "red"

    st.markdown(f"""
    <div class="card">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div>
                <div class="asset">{meta["icon"]} {r['asset']}</div>
                <div class="cat">{r['category']}</div>
            </div>
            <div class="{pnl_class}">
                {r['pnl_try']:,.0f} ₺<br/>
                <span style="font-size:12px;">{r['pnl_pct']:.1f}%</span>
            </div>
        </div>

        <div style="display:flex;justify-content:space-between;margin-top:10px;">
            <div>
                <div class="label">Adet</div>
                <div class="val">{r['amount']}</div>
            </div>
            <div>
                <div class="label">Fiyat</div>
                <div class="val">{r['price']}</div>
            </div>
            <div>
                <div class="label">Değer</div>
                <div class="val">{r['value_try']:,.0f} ₺</div>
            </div>
        </div>

        <div class="progress-wrap">
            <div class="progress-bar" style="width:{r['weight']:.1f}%;background:{meta['color']}"></div>
        </div>
        <div class="label">{r['weight']:.1f}% portföy ağırlığı</div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.caption(f"Son güncelleme: {datetime.now().strftime('%d.%m.%Y %H:%M')}")