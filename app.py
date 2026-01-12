import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="TradeMaster", layout="wide")

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
}

.asset { font-size:20px; font-weight:600; color:white; }
.cat { font-size:13px; color:#94a3b8; }
.label { font-size:12px; color:#64748b; }
.val { font-size:16px; color:#e5e7eb; }

.green { color:#22c55e; font-weight:600; }
.red { color:#ef4444; font-weight:600; }

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
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.title("🧠 TradeMaster")
st.caption("Kişisel Portföy Yönetimi")

# -------------------- SIDEBAR --------------------
usdtry = st.sidebar.number_input("USD / TRY", value=32.0, step=0.1)
target = st.sidebar.number_input("🎯 Hedef (TRY)", value=5_000_000)

# -------------------- DATA --------------------
data = [
    {"asset":"BTC","category":"Kripto","amount":0.15,"price":91700,"currency":"USD"},
    {"asset":"THETA","category":"Kripto","amount":56400,"price":0.34,"currency":"USD"},
    {"asset":"AAPL","category":"Hisse","amount":15,"price":259,"currency":"USD"},
    {"asset":"Altın (gr)","category":"Altın","amount":100,"price":3000,"currency":"TRY"},
]

df = pd.DataFrame(data)

def calc_value(row):
    return row["amount"] * row["price"] * (usdtry if row["currency"]=="USD" else 1)

df["value_try"] = df.apply(calc_value, axis=1)
df["cost_try"] = df["value_try"] * 0.8
df["pnl_try"] = df["value_try"] - df["cost_try"]

total = df["value_try"].sum()
df["weight"] = (df["value_try"] / total) * 100

# -------------------- METRICS --------------------
c1,c2,c3 = st.columns(3)
c1.metric("💰 Toplam", f"{total:,.0f} ₺")
c2.metric("🎯 Hedef", f"{target:,.0f} ₺")
c3.metric("📉 Kalan", f"{max(target-total,0):,.0f} ₺")
st.progress(min(total/target,1.0))

# -------------------- PIE --------------------
fig = px.pie(df, values="value_try", names="category", hole=0.6)
st.plotly_chart(fig, use_container_width=True)

# -------------------- CARDS --------------------
st.subheader("📋 Portföy Detayı")

for _, r in df.sort_values("value_try", ascending=False).iterrows():
    pnl_class = "green" if r["pnl_try"] >= 0 else "red"

    st.markdown(f"""
    <div class="card">
        <div style="display:flex;justify-content:space-between;">
            <div>
                <div class="asset">{r['asset']}</div>
                <div class="cat">{r['category']}</div>
            </div>
            <div class="{pnl_class}">
                {r['pnl_try']:,.0f} ₺
            </div>
        </div>

        <div style="display:flex;justify-content:space-between;margin-top:12px;">
            <div><div class="label">Adet</div><div class="val">{r['amount']}</div></div>
            <div><div class="label">Fiyat</div><div class="val">{r['price']}</div></div>
            <div><div class="label">Değer</div><div class="val">{r['value_try']:,.0f} ₺</div></div>
        </div>

        <div class="progress-wrap">
            <div class="progress-bar" style="width:{r['weight']:.1f}%"></div>
        </div>
        <div class="label">{r['weight']:.1f}% portföy ağırlığı</div>
    </div>
    """, unsafe_allow_html=True)

st.caption(f"Son güncelleme: {datetime.now().strftime('%d.%m.%Y %H:%M')}")