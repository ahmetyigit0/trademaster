import streamlit as st
import numpy as np
import json
import os

st.set_page_config(layout="wide")

# ------------------ STORAGE ------------------
FILE = "trades.json"

def load_data():
    if os.path.exists(FILE):
        with open(FILE, "r") as f:
            return json.load(f)
    return {"active": [], "closed": []}

def save_data(data):
    with open(FILE, "w") as f:
        json.dump(data, f)

data = load_data()

# ------------------ STYLE ------------------
st.markdown("""
<style>
.card {
    background:#111;
    padding:15px;
    border-radius:12px;
    margin-bottom:10px;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 PRO Trade Journal")

# ------------------ ADD TRADE ------------------
if st.button("➕ Yeni Trade"):
    data["active"].append({
        "symbol":"BTC",
        "side":"LONG",
        "capital":18400,
        "risk":2.0,
        "entries":[78000],
        "weights":[100],
        "stop":76000,
        "tp":[80000],
        "comment":""
    })
    save_data(data)

# ------------------ ACTIVE TRADES ------------------
st.subheader("🟢 Active Trades")

for i, t in enumerate(data["active"]):

    title = f"#{i+1}-{t['symbol']}-{t['side']}"

    with st.expander(title):

        col1, col2 = st.columns(2)

        with col1:
            t["symbol"] = st.text_input("Symbol", t["symbol"], key=f"s{i}")
            t["side"] = st.selectbox("Side", ["LONG","SHORT"], index=0 if t["side"]=="LONG" else 1, key=f"side{i}")

        with col2:
            t["capital"] = st.number_input("Capital", value=t["capital"], key=f"cap{i}")
            t["risk"] = st.number_input("Risk %", value=t["risk"], key=f"risk{i}")

        split = st.checkbox("Parçalı Entry", key=f"split{i}")

        entries = []
        weights = []

        if split:
            cols = st.columns(3)
            for j in range(3):
                with cols[j]:
                    e = st.number_input(f"E{j+1}", value=78000+(j*1000), key=f"e{i}{j}")
                    w = st.number_input(f"W{j+1}", value=30, key=f"w{i}{j}")
                    entries.append(e)
                    weights.append(w)
        else:
            e = st.number_input("Entry", value=78000, key=f"e_single{i}")
            entries=[e]
            weights=[100]

        t["stop"] = st.number_input("Stop", value=t["stop"], key=f"stop{i}")

        # -------- CALCULATION --------
        weights = np.array(weights)
        weights = weights / weights.sum()
        entries = np.array(entries)

        risk_amount = t["capital"] * (t["risk"]/100)
        avg_entry = np.sum(entries * weights)
        distance = abs(t["stop"] - avg_entry)

        btc_size = risk_amount / distance if distance != 0 else 0
        usd_pos = btc_size * avg_entry

        st.markdown(f"💰 **Pozisyon:** ${usd_pos:.2f}")
        st.markdown(f"📍 Avg Entry: {avg_entry:.0f}")

        # -------- TP --------
        tp_split = st.checkbox("Parçalı TP", key=f"tp{i}")

        if tp_split:
            tp_vals = [st.number_input(f"TP{j+1}", key=f"tp{i}{j}") for j in range(2)]
        else:
            tp_vals = [st.number_input("TP", key=f"tp_single{i}")]

        pnl = st.number_input("PnL ($)", key=f"pnl{i}")
        comment = st.text_area("Trade Yorumu", key=f"c{i}")

        if st.button(f"❌ Pozisyonu Kapat #{i+1}"):
            rr = abs((tp_vals[0]-avg_entry)/(t["stop"]-avg_entry)) if (t["stop"]-avg_entry)!=0 else 0

            data["closed"].append({
                "title":f"#{i+1}-{t['symbol']}-{t['side']}-1:{rr:.2f}",
                "pnl":pnl,
                "comment":comment
            })

            data["active"].pop(i)
            save_data(data)
            st.rerun()

# ------------------ FILTER ------------------
st.subheader("📉 Closed Trades")

filter_side = st.selectbox("Side Filter", ["ALL","LONG","SHORT"])
filter_pnl = st.selectbox("PnL Filter", ["ALL","WIN","LOSS"])

for t in data["closed"]:

    show=True

    if filter_pnl=="WIN" and t["pnl"]<=0:
        show=False
    if filter_pnl=="LOSS" and t["pnl"]>0:
        show=False

    if show:
        color = "🟢" if t["pnl"]>0 else "🔴"

        with st.expander(f"{t['title']} | {color} ${t['pnl']}"):
            st.write("📝 Comment:", t["comment"])
