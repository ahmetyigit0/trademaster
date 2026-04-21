import streamlit as st
import numpy as np
import json
import os

st.set_page_config(layout="wide")

# ---------------- STORAGE ----------------
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

# ---------------- UI ----------------
st.title("📊 Trade Journal PRO MAX")

# ---------------- ADD ----------------
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
    })
    save_data(data)

# ---------------- ACTIVE ----------------
st.subheader("🟢 Active Trades")

for i, t in enumerate(data["active"]):

    with st.expander(f"#{i+1} {t['symbol']} {t['side']}"):

        col1, col2 = st.columns(2)

        with col1:
            t["symbol"] = st.text_input("Symbol", t["symbol"], key=f"s{i}")
            t["side"] = st.selectbox("Side", ["LONG","SHORT"], key=f"side{i}")

        with col2:
            t["capital"] = st.number_input("Capital", value=t["capital"], key=f"cap{i}")
            t["risk"] = st.number_input("Risk %", value=t["risk"], key=f"risk{i}")

        # ENTRY
        split = st.checkbox("Parçalı Entry", key=f"split{i}")

        entries=[]
        weights=[]

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

        stop = st.number_input("Stop Loss", value=t["stop"], key=f"stop{i}")

        # ---- CALC ----
        weights = np.array(weights)
        weights = weights / weights.sum()
        entries = np.array(entries)

        avg_entry = np.sum(entries * weights)
        distance = abs(stop - avg_entry)

        risk_amount = t["capital"] * (t["risk"]/100)

        # FULL POSITION RISK
        full_position_loss = (distance / avg_entry) * t["capital"]

        st.markdown(f"📍 Avg Entry: {avg_entry:.0f}")

        if full_position_loss <= risk_amount:
            st.success("✅ Full sermaye ile girilebilir")
            usd_pos = t["capital"]
        else:
            st.error("⚠️ Risk fazla")

            btc_size = risk_amount / distance if distance != 0 else 0
            usd_pos = btc_size * avg_entry

            st.warning(f"👉 Önerilen Pozisyon: ${usd_pos:.2f}")

            if st.button(f"Öneriyi Uygula #{i}"):
                t["capital"] = usd_pos
                save_data(data)
                st.rerun()

        st.markdown(f"💰 Pozisyon: ${usd_pos:.2f}")

        # TP
        tp = st.number_input("Take Profit", key=f"tp{i}")

        pnl = st.number_input("PnL", key=f"pnl{i}")
        comment = st.text_area("Yorum", key=f"c{i}")

        colA, colB = st.columns(2)

        if colA.button(f"❌ Kapat #{i}"):
            rr = abs((tp-avg_entry)/(stop-avg_entry)) if distance!=0 else 0

            data["closed"].append({
                "title":f"#{i+1}-{t['symbol']}-{t['side']}-1:{rr:.2f}",
                "pnl":pnl,
                "entries":entries.tolist(),
                "avg":avg_entry,
                "stop":stop,
                "tp":tp,
                "comment":comment
            })

            data["active"].pop(i)
            save_data(data)
            st.rerun()

        if colB.button(f"🗑 Sil #{i}"):
            data["active"].pop(i)
            save_data(data)
            st.rerun()

# ---------------- CLOSED ----------------
st.subheader("📉 Closed Trades")

for t in data["closed"]:
    color = "🟢" if t["pnl"]>0 else "🔴"

    with st.expander(f"{t['title']} | {color} ${t['pnl']}"):
        st.write("Entries:", t["entries"])
        st.write("Avg:", round(t["avg"]))
        st.write("Stop:", t["stop"])
        st.write("TP:", t["tp"])
        st.write("Comment:", t["comment"])