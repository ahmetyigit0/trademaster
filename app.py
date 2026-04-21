import streamlit as st
import numpy as np

st.set_page_config(layout="wide")

# --- SESSION STATE ---
if "active_trades" not in st.session_state:
    st.session_state.active_trades = []

if "closed_trades" not in st.session_state:
    st.session_state.closed_trades = []

# --- HEADER ---
st.title("📊 Trade Journal + Position Manager")

# --- ADD NEW TRADE ---
if st.button("➕ Yeni Pozisyon Ekle"):
    st.session_state.active_trades.append({
        "capital": 18400,
        "risk": 2.0,
        "entries": [78600, 79400, 80200],
        "weights": [30, 40, 30],
        "stop": 81500
    })

st.divider()

# --- ACTIVE TRADES ---
st.subheader("🟢 Active Positions")

for i, trade in enumerate(st.session_state.active_trades):
    with st.expander(f"#{i+1} Active Position"):

        col1, col2 = st.columns(2)

        with col1:
            trade["capital"] = st.number_input("Capital", value=trade["capital"], key=f"cap{i}")
            trade["risk"] = st.number_input("Risk %", value=trade["risk"], key=f"risk{i}")

        with col2:
            trade["stop"] = st.number_input("Stop", value=trade["stop"], key=f"stop{i}")

        col1, col2, col3 = st.columns(3)

        entries = []
        weights = []

        for j in range(3):
            with [col1, col2, col3][j]:
                e = st.number_input(f"Entry {j+1}", value=trade["entries"][j], key=f"e{i}{j}")
                w = st.number_input(f"W{j+1}%", value=trade["weights"][j], key=f"w{i}{j}")
                entries.append(e)
                weights.append(w)

        weights = np.array(weights)
        weights = weights / weights.sum()
        entries = np.array(entries)

        # CALC
        risk_amount = trade["capital"] * (trade["risk"] / 100)
        avg_entry = np.sum(entries * weights)
        distance = abs(trade["stop"] - avg_entry)

        btc_size = risk_amount / distance
        usd_position = btc_size * avg_entry

        st.markdown(f"💰 **Position:** ${usd_position:.2f}")
        st.markdown(f"📍 **Avg Entry:** {avg_entry:.0f}")

        # CLOSE TRADE
        pnl = st.number_input(f"PnL ($)", value=0.0, key=f"pnl{i}")

        if st.button(f"❌ Pozisyonu Kapat #{i+1}"):
            st.session_state.closed_trades.append({
                "id": i+1,
                "pnl": pnl,
                "details": trade
            })
            st.session_state.active_trades.pop(i)
            st.experimental_rerun()

st.divider()

# --- CLOSED TRADES ---
st.subheader("📉 Closed Positions")

for i, trade in enumerate(st.session_state.closed_trades):
    pnl = trade["pnl"]
    color = "🟢" if pnl >= 0 else "🔴"

    with st.expander(f"#{trade['id']} Result = {color} ${pnl:.2f}"):

        details = trade["details"]

        st.write(f"Capital: {details['capital']}")
        st.write(f"Risk: %{details['risk']}")
        st.write(f"Entries: {details['entries']}")
        st.write(f"Weights: {details['weights']}")
        st.write(f"Stop: {details['stop']}")
