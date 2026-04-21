import streamlit as st
import numpy as np

st.title("📊 Risk Based Position Calculator")

# INPUTS
capital = st.number_input("Total Capital ($)", value=18400)
risk_percent = st.number_input("Risk (%)", value=2.0)

entry1 = st.number_input("Entry 1 Price", value=78600)
entry2 = st.number_input("Entry 2 Price", value=79400)
entry3 = st.number_input("Entry 3 Price", value=80200)

w1 = st.number_input("Weight 1 (%)", value=30.0)
w2 = st.number_input("Weight 2 (%)", value=40.0)
w3 = st.number_input("Weight 3 (%)", value=30.0)

stop_price = st.number_input("Stop Loss Price", value=81500)

# NORMALIZE WEIGHTS
weights = np.array([w1, w2, w3])
weights = weights / weights.sum()

entries = np.array([entry1, entry2, entry3])

# CALCULATIONS
risk_amount = capital * (risk_percent / 100)

avg_entry = np.sum(entries * weights)
distance = abs(stop_price - avg_entry)

btc_size = risk_amount / distance
usd_position = btc_size * avg_entry

# SPLIT ORDERS
order_btc = btc_size * weights
order_usd = order_btc * entries

# OUTPUT
st.subheader("📌 RESULT")

st.write(f"Risk Amount: ${risk_amount:.2f}")
st.write(f"Average Entry: {avg_entry:.2f}")
st.write(f"Total Position: {btc_size:.4f} BTC (~${usd_position:.2f})")

st.subheader("📊 Orders")

for i in range(3):
    st.write(f"{i+1}. Price: {entries[i]:.0f} | Size: {order_btc[i]:.4f} BTC (~${order_usd[i]:.2f})")
