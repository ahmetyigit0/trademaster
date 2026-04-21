import streamlit as st
import numpy as np

st.title("📊 Trade Order Calculator")

# INPUTS
capital = st.number_input("Total Capital ($)", value=18400)
risk_percent = st.number_input("Risk (%)", value=2.0)
entry_min = st.number_input("Entry Min Price", value=78600)
entry_max = st.number_input("Entry Max Price", value=80200)
stop_price = st.number_input("Stop Loss Price", value=81500)
orders = st.number_input("Number of Orders", value=3)

# CALCULATIONS
risk_amount = capital * (risk_percent / 100)

prices = np.linspace(entry_min, entry_max, int(orders))

avg_entry = np.mean(prices)
distance = abs(stop_price - avg_entry)

btc_size = risk_amount / distance
usd_position = btc_size * avg_entry

order_sizes_btc = btc_size / orders
order_sizes_usd = usd_position / orders

# OUTPUT
st.subheader("📌 Results")

st.write(f"Risk Amount: ${risk_amount:.2f}")
st.write(f"Average Entry: {avg_entry:.2f}")
st.write(f"Total Position: {btc_size:.4f} BTC (~${usd_position:.2f})")
st.write(f"Per Order: {order_sizes_btc:.4f} BTC (~${order_sizes_usd:.2f})")

st.subheader("📊 Orders")

for i, price in enumerate(prices):
    st.write(f"{i+1}. Price: {price:.0f} | Size: {order_sizes_btc:.4f} BTC")
