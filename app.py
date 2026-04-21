import streamlit as st
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Trade Calculator", layout="wide")

# --- STYLE ---
st.markdown("""
    <style>
    .card {
        background-color: #111;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 0 10px rgba(0,0,0,0.5);
    }
    .title {
        font-size: 22px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .result {
        font-size: 20px;
        font-weight: bold;
        color: #00ff99;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Risk Based Trade Calculator")

# --- INPUT CARDS ---
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="title">💰 Account</div>', unsafe_allow_html=True)

    capital = st.number_input("Total Capital ($)", value=18400)
    risk_percent = st.number_input("Risk (%)", value=2.0)

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="title">📉 Stop</div>', unsafe_allow_html=True)

    stop_price = st.number_input("Stop Loss Price", value=81500)

    st.markdown('</div>', unsafe_allow_html=True)

# --- ENTRY CARD ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="title">🎯 Entries</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    entry1 = st.number_input("Entry 1", value=78600)
    w1 = st.number_input("Weight 1 (%)", value=30.0)

with col2:
    entry2 = st.number_input("Entry 2", value=79400)
    w2 = st.number_input("Weight 2 (%)", value=40.0)

with col3:
    entry3 = st.number_input("Entry 3", value=80200)
    w3 = st.number_input("Weight 3 (%)", value=30.0)

st.markdown('</div>', unsafe_allow_html=True)

# --- CALCULATIONS ---
weights = np.array([w1, w2, w3])
weights = weights / weights.sum()

entries = np.array([entry1, entry2, entry3])

risk_amount = capital * (risk_percent / 100)
avg_entry = np.sum(entries * weights)
distance = abs(stop_price - avg_entry)

btc_size = risk_amount / distance
usd_position = btc_size * avg_entry

order_btc = btc_size * weights
order_usd = order_btc * entries

# --- RESULT CARD ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="title">📊 Results</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"<div>Risk Amount</div><div class='result'>${risk_amount:.2f}</div>", unsafe_allow_html=True)

with col2:
    st.markdown(f"<div>Total Position</div><div class='result'>${usd_position:.2f}</div>", unsafe_allow_html=True)

with col3:
    st.markdown(f"<div>Average Entry</div><div class='result'>{avg_entry:.0f}</div>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- ORDER TABLE ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="title">📦 Orders</div>', unsafe_allow_html=True)

for i in range(3):
    st.write(f"{i+1}. Price: {entries[i]:.0f} | Size: {order_btc[i]:.4f} BTC (~${order_usd[i]:.2f})")

st.markdown('</div>', unsafe_allow_html=True)
