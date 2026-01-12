import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
import random

# -------------------- CONFIG --------------------
st.set_page_config(page_title="TradeMaster Pro", layout="wide")

# -------------------- ULTRA MODERN CSS --------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0d1117;
        font-family: 'Inter', sans-serif;
        color: #e6edf3;
    }

    /* Üst Metrik Kartları */
    .metric-card {
        background: #161b22;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #30363d;
        text-align: left;
    }
    .metric-val { font-size: 24px; font-weight: 700; margin-top: 5px; }
    .metric-label { color: #8b949e; font-size: 14px; }

    /* Sinyal Kutuları (Görseldeki gibi) */
    .signal-box {
        padding: 20px;
        border-radius: 15px;
        color: white;
        font-weight: 600;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        min-height: 120px;
    }
    .sig-blue { background: linear-gradient(135deg, #007aff, #0056b3); }
    .sig-orange { background: linear-gradient(135deg, #ff9500, #ff5e00); }
    .sig-green { background: linear-gradient(135deg, #34c759, #248a3d); }

    /* Varlık Kartları */
    .asset-card {
        background: #161b22;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #30363d;
        margin-bottom: 10px;
    }
    .progress-bar-bg {
        background: #30363d;
        height: 6px;
        border-radius: 3px;
        margin-top: 8px;
    }
    .progress-bar-fill {
        background: #007aff;
        height: 6px;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- MOCK ALGORİTMA --------------------
def get_ai_signal(asset):
    # Gerçekte burası bir API'ye veya TA-Lib'e bağlanabilir
    rsi = random.randint(25, 75)
    if rsi < 35: return "GÜÇLÜ AL", "sig-blue", rsi
    elif rsi > 65: return "AŞIRI ALIM / SAT", "sig-orange", rsi
    else: return "YATAY / BEKLE", "sig-green", rsi

# -------------------- DATA LOAD --------------------
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = [
        {"asset": "BTC", "cat": "Kripto", "amount": 0.12, "price": 95000, "cost": 88000},
        {"asset": "AAPL", "cat": "Hisse", "amount": 10, "price": 240, "cost": 210},
        {"asset": "GOLD", "cat": "Emtia", "amount": 50, "price": 3000, "cost": 2850}
    ]

df = pd.DataFrame(st.session_state.portfolio)
usd_rate = 32.5
df['value_try'] = df['amount'] * df['price'] * usd_rate
total_val = df['value_try'].sum()

# -------------------- HEADER --------------------
st.markdown("<h1 style='color:#a371f7;'>🧠 TradeMaster Pro</h1>", unsafe_allow_html=True)

# TOP METRICS
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">💰 Toplam Portföy</div><div class="metric-val">{total_val/1e6:.1f}M ₺</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">🎯 Hedef</div><div class="metric-val">10.0M ₺</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">📈 Toplam K/Z</div><div class="metric-val" style="color:#34c759">+142K ₺</div></div>', unsafe_allow_html=True)

st.write("")

# -------------------- MIDDLE SECTION: SIGNALS --------------------
st.subheader("⚡ Akıllı Sinyaller")
s1, s2, s3 = st.columns(3)

# Örnek 3 varlık için sinyal üretelim
assets_to_watch = ["BTC", "ETH", "AAPL"]
cols = [s1, s2, s3]

for i, asset in enumerate(assets_to_watch):
    sig_text, sig_class, rsi_val = get_ai_signal(asset)
    with cols[i]:
        st.markdown(f"""
        <div class="signal-box {sig_class}">
            <div style="font-size:12px; opacity:0.8;">{asset} SINYAL</div>
            <div style="font-size:18px; margin:5px 0;">{sig_text}</div>
            <div style="font-size:12px;">RSI: {rsi_val}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")

# -------------------- BOTTOM SECTION: ASSETS & CHARTS --------------------
col_list, col_chart = st.columns([1.5, 1])

with col_list:
    st.subheader("📋 Varlıklarım")
    for _, row in df.iterrows():
        pnl = ((row['price'] - row['cost']) / row['cost']) * 100
        weight = (row['value_try'] / total_val) * 100
        
        st.markdown(f"""
        <div class="asset-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="font-weight:700; font-size:18px;">{row['asset']}</span>
                    <br><span style="color:#8b949e; font-size:12px;">{row['cat']}</span>
                </div>
                <div style="text-align:right;">
                    <div style="color:{'#34c759' if pnl > 0 else '#ff453a'}; font-weight:700;">%{pnl:.2f}</div>
                    <div style="color:#8b949e; font-size:12px;">{row['value_try']:,.0f} ₺</div>
                </div>
            </div>
            <div class="progress-bar-bg">
                <div class="progress-bar-fill" style="width:{weight}%;"></div>
            </div>
            <div style="font-size:10px; color:#8b949e; margin-top:4px;">Portföy Ağırlığı: %{weight:.1f}</div>
        </div>
        """, unsafe_allow_html=True)

with col_chart:
    st.subheader("📊 Dağılım")
    fig = px.pie(df, values='value_try', names='cat', hole=0.7,
                 color_discrete_sequence=['#007aff', '#a371f7', '#34c759'])
    fig.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', 
                      margin=dict(t=0, b=0, l=0, r=0), height=250)
    st.plotly_chart(fig, use_container_width=True)
    
    # Küçük bir performans grafiği
    st.subheader("📈 Trend")
    hist_data = pd.DataFrame({"Gün": range(10), "Değer": [random.randint(100, 120) for _ in range(10)]})
    fig_line = px.line(hist_data, x="Gün", y="Değer")
    fig_line.update_traces(line_color='#007aff', line_width=3)
    fig_line.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                           height=150, margin=dict(t=0, b=0, l=0, r=0),
                           xaxis_visible=False, yaxis_visible=False)
    st.plotly_chart(fig_line, use_container_width=True)

# -------------------- SIDEBAR SETTINGS --------------------
with st.sidebar:
    st.markdown("### ⚙️ Ayarlar")
    st.number_input("USD/TRY", value=32.5)
    st.selectbox("Risk Profili", ["Agresif", "Dengeli", "Muhafazakar"])
    if st.button("🔄 Verileri Güncelle"):
        st.rerun()
