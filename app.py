import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import time

# -------------------- PREMIUM CONFIG --------------------
st.set_page_config(page_title="TradeMaster Pro Premium", layout="wide", initial_sidebar_state="collapsed")

# -------------------- PREMIUM CSS (Glassmorphism & Neon) --------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
    
    * { font-family: 'Plus Jakarta Sans', sans-serif; }
    [data-testid="stAppViewContainer"] { background: #05070a; }
    
    /* Premium Kart Yapısı */
    .premium-card {
        background: rgba(22, 27, 34, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 24px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }

    /* Üst Metrikler */
    .stat-label { color: #8b949e; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
    .stat-value { font-size: 28px; font-weight: 800; color: #fff; margin-top: 5px; }
    .stat-delta { font-size: 14px; font-weight: 600; margin-top: 5px; }

    /* Neon Sinyal Kutuları */
    .signal-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin: 20px 0; }
    .sig-card {
        padding: 25px; border-radius: 20px; text-align: center; font-weight: 700;
        transition: all 0.3s ease; border: 1px solid rgba(255,255,255,0.1);
    }
    .sig-buy { background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); box-shadow: 0 0 20px rgba(59, 130, 246, 0.3); }
    .sig-sell { background: linear-gradient(135deg, #7c2d12 0%, #ea580c 100%); box-shadow: 0 0 20px rgba(234, 88, 12, 0.3); }
    .sig-neutral { background: linear-gradient(135deg, #064e3b 0%, #10b981 100%); box-shadow: 0 0 20px rgba(16, 185, 129, 0.3); }

    /* Custom Progress Bar */
    .custom-progress { background: #30363d; height: 8px; border-radius: 4px; overflow: hidden; margin-top: 10px; }
    .custom-progress-fill { background: linear-gradient(90deg, #3b82f6, #8b5cf6); height: 100%; border-radius: 4px; }

    /* Gizli Sidebar ve Temiz Arayüz */
    [data-testid="stSidebar"] { background-color: #0d1117; }
</style>
""", unsafe_allow_html=True)

# -------------------- PREMIUM LOGIC (Sinyal Motoru) --------------------
def analyze_market(price, cost, rsi):
    pnl = (price - cost) / cost * 100
    if rsi < 30: return "GÜÇLÜ ALIM", "sig-buy", "RSI Aşırı Satım Bölgesinde"
    if rsi > 70: return "KÂR REALİZE", "sig-sell", "RSI Aşırı Alım Bölgesinde"
    if pnl > 10: return "TREND TAKİP", "sig-neutral", "Yükseliş Trendi Korunuyor"
    return "BEKLE", "sig-neutral", "Piyasa Doygunluk Noktasında"

# -------------------- MOCK DATA --------------------
portfolio = [
    {"symbol": "BTC", "name": "Bitcoin", "amt": 0.45, "price": 96200.0, "cost": 82000.0, "rsi": 28},
    {"symbol": "ETH", "name": "Ethereum", "amt": 4.2, "price": 3850.0, "cost": 3100.0, "rsi": 72},
    {"symbol": "AAPL", "name": "Apple Inc.", "amt": 25, "price": 242.0, "cost": 195.0, "rsi": 55}
]
usd_try = 32.80

# -------------------- APP LAYOUT --------------------

# 1. HEADER & TİCKER
st.markdown("<div style='text-align: center; padding: 10px 0;'><h2 style='color: #fff; letter-spacing: -1px;'>TRADEMASTER <span style='color: #3b82f6;'>PREMIUM</span></h2></div>", unsafe_allow_html=True)

# 2. TOP METRICS (Üçlü Özet)
m1, m2, m3 = st.columns(3)
total_val = sum(p['amt'] * p['price'] * usd_try for p in portfolio)
total_pnl = sum((p['price'] - p['cost']) * p['amt'] * usd_try for p in portfolio)

with m1:
    st.markdown(f'<div class="premium-card"><div class="stat-label">Toplam Portföy</div><div class="stat-value">{total_val/1e6:.2f}M ₺</div><div class="stat-delta" style="color:#3b82f6;">↗ %12.4 Bu Ay</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="premium-card"><div class="stat-label">Net Kar/Zarar</div><div class="stat-value" style="color:#10b981;">+{total_pnl/1e3:.1f}K ₺</div><div class="stat-delta" style="color:#10b981;">🚀 Rekor Seviye</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="premium-card"><div class="stat-label">Hedef (5M ₺)</div><div class="stat-value">{int(total_val/5000000*100)}%</div><div class="stat-delta" style="color:#8b949e;">Kalan: {(5e6-total_val)/1e3:.0f}K ₺</div></div>', unsafe_allow_html=True)

# 3. AI SİNYAL MERKEZİ (Görseldeki 3'lü renkli kutu)
st.markdown("<h4 style='color:#fff; margin-bottom:15px;'>⚡ Algoritmik Sinyaller</h4>", unsafe_allow_html=True)
s1, s2, s3 = st.columns(3)
cols = [s1, s2, s3]

for i, p in enumerate(portfolio):
    action, style, reason = analyze_market(p['price'], p['cost'], p['rsi'])
    with cols[i]:
        st.markdown(f"""
        <div class="sig-card {style}">
            <div style="font-size:14px; opacity:0.8; margin-bottom:5px;">{p['symbol']} / USD</div>
            <div style="font-size:22px; margin-bottom:5px;">{action}</div>
            <div style="font-size:11px; opacity:0.9; font-weight:400;">{reason}</div>
        </div>
        """, unsafe_allow_html=True)

# 4. PORTFÖY DETAY & GRAFİK
st.write("")
c_left, c_right = st.columns([1.5, 1])

with c_left:
    st.markdown("<h4 style='color:#fff;'>📋 Varlık Yönetimi</h4>", unsafe_allow_html=True)
    for p in portfolio:
        val = p['amt'] * p['price'] * usd_try
        weight = (val / total_val) * 100
        pnl_p = (p['price'] - p['cost']) / p['cost'] * 100
        
        st.markdown(f"""
        <div class="premium-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div style="display:flex; align-items:center; gap:15px;">
                    <div style="background:#3b82f6; width:40px; height:40px; border-radius:12px; display:flex; align-items:center; justify-content:center; font-weight:800;">{p['symbol'][0]}</div>
                    <div>
                        <div style="font-weight:700; color:#fff;">{p['name']}</div>
                        <div style="font-size:12px; color:#8b949e;">{p['amt']} {p['symbol']} @ ${p['price']:,}</div>
                    </div>
                </div>
                <div style="text-align:right;">
                    <div style="color:#10b981; font-weight:700;">+{pnl_p:.1f}%</div>
                    <div style="font-size:14px; color:#fff;">{val:,.0f} ₺</div>
                </div>
            </div>
            <div class="custom-progress"><div class="custom-progress-fill" style="width:{weight}%;"></div></div>
            <div style="display:flex; justify-content:space-between; margin-top:8px; font-size:11px; color:#8b949e;">
                <span>Portföy Ağırlığı: %{weight:.1f}</span>
                <span>Maliyet: ${p['cost']:,}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

with c_right:
    st.markdown("<h4 style='color:#fff;'>📊 Dağılım Analizi</h4>", unsafe_allow_html=True)
    # Donut Chart
    fig = px.pie(portfolio, values=[p['amt']*p['price'] for p in portfolio], names=[p['symbol'] for p in portfolio], hole=0.7)
    fig.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                      margin=dict(t=0, b=0, l=0, r=0), height=250)
    fig.update_traces(marker=dict(colors=['#3b82f6', '#8b5cf6', '#10b981']))
    st.plotly_chart(fig, use_container_width=True)

    # Performans Line Chart
    st.markdown("<h4 style='color:#fff; margin-top:20px;'>📈 Haftalık Trend</h4>", unsafe_allow_html=True)
    df_line = pd.DataFrame({'x': range(10), 'y': np.random.randint(90, 110, size=10).cumsum()})
    fig_line = px.line(df_line, x='x', y='y')
    fig_line.update_traces(line_color='#3b82f6', line_width=4)
    fig_line.update_layout(xaxis_visible=False, yaxis_visible=False, paper_bgcolor='rgba(0,0,0,0)', 
                           plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=0, b=0, l=0, r=0), height=150)
    st.plotly_chart(fig_line, use_container_width=True)

# -------------------- FOOTER --------------------
st.markdown(f"<div style='text-align:center; color:#444; font-size:12px; margin-top:50px;'>TradeMaster AI Engine v2.4 Premium • Son Güncelleme: {datetime.now().strftime('%H:%M:%S')}</div>", unsafe_allow_html=True)
