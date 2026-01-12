import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
from datetime import datetime

# -------------------- PREMIUM CONFIG --------------------
st.set_page_config(page_title="TradeMaster Pro Premium", layout="wide")

# -------------------- PREMIUM CSS --------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
    * { font-family: 'Plus Jakarta Sans', sans-serif; }
    [data-testid="stAppViewContainer"] { background: #05070a; }
    
    .premium-card {
        background: rgba(22, 27, 34, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 24px; padding: 20px; margin-bottom: 20px;
    }
    .stat-label { color: #8b949e; font-size: 13px; font-weight: 600; text-transform: uppercase; }
    .stat-value { font-size: 28px; font-weight: 800; color: #fff; }

    .sig-card {
        padding: 25px; border-radius: 20px; text-align: center; font-weight: 700;
        border: 1px solid rgba(255,255,255,0.1); color: white;
    }
    .sig-buy { background: linear-gradient(135deg, #1e3a8a, #3b82f6); box-shadow: 0 0 20px rgba(59, 130, 246, 0.3); }
    .sig-sell { background: linear-gradient(135deg, #7c2d12, #ea580c); box-shadow: 0 0 20px rgba(234, 88, 12, 0.3); }
    .sig-neutral { background: linear-gradient(135deg, #064e3b, #10b981); box-shadow: 0 0 20px rgba(16, 185, 129, 0.3); }

    .custom-progress { background: #30363d; height: 8px; border-radius: 4px; margin-top: 10px; }
    .custom-progress-fill { background: linear-gradient(90deg, #3b82f6, #8b5cf6); height: 100%; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# -------------------- LIVE DATA ENGINE --------------------
@st.cache_data(ttl=60)  # Veriyi 60 saniyede bir güncelle
def get_live_data(symbols):
    data = {}
    for sym in symbols:
        ticker = yf.Ticker(sym)
        hist = ticker.history(period="1mo")
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]
            # Basit RSI Hesaplama
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1+rs)).iloc[-1]
            data[sym] = {"price": current_price, "rsi": rsi, "change": ((current_price/hist['Close'].iloc[-2])-1)*100}
    return data

# -------------------- ANALİZ MOTORU --------------------
def analyze_market(price, cost, rsi):
    if rsi < 35: return "GÜÇLÜ ALIM", "sig-buy", "RSI Aşırı Satımda"
    if rsi > 65: return "KÂR REALİZE", "sig-sell", "RSI Aşırı Alımda"
    return "BEKLE / TAKİP", "sig-neutral", "Trend Stabil"

# -------------------- MAIN APP --------------------
st.markdown("<div style='text-align: center;'><h2 style='color: #fff;'>TRADEMASTER <span style='color: #3b82f6;'>PREMIUM</span></h2></div>", unsafe_allow_html=True)

# Portföy Tanımı (Buraya kendi miktarlarını ve maliyetlerini yaz)
my_portfolio = [
    {"symbol": "BTC-USD", "name": "Bitcoin", "amt": 0.05, "cost": 65000},
    {"symbol": "ETH-USD", "name": "Ethereum", "amt": 1.2, "cost": 2400},
    {"symbol": "AAPL", "name": "Apple", "amt": 10, "cost": 180}
]

symbols = [p["symbol"] for p in my_portfolio]
with st.spinner('Canlı veriler çekiliyor...'):
    live_prices = get_live_data(symbols)

usd_try = 32.90 # Manuel kur veya yf.Ticker("TRY=X") ile çekilebilir

# ÜST METRİKLER
total_val_try = sum(live_prices[p["symbol"]]["price"] * p["amt"] * usd_try for p in my_portfolio if p["symbol"] in live_prices)
total_cost_try = sum(p["cost"] * p["amt"] * usd_try for p in my_portfolio)
total_pnl = total_val_try - total_cost_try

c1, c2, c3 = st.columns(3)
c1.markdown(f'<div class="premium-card"><div class="stat-label">Toplam Varlık</div><div class="stat-value">{total_val_try/1e3:.1f}K ₺</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="premium-card"><div class="stat-label">Net Kar/Zarar</div><div class="stat-value" style="color:#10b981;">{total_pnl/1e3:+.1f}K ₺</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="premium-card"><div class="stat-label">Piyasa Durumu</div><div class="stat-value" style="font-size:18px; margin-top:12px;">CANLI VERİ AKTİF ✅</div></div>', unsafe_allow_html=True)

# SİNYAL KUTULARI
st.markdown("<h4>⚡ Akıllı Sinyaller (AI Analysis)</h4>", unsafe_allow_html=True)
s_cols = st.columns(len(my_portfolio))

for i, p in enumerate(my_portfolio):
    sym = p["symbol"]
    if sym in live_prices:
        ld = live_prices[sym]
        action, style, reason = analyze_market(ld["price"], p["cost"], ld["rsi"])
        with s_cols[i]:
            st.markdown(f"""
            <div class="sig-card {style}">
                <div style="font-size:12px; opacity:0.8;">{sym}</div>
                <div style="font-size:20px; margin:5px 0;">{action}</div>
                <div style="font-size:11px;">RSI: {ld['rsi']:.1f} • {reason}</div>
            </div>
            """, unsafe_allow_html=True)

# DETAY LİSTESİ
st.write("")
col_left, col_right = st.columns([1.5, 1])

with col_left:
    st.markdown("<h4>📋 Portföy Detayı</h4>", unsafe_allow_html=True)
    for p in my_portfolio:
        sym = p["symbol"]
        if sym in live_prices:
            ld = live_prices[sym]
            val_try = ld["price"] * p["amt"] * usd_try
            weight = (val_try / total_val_try) * 100
            
            st.markdown(f"""
            <div class="premium-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div><b>{p['name']}</b> <span style="color:#8b949e; font-size:12px;">{sym}</span></div>
                    <div style="text-align:right;">
                        <div style="color:#10b981;">%{ld['change']:+.2f} (24s)</div>
                        <div style="color:#fff;">{val_try:,.0f} ₺</div>
                    </div>
                </div>
                <div class="custom-progress"><div class="custom-progress-fill" style="width:{weight}%;"></div></div>
            </div>
            """, unsafe_allow_html=True)

with col_right:
    st.markdown("<h4>📊 Dağılım</h4>", unsafe_allow_html=True)
    fig = px.pie(values=[live_prices[p["symbol"]]["price"]*p["amt"] for p in my_portfolio], 
                 names=[p["name"] for p in my_portfolio], hole=0.7)
    fig.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=250, margin=dict(t=0,b=0,l=0,r=0))
    st.plotly_chart(fig, use_container_width=True)

st.caption(f"Veriler Yahoo Finance üzerinden canlı çekilmektedir. Son Senkronizasyon: {datetime.now().strftime('%H:%M:%S')}")
