import streamlit as st
import pandas as pd
import yfinance as yf
import json
import os
import plotly.express as px
import numpy as np
from datetime import datetime

# --- KONFİGÜRASYON ---
DATA_FILE = "orion_v6_final.json"
st.set_page_config(page_title="Orion V6 Terminal", layout="wide")

# --- VERİ YÖNETİMİ ---
def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f: return json.load(f)
        except: return []
    return []

def save_data(data):
    with open(DATA_FILE, "w") as f: json.dump(data, f, indent=4)

# --- CSS (GRID ZORLAMA & PREMIUM LOOK) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
    * { font-family: 'Plus Jakarta Sans', sans-serif; }
    [data-testid="stAppViewContainer"] { background: #05070a; }
    
    /* Kartların yan yana durması için konteyner */
    .asset-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 15px;
        margin-bottom: 20px;
    }
    
    .glass-card {
        background: rgba(23, 28, 36, 0.8);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 18px; padding: 15px;
    }
    .symbol-badge { background: #1e293b; color: #3b82f6; padding: 3px 8px; border-radius: 6px; font-weight: 800; font-size: 11px; }
    .price-val { font-size: 20px; font-weight: 800; color: #fff; margin: 5px 0; }
    .analysis-box { background: rgba(59, 130, 246, 0.1); border: 1px solid #3b82f6; border-radius: 12px; padding: 10px; margin-top: 10px; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# --- TEKNİK ANALİZ MOTORU ---
def run_technical_analysis(symbol):
    t = yf.Ticker(symbol)
    df = t.history(period="60d")
    if len(df) < 30: return "Yetersiz veri kanka."
    
    # RSI Hesaplama
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rsi = 100 - (100 / (1 + (gain / loss))).iloc[-1]
    
    # EMA 20
    ema20 = df['Close'].ewm(span=20, adjust=False).mean().iloc[-1]
    curr_price = df['Close'].iloc[-1]
    
    # Bollinger Bantları
    sma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    upper_b = sma20.iloc[-1] + (std20.iloc[-1] * 2)
    lower_b = sma20.iloc[-1] - (std20.iloc[-1] * 2)
    
    # Yorumlama
    yorum = f"📉 **RSI:** {rsi:.1f}. "
    if rsi > 70: yorum += "Aşırı alım bölgesinde, dikkat! "
    elif rsi < 30: yorum += "Aşırı satım, tepki gelebilir. "
    else: yorum += "Nötr bölge. "
    
    if curr_price > upper_b: yorum += "\n⚠️ Bollinger üst bandı kırıldı, kâr satışı gelebilir."
    elif curr_price < lower_b: yorum += "\n🚀 Bollinger alt bandına çarptı, destek bulabilir."
    
    if curr_price > ema20: yorum += "\n📈 EMA 20 üzerinde, trend yukarı."
    else: yorum += "\n📉 EMA 20 altında, baskı devam ediyor."
    
    return yorum

# --- TEST VERİSİ ---
if not os.path.exists(DATA_FILE):
    test_data = [
        {"symbol": "NVDA", "name": "Nvidia", "amount": 5.0, "cost": 105.0, "date": "2024-06-01"},
        {"symbol": "BTC-USD", "name": "Bitcoin", "amount": 0.1, "cost": 55000.0, "date": "2023-12-15"},
        {"symbol": "THYAO.IS", "name": "THY", "amount": 200.0, "cost": 240.0, "date": "2024-02-10"}
    ]
    save_data(test_data)

# --- ANA PANEL ---
st.markdown("<h2 style='color: #fff; font-weight:800;'>ORION <span style='color:#3b82f6;'>V6 PRO</span></h2>", unsafe_allow_html=True)

portfolio = load_data()
if portfolio:
    # Yan yana dizilim için kolonları dinamik oluşturuyoruz
    rows = [portfolio[i:i + 3] for i in range(0, len(portfolio), 3)]
    
    total_val = 0
    for row_items in rows:
        cols = st.columns(3) # Her satırda 3 kolon
        for idx, item in enumerate(row_items):
            with cols[idx]:
                try:
                    t = yf.Ticker(item['symbol'])
                    curr_p = t.history(period="1d")['Close'].iloc[-1]
                    total_val += (curr_p * item['amount'])
                    
                    st.markdown(f"""
                    <div class="glass-card">
                        <div style="display:flex; justify-content:space-between;">
                            <span class="symbol-badge">{item['symbol']}</span>
                            <span style="color:#8b949e; font-size:10px;">{item['date']}</span>
                        </div>
                        <div class="price-val">${curr_p:,.2f}</div>
                        <div style="color:#3b82f6; font-size:13px; font-weight:600;">Portföy: ${(curr_p * item['amount']):,.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Buton Grubu
                    b1, b2, b3, b4 = st.columns(4)
                    if b1.button("📝", key=f"e_{item['symbol']}"): st.session_state[f"edit_{i}"] = True
                    if b2.button("🗑️", key=f"d_{item['symbol']}"): st.session_state[f"del_{i}"] = True
                    if b3.button("🔍", key=f"dt_{item['symbol']}"): st.info(f"📍 {item['name']} - {item['amount']} Adet")
                    if b4.button("📊", key=f"ta_{item['symbol']}"):
                        analysis_res = run_technical_analysis(item['symbol'])
                        st.markdown(f'<div class="analysis-box">{analysis_res}</div>', unsafe_allow_html=True)
                except:
                    st.error(f"{item['symbol']} hatası")

    st.markdown(f"### 💰 Toplam Servet: ${total_val:,.2f}")
    
    # Dağılım Grafiği
    fig = px.pie(names=[x['symbol'] for x in portfolio], 
                 values=[yf.Ticker(x['symbol']).history(period="1d")['Close'].iloc[-1] * x['amount'] for x in portfolio],
                 hole=0.7, template="plotly_dark")
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', showlegend=False, height=300)
    st.plotly_chart(fig, use_container_width=True)

with st.sidebar:
    st.markdown("### ➕ Varlık Ekle")
    with st.form("sidebar_add"):
        s = st.text_input("Sembol (Örn: AAPL, BTC-USD)").upper()
        a = st.number_input("Adet", min_value=0.0)
        c = st.number_input("Maliyet", min_value=0.0)
        if st.form_submit_button("Sisteme Ekle"):
            p = load_data(); p.append({"symbol": s, "name": s, "amount": a, "cost": c, "date": str(datetime.now().date())})
            save_data(p); st.rerun()
