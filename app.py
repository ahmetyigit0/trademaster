import streamlit as st
import pandas as pd
import yfinance as yf
import json
import os
import plotly.express as px
from datetime import datetime, timedelta

# --- KONFİGÜRASYON ---
DATA_FILE = "orion_test_v5.json"
st.set_page_config(page_title="Orion Ultimate Test", layout="wide")

# --- VERİ YÖNETİMİ ---
def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f: return json.load(f)
        except: return []
    return []

def save_data(data):
    with open(DATA_FILE, "w") as f: json.dump(data, f, indent=4)

# --- TEST VERİLERİNİ OLUŞTURMA (İLK AÇILIŞ) ---
if not os.path.exists(DATA_FILE):
    test_assets = [
        {"symbol": "NVDA", "name": "Nvidia", "amount": 10.0, "cost": 95.0, "date": "2024-05-15"},
        {"symbol": "BTC-USD", "name": "Bitcoin", "amount": 0.25, "cost": 42000.0, "date": "2023-11-10"},
        {"symbol": "THYAO.IS", "name": "Türk Hava Yolları", "amount": 100.0, "cost": 210.0, "date": "2024-01-20"},
        {"symbol": "TUPRS.IS", "name": "Tüpraş", "amount": 150.0, "cost": 145.0, "date": "2024-03-05"},
        {"symbol": "GOOGL", "name": "Google", "amount": 5.0, "cost": 130.0, "date": "2023-08-12"},
        {"symbol": "GC=F", "name": "Altın ONS", "amount": 2.0, "cost": 1950.0, "date": "2023-06-01"}
    ]
    save_data(test_assets)

# --- CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
    * { font-family: 'Plus Jakarta Sans', sans-serif; }
    [data-testid="stAppViewContainer"] { background: #05070a; color: #e6edf3; }
    .glass-card {
        background: rgba(23, 28, 36, 0.7);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px; padding: 20px; margin-bottom: 15px;
        min-height: 160px;
    }
    .symbol-badge { background: #1e293b; color: #3b82f6; padding: 4px 8px; border-radius: 8px; font-weight: 800; font-size: 11px; }
    .price-text { font-size: 20px; font-weight: 800; color: #fff; margin-top: 5px; }
</style>
""", unsafe_allow_html=True)

# --- ANALİZ ---
@st.cache_data(ttl=300)
def get_performance(symbol, period_choice):
    try:
        t = yf.Ticker(symbol)
        p_map = {"1 Gün": "2d", "1 Ay": "1mo", "1 Yıl": "1y"}
        hist = t.history(period=p_map[period_choice])
        if len(hist) < 2: return 0.0, 0.0
        start_p, curr_p = hist['Close'].iloc[0], hist['Close'].iloc[-1]
        return curr_p, ((curr_p - start_p) / start_p) * 100
    except: return 0.0, 0.0

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🛡️ KONSEY KONTROL")
    period_btn = st.radio("Zaman Aralığı", ["1 Gün", "1 Ay", "1 Yıl"], horizontal=True)
    st.divider()
    if st.button("♻️ Verileri Resetle"):
        if os.path.exists(DATA_FILE): os.remove(DATA_FILE)
        st.rerun()

# --- ANA EKRAN ---
st.markdown("<h1 style='color: #fff; font-weight:800;'>ORION <span style='color:#3b82f6;'>ULTIMATE</span></h1>", unsafe_allow_html=True)

portfolio = load_data()

if portfolio:
    total_val = 0
    cols = st.columns(3) # Semboller yan yana
    
    for i, item in enumerate(portfolio):
        curr_p, p_perc = get_performance(item['symbol'], period_btn)
        t_val = curr_p * item['amount']
        total_val += t_val
        
        with cols[i % 3]:
            color = "#10b981" if p_perc >= 0 else "#ef4444"
            st.markdown(f"""
            <div class="glass-card">
                <div style="display:flex; justify-content:space-between; align-items:start;">
                    <span class="symbol-badge">{item['symbol']}</span>
                    <span style="color:{color}; font-size:12px; font-weight:700;">{p_perc:+.2f}%</span>
                </div>
                <div class="price-text">${curr_p:,.2f}</div>
                <small style="color:#8b949e;">{item['name']}</small><br>
                <small style="color:#3b82f6; font-weight:600;">Portföy: ${t_val:,.2f}</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Aksiyonlar
            b_edit, b_del, b_det = st.columns(3)
            if b_edit.button("📝", key=f"ed_{i}"): st.session_state[f"m_ed_{i}"] = True
            if b_del.button("🗑️", key=f"de_{i}"): st.session_state[f"m_cf_{i}"] = True
            if b_det.button("🔍", key=f"dt_{i}"): st.session_state[f"m_dt_{i}"] = True

            # DETAY MODAL
            if st.session_state.get(f"m_dt_{i}", False):
                st.info(f"🔍 **{item['name']}** Analizi")
                b_date = datetime.strptime(item['date'], '%Y-%m-%d')
                days = (datetime.now() - b_date).days
                st.write(f"📅 **{days // 30} aydır** cüzdanında.")
                st.write(f"📍 Giriş Tarihi: {item['date']}")
                st.write(f"💵 Maliyet: ${item['cost']:.2f}")
                if st.button("Kapat", key=f"c_dt_{i}"):
                    st.session_state[f"m_dt_{i}"] = False; st.rerun()

            # DÜZENLE MODAL
            if st.session_state.get(f"m_ed_{i}", False):
                with st.form(f"f_ed_{i}"):
                    na, nc = st.number_input("Adet", value=item['amount']), st.number_input("Maliyet", value=item['cost'])
                    if st.form_submit_button("Güncelle"):
                        portfolio[i].update({"amount": na, "cost": nc})
                        save_data(portfolio); st.rerun()

            # SİL MODAL
            if st.session_state.get(f"m_cf_{i}", False):
                st.error("Emin misin?")
                if st.button("Evet, SİL", key=f"y_{i}"):
                    portfolio.pop(i); save_data(portfolio); st.rerun()
                if st.button("İptal", key=f"n_{i}"):
                    st.session_state[f"m_cf_{i}"] = False; st.rerun()

    st.markdown(f"### 💰 Toplam Servet: ${total_val:,.2f}")
    
    # Pasta Grafiği
    fig = px.pie(names=[x['symbol'] for x in portfolio], 
                 values=[yf.Ticker(x['symbol']).history(period="1d")['Close'].iloc[-1] * x['amount'] for x in portfolio],
                 hole=0.7, template="plotly_dark")
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', showlegend=False, height=350)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Veri yok kanka. Resetle butonuna bas veya ekle.")
