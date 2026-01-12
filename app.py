import streamlit as st
import pandas as pd
import yfinance as yf
import json
import os
import plotly.express as px
from datetime import datetime, timedelta

# --- KONFİGÜRASYON ---
DATA_FILE = "orion_ultimate_v5.json"
st.set_page_config(page_title="Orion Ultimate", layout="wide")

# --- VERİ YÖNETİMİ ---
def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f: return json.load(f)
        except: return []
    return []

def save_data(data):
    with open(DATA_FILE, "w") as f: json.dump(data, f, indent=4)

# --- CSS (ULTRA MODERN GRID & PREMIUM DARK) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
    * { font-family: 'Plus Jakarta Sans', sans-serif; }
    [data-testid="stAppViewContainer"] { background: #05070a; color: #e6edf3; }
    
    .glass-card {
        background: rgba(23, 28, 36, 0.7);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px; padding: 20px; margin-bottom: 10px;
        min-height: 180px;
    }
    .symbol-badge {
        background: #1e293b; color: #3b82f6; padding: 4px 8px;
        border-radius: 8px; font-weight: 800; font-size: 12px;
    }
    .price-text { font-size: 22px; font-weight: 800; color: #fff; }
    .pnl-text { font-size: 14px; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# --- ANALİZ MOTORU (1G, 1A, 1Y PERFORMANS) ---
@st.cache_data(ttl=600)
def get_performance(symbol, period_choice):
    try:
        t = yf.Ticker(symbol)
        # Kullanıcının seçimine göre veri çekme
        periods = {"1 Gün": "2d", "1 Ay": "1mo", "1 Yıl": "1y"}
        hist = t.history(period=periods[period_choice])
        if len(hist) < 2: return 0.0, 0.0
        
        start_price = hist['Close'].iloc[0]
        curr_price = hist['Close'].iloc[-1]
        diff = curr_price - start_price
        perc = (diff / start_price) * 100
        return curr_price, perc
    except: return 0.0, 0.0

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🛡️ KONSEY KONTROL")
    period_btn = st.radio("Performans Görünümü", ["1 Gün", "1 Ay", "1 Yıl"], horizontal=True)
    
    with st.expander("➕ Varlık Ekle"):
        q = st.text_input("Arama")
        if q:
            res = yf.Search(q, max_results=3).quotes
            if res:
                opt = {f"{r.get('shortname','')} ({r['symbol']})": r['symbol'] for r in res}
                sel = st.selectbox("Seç", list(opt.keys()))
                with st.form("add_v5"):
                    a = st.number_input("Adet", value=1.0)
                    c = st.number_input("Maliyet", value=0.0)
                    d = st.date_input("İlk Alım Tarihi")
                    if st.form_submit_button("Sisteme İşle"):
                        p = load_data()
                        p.append({
                            "symbol": opt[sel], "name": sel.split(' (')[0],
                            "amount": a, "cost": c, "date": str(d),
                            "history": [{"type": "Alım", "date": str(d), "amt": a}]
                        })
                        save_data(p); st.rerun()

# --- ANA EKRAN ---
st.markdown("<h1 style='color: #fff; font-weight:800;'>ORION <span style='color:#3b82f6;'>ULTIMATE</span></h1>", unsafe_allow_html=True)

portfolio = load_data()

if portfolio:
    # --- PORTFÖY ÖZETİ ---
    total_val = 0
    df_list = []
    
    # Grid Düzeni: Yan yana 3 sembol
    cols = st.columns(3)
    
    for i, item in enumerate(portfolio):
        curr_p, p_perc = get_performance(item['symbol'], period_btn)
        t_val = curr_p * item['amount']
        total_val += t_val
        
        # Grid yerleşimi (i % 3 ile kolon seçimi)
        with cols[i % 3]:
            color = "#10b981" if p_perc >= 0 else "#ef4444"
            st.markdown(f"""
            <div class="glass-card">
                <div style="display:flex; justify-content:space-between; align-items:start;">
                    <span class="symbol-badge">{item['symbol']}</span>
                    <span class="pnl-text" style="color:{color};">{p_perc:+.2f}% ({period_btn})</span>
                </div>
                <div style="margin-top:15px;">
                    <small style="color:#8b949e;">{item['name']}</small>
                    <div class="price-text">${curr_p:,.2f}</div>
                    <small style="color:#3b82f6; font-weight:600;">Bakiye: ${t_val:,.2f}</small>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Aksiyon Butonları (Düzenle, Sil, Detay)
            b_edit, b_del, b_det = st.columns(3)
            if b_edit.button("📝", key=f"ed_{i}"): st.session_state[f"mode_ed_{i}"] = True
            if b_del.button("🗑️", key=f"del_{i}"): st.session_state[f"mode_cf_{i}"] = True
            if b_det.button("🔍", key=f"det_{i}"): st.session_state[f"mode_dt_{i}"] = True

            # --- MODAL: DETAY (ZAMAN ANALİZİ) ---
            if st.session_state.get(f"mode_dt_{i}", False):
                st.info(f"📜 **{item['name']} Analiz Raporu**")
                buy_date = datetime.strptime(item['date'], '%Y-%m-%d')
                hold_days = (datetime.now() - buy_date).days
                st.write(f"📅 **{hold_days // 30} aydır** tutuyorsun.")
                st.write(f"📥 İlk ekleme: {item['date']}")
                st.write(f"💰 Ortalama Maliyet: ${item['cost']:.2f}")
                if st.button("Kapat", key=f"cls_dt_{i}"):
                    st.session_state[f"mode_dt_{i}"] = False; st.rerun()

            # --- MODAL: DÜZENLE ---
            if st.session_state.get(f"mode_ed_{i}", False):
                with st.form(f"f_ed_{i}"):
                    new_a = st.number_input("Adet", value=item['amount'])
                    new_c = st.number_input("Maliyet", value=item['cost'])
                    if st.form_submit_button("Güncelle"):
                        portfolio[i].update({"amount": new_a, "cost": new_c})
                        save_data(portfolio); st.rerun()

            # --- MODAL: SİL ---
            if st.session_state.get(f"mode_cf_{i}", False):
                st.error("Emin misin?")
                if st.button("Evet, SİL", key=f"y_{i}"):
                    portfolio.pop(i); save_data(portfolio); st.rerun()
                if st.button("İptal", key=f"n_{i}"):
                    st.session_state[f"mode_cf_{i}"] = False; st.rerun()

    # --- ALT GRAFİK: PORTFÖY DEĞERİ ---
    st.markdown(f"### 💰 Toplam Servet: ${total_val:,.2f}")
    # Pasta grafiği ile dağılım
    fig_pie = px.pie(names=[x['symbol'] for x in portfolio], 
                     values=[yf.Ticker(x['symbol']).history(period="1d")['Close'].iloc[-1] * x['amount'] for x in portfolio],
                     hole=0.7, template="plotly_dark")
    fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', showlegend=False, height=300)
    st.plotly_chart(fig_pie, use_container_width=True)

else:
    st.warning("Konsey boş. Sidebar'dan ekleme yap kanka.")
