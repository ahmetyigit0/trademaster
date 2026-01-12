import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
from datetime import datetime
import json
import os

# -------------------- KONFİGÜRASYON --------------------
st.set_page_config(page_title="TradeMaster: Orion", layout="wide")

DATA_FILE = "my_personal_portfolio.json"

# -------------------- CSS (ORION UI - GÖRSELDEKİ TEMA) --------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
    * { font-family: 'Plus Jakarta Sans', sans-serif; }
    [data-testid="stAppViewContainer"] { background: #05070a; color: #e6edf3; }
    
    /* Konsey Kartları */
    .orion-card {
        background: rgba(17, 24, 39, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px; padding: 15px; margin-bottom: 10px;
    }
    
    /* Momentum/Trend İbreleri için Küçük Kartlar */
    .indicator-box {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 10px;
        text-align: center;
    }

    /* Sinyal Renkleri (Görseldeki gibi Soft) */
    .tut-btn { background: #ff4b4b; color: white; border-radius: 8px; padding: 2px 10px; font-weight: 800; font-size: 12px; }
    .sig-buy { background: linear-gradient(135deg, #1e40af, #3b82f6); color: white; }
    .sig-sell { background: linear-gradient(135deg, #991b1b, #ef4444); color: white; }
    .sig-neutral { background: linear-gradient(135deg, #065f46, #10b981); color: white; }
    
    .status-text { font-size: 24px; font-weight: 800; color: #fff; margin-bottom: 0; }
</style>
""", unsafe_allow_html=True)

# -------------------- VERİ YÖNETİMİ --------------------
def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                return json.load(f)
        except: return []
    return []

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

# -------------------- TEKNİK ANALİZ MOTORU --------------------
@st.cache_data(ttl=300)
def get_orion_analysis(symbol):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1mo")
        if hist.empty: return None
        
        current_price = hist['Close'].iloc[-1]
        # RSI Hesaplama
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rsi = 100 - (100 / (1 + (gain / loss))).iloc[-1]
        
        # Momentum ve Trend (Görseldeki mantık)
        momentum = "Pozitif" if current_price > hist['Close'].mean() else "Zayıf"
        trend_dir = "YUKARI" if current_price > hist['Close'].iloc[-5] else "AŞAĞI"
        
        return {
            "price": current_price, "rsi": rsi, 
            "momentum": momentum, "trend": trend_dir,
            "change": ((current_price / hist['Close'].iloc[-2]) - 1) * 100
        }
    except: return None

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.title("🛡️ Konsey Paneli")
    with st.form("new_asset"):
        sym = st.text_input("Sembol (BTC-USD, AAPL, THYAO.IS)").upper()
        amt = st.number_input("Adet", min_value=0.0)
        cst = st.number_input("Maliyet", min_value=0.0)
        curr = st.radio("Birim", ["USD", "TRY"])
        if st.form_submit_button("Sisteme Ekle") and sym:
            data = load_data()
            data.append({"symbol": sym, "amount": amt, "cost": cst, "currency": curr})
            save_data(data)
            st.rerun()
    usd_try = st.number_input("USD/TRY", value=33.15)

# -------------------- ANA EKRAN --------------------
st.markdown("<h2 style='color:#10b981;'>KONSEY KARARI: <span style='color:#fff;'>BİRİKTİR</span></h2>", unsafe_allow_html=True)

portfolio = load_data()

if not portfolio:
    st.warning("Konsey henüz toplanmadı. Varlık ekle kanka.")
else:
    processed = []
    with st.spinner('Analiz yapılıyor...'):
        for i, item in enumerate(portfolio):
            analysis = get_orion_analysis(item['symbol'])
            if analysis:
                item.update(analysis)
                mult = usd_try if item['currency'] == "USD" else 1
                item['val_try'] = item['amount'] * item['price'] * mult
                item['cost_try'] = item['amount'] * item['cost'] * mult
                
                # SIFIRA BÖLME HATASI (ZeroDivisionError) ÇÖZÜMÜ:
                if item['cost'] > 0:
                    item['pnl_perc'] = ((item['price'] - item['cost']) / item['cost']) * 100
                else:
                    item['pnl_perc'] = 0.0
                
                item['id'] = i
                processed.append(item)

    # ÜST METRİKLER
    t_val = sum(x['val_try'] for x in processed)
    t_pnl = t_val - sum(x['cost_try'] for x in processed)
    
    m1, m2, m3 = st.columns(3)
    m1.markdown(f'<div class="orion-card"><small>TOPLAM PORTFÖY</small><div class="status-text">{t_val:,.0f} ₺</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="orion-card"><small>GÜVEN ENDEKSİ</small><div class="status-text" style="color:#10b981;">%72</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="orion-card"><small>NET K/Z</small><div class="status-text" style="color:#3b82f6;">{t_pnl:,.0f} ₺</div></div>', unsafe_allow_html=True)

    # ORION TEKNİK ANALİZ KARTLARI (Görseldeki 4'lü yapı gibi)
    st.subheader("🔭 Orion Teknik Analiz")
    for row in processed:
        with st.expander(f"🔍 {row['symbol']} Analiz Detayı", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            
            # Momentum
            c1.markdown(f"""<div class="indicator-box"><small>Momentum</small><br><b>{row['momentum']}</b><br><small>RSI: {row['rsi']:.1f}</small></div>""", unsafe_allow_html=True)
            # Yapı
            c2.markdown(f"""<div class="indicator-box"><small>Piyasa Yapısı</small><br><b>Kanal İçi</b><br><small>S-R Dengeli</small></div>""", unsafe_allow_html=True)
            # Trend
            c3.markdown(f"""<div class="indicator-box"><small>Trend Yönü</small><br><b style="color:#ef4444;">{row['trend']}</b><br><small>Kısa Vade</small></div>""", unsafe_allow_html=True)
            # Karar
            karar = "TUT" if 40 < row['rsi'] < 60 else ("AL" if row['rsi'] < 40 else "SAT")
            c4.markdown(f"""<div class="indicator-box"><small>Konsey Kararı</small><br><span class="tut-btn">{karar}</span><br><small>Skor: 43/100</small></div>""", unsafe_allow_html=True)

            # Silme Butonu
            if st.button(f"🗑️ {row['symbol']} Sistemden Çıkar", key=f"del_{row['id']}"):
                portfolio.pop(row['id'])
                save_data(portfolio)
                st.rerun()

    # PORTFÖY GRAFİĞİ
    st.write("")
    fig = px.area(pd.DataFrame(processed), x="symbol", y="val_try", title="Varlık Dağılımı (Konsey Stratejisi)")
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#fff")
    st.plotly_chart(fig, use_container_width=True)

st.caption("Veriler Orion Algoritması ile senkronize edilmiştir.")
