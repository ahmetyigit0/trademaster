import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
from datetime import datetime
import json
import os

# -------------------- KONFİGÜRASYON --------------------
st.set_page_config(page_title="TradeMaster Pro Premium", layout="wide")

DATA_FILE = "my_personal_portfolio.json"

# -------------------- CSS (PREMIUM LOOK) --------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
    * { font-family: 'Plus Jakarta Sans', sans-serif; }
    [data-testid="stAppViewContainer"] { background: #05070a; color: #e6edf3; }
    
    .premium-card {
        background: rgba(22, 27, 34, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 24px; padding: 20px; margin-bottom: 15px;
    }
    .sig-card {
        padding: 20px; border-radius: 18px; text-align: center; font-weight: 700;
        border: 1px solid rgba(255,255,255,0.1); color: white;
    }
    .sig-buy { background: linear-gradient(135deg, #1e3a8a, #3b82f6); box-shadow: 0 0 15px rgba(59, 130, 246, 0.3); }
    .sig-sell { background: linear-gradient(135deg, #7c2d12, #ea580c); box-shadow: 0 0 15px rgba(234, 88, 12, 0.3); }
    .sig-neutral { background: linear-gradient(135deg, #064e3b, #10b981); box-shadow: 0 0 15px rgba(16, 185, 129, 0.3); }
    
    .stButton>button { border-radius: 12px; background: #1e293b; color: white; border: 1px solid #334155; width: 100%; }
</style>
""", unsafe_allow_html=True)

# -------------------- VERİ YÖNETİMİ --------------------
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

# -------------------- CANLI VERİ & ANALİZ --------------------
@st.cache_data(ttl=300)
def get_live_stats(symbol):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1mo")
        if hist.empty: return None
        
        current_price = hist['Close'].iloc[-1]
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1+rs)).iloc[-1]
        change = ((current_price / hist['Close'].iloc[-2]) - 1) * 100
        return {"price": current_price, "rsi": rsi, "change": change}
    except:
        return None

def get_signal(rsi):
    if rsi < 35: return "GÜÇLÜ ALIM", "sig-buy"
    if rsi > 65: return "KÂR AL / SAT", "sig-sell"
    return "BEKLE / NÖTR", "sig-neutral"

# -------------------- SIDEBAR (VERİ GİRİŞİ) --------------------
with st.sidebar:
    st.title("⚙️ Portföy Yönetimi")
    with st.form("add_asset", clear_on_submit=True):
        st.subheader("Yeni Varlık")
        sym = st.text_input("Sembol (Örn: BTC-USD, THYAO.IS, GC=F)").upper()
        amt = st.number_input("Adet / Miktar", min_value=0.0, step=0.01)
        cst = st.number_input("Birim Maliyet ($ veya ₺)", min_value=0.0)
        curr = st.radio("Para Birimi", ["USD", "TRY"], horizontal=True)
        
        if st.form_submit_button("Portföye Ekle"):
            current_portfolio = load_data()
            current_portfolio.append({"symbol": sym, "amount": amt, "cost": cst, "currency": curr})
            save_data(current_portfolio)
            st.rerun()
            
    usd_try = st.number_input("USD/TRY Kuru", value=33.10)
    if st.button("🗑️ Tümünü Temizle"):
        save_data([])
        st.rerun()

# -------------------- ANA EKRAN --------------------
portfolio_list = load_data()
st.markdown("<h2 style='text-align: center; color: #3b82f6;'>TRADEMASTER PREMIUM</h2>", unsafe_allow_html=True)

if not portfolio_list:
    st.info("Portföyün henüz boş kanka. Kenar çubuğundan varlık ekleyerek başlayabilirsin! (Örn: BTC-USD)")
else:
    # Verileri Çek
    processed_data = []
    with st.spinner('Piyasa verileri senkronize ediliyor...'):
        for item in portfolio_list:
            stats = get_live_stats(item['symbol'])
            if stats:
                item.update(stats)
                # TRY cinsinden toplam değerleri hesapla
                multiplier = usd_try if item['currency'] == "USD" else 1
                item['val_try'] = item['amount'] * item['price'] * multiplier
                item['cost_try'] = item['amount'] * item['cost'] * multiplier
                processed_data.append(item)

    df = pd.DataFrame(processed_data)
    total_val = df['val_try'].sum()
    total_pnl = total_val - df['cost_try'].sum()

    # ÜST ÖZET
    m1, m2, m3 = st.columns(3)
    m1.markdown(f'<div class="premium-card"><div class="stat-label">Toplam Varlık</div><div class="stat-value">{total_val:,.0f} ₺</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="premium-card"><div class="stat-label">Net Kar/Zarar</div><div class="stat-value" style="color:#10b981;">{total_pnl:+,.0f} ₺</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="premium-card"><div class="stat-label">Aktif Sinyal</div><div class="stat-value" style="font-size:20px; color:#3b82f6;">{len(df)} VARLIK</div></div>', unsafe_allow_html=True)

    # SİNYAL KARTLARI
    st.subheader("⚡ Akıllı Sinyaller")
    sig_cols = st.columns(min(len(df), 4))
    for i, row in df.iterrows():
        action, style = get_signal(row['rsi'])
        with sig_cols[i % 4]:
            st.markdown(f"""
            <div class="sig-card {style}">
                <div style="font-size:11px; opacity:0.8;">{row['symbol']}</div>
                <div style="font-size:16px;">{action}</div>
                <div style="font-size:10px;">RSI: {row['rsi']:.1f}</div>
            </div>
            """, unsafe_allow_html=True)

    # DETAY LİSTESİ
    st.write("")
    c_left, c_right = st.columns([1.5, 1])
    
    with c_left:
        st.subheader("📋 Varlıklarım")
        for i, row in df.iterrows():
            weight = (row['val_try'] / total_val) * 100
            pnl_perc = ((row['price'] - row['cost']) / row['cost']) * 100
            
            with st.container():
                st.markdown(f"""
                <div class="premium-card">
                    <div style="display:flex; justify-content:space-between;">
                        <div><b>{row['symbol']}</b> <span style="font-size:12px; color:#8b949e;">{row['amount']} Adet</span></div>
                        <div style="text-align:right;">
                            <div style="color:{'#10b981' if pnl_perc > 0 else '#ef4444'}; font-weight:700;">%{pnl_perc:+.2f}</div>
                            <div style="font-size:15px;">{row['val_try']:,.0f} ₺</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"🗑️ {row['symbol']} Sil", key=f"del_{i}"):
                    current_data = load_data()
                    current_data.pop(i)
                    save_data(current_data)
                    st.rerun()

    with c_right:
        st.subheader("📊 Dağılım")
        fig = px.pie(df, values='val_try', names='symbol', hole=0.7)
        fig.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(t=0,b=0,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)

st.caption(f"Veriler otomatik kaydedilir. • {datetime.now().strftime('%H:%M:%S')}")
