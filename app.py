import streamlit as st
import pandas as pd
import yfinance as yf
import json
import os
import plotly.express as px
from datetime import datetime

# --- AYARLAR ---
DATA_FILE = "premium_portfolio.json"
st.set_page_config(page_title="TradeMaster Ultra", layout="wide")

# --- VERİ YÖNETİMİ ---
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

# --- CSS (PREMIUM DARK MODE) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    html, body, [data-testid="stAppViewContainer"] { font-family: 'Inter', sans-serif; background-color: #0d1117; color: white; }
    .metric-box { background: #161b22; padding: 20px; border-radius: 15px; border: 1px solid #30363d; text-align: center; }
    .asset-card { background: #1c2128; padding: 20px; border-radius: 15px; margin-bottom: 15px; border: 1px solid #30363d; transition: 0.3s; }
    .asset-card:hover { border-color: #58a6ff; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR (VARLIK EKLEME) ---
with st.sidebar:
    st.title("➕ Varlık Ekle")
    st.markdown("Hisse, Kripto veya Emtia ara:")
    search_query = st.text_input("Arama (Örn: THY, BTC, Altın, Tesla)", key="search_box")
    
    if search_query:
        search_results = yf.Search(search_query, max_results=5).quotes
        if search_results:
            options = {f"{res.get('shortname', 'Bilinmeyen')} ({res['symbol']})": res['symbol'] for res in search_results}
            selected_display = st.selectbox("Sonuç Seç:", list(options.keys()))
            ticker_symbol = options[selected_display]
            
            with st.form("add_form", clear_on_submit=True):
                amount = st.number_input("Adet / Miktar", min_value=0.0, step=0.01)
                cost = st.number_input("Birim Maliyet", min_value=0.0)
                buy_date = st.date_input("Alım Tarihi", datetime.now())
                
                if st.form_submit_button("Portföye Ekle"):
                    current_portfolio = load_data()
                    current_portfolio.append({
                        "symbol": ticker_symbol,
                        "name": selected_display.split(' (')[0],
                        "amount": amount,
                        "cost": cost,
                        "date": str(buy_date)
                    })
                    save_data(current_portfolio)
                    st.success("Eklendi! Sayfa güncelleniyor...")
                    st.rerun()
        else:
            st.error("Sonuç bulunamadı.")

    st.divider()
    if st.button("🔴 Tüm Portföyü Sıfırla"):
        save_data([])
        st.rerun()

# --- ANA SAYFA ---
st.markdown("<h2 style='text-align: center; color: #58a6ff;'>📊 TRADEMASTER PREMIUM</h2>", unsafe_allow_html=True)
portfolio_list = load_data()

if not portfolio_list:
    st.info("Kanka yan taraftan varlık ekle de portföyünü analiz edelim! 🔥")
else:
    # Verileri Hazırla ve Canlı Fiyat Çek
    processed_data = []
    with st.spinner('Piyasa verileri canlı çekiliyor...'):
        for item in portfolio_list:
            try:
                ticker = yf.Ticker(item['symbol'])
                live_price = ticker.history(period="1d")['Close'].iloc[-1]
                
                total_value = live_price * item['amount']
                total_cost = item['cost'] * item['amount']
                pnl = total_value - total_cost
                pnl_perc = (pnl / total_cost * 100) if total_cost > 0 else 0
                
                item.update({
                    "live_price": live_price,
                    "total_value": total_value,
                    "pnl": pnl,
                    "pnl_perc": pnl_perc
                })
                processed_data.append(item)
            except:
                st.warning(f"{item['symbol']} için veri alınamadı.")

    df = pd.DataFrame(processed_data)
    
    # --- ÜST ÖZET METRİKLER ---
    total_port_val = df['total_value'].sum()
    total_port_pnl = df['pnl'].sum()
    total_pnl_perc = (total_port_pnl / (total_port_val - total_port_pnl) * 100) if (total_port_val - total_port_pnl) > 0 else 0

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="metric-box"><small>TOPLAM PORTFÖY DEĞERİ</small><br><span style="font-size:24px; font-weight:bold;">{total_port_val:,.2f}</span></div>', unsafe_allow_html=True)
    with m2:
        color = "#238636" if total_port_pnl >= 0 else "#da3633"
        st.markdown(f'<div class="metric-box"><small>TOPLAM KAR/ZARAR</small><br><span style="font-size:24px; font-weight:bold; color:{color};">{total_port_pnl:+,.2f} (%{total_pnl_perc:+.2f})</span></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-box"><small>VARLIK SAYISI</small><br><span style="font-size:24px; font-weight:bold;">{len(df)}</span></div>', unsafe_allow_html=True)

    # --- GRAFİKLER (DAĞILIM) ---
    st.write("")
    col_chart, col_list = st.columns([1, 1.2])

    with col_chart:
        st.subheader("🥧 Portföy Dağılımı")
        fig = px.pie(df, values='total_value', names='symbol', hole=0.6, color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(showlegend=True, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    with col_list:
        st.subheader("📋 Varlık Detayları")
        for i, row in df.iterrows():
            color = "#34d399" if row['pnl'] >= 0 else "#f87171"
            st.markdown(f"""
            <div class="asset-card">
                <div style="display:flex; justify-content:space-between;">
                    <div>
                        <span style="font-size:18px; font-weight:bold;">{row['name']}</span> <small>({row['symbol']})</small><br>
                        <small style="color:#8b949e;">{row['amount']} Adet x {row['live_price']:.2f}</small>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:18px; font-weight:bold;">{row['total_value']:,.2f}</div>
                        <div style="color:{color}; font-weight:bold;">{row['pnl']:+,.2f} (%{row['pnl_perc']:+.2f})</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"🗑️ {row['symbol']} Sil", key=f"del_{i}"):
                portfolio_list.pop(i)
                save_data(portfolio_list)
                st.rerun()

st.caption(f"Veriler Yahoo Finance üzerinden canlı çekilmektedir. Son Güncelleme: {datetime.now().strftime('%H:%M:%S')}")
