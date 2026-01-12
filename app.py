import streamlit as st
import pandas as pd
import yfinance as yf
import json
import os
from datetime import datetime

# --- AYARLAR ---
DATA_FILE = "smart_portfolio.json"
st.set_page_config(page_title="TradeMaster Pro", layout="wide")

# --- VERİ FONKSİYONLARI ---
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

# --- CSS (Şık Arayüz) ---
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    .asset-card { background: #1f2937; padding: 15px; border-radius: 12px; margin-bottom: 10px; border-left: 5px solid #3b82f6; }
</style>
""", unsafe_allow_html=True)

# --- ANA PANEL ---
st.title("🛡️ Akıllı Portföy Yönetimi")

# 1. AKILLI ARAMA VE EKLEME BÖLÜMÜ
with st.expander("➕ Yeni Varlık Ekle", expanded=True):
    search_query = st.text_input("Hisse veya Kripto Ara (Örn: THY, Apple, Bitcoin, Altın)")
    
    if search_query:
        # yfinance üzerinden akıllı arama yapıyoruz
        search_results = yf.Search(search_query, max_results=5).quotes
        
        if not search_results:
            st.error("Böyle bir hisse/varlık bulunamadı kanka. Farklı bir isim dene.")
        else:
            # Bulunan sonuçları bir listeye koyuyoruz
            options = {f"{res['shortname']} ({res['symbol']})": res['symbol'] for res in search_results}
            selected_display = st.selectbox("Sonuçlar arasından seç:", list(options.keys()))
            ticker_symbol = options[selected_display]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                amount = st.number_input("Adet", min_value=0.01, step=0.01)
            with col2:
                cost = st.number_input("Birim Alım Fiyatı", min_value=0.0)
            with col3:
                buy_date = st.date_input("Alım Tarihi", datetime.now())
            
            if st.button("Portföye Kaydet"):
                new_entry = {
                    "symbol": ticker_symbol,
                    "name": selected_display.split(' (')[0],
                    "amount": amount,
                    "cost": cost,
                    "date": str(buy_date)
                }
                current_portfolio = load_data()
                current_portfolio.append(new_entry)
                save_data(current_portfolio)
                st.success(f"{ticker_symbol} portföyüne başarıyla eklendi!")
                st.rerun()

# 2. PORTFÖY LİSTESİ VE CANLI VERİ
st.divider()
st.subheader("📋 Mevcut Varlıklarım")

portfolio_data = load_data()

if not portfolio_data:
    st.info("Henüz portföyünde varlık yok. Yukarıdan ekleme yapabilirsin.")
else:
    for i, item in enumerate(portfolio_data):
        # Canlı fiyatı çek
        try:
            live_price = yf.Ticker(item['symbol']).history(period="1d")['Close'].iloc[-1]
            total_value = live_price * item['amount']
            pnl = (live_price - item['cost']) * item['amount']
            pnl_perc = ((live_price - item['cost']) / item['cost'] * 100) if item['cost'] > 0 else 0
            
            with st.container():
                st.markdown(f"""
                <div class="asset-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span style="font-size:18px; font-weight:bold;">{item['name']}</span> 
                            <span style="color:#8b949e;">({item['symbol']})</span><br>
                            <small>Alım Tarihi: {item['date']}</small>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-size:16px; font-weight:bold;">{total_value:,.2f} USD/TRY</div>
                            <div style="color:{'#10b981' if pnl >= 0 else '#ef4444'}; font-size:14px;">
                                {pnl:+,.2f} ({pnl_perc:+.2f}%)
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Silme butonu
                if st.button(f"🗑️ {item['symbol']} Sil", key=f"del_{i}"):
                    portfolio_data.pop(i)
                    save_data(portfolio_data)
                    st.rerun()
        except:
            st.warning(f"{item['symbol']} için veri çekilemedi.")

st.caption("Veriler JSON dosyasında saklanır, sayfayı kapatsan da silinmez.")
