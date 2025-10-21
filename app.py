import streamlit as st
import yfinance as yf
import pandas as pd

st.title("🚀 Crypto Test")
st.write("Yahoo Finance veri çekme testi")

# Yahoo Finance denemesi
try:
    # BTC verisini çek
    data = yf.download("BTC-USD", period="1d", interval="1h", progress=False)
    
    if data.empty:
        st.error("❌ Veri boş geldi!")
    else:
        st.success(f"✅ Veri çekildi! {len(data)} kayıt")
        
        # Son 5 saatlik veriyi göster
        st.subheader("Son 5 Saatlik BTC Fiyatları")
        latest_data = data.tail()[['Open', 'High', 'Low', 'Close', 'Volume']]
        st.dataframe(latest_data)
        
        # Mevcut fiyat
        current_price = data['Close'].iloc[-1]
        st.metric("💰 Mevcut BTC Fiyatı", f"${current_price:.2f}")
        
except Exception as e:
    st.error(f"❌ Hata: {e}")
    st.info("İnternet bağlantını kontrol et!")