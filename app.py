import streamlit as st
import yfinance as yf
import pandas as pd

st.title("Crypto Test")
st.write("Yahoo Finance veri çekme testi")

try:
    # BTC verisini çek
    data = yf.download("BTC-USD", period="1d", progress=False)
    
    if data.empty:
        st.error("Veri boş geldi!")
    else:
        st.success(f"Veri çekildi! {len(data)} kayıt")
        
        # Son veriyi göster
        st.subheader("BTC Fiyatları")
        st.dataframe(data.tail())
        
        # Basit fiyat gösterimi
        current_price = data['Close'].iloc[-1]
        st.write(f"Son Fiyat: ${current_price:.2f}")
        
except Exception as e:
    st.error(f"Hata: {str(e)}")