import streamlit as st
import yfinance as yf
import pandas as pd

st.title("🔍 yFinance Test")

crypto_symbol = st.text_input("Sembol:", "BTC-USD")
period_days = st.slider("Gün:", 7, 90, 30)
interval_type = st.selectbox("Interval:", ["1h", "4h", "1d"])

if st.button("Test Et"):
    try:
        st.write("📡 Veri çekiliyor...")
        
        # yfinance'dan veri çek
        data = yf.download(crypto_symbol, period=f"{period_days}d", interval=interval_type, progress=False)
        
        st.write("📊 Veri Bilgisi:")
        st.write(f"- DataFrame boyutu: {data.shape}")
        st.write(f"- Boş mu: {data.empty}")
        st.write(f"- Sütunlar: {list(data.columns)}")
        
        if not data.empty:
            st.write("✅ İlk 5 satır:")
            st.dataframe(data.head())
            
            st.write("💰 Son fiyat bilgisi:")
            st.write(f"- Son kapanış: {data['Close'].iloc[-1]}")
            st.write(f"- Veri tipi: {type(data['Close'].iloc[-1])}")
        else:
            st.error("❌ DataFrame boş!")
            
    except Exception as e:
        st.error(f"❌ Hata: {e}")