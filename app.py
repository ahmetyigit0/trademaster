import streamlit as st
import yfinance as yf
import pandas as pd

st.title("Crypto Test")

try:
    # Daha basit şekilde veri çek
    ticker = yf.Ticker("BTC-USD")
    data = ticker.history(period="1d")
    
    if data.empty:
        st.error("Veri bos geldi")
    else:
        st.success("Veri cekildi")
        st.write("Son BTC Fiyati:", data['Close'].iloc[-1])
        st.dataframe(data)
        
except Exception as e:
    st.error("Hata var")
    st.write(str(e))