import streamlit as st
import requests
import pandas as pd

st.title("🚀 Crypto Test")
st.write("Basit veri çekme testi")

# Binance API denemesi
try:
    url = "https://api.binance.com/api/v3/ticker/price"
    response = requests.get(url, timeout=5)
    data = response.json()
    
    # Sadece ilk 10 coin'i göster
    coins = []
    for coin in data[:10]:
        if 'USDT' in coin['symbol']:
            coins.append({
                'Coin': coin['symbol'],
                'Fiyat': f"${float(coin['price']):.2f}"
            })
    
    df = pd.DataFrame(coins)
    st.success("✅ Veri başarıyla çekildi!")
    st.dataframe(df)
    
except Exception as e:
    st.error(f"❌ Hata: {e}")
    st.info("İnternet bağlantını kontrol et!")