import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import numpy as np

st.set_page_config(page_title="Altcoin Analiz", layout="wide")

@st.cache_data(ttl=300)
def fetch_data(coin="ethereum", days=180):
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days={days}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df['open'] = df['price'].shift(1)
        df['high'] = df['price'].rolling(2).max()
        df['low'] = df['price'].rolling(2).min()
        df = df.dropna()
        return df
    except:
        return None

def calculate_indicators(df):
    df['ema12'] = df['price'].ewm(span=12).mean()
    df['ema26'] = df['price'].ewm(span=26).mean()
    
    delta = df['price'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df['bb_mid'] = df['price'].rolling(20).mean()
    df['bb_std'] = df['price'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * 2)
    
    return df

def signals(df):
    sig = []
    if df['ema12'].iloc[-1] > df['ema26'].iloc[-1]:
        sig.append("🟢 EMA: AL")
    else:
        sig.append("🔴 EMA: SAT")
        
    rsi = df['rsi'].iloc[-1]
    if rsi < 30: sig.append("🟢 RSI: AL")
    elif rsi > 70: sig.append("🔴 RSI: SAT")
    
    price = df['price'].iloc[-1]
    if price < df['bb_lower'].iloc[-1]: sig.append("🟢 BB: AL")
    elif price > df['bb_upper'].iloc[-1]: sig.append("🔴 BB: SAT")
    
    return sig

def plot_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['price'], name='Fiyat', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['ema12'], name='EMA12', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df.index, y=df['ema26'], name='EMA26', line=dict(color='red')))
    fig.update_layout(title='Altcoin Grafik', height=400)
    return fig

# UYGULAMA
st.title("🚀 Altcoin Analiz")

col1, col2 = st.columns(2)
with col1:
    coin = st.text_input("Coin ID", "ethereum")
with col2:
    days = st.slider("Gün", 30, 365, 180)

if st.button("📊 ANALİZ ET"):
    with st.spinner("Veri yükleniyor..."):
        df = fetch_data(coin, days)
        if df is not None:
            df = calculate_indicators(df)
            signals_list = signals(df)
            
            # Grafik
            fig = plot_chart(df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Sinyaller
            st.subheader("📈 SİNYALLER")
            for s in signals_list:
                st.success(s)
                
            # Bilgi
            st.info(f"Fiyat: ${df['price'].iloc[-1]:.2f}")
