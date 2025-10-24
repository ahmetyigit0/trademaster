import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Mum Grafiği", layout="wide")

st.title("📊 BASİT MUM GRAFİĞİ")

# Sidebar
with st.sidebar:
    crypto_symbol = st.text_input("Sembol", "BTC-USD")
    if st.button("Grafik Göster"):
        st.session_state.show_chart = True

# BASİT MUM GRAFİĞİ - KESİN ÇALIŞIR
def create_simple_candlestick():
    # Örnek veri oluştur
    dates = pd.date_range(start='2024-01-20', periods=20, freq='4h')
    
    data = []
    base_price = 45000
    
    for i in range(20):
        # Rastgele fiyat hareketi
        change = np.random.uniform(-0.03, 0.03)
        base_price = base_price * (1 + change)
        
        open_price = base_price
        close_price = base_price * (1 + np.random.uniform(-0.02, 0.02))
        
        # High ve Low'u belirle
        high = max(open_price, close_price) + np.random.uniform(100, 500)
        low = min(open_price, close_price) - np.random.uniform(100, 500)
        
        data.append({
            'Date': dates[i],
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close_price
        })
    
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    return df

def plot_candlestick(data):
    fig = go.Figure()
    
    # HER MUM İÇİN AYRI AYRI ÇİZ
    for i in range(len(data)):
        row = data.iloc[i]
        open_price = row['Open']
        high = row['High']
        low = row['Low']
        close_price = row['Close']
        
        # Renk belirle: Kapanış > Açılış ise yeşil, değilse kırmızı
        color = 'green' if close_price > open_price else 'red'
        
        # DİKDORTGEN MUM GÖVDESİ
        fig.add_trace(go.Scatter(
            x=[data.index[i], data.index[i]],
            y=[open_price, close_price],
            mode='lines',
            line=dict(color=color, width=10),  # KALIN çizgi
            showlegend=False
        ))
        
        # ÜST İĞNE (High)
        fig.add_trace(go.Scatter(
            x=[data.index[i], data.index[i]],
            y=[close_price if close_price > open_price else open_price, high],
            mode='lines',
            line=dict(color=color, width=2),
            showlegend=False
        ))
        
        # ALT İĞNE (Low)
        fig.add_trace(go.Scatter(
            x=[data.index[i], data.index[i]],
            y=[open_price if close_price > open_price else close_price, low],
            mode='lines',
            line=dict(color=color, width=2),
            showlegend=False
        ))
    
    fig.update_layout(
        title="BASİT MUM GRAFİĞİ - KESİN GÖRÜNÜR",
        xaxis_title="Zaman",
        yaxis_title="Fiyat",
        height=500
    )
    
    return fig

# Ana uygulama
def main():
    st.write("**Mum Grafiği Nasıl Olmalı:**")
    st.write("- 🟢 **Yeşil Mum:** Kapanış > Açılış (Yükseliş)")
    st.write("- 🔴 **Kırmızı Mum:** Kapanış < Açılış (Düşüş)")
    st.write("- 📏 **Dikdörtgen Gövde:** Açılış-Kapanış arası")
    st.write("- 📍 **İğneler:** High-Low arası")
    
    if st.button("MUM GRAFİĞİNİ GÖSTER"):
        # Veri oluştur
        data = create_simple_candlestick()
        
        # Grafik oluştur
        fig = plot_candlestick(data)
        
        # Göster
        st.plotly_chart(fig, use_container_width=True)
        
        # Veriyi göster
        st.subheader("Mum Verileri")
        st.dataframe(data.tail(10))
        
        st.success("✅ İŞTE BU! MUM GRAFİĞİ GÖRÜNÜYOR!")

if __name__ == "__main__":
    main()