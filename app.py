import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Profesyonel Mum Analizi", layout="wide")

st.title("🎯 PROFESYONEL MUM GRAFİĞİ ANALİZİ")

# Sidebar
with st.sidebar:
    st.header("⚙️ Ayarlar")
    crypto_symbol = st.text_input("Kripto Sembolü", "BTC-USD")
    days = st.slider("Gün Sayısı", 1, 30, 7)

# Veri çekme
@st.cache_data
def get_crypto_data(symbol, days):
    try:
        data = yf.download(symbol, period=f"{days}d", interval="4h", progress=False)
        return data if not data.empty else None
    except:
        return None

# MUM GRAFİĞİ çizimi - KESİN ÇALIŞAN
def plot_candlestick_chart(data, symbol):
    fig = go.Figure()
    
    # HER MUMU AYRI AYRI ÇİZ
    for i in range(len(data)):
        row = data.iloc[i]
        open_price = row['Open']
        high = row['High']
        low = row['Low']
        close_price = row['Close']
        
        # Renk belirle
        color = 'green' if close_price > open_price else 'red'
        
        # MUM GÖVDESİ (kalın dikdörtgen)
        fig.add_trace(go.Scatter(
            x=[data.index[i], data.index[i]],
            y=[open_price, close_price],
            mode='lines',
            line=dict(color=color, width=12),
            showlegend=False
        ))
        
        # ÜST İĞNE (High)
        fig.add_trace(go.Scatter(
            x=[data.index[i], data.index[i]],
            y=[max(open_price, close_price), high],
            mode='lines',
            line=dict(color=color, width=2),
            showlegend=False
        ))
        
        # ALT İĞNE (Low)
        fig.add_trace(go.Scatter(
            x=[data.index[i], data.index[i]],
            y=[min(open_price, close_price), low],
            mode='lines',
            line=dict(color=color, width=2),
            showlegend=False
        ))
    
    fig.update_layout(
        title=f"{symbol} - {days} Günlük 4 Saatlik Mum Grafiği",
        xaxis_title="Tarih",
        yaxis_title="Fiyat (USD)",
        height=600,
        showlegend=False,
        xaxis_rangeslider_visible=False
    )
    
    return fig

# Basit destek/direnç bulma
def find_support_resistance(data):
    current_price = data['Close'].iloc[-1]
    
    # Son 20 mumun en düşük ve en yüksekleri
    recent_lows = data['Low'].tail(20).nsmallest(3)
    recent_highs = data['High'].tail(20).nlargest(3)
    
    support_levels = [float(low) for low in recent_lows if low < current_price]
    resistance_levels = [float(high) for high in recent_highs if high > current_price]
    
    return support_levels[:3], resistance_levels[:3]

# Ana uygulama
def main():
    # Veri yükleme
    with st.spinner('Veriler yükleniyor...'):
        data = get_crypto_data(crypto_symbol, days)
    
    if data is None or data.empty:
        st.error("❌ Veri çekilemedi! Sembolü kontrol edin.")
        return
    
    st.success(f"✅ {len(data)} mum verisi yüklendi")
    
    # Grafik oluştur
    fig = plot_candlestick_chart(data, crypto_symbol)
    
    # Destek/direnç seviyeleri
    support_levels, resistance_levels = find_support_resistance(data)
    current_price = data['Close'].iloc[-1]
    
    # Destek çizgileri ekle
    for i, level in enumerate(support_levels):
        fig.add_hline(y=level, line_dash="solid", line_color="lime", line_width=2,
                     annotation_text=f"S{i+1}", annotation_position="left")
    
    # Direnç çizgileri ekle
    for i, level in enumerate(resistance_levels):
        fig.add_hline(y=level, line_dash="solid", line_color="red", line_width=2,
                     annotation_text=f"R{i+1}", annotation_position="right")
    
    # Mevcut fiyat çizgisi
    fig.add_hline(y=current_price, line_dash="dot", line_color="yellow", line_width=2,
                 annotation_text=f"Şimdi: ${current_price:,.0f}")
    
    # Grafiği göster
    st.plotly_chart(fig, use_container_width=True)
    
    # Bilgi paneli
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Mevcut Fiyat", f"${current_price:,.0f}")
    
    with col2:
        st.metric("Toplam Mum", len(data))
    
    with col3:
        change = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
        st.metric("Değişim", f"%{change:.1f}")
    
    # Seviyeler
    col4, col5 = st.columns(2)
    
    with col4:
        st.subheader("🟢 Destek Seviyeleri")
        for i, level in enumerate(support_levels):
            st.write(f"S{i+1}: ${level:,.0f}")
    
    with col5:
        st.subheader("🔴 Direnç Seviyeleri")
        for i, level in enumerate(resistance_levels):
            st.write(f"R{i+1}: ${level:,.0f}")
    
    # Açıklama
    st.info("""
    **📊 Mum Grafiği Okuma:**
    - 🟢 **Yeşil Mum:** Kapanış > Açılış (Yükseliş)
    - 🔴 **Kırmızı Mum:** Kapanış < Açılış (Düşüş) 
    - 🟢 **S1,S2,S3:** Destek Seviyeleri
    - 🔴 **R1,R2,R3:** Direnç Seviyeleri
    - 🟡 **Sarı Çizgi:** Mevcut Fiyat
    """)

if __name__ == "__main__":
    main()