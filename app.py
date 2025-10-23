import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Streamlit arayüzü
st.set_page_config(page_title="Kripto Destek & Direnç Analizi", layout="wide")
st.title("Kripto Destek ve Direnç Seviyeleri Analizi")

# Sidebar
st.sidebar.header("Kripto Sembolü Girin")
crypto_symbol = st.sidebar.text_input("Kripto Sembolü (Örn: BTC-USD, ETH-USD, ADA-USD):", "BTC-USD")

# Analiz parametreleri
lookback_period = st.sidebar.slider("Mum Sayısı (Max 100)", 50, 100, 100)
pinbar_sensitivity = st.sidebar.slider("İğne (Pinbar) Hassasiyeti", 0.1, 0.5, 0.3)

def calculate_support_resistance(df, lookback=100, pinbar_sensitivity=0.3):
    """
    Destek ve direnç seviyelerini hesaplar
    """
    # Son lookback_period kadar mumu al
    df = df.tail(lookback).copy()
    
    support_levels = []
    resistance_levels = []
    
    # Yüksek ve düşük seviyelerdeki potansiyel direnç/destek noktaları
    for i in range(2, len(df)-2):
        # Direnç seviyeleri (tepe noktaları)
        if (df['High'].iloc[i] > df['High'].iloc[i-1] and 
            df['High'].iloc[i] > df['High'].iloc[i-2] and 
            df['High'].iloc[i] > df['High'].iloc[i+1] and 
            df['High'].iloc[i] > df['High'].iloc[i+2]):
            resistance_levels.append(df['High'].iloc[i])
        
        # Destek seviyeleri (dip noktaları)
        if (df['Low'].iloc[i] < df['Low'].iloc[i-1] and 
            df['Low'].iloc[i] < df['Low'].iloc[i-2] and 
            df['Low'].iloc[i] < df['Low'].iloc[i+1] and 
            df['Low'].iloc[i] < df['Low'].iloc[i+2]):
            support_levels.append(df['Low'].iloc[i])
    
    # İğne (Pinbar) formasyonu tespiti
    pinbar_support = []
    pinbar_resistance = []
    
    for i in range(1, len(df)):
        body_size = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
        total_range = df['High'].iloc[i] - df['Low'].iloc[i]
        
        if total_range > 0:
            # Bullish pinbar (destek sinyali)
            upper_shadow = df['High'].iloc[i] - max(df['Open'].iloc[i], df['Close'].iloc[i])
            lower_shadow = min(df['Open'].iloc[i], df['Close'].iloc[i]) - df['Low'].iloc[i]
            
            if lower_shadow > total_range * pinbar_sensitivity and upper_shadow < lower_shadow * 0.5:
                pinbar_support.append(df['Low'].iloc[i])
            
            # Bearish pinbar (direnç sinyali)
            if upper_shadow > total_range * pinbar_sensitivity and lower_shadow < upper_shadow * 0.5:
                pinbar_resistance.append(df['High'].iloc[i])
    
    return support_levels, resistance_levels, pinbar_support, pinbar_resistance

def filter_close_levels(levels, threshold_percent=1.0):
    """
    Birbirine çok yakın seviyeleri filtreler
    """
    if not levels:
        return []
    
    levels.sort()
    filtered = [levels[0]]
    
    for level in levels[1:]:
        if abs(level - filtered[-1]) / filtered[-1] * 100 > threshold_percent:
            filtered.append(level)
    
    return filtered

def main():
    try:
        # Veri çekme
        st.write(f"**{crypto_symbol}** için 4 saatlik veriler çekiliyor...")
        
        # 4 saatlik verileri çek (100 mum + buffer)
        data = yf.download(crypto_symbol, period="60d", interval="4h")
        
        if data.empty:
            st.error("Veri çekilemedi. Lütfen sembolü kontrol edin.")
            return
        
        # Hesaplamalar
        support, resistance, pinbar_support, pinbar_resistance = calculate_support_resistance(
            data, lookback_period, pinbar_sensitivity
        )
        
        # Seviyeleri filtrele
        key_support = filter_close_levels(support)[-5:]  # Son 5 önemli destek
        key_resistance = filter_close_levels(resistance)[-5:]  # Son 5 önemli direnç
        
        # Mevcut fiyat
        current_price = data['Close'].iloc[-1]
        
        # Sonuçları göster
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Destek Seviyeleri")
            if key_support:
                for i, level in enumerate(reversed(key_support)):
                    distance_pct = ((current_price - level) / current_price) * 100
                    if level < current_price:
                        st.write(f"🟢 **Destek {i+1}:** ${level:.2f} (%{distance_pct:.2f} altında)")
                    else:
                        st.write(f"🔴 **Destek {i+1}:** ${level:.2f} (%{abs(distance_pct):.2f} üstünde)")
            else:
                st.write("Destek seviyesi bulunamadı")
            
            st.subheader("📊 İğne (Pinbar) Destekleri")
            if pinbar_support:
                for level in pinbar_support[-3:]:
                    st.write(f"📍 ${level:.2f}")
            else:
                st.write("İğne destek seviyesi bulunamadı")
        
        with col2:
            st.subheader("📉 Direnç Seviyeleri")
            if key_resistance:
                for i, level in enumerate(key_resistance):
                    distance_pct = ((level - current_price) / current_price) * 100
                    if level > current_price:
                        st.write(f"🔴 **Direnç {i+1}:** ${level:.2f} (%{distance_pct:.2f} üstünde)")
                    else:
                        st.write(f"🟢 **Direnç {i+1}:** ${level:.2f} (%{abs(distance_pct):.2f} altında)")
            else:
                st.write("Direnç seviyesi bulunamadı")
            
            st.subheader("📊 İğne (Pinbar) Dirençleri")
            if pinbar_resistance:
                for level in pinbar_resistance[-3:]:
                    st.write(f"📍 ${level:.2f}")
            else:
                st.write("İğne direnç seviyesi bulunamadı")
        
        # Grafik
        st.subheader("🎯 Fiyat Grafiği ve Seviyeler")
        
        fig = go.Figure()
        
        # Mum grafiği
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ))
        
        # Destek seviyeleri
        for level in key_support:
            fig.add_hline(y=level, line_dash="dash", line_color="green", 
                         annotation_text=f"Destek: ${level:.2f}")
        
        # Direnç seviyeleri
        for level in key_resistance:
            fig.add_hline(y=level, line_dash="dash", line_color="red", 
                         annotation_text=f"Direnç: ${level:.2f}")
        
        # İğne seviyeleri
        for level in pinbar_support[-3:]:
            fig.add_hline(y=level, line_dash="dot", line_color="lightgreen", 
                         annotation_text=f"Pinbar D: ${level:.2f}")
        
        for level in pinbar_resistance[-3:]:
            fig.add_hline(y=level, line_dash="dot", line_color="lightcoral", 
                         annotation_text=f"Pinbar R: ${level:.2f}")
        
        # Mevcut fiyat çizgisi
        fig.add_hline(y=current_price, line_color="blue", 
                     annotation_text=f"Mevcut: ${current_price:.2f}")
        
        fig.update_layout(
            title=f"{crypto_symbol} 4 Saatlik Grafik - Destek & Direnç Seviyeleri",
            xaxis_title="Tarih",
            yaxis_title="Fiyat (USD)",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Veri önizleme
        with st.expander("Son 10 Mum Verisi"):
            st.dataframe(data.tail(10))
            
    except Exception as e:
        st.error(f"Hata oluştu: {str(e)}")
        st.info("Lütfen sembol formatını kontrol edin (Örn: BTC-USD)")

if __name__ == "__main__":
    main()