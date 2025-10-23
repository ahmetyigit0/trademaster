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
        current_high = df['High'].iloc[i]
        current_low = df['Low'].iloc[i]
        
        # Direnç seviyeleri (tepe noktaları)
        if (current_high > df['High'].iloc[i-1] and 
            current_high > df['High'].iloc[i-2] and 
            current_high > df['High'].iloc[i+1] and 
            current_high > df['High'].iloc[i+2]):
            resistance_levels.append(current_high)
        
        # Destek seviyeleri (dip noktaları)
        if (current_low < df['Low'].iloc[i-1] and 
            current_low < df['Low'].iloc[i-2] and 
            current_low < df['Low'].iloc[i+1] and 
            current_low < df['Low'].iloc[i+2]):
            support_levels.append(current_low)
    
    # İğne (Pinbar) formasyonu tespiti
    pinbar_support = []
    pinbar_resistance = []
    
    for i in range(len(df)):
        open_price = df['Open'].iloc[i]
        close_price = df['Close'].iloc[i]
        high_price = df['High'].iloc[i]
        low_price = df['Low'].iloc[i]
        
        body_size = abs(close_price - open_price)
        total_range = high_price - low_price
        
        if total_range > 0:
            # Bullish pinbar (destek sinyali)
            upper_shadow = high_price - max(open_price, close_price)
            lower_shadow = min(open_price, close_price) - low_price
            
            if lower_shadow > total_range * pinbar_sensitivity and upper_shadow < lower_shadow * 0.5:
                pinbar_support.append(low_price)
            
            # Bearish pinbar (direnç sinyali)
            if upper_shadow > total_range * pinbar_sensitivity and lower_shadow < upper_shadow * 0.5:
                pinbar_resistance.append(high_price)
    
    return support_levels, resistance_levels, pinbar_support, pinbar_resistance

def filter_close_levels(levels, threshold_percent=1.0):
    """
    Birbirine çok yakın seviyeleri filtreler
    """
    if not levels:
        return []
    
    # Seviyeleri sırala ve benzersiz yap
    levels = sorted(set(levels))
    filtered = [levels[0]]
    
    for level in levels[1:]:
        # Yüzde farkı hesapla
        if abs(level - filtered[-1]) / filtered[-1] * 100 > threshold_percent:
            filtered.append(level)
    
    return filtered

def group_support_resistance(levels, tolerance_percent=1.0):
    """
    Benzer seviyeleri gruplandırır
    """
    if not levels:
        return []
    
    levels = sorted(levels)
    groups = []
    current_group = [levels[0]]
    
    for level in levels[1:]:
        # Eğer seviye mevcut gruba yakınsa, gruba ekle
        if abs(level - np.mean(current_group)) / np.mean(current_group) * 100 <= tolerance_percent:
            current_group.append(level)
        else:
            # Yeni grup başlat
            groups.append(np.mean(current_group))
            current_group = [level]
    
    if current_group:
        groups.append(np.mean(current_group))
    
    return groups

def main():
    try:
        # Veri çekme
        st.write(f"**{crypto_symbol}** için 4 saatlik veriler çekiliyor...")
        
        # 4 saatlik verileri çek (daha uzun periyot vererek daha fazla 4h mumu alabiliriz)
        data = yf.download(crypto_symbol, period="60d", interval="4h")
        
        if data.empty:
            st.error("Veri çekilemedi. Lütfen sembolü kontrol edin.")
            return
        
        st.success(f"{len(data)} adet 4 saatlik mum verisi çekildi.")
        
        # Hesaplamalar
        support, resistance, pinbar_support, pinbar_resistance = calculate_support_resistance(
            data, lookback_period, pinbar_sensitivity
        )
        
        # Seviyeleri gruplandır ve filtrele
        key_support = group_support_resistance(support)[-5:]  # Son 5 önemli destek
        key_resistance = group_support_resistance(resistance)[-5:]  # Son 5 önemli direnç
        key_pinbar_support = group_support_resistance(pinbar_support)[-3:]  # Son 3 pinbar destek
        key_pinbar_resistance = group_support_resistance(pinbar_resistance)[-3:]  # Son 3 pinbar direnç
        
        # Mevcut fiyat
        current_price = data['Close'].iloc[-1]
        
        # Sonuçları göster
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Destek Seviyeleri")
            if key_support:
                for i, level in enumerate(reversed(key_support)):
                    distance_pct = ((current_price - level) / current_price) * 100
                    color = "🟢" if level < current_price else "🔴"
                    position = "altında" if level < current_price else "üstünde"
                    st.write(f"{color} **Destek {i+1}:** ${level:.2f} (%{abs(distance_pct):.2f} {position})")
            else:
                st.write("Destek seviyesi bulunamadı")
            
            st.subheader("📊 İğne (Pinbar) Destekleri")
            if key_pinbar_support:
                for i, level in enumerate(reversed(key_pinbar_support)):
                    distance_pct = ((current_price - level) / current_price) * 100
                    st.write(f"📍 **Pinbar D {i+1}:** ${level:.2f} (%{abs(distance_pct):.2f} {('altında' if level < current_price else 'üstünde')})")
            else:
                st.write("İğne destek seviyesi bulunamadı")
        
        with col2:
            st.subheader("📉 Direnç Seviyeleri")
            if key_resistance:
                for i, level in enumerate(key_resistance):
                    distance_pct = ((level - current_price) / current_price) * 100
                    color = "🔴" if level > current_price else "🟢"
                    position = "üstünde" if level > current_price else "altında"
                    st.write(f"{color} **Direnç {i+1}:** ${level:.2f} (%{abs(distance_pct):.2f} {position})")
            else:
                st.write("Direnç seviyesi bulunamadı")
            
            st.subheader("📊 İğne (Pinbar) Dirençleri")
            if key_pinbar_resistance:
                for i, level in enumerate(key_pinbar_resistance):
                    distance_pct = ((level - current_price) / current_price) * 100
                    st.write(f"📍 **Pinbar R {i+1}:** ${level:.2f} (%{abs(distance_pct):.2f} {('üstünde' if level > current_price else 'altında')})")
            else:
                st.write("İğne direnç seviyesi bulunamadı")
        
        # Özet bilgiler
        st.subheader("📋 Özet Bilgiler")
        col3, col4, col5 = st.columns(3)
        
        with col3:
            st.metric("Mevcut Fiyat", f"${current_price:.2f}")
        with col4:
            st.metric("Toplam Destek Seviyesi", len(key_support))
        with col5:
            st.metric("Toplam Direnç Seviyesi", len(key_resistance))
        
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
                         annotation_text=f"D: ${level:.2f}")
        
        # Direnç seviyeleri
        for level in key_resistance:
            fig.add_hline(y=level, line_dash="dash", line_color="red", 
                         annotation_text=f"R: ${level:.2f}")
        
        # İğne seviyeleri
        for level in key_pinbar_support:
            fig.add_hline(y=level, line_dash="dot", line_color="lightgreen", 
                         annotation_text=f"PD: ${level:.2f}")
        
        for level in key_pinbar_resistance:
            fig.add_hline(y=level, line_dash="dot", line_color="lightcoral", 
                         annotation_text=f"PR: ${level:.2f}")
        
        # Mevcut fiyat çizgisi
        fig.add_hline(y=current_price, line_color="blue", 
                     annotation_text=f"Şimdi: ${current_price:.2f}")
        
        fig.update_layout(
            title=f"{crypto_symbol} 4 Saatlik Grafik - Destek & Direnç Seviyeleri",
            xaxis_title="Tarih",
            yaxis_title="Fiyat (USD)",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Veri önizleme
        with st.expander("Son 10 Mum Verisi"):
            st.dataframe(data.tail(10)[['Open', 'High', 'Low', 'Close', 'Volume']])
            
    except Exception as e:
        st.error(f"Hata oluştu: {str(e)}")
        st.info("Lütfen sembol formatını kontrol edin (Örn: BTC-USD) ve internet bağlantınızı kontrol edin.")

if __name__ == "__main__":
    main()