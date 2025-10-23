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
sensitivity = st.sidebar.slider("Seviye Hassasiyeti", 0.5, 3.0, 1.0)

def calculate_pivot_points(df, lookback=100):
    """
    Basit pivot point hesaplama ile destek ve direnç seviyeleri
    """
    df = df.tail(lookback).copy()
    
    support_levels = []
    resistance_levels = []
    
    # Yerel maksimum ve minimumları bul
    for i in range(3, len(df)-3):
        high = df['High'].iloc[i]
        low = df['Low'].iloc[i]
        
        # Yerel maksimum kontrolü (direnç)
        if (high > df['High'].iloc[i-1] and 
            high > df['High'].iloc[i-2] and 
            high > df['High'].iloc[i-3] and
            high > df['High'].iloc[i+1] and 
            high > df['High'].iloc[i+2] and
            high > df['High'].iloc[i+3]):
            resistance_levels.append(high)
        
        # Yerel minimum kontrolü (destek)
        if (low < df['Low'].iloc[i-1] and 
            low < df['Low'].iloc[i-2] and 
            low < df['Low'].iloc[i-3] and
            low < df['Low'].iloc[i+1] and 
            low < df['Low'].iloc[i+2] and
            low < df['Low'].iloc[i+3]):
            support_levels.append(low)
    
    return support_levels, resistance_levels

def calculate_fibonacci_levels(df, lookback=100):
    """
    Fibonacci destek ve direnç seviyeleri
    """
    df_lookback = df.tail(lookback)
    high = df_lookback['High'].max()
    low = df_lookback['Low'].min()
    diff = high - low
    
    fib_levels = {
        '0.236': high - diff * 0.236,
        '0.382': high - diff * 0.382,
        '0.5': high - diff * 0.5,
        '0.618': high - diff * 0.618,
        '0.786': high - diff * 0.786
    }
    
    return fib_levels

def group_levels(levels, tolerance_percent=1.0):
    """
    Benzer seviyeleri gruplandırır
    """
    if not levels:
        return []
    
    levels = sorted(levels)
    grouped = []
    current_group = [levels[0]]
    
    for level in levels[1:]:
        # Mevcut grubun ortalamasına göre yüzde farkı kontrol et
        avg_current = np.mean(current_group)
        percent_diff = abs(level - avg_current) / avg_current * 100
        
        if percent_diff <= tolerance_percent:
            current_group.append(level)
        else:
            grouped.append(np.mean(current_group))
            current_group = [level]
    
    if current_group:
        grouped.append(np.mean(current_group))
    
    return grouped

def main():
    try:
        # Veri çekme
        st.write(f"**{crypto_symbol}** için 4 saatlik veriler çekiliyor...")
        
        # 4 saatlik verileri çek
        data = yf.download(crypto_symbol, period="60d", interval="4h", progress=False)
        
        if data.empty:
            st.error("Veri çekilemedi. Lütfen sembolü kontrol edin.")
            return
        
        st.success(f"{len(data)} adet 4 saatlik mum verisi çekildi.")
        
        # Mevcut fiyat
        current_price = data['Close'].iloc[-1]
        
        # Destek ve direnç seviyelerini hesapla
        support_levels, resistance_levels = calculate_pivot_points(data, lookback_period)
        
        # Fibonacci seviyeleri
        fib_levels = calculate_fibonacci_levels(data, lookback_period)
        
        # Seviyeleri gruplandır
        key_support = group_levels(support_levels, sensitivity)
        key_resistance = group_levels(resistance_levels, sensitivity)
        
        # Sadece önemli seviyeleri al (mevcut fiyata yakın olanlar)
        def get_relevant_levels(levels, current_price, percent_range=20):
            relevant = []
            for level in levels:
                percent_diff = abs(level - current_price) / current_price * 100
                if percent_diff <= percent_range:
                    relevant.append(level)
            return sorted(relevant)
        
        relevant_support = get_relevant_levels(key_support, current_price)
        relevant_resistance = get_relevant_levels(key_resistance, current_price)
        
        # Sonuçları göster
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Destek Seviyeleri")
            if relevant_support:
                for i, level in enumerate(relevant_support):
                    distance_pct = ((current_price - level) / current_price) * 100
                    color = "🟢" if level < current_price else "🔴"
                    position = "altında" if level < current_price else "üstünde"
                    st.write(f"{color} **Destek {i+1}:** ${level:.2f} (%{abs(distance_pct):.2f} {position})")
            else:
                st.write("Destek seviyesi bulunamadı")
            
            st.subheader("🔮 Fibonacci Destekleri")
            for level_name, level_value in fib_levels.items():
                if level_value < current_price:
                    distance_pct = ((current_price - level_value) / current_price) * 100
                    st.write(f"📊 **Fib {level_name}:** ${level_value:.2f} (%{abs(distance_pct):.2f} altında)")
        
        with col2:
            st.subheader("📉 Direnç Seviyeleri")
            if relevant_resistance:
                for i, level in enumerate(relevant_resistance):
                    distance_pct = ((level - current_price) / current_price) * 100
                    color = "🔴" if level > current_price else "🟢"
                    position = "üstünde" if level > current_price else "altında"
                    st.write(f"{color} **Direnç {i+1}:** ${level:.2f} (%{abs(distance_pct):.2f} {position})")
            else:
                st.write("Direnç seviyesi bulunamadı")
            
            st.subheader("🔮 Fibonacci Dirençleri")
            for level_name, level_value in fib_levels.items():
                if level_value > current_price:
                    distance_pct = ((level_value - current_price) / current_price) * 100
                    st.write(f"📊 **Fib {level_name}:** ${level_value:.2f} (%{abs(distance_pct):.2f} üstünde)")
        
        # Özet bilgiler
        st.subheader("📋 Özet Bilgiler")
        col3, col4, col5, col6 = st.columns(4)
        
        with col3:
            st.metric("Mevcut Fiyat", f"${current_price:.2f}")
        with col4:
            st.metric("Destek Seviyeleri", len(relevant_support))
        with col5:
            st.metric("Direnç Seviyeleri", len(relevant_resistance))
        with col6:
            change = ((current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100
            st.metric("Son Mum Değişim", f"%{change:.2f}")
        
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
            name='Fiyat'
        ))
        
        # Destek seviyeleri
        for level in relevant_support:
            fig.add_hline(y=level, line_dash="dash", line_color="green", 
                         line_width=1, opacity=0.7,
                         annotation_text=f"D: ${level:.2f}")
        
        # Direnç seviyeleri
        for level in relevant_resistance:
            fig.add_hline(y=level, line_dash="dash", line_color="red", 
                         line_width=1, opacity=0.7,
                         annotation_text=f"R: ${level:.2f}")
        
        # Fibonacci seviyeleri
        for level_name, level_value in fib_levels.items():
            if abs(level_value - current_price) / current_price * 100 <= 15:  # Sadece yakın seviyeler
                fig.add_hline(y=level_value, line_dash="dot", line_color="orange", 
                             line_width=1, opacity=0.5,
                             annotation_text=f"Fib {level_name}")
        
        # Mevcut fiyat çizgisi
        fig.add_hline(y=current_price, line_color="blue", line_width=2,
                     annotation_text=f"Mevcut: ${current_price:.2f}")
        
        fig.update_layout(
            title=f"{crypto_symbol} 4 Saatlik Grafik - Destek & Direnç Seviyeleri",
            xaxis_title="Tarih",
            yaxis_title="Fiyat (USD)",
            height=600,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detaylı analiz
        with st.expander("📊 Detaylı Analiz Verileri"):
            col7, col8 = st.columns(2)
            
            with col7:
                st.write("**Tüm Tespit Edilen Destekler:**")
                if support_levels:
                    for level in sorted(support_levels)[-10:]:
                        st.write(f"${level:.2f}")
                else:
                    st.write("Bulunamadı")
            
            with col8:
                st.write("**Tüm Tespit Edilen Dirençler:**")
                if resistance_levels:
                    for level in sorted(resistance_levels)[-10:]:
                        st.write(f"${level:.2f}")
                else:
                    st.write("Bulunamadı")
            
            st.write("**Son 5 Mum:**")
            st.dataframe(data.tail(5)[['Open', 'High', 'Low', 'Close', 'Volume']].round(2))
            
    except Exception as e:
        st.error(f"Hata oluştu: {str(e)}")
        st.info("""
        **Olası Çözümler:**
        - Sembol formatını kontrol edin (BTC-USD, ETH-USD vb.)
        - İnternet bağlantınızı kontrol edin
        - Daha kısa bir periyot deneyin
        - Farklı bir kripto sembolü deneyin
        """)

if __name__ == "__main__":
    main()