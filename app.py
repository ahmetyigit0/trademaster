import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Kripto Teknik Analiz", layout="wide")
st.title("🎯 Kripto Teknik Analiz")

# Sidebar
crypto_symbol = st.sidebar.text_input("Kripto Sembolü:", "BTC-USD")
lookback_days = st.sidebar.slider("Gün Sayısı", 30, 365, 90)
analysis_type = st.sidebar.selectbox("Analiz Türü", ["4 Saatlik", "1 Günlük", "1 Saatlik"])

interval_map = {"4 Saatlik": "4h", "1 Günlük": "1d", "1 Saatlik": "1h"}

def get_crypto_data(symbol, days, interval):
    try:
        data = yf.download(symbol, period=f"{days}d", interval=interval, progress=False)
        return data
    except Exception as e:
        st.error(f"Veri çekilemedi: {e}")
        return None

def calculate_technical_indicators(data):
    """Teknik göstergeleri hesapla"""
    df = data.copy()
    
    # Moving Average'lar
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI Hesaplama
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['RSI'] = calculate_rsi(df['Close'])
    
    # MACD Hesaplama
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = calculate_macd(df['Close'])
    
    # Bollinger Bands - DÜZELTİLDİ
    def calculate_bollinger_bands(prices, window=20, num_std=2):
        middle_band = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        return upper_band, lower_band
    
    # DÜZELTME: Tek tek atama yap
    bb_upper, bb_lower = calculate_bollinger_bands(df['Close'])
    df['BB_Upper'] = bb_upper
    df['BB_Lower'] = bb_lower
    
    return df

def identify_candlestick_patterns(data):
    """Mum formasyonlarını tespit et"""
    df = data.copy()
    patterns = []
    
    if len(df) < 2:
        return patterns
    
    curr_open, curr_high, curr_low, curr_close = float(df['Open'].iloc[-1]), float(df['High'].iloc[-1]), float(df['Low'].iloc[-1]), float(df['Close'].iloc[-1])
    prev_open, prev_high, prev_low, prev_close = float(df['Open'].iloc[-2]), float(df['High'].iloc[-2]), float(df['Low'].iloc[-2]), float(df['Close'].iloc[-2])
    
    # Doji
    body_size = abs(curr_close - curr_open)
    total_range = curr_high - curr_low
    if total_range > 0 and (body_size / total_range) < 0.1:
        patterns.append("DOJI - Kararsızlık sinyali")
    
    # Bullish Engulfing
    if (prev_close < prev_open and curr_close > curr_open and 
        curr_open < prev_close and curr_close > prev_open):
        patterns.append("BULLISH ENGULFING - Güçlü yükseliş sinyali")
    
    return patterns

def calculate_support_resistance(data, window=10):
    """Destek ve direnç seviyelerini hesapla"""
    highs = data['High'].astype(float).values
    lows = data['Low'].astype(float).values
    
    support_levels = []
    resistance_levels = []
    
    for i in range(window, len(data)-window):
        current_high = float(highs[i])
        current_low = float(lows[i])
        
        # Direnç
        is_resistance = True
        for j in range(1, window+1):
            if current_high <= float(highs[i-j]) or current_high <= float(highs[i+j]):
                is_resistance = False
                break
        
        if is_resistance:
            resistance_levels.append(current_high)
        
        # Destek
        is_support = True
        for j in range(1, window+1):
            if current_low >= float(lows[i-j]) or current_low >= float(lows[i+j]):
                is_support = False
                break
        
        if is_support:
            support_levels.append(current_low)
    
    return support_levels, resistance_levels

def main():
    try:
        interval = interval_map[analysis_type]
        st.write(f"**{crypto_symbol}** için veriler çekiliyor...")
        
        data = get_crypto_data(crypto_symbol, lookback_days, interval)
        
        if data is None or data.empty:
            st.error("Veri çekilemedi.")
            return
        
        st.success(f"✅ {len(data)} adet mum verisi çekildi")
        
        # Teknik göstergeleri hesapla
        data = calculate_technical_indicators(data)
        
        # Analizleri yap
        patterns = identify_candlestick_patterns(data)
        support_levels, resistance_levels = calculate_support_resistance(data)
        
        # Mevcut fiyat
        current_price = float(data['Close'].iloc[-1])
        
        # Sonuçları göster
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Destek Seviyeleri")
            key_support = [level for level in support_levels if abs(level - current_price) / current_price * 100 <= 15]
            if key_support:
                for i, level in enumerate(reversed(key_support[-3:])):
                    st.write(f"🟢 **Destek {i+1}:** ${level:.2f}")
            else:
                st.write("Destek seviyesi bulunamadı")
            
            st.subheader("🕯️ Mum Formasyonları")
            if patterns:
                for pattern in patterns:
                    st.write(f"📍 {pattern}")
            else:
                st.write("Belirgin formasyon yok")
        
        with col2:
            st.subheader("📉 Direnç Seviyeleri")
            key_resistance = [level for level in resistance_levels if abs(level - current_price) / current_price * 100 <= 15]
            if key_resistance:
                for i, level in enumerate(key_resistance[-3:]):
                    st.write(f"🔴 **Direnç {i+1}:** ${level:.2f}")
            else:
                st.write("Direnç seviyesi bulunamadı")
            
            st.subheader("📊 Göstergeler")
            st.metric("Mevcut Fiyat", f"${current_price:.2f}")
            rsi = data['RSI'].iloc[-1]
            if not pd.isna(rsi):
                st.metric("RSI", f"{rsi:.1f}")
        
        # Grafik
        st.subheader("📊 Fiyat Grafiği")
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
        
        # Moving Average'lar
        fig.add_trace(go.Scatter(x=data.index, y=data['MA_20'], name='MA 20', line=dict(color='orange', width=1)))
        fig.add_trace(go.Scatter(x=data.index, y=data['MA_50'], name='MA 50', line=dict(color='red', width=1)))
        
        # Destek seviyeleri
        for level in key_support[-3:]:
            fig.add_hline(y=level, line_dash="dash", line_color="green", line_width=1, opacity=0.7)
        
        # Direnç seviyeleri
        for level in key_resistance[-3:]:
            fig.add_hline(y=level, line_dash="dash", line_color="red", line_width=1, opacity=0.7)
        
        fig.update_layout(height=500, title=f"{crypto_symbol} {analysis_type} Grafik")
        st.plotly_chart(fig, use_container_width=True)
        
        # Son veriler - FORMAT HATASI YOK
        st.subheader("📜 Son Mum Verileri")
        display_data = data.tail(10)[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # Formatlama - KESİN ÇÖZÜM
        display_data['Open'] = display_data['Open'].map(lambda x: f"${x:.2f}")
        display_data['High'] = display_data['High'].map(lambda x: f"${x:.2f}")
        display_data['Low'] = display_data['Low'].map(lambda x: f"${x:.2f}")
        display_data['Close'] = display_data['Close'].map(lambda x: f"${x:.2f}")
        display_data['Volume'] = display_data['Volume'].map(lambda x: f"{x:,.0f}")
        
        st.dataframe(display_data)
        
    except Exception as e:
        st.error(f"❌ Hata oluştu: {str(e)}")

if __name__ == "__main__":
    main()