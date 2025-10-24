import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Streamlit arayüzü
st.set_page_config(page_title="Kripto Teknik Analiz", layout="wide")
st.title("🎯 Kripto Teknik Analiz - Mum, Trend, Sinyal ve Yorum")

# Sidebar
st.sidebar.header("⚙️ Analiz Ayarları")
crypto_symbol = st.sidebar.text_input("Kripto Sembolü (Örn: BTC-USD, ETH-USD):", "BTC-USD")
lookback_days = st.sidebar.slider("Gün Sayısı", 30, 365, 90)
analysis_type = st.sidebar.selectbox("Analiz Türü", ["4 Saatlik", "1 Günlük", "1 Saatlik"])

# Analiz periyodu mapping
interval_map = {"4 Saatlik": "4h", "1 Günlük": "1d", "1 Saatlik": "1h"}

def get_crypto_data(symbol, days, interval):
    """Kripto verilerini çek"""
    try:
        data = yf.download(symbol, period=f"{days}d", interval=interval, progress=False)
        # None değerleri temizle
        if data is not None and not data.empty:
            data = data.dropna()
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
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    
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
    
    return df

def identify_candlestick_patterns(data):
    """Mum formasyonlarını tespit et"""
    df = data.copy()
    patterns = []
    
    if len(df) < 2:
        return patterns
    
    # Tüm değerleri float'a çevir
    curr_open, curr_high, curr_low, curr_close = float(df['Open'].iloc[-1]), float(df['High'].iloc[-1]), float(df['Low'].iloc[-1]), float(df['Close'].iloc[-1])
    prev_open, prev_high, prev_low, prev_close = float(df['Open'].iloc[-2]), float(df['High'].iloc[-2]), float(df['Low'].iloc[-2]), float(df['Close'].iloc[-2])
    
    # Doji - Açılış ve kapanış çok yakın
    body_size = abs(curr_close - curr_open)
    total_range = curr_high - curr_low
    if total_range > 0 and (body_size / total_range) < 0.1:
        patterns.append("DOJI - Kararsızlık sinyali")
    
    # Bullish Engulfing
    if (prev_close < prev_open and
        curr_close > curr_open and
        curr_open < prev_close and
        curr_close > prev_open):
        patterns.append("BULLISH ENGULFING - Güçlü yükseliş sinyali")
    
    # Bearish Engulfing
    if (prev_close > prev_open and
        curr_close < curr_open and
        curr_open > prev_close and
        curr_close < prev_open):
        patterns.append("BEARISH ENGULFING - Güçlü düşüş sinyali")
    
    # Hammer
    lower_shadow = min(curr_open, curr_close) - curr_low
    upper_shadow = curr_high - max(curr_open, curr_close)
    body = abs(curr_close - curr_open)
    
    if lower_shadow > 2 * body and upper_shadow < body and curr_close > curr_open:
        patterns.append("HAMMER - Dip reversal sinyali")
    
    return patterns

def calculate_trend_lines(data):
    """Trend çizgilerini hesapla"""
    try:
        # Son 20 mumun kapanış fiyatlarını al
        closes = data['Close'].tail(20).astype(float).values
        
        if len(closes) < 2:
            return None, "YETERSİZ VERİ", 0
        
        # Basit lineer regresyon
        x = np.arange(len(closes))
        slope = np.polyfit(x, closes, 1)[0]
        
        # Trend yönü
        if slope > 0:
            trend_dir = "📈 YÜKSELİŞ"
            strength = "GÜÇLÜ" if slope > np.std(closes) * 0.05 else "ZAYIF"
        elif slope < 0:
            trend_dir = "📉 DÜŞÜŞ"
            strength = "GÜÇLÜ" if abs(slope) > np.std(closes) * 0.05 else "ZAYIF"
        else:
            trend_dir = "➡️ YATAY"
            strength = "NÖTR"
        
        # Trend çizgisi oluştur
        trend_line = np.poly1d([slope, closes[0]])(x)
        
        return trend_line, f"{trend_dir} TRENDİ ({strength})", slope
        
    except Exception:
        return None, "TREND HESAPLANAMADI", 0

def generate_trading_signals(data):
    """Alım-satım sinyalleri üret"""
    df = data.copy()
    signals = []
    
    if len(df) < 2:
        return signals
    
    # Tüm değerleri float'a çevir
    rsi = float(df['RSI'].iloc[-1]) if not pd.isna(df['RSI'].iloc[-1]) else None
    macd = float(df['MACD'].iloc[-1]) if not pd.isna(df['MACD'].iloc[-1]) else None
    macd_signal = float(df['MACD_Signal'].iloc[-1]) if not pd.isna(df['MACD_Signal'].iloc[-1]) else None
    
    # RSI Sinyalleri
    if rsi is not None:
        if rsi < 30:
            signals.append("🎯 RSI AŞIRI SATIM - Potansiyel ALIM fırsatı")
        elif rsi > 70:
            signals.append("⚠️ RSI AŞIRI ALIM - Potansiyel SATIM sinyali")
    
    # MACD Sinyalleri
    if macd is not None and macd_signal is not None:
        if macd > macd_signal:
            signals.append("✅ MACD POZİTİF - ALIM eğilimi")
        else:
            signals.append("❌ MACD NEGATİF - SATIM eğilimi")
    
    return signals

def calculate_support_resistance(data, window=10):
    """Destek ve direnç seviyelerini hesapla"""
    highs = data['High'].astype(float).values
    lows = data['Low'].astype(float).values
    
    support_levels = []
    resistance_levels = []
    
    for i in range(window, len(data)-window):
        current_high = float(highs[i])
        current_low = float(lows[i])
        
        # Direnç - yerel maksimum
        is_resistance = True
        for j in range(1, window+1):
            if current_high <= float(highs[i-j]) or current_high <= float(highs[i+j]):
                is_resistance = False
                break
        
        if is_resistance:
            resistance_levels.append(current_high)
        
        # Destek - yerel minimum
        is_support = True
        for j in range(1, window+1):
            if current_low >= float(lows[i-j]) or current_low >= float(lows[i+j]):
                is_support = False
                break
        
        if is_support:
            support_levels.append(current_low)
    
    return support_levels, resistance_levels

def safe_float_format(value):
    """Güvenli float formatlama"""
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def main():
    try:
        # Veri çekme
        interval = interval_map[analysis_type]
        st.write(f"**{crypto_symbol}** için {analysis_type} veriler çekiliyor...")
        
        data = get_crypto_data(crypto_symbol, lookback_days, interval)
        
        if data is None or data.empty:
            st.error("Veri çekilemedi. Lütfen sembolü kontrol edin.")
            return
        
        st.success(f"✅ {len(data)} adet {analysis_type} mum verisi çekildi")
        
        # Teknik göstergeleri hesapla
        data = calculate_technical_indicators(data)
        
        # Analizleri yap
        patterns = identify_candlestick_patterns(data)
        signals = generate_trading_signals(data)
        trend_line, trend_direction, trend_slope = calculate_trend_lines(data)
        support_levels, resistance_levels = calculate_support_resistance(data)
        
        # Seviyeleri filtrele
        current_price = float(data['Close'].iloc[-1])
        key_support = [level for level in support_levels if abs(level - current_price) / current_price * 100 <= 15]
        key_resistance = [level for level in resistance_levels if abs(level - current_price) / current_price * 100 <= 15]
        
        # Benzersiz seviyeler
        key_support = list(set(key_support))
        key_resistance = list(set(key_resistance))
        key_support.sort()
        key_resistance.sort()
        
        # Analiz raporu
        st.subheader("📊 Teknik Analiz Raporu")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Mevcut Fiyat", f"${current_price:.2f}")
            st.metric("Trend", trend_direction)
            
            st.write("**📈 Destek Seviyeleri:**")
            if key_support:
                for level in key_support[-3:]:
                    st.write(f"🟢 ${level:.2f}")
            else:
                st.write("Destek bulunamadı")
        
        with col2:
            price_change = ((current_price - float(data['Close'].iloc[-2])) / float(data['Close'].iloc[-2])) * 100
            st.metric("24s Değişim", f"%{price_change:.2f}")
            
            st.write("**📉 Direnç Seviyeleri:**")
            if key_resistance:
                for level in key_resistance[-3:]:
                    st.write(f"🔴 ${level:.2f}")
            else:
                st.write("Direnç bulunamadı")
        
        # Mum Formasyonları
        if patterns:
            st.write("**🕯️ Mum Formasyonları:**")
            for pattern in patterns:
                st.write(f"- {pattern}")
        
        # Trading Sinyalleri
        if signals:
            st.write("**🔔 Trading Sinyalleri:**")
            for signal in signals:
                st.write(f"- {signal}")
        
        # Grafikler
        st.subheader("📈 Görsel Analiz")
        
        # Ana grafik
        fig1 = go.Figure()
        
        # Mum grafiği
        fig1.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ))
        
        # Moving Average'lar
        fig1.add_trace(go.Scatter(x=data.index, y=data['MA_20'], name='MA 20', line=dict(color='orange')))
        fig1.add_trace(go.Scatter(x=data.index, y=data['MA_50'], name='MA 50', line=dict(color='red')))
        
        # Destek seviyeleri
        for level in key_support[-3:]:
            fig1.add_hline(y=level, line_dash="dash", line_color="green", annotation_text=f"D: ${level:.2f}")
        
        # Direnç seviyeleri
        for level in key_resistance[-3:]:
            fig1.add_hline(y=level, line_dash="dash", line_color="red", annotation_text=f"R: ${level:.2f}")
        
        fig1.update_layout(height=500, title=f"{crypto_symbol} {analysis_type} Grafik")
        st.plotly_chart(fig1, use_container_width=True)
        
        # Göstergeler
        col3, col4 = st.columns(2)
        
        with col3:
            # RSI Grafiği
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')))
            fig2.add_hline(y=70, line_dash="dash", line_color="red")
            fig2.add_hline(y=30, line_dash="dash", line_color="green")
            fig2.update_layout(height=300, title="RSI")
            st.plotly_chart(fig2, use_container_width=True)
        
        with col4:
            # MACD Grafiği
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue')))
            fig3.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], name='Sinyal', line=dict(color='red')))
            fig3.update_layout(height=300, title="MACD")
            st.plotly_chart(fig3, use_container_width=True)
        
        # Son mum verileri
        with st.expander("📜 Son Mum Verileri"):
            display_data = data.tail(10)[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            
            # Güvenli formatlama
            for col in ['Open', 'High', 'Low', 'Close']:
                display_data[col] = display_data[col].apply(lambda x: f"${safe_float_format(x):.2f}")
            
            display_data['Volume'] = display_data['Volume'].apply(lambda x: f"{safe_float_format(x):,.0f}")
            
            st.dataframe(display_data)
            
    except Exception as e:
        st.error(f"❌ Hata oluştu: {str(e)}")

if __name__ == "__main__":
    main()