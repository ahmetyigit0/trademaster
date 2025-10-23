import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import ta

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
        return data
    except Exception as e:
        st.error(f"Veri çekilemedi: {e}")
        return None

def calculate_technical_indicators(data):
    """Teknik göstergeleri hesapla"""
    df = data.copy()
    
    # Moving Average'lar
    df['MA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['MA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['MA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
    
    # RSI
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    
    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Histogram'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['BB_Upper'] = bollinger.bollinger_hband()
    df['BB_Lower'] = bollinger.bollinger_lband()
    df['BB_Middle'] = bollinger.bollinger_mavg()
    
    return df

def identify_candlestick_patterns(data):
    """Mum formasyonlarını tespit et"""
    df = data.copy()
    patterns = []
    
    # Doji
    doji = abs(df['Close'] - df['Open']) / (df['High'] - df['Low']) < 0.1
    if doji.iloc[-1]:
        patterns.append("DOJI - Kararsızlık")
    
    # Bullish Engulfing
    if len(df) >= 2:
        prev_open, prev_close = df['Open'].iloc[-2], df['Close'].iloc[-2]
        curr_open, curr_close = df['Open'].iloc[-1], df['Close'].iloc[-1]
        
        if prev_close < prev_open and curr_close > curr_open and curr_open < prev_close and curr_close > prev_open:
            patterns.append("BULLISH ENGULFING - Yükseliş sinyali")
    
    # Hammer
    body = abs(df['Close'].iloc[-1] - df['Open'].iloc[-1])
    lower_shadow = min(df['Open'].iloc[-1], df['Close'].iloc[-1]) - df['Low'].iloc[-1]
    upper_shadow = df['High'].iloc[-1] - max(df['Open'].iloc[-1], df['Close'].iloc[-1])
    
    if lower_shadow > 2 * body and upper_shadow < body:
        patterns.append("HAMMER - Dip reversal sinyali")
    
    return patterns

def calculate_trend_lines(data):
    """Trend çizgilerini hesapla"""
    closes = data['Close'].values
    dates = np.arange(len(closes))
    
    # Basit lineer regresyon ile trend
    if len(dates) > 1:
        z = np.polyfit(dates, closes, 1)
        trend_line = np.poly1d(z)(dates)
        trend_slope = z[0]
        
        if trend_slope > 0:
            trend_direction = "YÜKSELİŞ"
        elif trend_slope < 0:
            trend_direction = "DÜŞÜŞ"
        else:
            trend_direction = "YATAY"
        
        return trend_line, trend_direction, trend_slope
    
    return None, "BELİRSİZ", 0

def generate_trading_signals(data):
    """Alım-satım sinyalleri üret"""
    df = data.copy()
    signals = []
    
    # RSI Sinyalleri
    rsi = df['RSI'].iloc[-1]
    if rsi < 30:
        signals.append("RSI AŞIRI SATIM - Alım fırsatı")
    elif rsi > 70:
        signals.append("RSI AŞIRI ALIM - Satım sinyali")
    
    # MACD Sinyalleri
    macd = df['MACD'].iloc[-1]
    macd_signal = df['MACD_Signal'].iloc[-1]
    if macd > macd_signal and df['MACD'].iloc[-2] <= df['MACD_Signal'].iloc[-2]:
        signals.append("MACD ALTI - Alım sinyali")
    elif macd < macd_signal and df['MACD'].iloc[-2] >= df['MACD_Signal'].iloc[-2]:
        signals.append("MACD ÜSTÜ - Satım sinyali")
    
    # Moving Average Sinyalleri
    price = df['Close'].iloc[-1]
    ma_20 = df['MA_20'].iloc[-1]
    ma_50 = df['MA_50'].iloc[-1]
    
    if price > ma_20 > ma_50:
        signals.append("GÜÇLÜ YÜKSELİŞ TRENDİ")
    elif price < ma_20 < ma_50:
        signals.append("GÜÇLÜ DÜŞÜŞ TRENDİ")
    
    return signals

def calculate_support_resistance(data, window=10):
    """Destek ve direnç seviyelerini hesapla"""
    highs = data['High'].values
    lows = data['Low'].values
    
    support_levels = []
    resistance_levels = []
    
    for i in range(window, len(data)-window):
        current_high = highs[i]
        current_low = lows[i]
        
        # Direnç
        if all(current_high > highs[i-j] for j in range(1, window+1)) and \
           all(current_high > highs[i+j] for j in range(1, window+1)):
            resistance_levels.append(current_high)
        
        # Destek
        if all(current_low < lows[i-j] for j in range(1, window+1)) and \
           all(current_low < lows[i+j] for j in range(1, window+1)):
            support_levels.append(current_low)
    
    return support_levels, resistance_levels

def generate_analysis_report(data, patterns, signals, trend_direction):
    """Detaylı analiz raporu oluştur"""
    current_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2]
    change_pct = ((current_price - prev_price) / prev_price) * 100
    
    report = f"""
    ## 📊 Teknik Analiz Raporu
    
    **🎯 Mevcut Durum:**
    - Fiyat: ${current_price:.2f}
    - 1 Mum Önceki: ${prev_price:.2f}
    - Değişim: %{change_pct:.2f}
    - Trend: {trend_direction}
    
    **📈 Trend Analizi:**
    - Ana Trend: {trend_direction}
    - Momentum: {'Güçlü' if abs(change_pct) > 1 else 'Zayıf'}
    """
    
    if patterns:
        report += "\n**🕯️ Mum Formasyonları:**\n"
        for pattern in patterns:
            report += f"- {pattern}\n"
    
    if signals:
        report += "\n**🔔 Trading Sinyalleri:**\n"
        for signal in signals:
            report += f"- {signal}\n"
    
    # RSI Yorumu
    rsi = data['RSI'].iloc[-1]
    if rsi < 30:
        report += f"\n**📉 RSI Analizi:** AŞIRI SATIM bölgesinde (RSI: {rsi:.1f}) - Potansiyel alım fırsatı"
    elif rsi > 70:
        report += f"\n**📈 RSI Analizi:** AŞIRI ALIM bölgesinde (RSI: {rsi:.1f}) - Dikkatli olun"
    else:
        report += f"\n**⚖️ RSI Analizi:** Nötr bölgede (RSI: {rsi:.1f})"
    
    return report

def main():
    try:
        # Veri çekme
        interval = interval_map[analysis_type]
        st.write(f"**{crypto_symbol}** için {analysis_type} veriler çekiliyor...")
        
        data = get_crypto_data(crypto_symbol, lookback_days, interval)
        
        if data is None or data.empty:
            st.error("Veri çekilemedi. Lütfen sembolü kontrol edin.")
            return
        
        # Teknik göstergeleri hesapla
        data = calculate_technical_indicators(data)
        
        # Analizleri yap
        patterns = identify_candlestick_patterns(data)
        signals = generate_trading_signals(data)
        trend_line, trend_direction, trend_slope = calculate_trend_lines(data)
        support_levels, resistance_levels = calculate_support_resistance(data)
        
        # Seviyeleri filtrele
        current_price = data['Close'].iloc[-1]
        key_support = [level for level in support_levels if abs(level - current_price) / current_price * 100 <= 15]
        key_resistance = [level for level in resistance_levels if abs(level - current_price) / current_price * 100 <= 15]
        
        # Analiz raporu
        report = generate_analysis_report(data, patterns, signals, trend_direction)
        st.markdown(report)
        
        # Grafikler
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Fiyat ve Göstergeler")
            
            # Ana grafik
            fig1 = go.Figure()
            
            # Mum grafiği
            fig1.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Mum'
            ))
            
            # Moving Average'lar
            fig1.add_trace(go.Scatter(x=data.index, y=data['MA_20'], name='MA 20', line=dict(color='orange')))
            fig1.add_trace(go.Scatter(x=data.index, y=data['MA_50'], name='MA 50', line=dict(color='red')))
            
            # Trend çizgisi
            if trend_line is not None:
                fig1.add_trace(go.Scatter(x=data.index, y=trend_line, name='Trend', line=dict(color='blue', dash='dash')))
            
            # Destek seviyeleri
            for level in key_support[-3:]:
                fig1.add_hline(y=level, line_dash="dash", line_color="green", annotation_text=f"D: ${level:.2f}")
            
            # Direnç seviyeleri
            for level in key_resistance[-3:]:
                fig1.add_hline(y=level, line_dash="dash", line_color="red", annotation_text=f"R: ${level:.2f}")
            
            fig1.update_layout(height=500, title=f"{crypto_symbol} {analysis_type} Grafik")
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.subheader("📊 Teknik Göstergeler")
            
            # RSI Grafiği
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')))
            fig2.add_hline(y=70, line_dash="dash", line_color="red")
            fig2.add_hline(y=30, line_dash="dash", line_color="green")
            fig2.add_hline(y=50, line_dash="dot", line_color="gray")
            fig2.update_layout(height=250, title="RSI (14)")
            st.plotly_chart(fig2, use_container_width=True)
            
            # MACD Grafiği
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue')))
            fig3.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], name='Sinyal', line=dict(color='red')))
            fig3.update_layout(height=250, title="MACD")
            st.plotly_chart(fig3, use_container_width=True)
        
        # Detaylı bilgiler
        with st.expander("📋 Detaylı Teknik Veriler"):
            col3, col4, col5 = st.columns(3)
            
            with col3:
                st.write("**📈 Moving Average'lar:**")
                st.write(f"MA 20: ${data['MA_20'].iloc[-1]:.2f}")
                st.write(f"MA 50: ${data['MA_50'].iloc[-1]:.2f}")
                st.write(f"MA 200: ${data['MA_200'].iloc[-1]:.2f}")
            
            with col4:
                st.write("**🔍 Oscillator'lar:**")
                st.write(f"RSI: {data['RSI'].iloc[-1]:.1f}")
                st.write(f"MACD: {data['MACD'].iloc[-1]:.4f}")
                st.write(f"MACD Sinyal: {data['MACD_Signal'].iloc[-1]:.4f}")
            
            with col5:
                st.write("**💎 Seviyeler:**")
                st.write(f"Destekler: {len(key_support)}")
                st.write(f"Dirençler: {len(key_resistance)}")
                st.write(f"Trend Eğim: {trend_slope:.6f}")
        
        # Son 10 mum verisi
        with st.expander("📜 Son Mum Verileri"):
            display_data = data.tail(10)[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MA_20']].round(2)
            st.dataframe(display_data)
            
    except Exception as e:
        st.error(f"❌ Hata oluştu: {str(e)}")

if __name__ == "__main__":
    main()