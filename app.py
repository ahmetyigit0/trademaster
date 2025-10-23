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
        return data
    except Exception as e:
        st.error(f"Veri çekilemedi: {e}")
        return None

def calculate_technical_indicators(data):
    """Teknik göstergeleri hesapla (ta kütüphanesi olmadan)"""
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
    
    # Bollinger Bands
    def calculate_bollinger_bands(prices, window=20, num_std=2):
        middle_band = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        return upper_band, middle_band, lower_band
    
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
    
    return df

def identify_candlestick_patterns(data):
    """Mum formasyonlarını tespit et"""
    df = data.copy()
    patterns = []
    
    if len(df) < 2:
        return patterns
    
    # Tüm değerleri float'a çevir - HATA DÜZELTME
    curr_open, curr_high, curr_low, curr_close = float(df['Open'].iloc[-1]), float(df['High'].iloc[-1]), float(df['Low'].iloc[-1]), float(df['Close'].iloc[-1])
    prev_open, prev_high, prev_low, prev_close = float(df['Open'].iloc[-2]), float(df['High'].iloc[-2]), float(df['Low'].iloc[-2]), float(df['Close'].iloc[-2])
    
    # Doji - Açılış ve kapanış çok yakın
    body_size = abs(curr_close - curr_open)
    total_range = curr_high - curr_low
    if total_range > 0 and (body_size / total_range) < 0.1:
        patterns.append("DOJI - Kararsızlık sinyali")
    
    # Bullish Engulfing
    if (prev_close < prev_open and  # Önceki mum bearish
        curr_close > curr_open and  # Şimdiki mum bullish
        curr_open < prev_close and  # Şimdiki açılış önceki kapanıştan düşük
        curr_close > prev_open):    # Şimdiki kapanış önceki açılıştan yüksek
        patterns.append("BULLISH ENGULFING - Güçlü yükseliş sinyali")
    
    # Bearish Engulfing
    if (prev_close > prev_open and  # Önceki mum bullish
        curr_close < curr_open and  # Şimdiki mum bearish
        curr_open > prev_close and  # Şimdiki açılış önceki kapanıştan yüksek
        curr_close < prev_open):    # Şimdiki kapanış önceki açılıştan düşük
        patterns.append("BEARISH ENGULFING - Güçlü düşüş sinyali")
    
    # Hammer
    lower_shadow = min(curr_open, curr_close) - curr_low
    upper_shadow = curr_high - max(curr_open, curr_close)
    body = abs(curr_close - curr_open)
    
    if lower_shadow > 2 * body and upper_shadow < body and curr_close > curr_open:
        patterns.append("HAMMER - Dip reversal sinyali")
    
    # Shooting Star
    if upper_shadow > 2 * body and lower_shadow < body and curr_close < curr_open:
        patterns.append("SHOOTING STAR - Tepe reversal sinyali")
    
    return patterns

def calculate_trend_lines(data):
    """Trend çizgilerini hesapla"""
    # Float'a çevir - HATA DÜZELTME
    closes = data['Close'].astype(float).values
    dates = np.arange(len(closes))
    
    if len(dates) > 1:
        # Lineer regresyon ile trend
        z = np.polyfit(dates, closes, 1)
        trend_line = np.poly1d(z)(dates)
        trend_slope = float(z[0])  # float'a çevir
        
        # Trend yönünü belirle
        if trend_slope > 0:
            trend_direction = "📈 YÜKSELİŞ TRENDİ"
            trend_strength = "GÜÇLÜ" if trend_slope > np.std(closes) * 0.1 else "ZAYIF"
        elif trend_slope < 0:
            trend_direction = "📉 DÜŞÜŞ TRENDİ"
            trend_strength = "GÜÇLÜ" if abs(trend_slope) > np.std(closes) * 0.1 else "ZAYIF"
        else:
            trend_direction = "➡️ YATAY TREND"
            trend_strength = "NÖTR"
        
        return trend_line, f"{trend_direction} ({trend_strength})", trend_slope
    
    return None, "BELİRSİZ", 0

def generate_trading_signals(data):
    """Alım-satım sinyalleri üret"""
    df = data.copy()
    signals = []
    
    if len(df) < 2:
        return signals
    
    # Tüm değerleri float'a çevir - HATA DÜZELTME
    rsi = float(df['RSI'].iloc[-1]) if not pd.isna(df['RSI'].iloc[-1]) else None
    macd = float(df['MACD'].iloc[-1]) if not pd.isna(df['MACD'].iloc[-1]) else None
    macd_signal = float(df['MACD_Signal'].iloc[-1]) if not pd.isna(df['MACD_Signal'].iloc[-1]) else None
    price = float(df['Close'].iloc[-1])
    ma_20 = float(df['MA_20'].iloc[-1]) if not pd.isna(df['MA_20'].iloc[-1]) else None
    ma_50 = float(df['MA_50'].iloc[-1]) if not pd.isna(df['MA_50'].iloc[-1]) else None
    
    # RSI Sinyalleri
    if rsi is not None:
        if rsi < 30:
            signals.append("🎯 RSI AŞIRI SATIM - Potansiyel ALIM fırsatı")
        elif rsi > 70:
            signals.append("⚠️ RSI AŞIRI ALIM - Potansiyel SATIM sinyali")
    
    # MACD Sinyalleri
    if macd is not None and macd_signal is not None:
        prev_macd = float(df['MACD'].iloc[-2]) if not pd.isna(df['MACD'].iloc[-2]) else None
        prev_macd_signal = float(df['MACD_Signal'].iloc[-2]) if not pd.isna(df['MACD_Signal'].iloc[-2]) else None
        
        if prev_macd is not None and prev_macd_signal is not None:
            if macd > macd_signal and prev_macd <= prev_macd_signal:
                signals.append("✅ MACD ALTI KESİŞİM - ALIM sinyali")
            elif macd < macd_signal and prev_macd >= prev_macd_signal:
                signals.append("❌ MACD ÜSTÜ KESİŞİM - SATIM sinyali")
    
    # Moving Average Sinyalleri
    if ma_20 is not None and ma_50 is not None:
        if price > ma_20 > ma_50:
            signals.append("🚀 GÜÇLÜ YÜKSELİŞ TRENDİ - MA'lar destekliyor")
        elif price < ma_20 < ma_50:
            signals.append("🔻 GÜÇLÜ DÜŞÜŞ TRENDİ - MA'lar direnç gösteriyor")
        elif price > ma_20 and ma_20 > ma_50:
            signals.append("↗️ YÜKSELİŞ EĞİLİMİ - MA düzeni uygun")
        elif price < ma_20 and ma_20 < ma_50:
            signals.append("↘️ DÜŞÜŞ EĞİLİMİ - MA düzeni uygun")
    
    return signals

def calculate_support_resistance(data, window=10):
    """Destek ve direnç seviyelerini hesapla"""
    # Series'i numpy array'e çevir ve float'a dönüştür - HATA DÜZELTME
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
            prev_high = float(highs[i-j])
            next_high = float(highs[i+j])
            if current_high <= prev_high or current_high <= next_high:
                is_resistance = False
                break
        
        if is_resistance:
            resistance_levels.append(current_high)
        
        # Destek - yerel minimum
        is_support = True
        for j in range(1, window+1):
            prev_low = float(lows[i-j])
            next_low = float(lows[i+j])
            if current_low >= prev_low or current_low >= next_low:
                is_support = False
                break
        
        if is_support:
            support_levels.append(current_low)
    
    return support_levels, resistance_levels

def generate_analysis_report(data, patterns, signals, trend_direction):
    """Detaylı analiz raporu oluştur"""
    # Float'a çevir - HATA DÜZELTME
    current_price = float(data['Close'].iloc[-1])
    prev_price = float(data['Close'].iloc[-2])
    change_pct = ((current_price - prev_price) / prev_price) * 100
    
    # Volatilite hesapla
    volatility = data['Close'].astype(float).pct_change().std() * 100
    
    report = f"""
    ## 📊 Teknik Analiz Raporu
    
    **🎯 Mevcut Durum:**
    - **Fiyat:** ${current_price:.2f}
    - **Değişim:** %{change_pct:+.2f}
    - **Volatilite:** %{volatility:.2f}
    - **Trend:** {trend_direction}
    
    **📈 Trend Analizi:**
    - Ana Trend: {trend_direction.split('(')[0].strip()}
    - Momentum: {'YÜKSELİŞ' if change_pct > 0 else 'DÜŞÜŞ'}
    - Volatilite: {'YÜKSEK' if volatility > 3 else 'DÜŞÜK'}
    """
    
    if patterns:
        report += "\n**🕯️ Mum Formasyonları:**\n"
        for pattern in patterns:
            report += f"- {pattern}\n"
    else:
        report += "\n**🕯️ Mum Formasyonları:** Belirgin formasyon yok\n"
    
    if signals:
        report += "\n**🔔 Trading Sinyalleri:**\n"
        for signal in signals:
            report += f"- {signal}\n"
    else:
        report += "\n**🔔 Trading Sinyalleri:** Net sinyal yok\n"
    
    # RSI Detaylı Yorum
    rsi = float(data['RSI'].iloc[-1]) if not pd.isna(data['RSI'].iloc[-1]) else None
    if rsi is not None:
        if rsi < 30:
            report += f"\n**📉 RSI Analizi:** AŞIRI SATIM bölgesinde (RSI: {rsi:.1f}) - ⚠️ Potansiyel ALIM fırsatı"
        elif rsi > 70:
            report += f"\n**📈 RSI Analizi:** AŞIRI ALIM bölgesinde (RSI: {rsi:.1f}) - ⚠️ Dikkatli olun, SATIM sinyali"
        elif 30 <= rsi <= 70:
            report += f"\n**⚖️ RSI Analizi:** Nötr bölgede (RSI: {rsi:.1f}) - 🔄 Trend takibi önerilir"
    
    # Genel Öneri
    bullish_signals = len([s for s in signals if 'ALIM' in s or 'YÜKSELİŞ' in s])
    bearish_signals = len([s for s in signals if 'SATIM' in s or 'DÜŞÜŞ' in s])
    
    if bullish_signals > bearish_signals:
        report += "\n\n**💎 GENEL BAKIŞ:** YÜKSELİŞ eğilimi ağır basıyor"
    elif bearish_signals > bullish_signals:
        report += "\n\n**💎 GENEL BAKIŞ:** DÜŞÜŞ eğilimi ağır basıyor"
    else:
        report += "\n\n**💎 GENEL BAKIŞ:** KARARSIZ piyasa, bekleyin"
    
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
        
        st.success(f"✅ {len(data)} adet {analysis_type} mum verisi çekildi")
        
        # Teknik göstergeleri hesapla
        data = calculate_technical_indicators(data)
        
        # Analizleri yap
        patterns = identify_candlestick_patterns(data)
        signals = generate_trading_signals(data)
        trend_line, trend_direction, trend_slope = calculate_trend_lines(data)
        support_levels, resistance_levels = calculate_support_resistance(data)
        
        # Seviyeleri filtrele (mevcut fiyata yakın olanlar)
        current_price = float(data['Close'].iloc[-1])
        key_support = [level for level in support_levels if abs(level - current_price) / current_price * 100 <= 15]
        key_resistance = [level for level in resistance_levels if abs(level - current_price) / current_price * 100 <= 15]
        
        # Benzersiz seviyeler
        key_support = list(set(key_support))
        key_resistance = list(set(key_resistance))
        key_support.sort()
        key_resistance.sort()
        
        # Analiz raporu
        report = generate_analysis_report(data, patterns, signals, trend_direction)
        st.markdown(report)
        
        # Grafikler
        st.subheader("📊 Görsel Analiz")
        col1, col2 = st.columns(2)
        
        with col1:
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
            fig1.add_trace(go.Scatter(x=data.index, y=data['MA_20'], name='MA 20', line=dict(color='orange', width=2)))
            fig1.add_trace(go.Scatter(x=data.index, y=data['MA_50'], name='MA 50', line=dict(color='red', width=2)))
            
            # Trend çizgisi
            if trend_line is not None:
                fig1.add_trace(go.Scatter(x=data.index, y=trend_line, name='Trend Çizgisi', 
                                        line=dict(color='blue', dash='dash', width=3)))
            
            # Destek seviyeleri
            for level in key_support[-3:]:
                fig1.add_hline(y=level, line_dash="dash", line_color="green", 
                             line_width=2, opacity=0.7,
                             annotation_text=f"D: ${level:.2f}")
            
            # Direnç seviyeleri
            for level in key_resistance[-3:]:
                fig1.add_hline(y=level, line_dash="dash", line_color="red", 
                             line_width=2, opacity=0.7,
                             annotation_text=f"R: ${level:.2f}")
            
            fig1.update_layout(
                height=500, 
                title=f"{crypto_symbol} {analysis_type} Grafik - Mum Formasyonları ve Trend",
                xaxis_title="Tarih",
                yaxis_title="Fiyat (USD)"
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # RSI Grafiği
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', 
                                    line=dict(color='purple', width=2)))
            fig2.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Aşırı Alım")
            fig2.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Aşırı Satım")
            fig2.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="Orta")
            fig2.update_layout(height=250, title="RSI (14) - Momentum Göstergesi")
            st.plotly_chart(fig2, use_container_width=True)
            
            # MACD Grafiği
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD', 
                                    line=dict(color='blue', width=2)))
            fig3.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], name='Sinyal', 
                                    line=dict(color='red', width=2)))
            # Histogram'ı sadece NaN değilse ekle
            if not data['MACD_Histogram'].isna().all():
                fig3.add_trace(go.Bar(x=data.index, y=data['MACD_Histogram'], name='Histogram',
                                    marker_color='gray', opacity=0.3))
            fig3.update_layout(height=250, title="MACD - Trend Takip Göstergesi")
            st.plotly_chart(fig3, use_container_width=True)
        
        # Detaylı bilgiler
        with st.expander("📋 Detaylı Teknik Veriler"):
            col3, col4, col5 = st.columns(3)
            
            with col3:
                st.write("**📈 Moving Average'lar:**")
                ma_20_val = data['MA_20'].iloc[-1]
                ma_50_val = data['MA_50'].iloc[-1]
                ma_200_val = data['MA_200'].iloc[-1]
                
                st.metric("MA 20", f"${ma_20_val:.2f}" if not pd.isna(ma_20_val) else "Hesaplanıyor")
                st.metric("MA 50", f"${ma_50_val:.2f}" if not pd.isna(ma_50_val) else "Hesaplanıyor")
                st.metric("MA 200", f"${ma_200_val:.2f}" if not pd.isna(ma_200_val) else "Hesaplanıyor")
            
            with col4:
                st.write("**🔍 Oscillator'lar:**")
                rsi_val = data['RSI'].iloc[-1]
                macd_val = data['MACD'].iloc[-1]
                macd_signal_val = data['MACD_Signal'].iloc[-1]
                
                st.metric("RSI", f"{rsi_val:.1f}" if not pd.isna(rsi_val) else "Hesaplanıyor")
                st.metric("MACD", f"{macd_val:.4f}" if not pd.isna(macd_val) else "Hesaplanıyor")
                st.metric("MACD Sinyal", f"{macd_signal_val:.4f}" if not pd.isna(macd_signal_val) else "Hesaplanıyor")
            
            with col5:
                st.write("**💎 Piyasa Bilgileri:**")
                st.metric("Destek Seviyeleri", len(key_support))
                st.metric("Direnç Seviyeleri", len(key_resistance))
                st.metric("Trend Eğim", f"{trend_slope:.6f}")
        
        # Son 10 mum verisi
        with st.expander("📜 Son Mum Verileri"):
            display_data = data.tail(10)[['Open', 'High', 'Low', 'Close', 'Volume']].round(2)
            st.dataframe(display_data.style.format({
                'Open': '${:.2f}', 'High': '${:.2f}', 'Low': '${:.2f}', 'Close': '${:.2f}'
            }))
            
    except Exception as e:
        st.error(f"❌ Hata oluştu: {str(e)}")
        st.info("Lütfen sembolü kontrol edin ve internet bağlantınızı doğrulayın.")

if __name__ == "__main__":
    main()