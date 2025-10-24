import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Profesyonel Kripto Analiz", layout="wide")
st.title("🎯 Profesyonel Kripto Trading Analizi")

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

def calculate_advanced_indicators(data):
    """Gelişmiş teknik göstergeleri hesapla"""
    df = data.copy()
    
    # 1. EMA'lar - Trend analizi
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # 2. RSI - Momentum
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['RSI'] = calculate_rsi(df['Close'])
    
    # 3. Bollinger Bantları - Volatilite
    def calculate_bollinger_bands(prices, window=20, num_std=2):
        middle_band = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        return upper_band, lower_band
    
    bb_upper, bb_lower = calculate_bollinger_bands(df['Close'])
    df['BB_Upper'] = bb_upper
    df['BB_Lower'] = bb_lower
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    
    # 4. MACD - Momentum ve trend dönüşleri
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = calculate_macd(df['Close'])
    
    # 5. ATR - Volatilite ölçümü
    def calculate_atr(data, period=14):
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        true_range = np.maximum(np.maximum(high_low, high_close), low_close)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    df['ATR'] = calculate_atr(df)
    
    # 6. Volume analizi - DÜZELTME: Tek tek hesapla
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    # Volume_Ratio'yu ayrı satırda hesapla
    volume_ratio = df['Volume'] / df['Volume_MA']
    df['Volume_Ratio'] = volume_ratio
    
    return df

def generate_trading_signals(data):
    """Trading sinyalleri üret"""
    df = data.copy()
    signals = []
    
    if len(df) < 50:
        return signals
    
    # Mevcut değerler
    current_price = float(df['Close'].iloc[-1])
    ema_20 = float(df['EMA_20'].iloc[-1])
    ema_50 = float(df['EMA_50'].iloc[-1])
    ema_200 = float(df['EMA_200'].iloc[-1])
    rsi = float(df['RSI'].iloc[-1])
    bb_upper = float(df['BB_Upper'].iloc[-1])
    bb_lower = float(df['BB_Lower'].iloc[-1])
    macd = float(df['MACD'].iloc[-1])
    macd_signal = float(df['MACD_Signal'].iloc[-1])
    atr = float(df['ATR'].iloc[-1])
    volume_ratio = float(df['Volume_Ratio'].iloc[-1]) if not pd.isna(df['Volume_Ratio'].iloc[-1]) else 1.0
    
    # 1. TREND ANALİZİ - EMA
    trend_strength = 0
    if ema_20 > ema_50 > ema_200 and current_price > ema_20:
        signals.append("🚀 GÜÇLÜ YÜKSELİŞ TRENDİ - EMA'lar uyumlu")
        trend_strength = 2
    elif ema_20 < ema_50 < ema_200 and current_price < ema_20:
        signals.append("🔻 GÜÇLÜ DÜŞÜŞ TRENDİ - EMA'lar uyumlu")
        trend_strength = -2
    elif ema_20 > ema_50:
        signals.append("↗️ YÜKSELİŞ EĞİLİMİ - Kısa vade pozitif")
        trend_strength = 1
    elif ema_20 < ema_50:
        signals.append("↘️ DÜŞÜŞ EĞİLİMİ - Kısa vade negatif")
        trend_strength = -1
    
    # Golden Cross / Death Cross
    if len(df) >= 2:
        prev_ema_20 = float(df['EMA_20'].iloc[-2])
        prev_ema_50 = float(df['EMA_50'].iloc[-2])
        if prev_ema_20 <= prev_ema_50 and ema_20 > ema_50:
            signals.append("⭐ GOLDEN CROSS - EMA20, EMA50'yı yukarı kesti - ALIM")
        elif prev_ema_20 >= prev_ema_50 and ema_20 < ema_50:
            signals.append("💀 DEATH CROSS - EMA20, EMA50'yı aşağı kesti - SATIM")
    
    # 2. RSI SİNYALLERİ
    if rsi < 30:
        signals.append("🎯 RSI AŞIRI SATIM - Potansiyel ALIM fırsatı")
        if trend_strength > 0:
            signals.append("✅ RSI + Trend uyumu - GÜÇLÜ ALIM")
    elif rsi > 70:
        signals.append("⚠️ RSI AŞIRI ALIM - Potansiyel SATIM sinyali")
        if trend_strength < 0:
            signals.append("❌ RSI + Trend uyumu - GÜÇLÜ SATIM")
    elif 30 <= rsi <= 70:
        if rsi > 50 and trend_strength > 0:
            signals.append("🟢 RSI + Trend - Yükseliş destekli")
        elif rsi < 50 and trend_strength < 0:
            signals.append("🔴 RSI + Trend - Düşüş destekli")
    
    # 3. BOLLINGER BANT SİNYALLERİ
    bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) * 100 if (bb_upper - bb_lower) > 0 else 50
    
    if current_price <= bb_lower and rsi < 35:
        signals.append("📉 BOLLINGER ALT BANT + RSI - GÜÇLÜ ALIM SİNYALİ")
    elif current_price >= bb_upper and rsi > 65:
        signals.append("📈 BOLLINGER ÜST BANT + RSI - GÜÇLÜ SATIM SİNYALİ")
    
    # Bollinger Squeeze tespiti
    bb_width = (bb_upper - bb_lower) / df['BB_Middle'].iloc[-1] * 100 if not pd.isna(df['BB_Middle'].iloc[-1]) else 10
    if bb_width < 5:  # Dar bant
        signals.append("⚡ BOLLINGER SQUEEZE - Büyük hareket yakın!")
    
    # 4. MACD SİNYALLERİ
    if len(df) >= 2:
        prev_macd = float(df['MACD'].iloc[-2])
        prev_macd_signal = float(df['MACD_Signal'].iloc[-2])
        if macd > macd_signal and prev_macd <= prev_macd_signal:
            signals.append("✅ MACD ALTI KESİŞ - ALIM sinyali")
            if trend_strength > 0:
                signals.append("💪 MACD + Trend - ALIM güçlü")
        elif macd < macd_signal and prev_macd >= prev_macd_signal:
            signals.append("❌ MACD ÜSTÜ KESİŞ - SATIM sinyali")
            if trend_strength < 0:
                signals.append("💪 MACD + Trend - SATIM güçlü")
    
    # 5. VOLUME ANALİZİ
    if volume_ratio > 1.5:
        signals.append(f"📊 HACİM ARTIŞI - {volume_ratio:.1f}x - Sinyal güvenilir")
    
    # 6. ATR TABANLI STOP LOSS ÖNERİSİ
    stop_loss_pct = (atr / current_price) * 100 * 2  # 2x ATR
    signals.append(f"🛡️ Önerilen Stop Loss: %{stop_loss_pct:.1f} (2x ATR)")
    
    # Take Profit seviyeleri
    if trend_strength > 0:
        tp1 = current_price * 1.01  # %1
        tp2 = current_price * 1.02  # %2
        tp3 = current_price * 1.03  # %3
        signals.append(f"🎯 Take Profit Seviyeleri: ${tp1:.2f} | ${tp2:.2f} | ${tp3:.2f}")
    
    return signals

def calculate_support_resistance(data):
    """Destek ve direnç seviyelerini hesapla"""
    highs = data['High'].astype(float).values
    lows = data['Low'].astype(float).values
    
    support_levels = []
    resistance_levels = []
    
    # Basit pivot point hesaplama
    for i in range(2, len(data)-2):
        # Direnç
        if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and 
            highs[i] > highs[i+1] and highs[i] > highs[i+2]):
            resistance_levels.append(highs[i])
        
        # Destek
        if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and 
            lows[i] < lows[i+1] and lows[i] < lows[i+2]):
            support_levels.append(lows[i])
    
    return support_levels, resistance_levels

def main():
    try:
        interval = interval_map[analysis_type]
        st.write(f"**{crypto_symbol}** için {analysis_type} veriler çekiliyor...")
        
        data = get_crypto_data(crypto_symbol, lookback_days, interval)
        
        if data is None or data.empty:
            st.error("Veri çekilemedi.")
            return
        
        st.success(f"✅ {len(data)} adet mum verisi çekildi")
        
        # Gelişmiş göstergeleri hesapla
        data = calculate_advanced_indicators(data)
        
        # Trading sinyalleri üret
        signals = generate_trading_signals(data)
        
        # Destek/direnç seviyeleri
        support_levels, resistance_levels = calculate_support_resistance(data)
        
        # Mevcut fiyat ve analiz
        current_price = float(data['Close'].iloc[-1])
        key_support = [level for level in support_levels if abs(level - current_price) / current_price * 100 <= 10]
        key_resistance = [level for level in resistance_levels if abs(level - current_price) / current_price * 100 <= 10]
        
        # Ana panel
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📈 Detaylı Grafik Analizi")
            
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
            
            # EMA'lar
            fig.add_trace(go.Scatter(x=data.index, y=data['EMA_20'], name='EMA 20', line=dict(color='blue', width=2)))
            fig.add_trace(go.Scatter(x=data.index, y=data['EMA_50'], name='EMA 50', line=dict(color='orange', width=2)))
            fig.add_trace(go.Scatter(x=data.index, y=data['EMA_200'], name='EMA 200', line=dict(color='red', width=2)))
            
            # Bollinger Bantları
            fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], name='BB Upper', line=dict(color='gray', width=1, dash='dash')))
            fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], name='BB Lower', line=dict(color='gray', width=1, dash='dash')))
            
            # Destek/direnç çizgileri
            for level in key_support[-3:]:
                fig.add_hline(y=level, line_dash="dash", line_color="green", line_width=2, opacity=0.7)
            
            for level in key_resistance[-3:]:
                fig.add_hline(y=level, line_dash="dash", line_color="red", line_width=2, opacity=0.7)
            
            fig.update_layout(
                height=600,
                title=f"{crypto_symbol} {analysis_type} - Profesyonel Analiz",
                xaxis_title="Tarih",
                yaxis_title="Fiyat (USD)",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 TRADING SİNYALLERİ")
            
            if signals:
                for signal in signals:
                    if "ALIM" in signal or "GOLDEN" in signal:
                        st.success(f"✅ {signal}")
                    elif "SATIM" in signal or "DEATH" in signal:
                        st.error(f"❌ {signal}")
                    elif "GÜÇLÜ" in signal:
                        st.warning(f"⚠️ {signal}")
                    else:
                        st.info(f"📊 {signal}")
            else:
                st.info("📊 Net trading sinyali yok")
            
            st.subheader("📊 GÖSTERGELER")
            current = data.iloc[-1]
            st.metric("Fiyat", f"${current_price:.2f}")
            st.metric("RSI", f"{current['RSI']:.1f}" if not pd.isna(current['RSI']) else "N/A")
            st.metric("MACD", f"{current['MACD']:.4f}" if not pd.isna(current['MACD']) else "N/A")
            st.metric("ATR", f"{current['ATR']:.2f}" if not pd.isna(current['ATR']) else "N/A")
            st.metric("Volume Ratio", f"{current['Volume_Ratio']:.1f}x" if not pd.isna(current['Volume_Ratio']) else "N/A")
            
            st.subheader("💎 SEVİYELER")
            st.write("**Destek:**")
            for level in sorted(key_support)[-3:]:
                st.write(f"🟢 ${level:.2f}")
            
            st.write("**Direnç:**")
            for level in sorted(key_resistance)[-3:]:
                st.write(f"🔴 ${level:.2f}")
        
        # Alt grafikler
        st.subheader("📊 TEKNİK GÖSTERGELER")
        col3, col4 = st.columns(2)
        
        with col3:
            # RSI Grafiği
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray")
            fig_rsi.update_layout(height=300, title="RSI (14)")
            st.plotly_chart(fig_rsi, use_container_width=True)
        
        with col4:
            # MACD Grafiği
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue')))
            fig_macd.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], name='Sinyal', line=dict(color='red')))
            fig_macd.add_trace(go.Bar(x=data.index, y=data['MACD_Histogram'], name='Histogram', marker_color='gray'))
            fig_macd.update_layout(height=300, title="MACD")
            st.plotly_chart(fig_macd, use_container_width=True)
        
        # Son veriler
        with st.expander("📜 DETAYLI VERİLER"):
            display_data = data.tail(10)[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'EMA_20', 'EMA_50']].copy()
            
            # Formatlama
            for col in ['Open', 'High', 'Low', 'Close', 'EMA_20', 'EMA_50']:
                display_data[col] = display_data[col].map(lambda x: f"${x:.2f}" if not pd.isna(x) else "N/A")
            display_data['Volume'] = display_data['Volume'].map(lambda x: f"{x:,.0f}" if not pd.isna(x) else "N/A")
            display_data['RSI'] = display_data['RSI'].map(lambda x: f"{x:.1f}" if not pd.isna(x) else "N/A")
            
            st.dataframe(display_data)
            
    except Exception as e:
        st.error(f"❌ Hata oluştu: {str(e)}")

if __name__ == "__main__":
    main()