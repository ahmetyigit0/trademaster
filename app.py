import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Crypto AI Pro", layout="wide")
st.title("🚀 Crypto AI Pro")
st.markdown("**Eğitim amaçlıdır - Yatırım tavsiyesi DEĞİLDİR**")

# Yardımcı fonksiyonlar
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_ema(prices, period):
    return prices.ewm(span=period, adjust=False).mean()

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    macd = ema_fast - ema_slow
    signal_line = calculate_ema(macd, signal)
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(prices, period=20, std=2):
    sma = prices.rolling(period).mean()
    std_dev = prices.rolling(period).std()
    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)
    return upper_band, sma, lower_band

def calculate_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

# Sidebar
st.sidebar.header("⚙️ Analiz Ayarları")
ticker_input = st.sidebar.text_input("🎯 Kripto Sembolü", "BTC-USD", help="Örnek: BTC-USD, ETH-USD, ADA-USD, BNB-USD, XRP-USD, SOL-USD")

timeframe = st.sidebar.selectbox("⏰ Zaman Dilimi", ["1h", "4h", "1d", "1wk"], index=2)
period_map = {"1h": "1mo", "4h": "3mo", "1d": "6mo", "1wk": "1y"}
period = period_map[timeframe]

# Risk yönetimi
st.sidebar.header("🎯 Risk Yönetimi")
capital = st.sidebar.number_input("💰 Sermaye ($)", 1000, 1000000, 5000, step=1000)
risk_percent = st.sidebar.slider("📉 İşlem Risk %", 1.0, 5.0, 2.0, 0.1)
max_position = st.sidebar.slider("📊 Maks. Pozisyon %", 10.0, 50.0, 25.0, 5.0)

# Strateji ayarları
st.sidebar.header("🔧 Strateji Parametreleri")
rsi_oversold = st.sidebar.slider("📊 RSI Aşırı Satım", 20, 40, 30, 1)
rsi_overbought = st.sidebar.slider("📈 RSI Aşırı Alım", 60, 80, 70, 1)
atr_multiplier = st.sidebar.slider("🎯 ATR Çarpanı", 1.0, 3.0, 1.5, 0.1)

try:
    # Veri çek
    with st.spinner(f"🔄 {ticker_input} verileri çekiliyor..."):
        data = yf.download(ticker_input, period=period, interval=timeframe, progress=False)
    
    if data.empty:
        st.error("❌ Veri çekilemedi - Sembolü kontrol edin")
    else:
        # İndikatörleri hesapla
        data['RSI'] = calculate_rsi(data['Close'])
        data['EMA_20'] = calculate_ema(data['Close'], 20)
        data['EMA_50'] = calculate_ema(data['Close'], 50)
        data['EMA_200'] = calculate_ema(data['Close'], 200)
        data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = calculate_macd(data['Close'])
        data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = calculate_bollinger_bands(data['Close'])
        data['ATR'] = calculate_atr(data['High'], data['Low'], data['Close'])
        
        # Mevcut değerler
        current_price = float(data['Close'].iloc[-1])
        rsi = float(data['RSI'].iloc[-1])
        ema_20 = float(data['EMA_20'].iloc[-1])
        ema_50 = float(data['EMA_50'].iloc[-1])
        ema_200 = float(data['EMA_200'].iloc[-1])
        macd = float(data['MACD'].iloc[-1])
        macd_signal = float(data['MACD_Signal'].iloc[-1])
        macd_prev = float(data['MACD'].iloc[-2])
        macd_signal_prev = float(data['MACD_Signal'].iloc[-2])
        atr = float(data['ATR'].iloc[-1])
        bb_upper = float(data['BB_Upper'].iloc[-1])
        bb_lower = float(data['BB_Lower'].iloc[-1])
        
        # Sinyal belirleme
        buy_signals = 0
        sell_signals = 0
        signal_details = []
        
        # AL koşulları
        if rsi < rsi_oversold:
            buy_signals += 1
            signal_details.append(f"✅ RSI {rsi:.1f} - Aşırı satım bölgesi")
        
        if current_price > ema_20 and ema_20 > ema_50:
            buy_signals += 1
            signal_details.append("✅ EMA 20 > EMA 50 - Kısa vade trend yükseliş")
        
        if ema_50 > ema_200:
            buy_signals += 1
            signal_details.append("✅ EMA 50 > EMA 200 - Uzun vade trend yükseliş")
        
        if macd > macd_signal and macd_prev <= macd_signal_prev:
            buy_signals += 1
            signal_details.append("✅ MACD sinyali yukarı kesti - Momentum pozitif")
        
        if current_price < bb_lower:
            buy_signals += 1
            signal_details.append("✅ Fiyat Bollinger alt bandında - Potansiyel dip")
        
        # SAT koşulları
        if rsi > rsi_overbought:
            sell_signals += 1
            signal_details.append(f"❌ RSI {rsi:.1f} - Aşırı alım bölgesi")
        
        if current_price < ema_20 and ema_20 < ema_50:
            sell_signals += 1
            signal_details.append("❌ EMA 20 < EMA 50 - Kısa vade trend düşüş")
        
        if ema_50 < ema_200:
            sell_signals += 1
            signal_details.append("❌ EMA 50 < EMA 200 - Uzun vade trend düşüş")
        
        if macd < macd_signal and macd_prev >= macd_signal_prev:
            sell_signals += 1
            signal_details.append("❌ MACD sinyali aşağı kesti - Momentum negatif")
        
        if current_price > bb_upper:
            sell_signals += 1
            signal_details.append("❌ Fiyat Bollinger üst bandında - Potansiyel tepe")
        
        # Ana metrikler
        st.subheader(f"📊 {ticker_input} - Gerçek Zamanlı Analiz")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            price_change = ((current_price - float(data['Close'].iloc[-2])) / float(data['Close'].iloc[-2])) * 100
            st.metric("💰 Mevcut Fiyat", f"${current_price:.2f}", f"{price_change:+.2f}%")
        
        with col2:
            rsi_color = "🟢" if rsi < 30 else "🔴" if rsi > 70 else "🟡"
            st.metric("📊 RSI", f"{rsi:.1f}", f"{rsi_color}")
        
        with col3:
            trend = "🟢 YÜKSELİŞ" if ema_20 > ema_50 and ema_50 > ema_200 else "🔴 DÜŞÜŞ" if ema_20 < ema_50 and ema_50 < ema_200 else "🟡 YANAL"
            st.metric("🎯 Trend", trend)
        
        with col4:
            macd_trend = "🟢 YUKARI" if macd > macd_signal else "🔴 AŞAĞI"
            st.metric("📈 MACD", f"{macd:.4f}", macd_trend)
        
        st.markdown("---")
        
        # SİNYAL VE RİSK ANALİZİ
        total_signals = buy_signals + sell_signals
        buy_ratio = (buy_signals / total_signals * 100) if total_signals > 0 else 0
        risk_score = min(100, abs(buy_signals - sell_signals) * 20)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if buy_signals >= 4:
                st.success(f"🎯 **GÜÇLÜ AL SİNYALİ**\n\nAl: {buy_signals}/8 | Sat: {sell_signals}/8")
                signal_strength = "YÜKSEK"
                recommendation = "AL"
            elif sell_signals >= 4:
                st.error(f"🎯 **GÜÇLÜ SAT SİNYALİ**\n\nAl: {buy_signals}/8 | Sat: {sell_signals}/8")
                signal_strength = "YÜKSEK"
                recommendation = "SAT"
            elif buy_signals > sell_signals:
                st.warning(f"🎯 **ZAYIF AL SİNYALİ**\n\nAl: {buy_signals}/8 | Sat: {sell_signals}/8")
                signal_strength = "ORTA"
                recommendation = "AL"
            elif sell_signals > buy_signals:
                st.warning(f"🎯 **ZAYIF SAT SİNYALİ**\n\nAl: {buy_signals}/8 | Sat: {sell_signals}/8")
                signal_strength = "ORTA"
                recommendation = "SAT"
            else:
                st.info(f"🎯 **NÖTR SİNYAL**\n\nAl: {buy_signals}/8 | Sat: {sell_signals}/8")
                signal_strength = "DÜŞÜK"
                recommendation = "BEKLE"
        
        with col2:
            st.metric("📊 Sinyal Gücü", signal_strength)
            st.metric("🎯 Risk Skoru", f"%{risk_score:.0f}")
        
        with col3:
            st.metric("💰 Öneri", recommendation)
            st.metric("📈 Güven Oranı", f"%{buy_ratio:.0f}" if recommendation == "AL" else f"%{100-buy_ratio:.0f}")
        
        st.markdown("---")
        
        # DETAYLI STRATEJİ
        if recommendation in ["AL", "SAT"]:
            st.subheader("🎯 Detaylı İşlem Stratejisi")
            
            if recommendation == "AL":
                # AL stratejisi
                stop_loss = current_price - (atr * atr_multiplier)
                risk_per_coin = current_price - stop_loss
                
                tp1 = current_price + (risk_per_coin * 1.0)
                tp2 = current_price + (risk_per_coin * 2.0)
                tp3 = current_price + (risk_per_coin * 3.0)
                
                # Pozisyon büyüklüğü
                risk_amount = capital * (risk_percent / 100)
                position_size = risk_amount / risk_per_coin
                max_position_size = (capital * (max_position / 100)) / current_price
                final_position_size = min(position_size, max_position_size)
                position_value = final_position_size * current_price
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**🎯 Giriş ve Çıkış Seviyeleri:**")
                    st.write(f"- 📈 Mevcut Fiyat: `${current_price:.2f}`")
                    st.write(f"- 🛑 Stop Loss: `${stop_loss:.2f}` (%{((current_price-stop_loss)/current_price*100):.1f})")
                    st.write(f"- 🎯 TP1 (1:1): `${tp1:.2f}`")
                    st.write(f"- 🎯 TP2 (1:2): `${tp2:.2f}`")
                    st.write(f"- 🎯 TP3 (1:3): `${tp3:.2f}`")
                    
                with col2:
                    st.write("**💰 Pozisyon Bilgileri:**")
                    st.write(f"- 📊 Pozisyon Büyüklüğü: `{final_position_size:.4f} {ticker_input.split('-')[0]}`")
                    st.write(f"- 💰 Pozisyon Değeri: `${position_value:.2f}`")
                    st.write(f"- 📉 Risk Tutarı: `${risk_amount:.2f}`")
                    st.write(f"- ⚖️ Risk/Reward: `1:3`")
                    st.write(f"- 🎯 Başarı Şansı: `%{min(80, risk_score + 30):.0f}`")
            
            else:
                # SAT stratejisi
                resistance = float(data['High'].tail(10).max())
                stop_loss = resistance * 1.02
                risk_per_coin = stop_loss - current_price
                
                tp1 = current_price - (risk_per_coin * 1.0)
                tp2 = current_price - (risk_per_coin * 2.0)
                tp3 = current_price - (risk_per_coin * 3.0)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**🎯 Short Seviyeleri:**")
                    st.write(f"- 📉 Mevcut Fiyat: `${current_price:.2f}`")
                    st.write(f-