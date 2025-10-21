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

# Sidebar
st.sidebar.header("⚙️ Ayarlar")
ticker = st.sidebar.selectbox("Coin", ["BTC-USD", "ETH-USD", "ADA-USD", "BNB-USD", "XRP-USD", "SOL-USD", "DOT-USD"])
timeframe = st.sidebar.selectbox("Zaman Dilimi", ["1h", "4h", "1d", "1wk"])

# Risk yönetimi
st.sidebar.header("🎯 Risk Yönetimi")
capital = st.sidebar.number_input("Sermaye ($)", 1000, 100000, 5000)
risk_percent = st.sidebar.slider("Risk %", 1.0, 5.0, 2.0)

try:
    # Veri çek
    data = yf.download(ticker, period="3mo", interval=timeframe, progress=False)
    
    if data.empty:
        st.error("❌ Veri çekilemedi")
    else:
        # İndikatörleri hesapla
        data['RSI'] = calculate_rsi(data['Close'])
        data['EMA_20'] = calculate_ema(data['Close'], 20)
        data['EMA_50'] = calculate_ema(data['Close'], 50)
        data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = calculate_macd(data['Close'])
        
        current_price = data['Close'].iloc[-1]
        rsi = data['RSI'].iloc[-1]
        ema_20 = data['EMA_20'].iloc[-1]
        ema_50 = data['EMA_50'].iloc[-1]
        macd = data['MACD'].iloc[-1]
        macd_signal = data['MACD_Signal'].iloc[-1]
        
        # Sinyal belirleme
        buy_signals = 0
        sell_signals = 0
        
        # AL koşulları
        if rsi < 35:
            buy_signals += 1
        if current_price > ema_20 > ema_50:
            buy_signals += 1
        if macd > macd_signal and data['MACD'].iloc[-2] <= data['MACD_Signal'].iloc[-2]:
            buy_signals += 1
        if data['Close'].iloc[-1] > data['Close'].iloc[-2] > data['Close'].iloc[-3]:
            buy_signals += 1
            
        # SAT koşulları
        if rsi > 65:
            sell_signals += 1
        if current_price < ema_20 < ema_50:
            sell_signals += 1
        if macd < macd_signal and data['MACD'].iloc[-2] >= data['MACD_Signal'].iloc[-2]:
            sell_signals += 1
        if data['Close'].iloc[-1] < data['Close'].iloc[-2] < data['Close'].iloc[-3]:
            sell_signals += 1
        
        # Ana metrikler
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💰 Fiyat", f"${current_price:.2f}")
        with col2:
            st.metric("📊 RSI", f"{rsi:.1f}")
        with col3:
            trend = "YÜKSELİŞ" if ema_20 > ema_50 else "DÜŞÜŞ"
            st.metric("🎯 Trend", trend)
        with col4:
            st.metric("📈 MACD", f"{macd:.4f}")
        
        st.markdown("---")
        
        # SİNYAL BÖLÜMÜ
        if buy_signals >= 3:
            st.success("🎯 **AL SİNYALİ** - Güçlü alım fırsatı")
            
            # AL stratejisi detayları
            st.subheader("📈 AL Stratejisi Detayları")
            
            # Stop loss ve TP hesaplama
            atr = data['High'].subtract(data['Low']).rolling(14).mean().iloc[-1]
            stop_loss = current_price - (atr * 1.5)
            risk_per_coin = current_price - stop_loss
            
            tp1 = current_price + (risk_per_coin * 1.5)
            tp2 = current_price + (risk_per_coin * 2.5)
            tp3 = current_price + (risk_per_coin * 4)
            
            # Pozisyon büyüklüğü
            risk_amount = capital * (risk_percent / 100)
            position_size = risk_amount / risk_per_coin
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**🎯 Giriş Seviyeleri:**")
                st.write(f"- İlk Giriş: ${current_price:.2f}")
                st.write(f"- Ortalama: ${(current_price * 0.97):.2f} (dip alım)")
                
            with col2:
                st.write("**💰 Take Profit Seviyeleri:**")
                st.write(f"- TP1: ${tp1:.2f} (1.5R)")
                st.write(f"- TP2: ${tp2:.2f} (2.5R)") 
                st.write(f"- TP3: ${tp3:.2f} (4R)")
            
            st.write("**🛑 Stop Loss & Risk:**")
            st.write(f"- Stop Loss: ${stop_loss:.2f}")
            st.write(f"- Pozisyon Büyüklüğü: {position_size:.4f} coin")
            st.write(f"- Risk/Mükafat: 1:1.5 (TP1)")
            
        elif sell_signals >= 3:
            st.error("🎯 **SAT SİNYALİ** - Düşüş bekleniyor")
            
            # SAT stratejisi detayları
            st.subheader("📉 SAT Stratejisi Detayları")
            
            # Short için seviyeler
            resistance = data['High'].tail(10).max()
            entry_short = current_price
            stop_loss_short = resistance * 1.02  # %2 üstü
            risk_short = stop_loss_short - entry_short
            
            tp_short1 = entry_short - (risk_short * 1)
            tp_short2 = entry_short - (risk_short * 2)
            tp_short3 = entry_short - (risk_short * 3)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**🎯 Short Giriş Seviyeleri:**")
                st.write(f"- Mevcut: ${entry_short:.2f}")
                st.write(f"- Direniş: ${resistance:.2f}")
                
            with col2:
                st.write("**💰 Take Profit Seviyeleri:**")
                st.write(f"- TP1: ${tp_short1:.2f} (1R)")
                st.write(f"- TP2: ${tp_short2:.2f} (2R)")
                st.write(f"- TP3: ${tp_short3:.2f} (3R)")
            
            st.write("**🛑 Risk Yönetimi:**")
            st.write(f"- Stop Loss: ${stop_loss_short:.2f}")
            st.write(f"- Hedef Destek: ${data['Low'].tail(20).min():.2f}")
            
        else:
            st.info("🎯 **NÖTR** - Bekle ve izle")
            st.write("Yeterli sinyal yok. Piyasa belirsiz veya yanal hareket ediyor.")
        
        st.markdown("---")
        
        # GEREKÇELER
        st.subheader("🧠 Sinyal Gerekçeleri")
        
        st.write("**Teknik Göstergeler:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"- RSI: {rsi:.1f} {'(Aşırı Satım)' if rsi < 30 else '(Aşırı Alım)' if rsi > 70 else '(Nötr)'}")
            st.write(f"- EMA 20/50: {ema_20:.2f} / {ema_50:.2f}")
            st.write(f"- Trend: {'Yükseliş' if ema_20 > ema_50 else 'Düşüş'}")
            
        with col2:
            st.write(f"- MACD: {macd:.4f}")
            st.write(f"- MACD Sinyal: {macd_signal:.4f}")
            st.write(f"- MACD Hist: {(macd - macd_signal):.4f}")
        
        st.write("**Piyasa Dinamiği:**")
        st.write(f"- Al Sinyalleri: {buy_signals}/4")
        st.write(f"- Sat Sinyalleri: {sell_signals}/4")
        st.write(f"- Son Kapanış: {'Yükseliş' if data['Close'].iloc[-1] > data['Close'].iloc[-2] else 'Düşüş'}")
        
        # Son 5 mum
        st.write("**Son 5 Mum Performansı:**")
        recent_closes = data['Close'].tail(5)
        for i in range(len(recent_closes)-1):
            change = ((recent_closes.iloc[i+1] - recent_closes.iloc[i]) / recent_closes.iloc[i]) * 100
            st.write(f"- Mum {i+1}: {recent_closes.iloc[i+1]:.2f} ({change:+.2f}%)")

except Exception as e:
    st.error(f"❌ Hata oluştu: {str(e)}")
    st.info("İnternet bağlantınızı kontrol edin veya daha sonra tekrar deneyin.")