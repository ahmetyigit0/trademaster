import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =========================
# ŞİFRE KORUMASI
# =========================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    
    def password_entered():
        if st.session_state["password"] == "efe":
            st.session_state["password_correct"] = True
        else:
            st.session_state["password_correct"] = False
    
    if not st.session_state["password_correct"]:
        st.text_input("🔐 Şifre", type="password", on_change=password_entered, key="password")
        return False
    return True

if not check_password():
    st.stop()

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Crypto AI Pro", layout="wide")
st.title("🚀 Crypto AI Pro")
st.markdown("**Eğitim amaçlıdır; yatırım tavsiyesi değildir.**")

# =========================
# YARDIMCI FONKSİYONLAR
# =========================
@st.cache_data(ttl=600)
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=600)
def calculate_ema(prices, period):
    return prices.ewm(span=period, adjust=False).mean()

@st.cache_data(ttl=600)
def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    macd = ema_fast - ema_slow
    signal_line = calculate_ema(macd, signal)
    histogram = macd - signal_line
    return macd, signal_line, histogram

@st.cache_data(ttl=600)
def calculate_bollinger_bands(prices, period=20, std=2):
    sma = prices.rolling(period).mean()
    std_dev = prices.rolling(period).std()
    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)
    return upper_band, sma, lower_band

@st.cache_data(ttl=600)
def calculate_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

def create_bollinger_mini_chart(data, height=120):
    try:
        if len(data) < 20:
            return None
        
        recent_data = data.tail(20)
        fig, ax = plt.subplots(figsize=(6, height/80))
        
        bb_prox = (recent_data['Close'] - recent_data['BB_Lower']) / (recent_data['BB_Upper'] - recent_data['BB_Lower'])
        
        x_values = range(len(bb_prox))
        ax.plot(x_values, bb_prox.values, color='blue', linewidth=1.5)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        ax.axhline(y=1, color='green', linestyle='--', alpha=0.7, linewidth=1)
        ax.plot(len(bb_prox)-1, bb_prox.iloc[-1], 'ro', markersize=5)
        
        ax.set_ylim(-0.1, 1.1)
        ax.set_ylabel('BB Prox')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    except:
        return None

def create_rsi_mini_chart(data, height=120):
    try:
        if len(data) < 20:
            return None
        
        recent_data = data.tail(20)
        fig, ax = plt.subplots(figsize=(6, height/80))
        
        x_values = range(len(recent_data))
        ax.plot(x_values, recent_data['RSI'].values, color='purple', linewidth=1.5)
        ax.axhline(y=30, color='green', linestyle='--', alpha=0.8, linewidth=1.2)
        ax.axhline(y=70, color='red', linestyle='--', alpha=0.8, linewidth=1.2)
        ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
        ax.plot(len(recent_data)-1, recent_data['RSI'].iloc[-1], 'ro', markersize=5)
        
        ax.set_ylim(0, 100)
        ax.set_ylabel('RSI')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    except:
        return None

def create_macd_bars(data):
    try:
        recent_data = data.tail(5)
        macd_hist = recent_data['MACD_Hist']
        
        bars_html = ""
        for hist in macd_hist:
            color = "#00ff00" if hist > 0 else "#ff0000"
            height = min(abs(float(hist)) * 1000, 20)
            bars_html += f'<div style="display:inline-block; width:16px; height:{height}px; background-color:{color}; margin:0 3px; border: 1px solid #333;"></div>'
        
        return bars_html
    except:
        return ""

def create_price_mini_chart(data, height=120):
    try:
        if len(data) < 20:
            return None
        
        recent_data = data.tail(20)
        fig, ax = plt.subplots(figsize=(6, height/80))
        
        ax.plot(recent_data['Close'].values, color='blue', linewidth=2, label='Fiyat')
        ax.plot(recent_data['EMA_20'].values, color='orange', linewidth=1, label='EMA 20')
        ax.plot(recent_data['EMA_50'].values, color='red', linewidth=1, label='EMA 50')
        
        ax.set_ylabel('Fiyat')
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    except:
        return None

def create_fib_mini_chart(data, height=120):
    try:
        if len(data) < 10:
            return None
            
        recent_data = data.tail(10)
        high = recent_data['High'].max()
        low = recent_data['Low'].min()
        diff = high - low
        
        fib_levels = {
            '0.0': high,
            '0.236': high - diff * 0.236,
            '0.382': high - diff * 0.382, 
            '0.5': high - diff * 0.5,
            '0.618': high - diff * 0.618,
            '0.786': high - diff * 0.786,
            '1.0': low
        }
        
        current_price = data['Close'].iloc[-1]
        
        fig, ax = plt.subplots(figsize=(4, height/80))
        
        for level, price in fib_levels.items():
            color = 'red' if level in ['0.0', '1.0'] else 'gray'
            ax.axhline(y=price, color=color, linestyle='--', alpha=0.7, linewidth=0.8)
        
        ax.axhline(y=current_price, color='blue', linewidth=2, linestyle='-', alpha=0.8)
        
        ax.set_ylabel('Fib Seviyeleri')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    except:
        return None

def line(text, kind="neutral"):
    if kind == "pos":
        emoji, color = "🟢", "#0f9d58"
    elif kind == "neg":
        emoji, color = "🔴", "#d93025"
    else:
        emoji, color = "🟠", "#f29900"
    
    st.markdown(f"{emoji} <span style='color:{color}; font-weight:600;'>{text}</span>", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Analiz Ayarları")
ticker_input = st.sidebar.text_input("🎯 Kripto Sembolü", "BTC-USD")

timeframe = st.sidebar.selectbox("⏰ Zaman Dilimi", ["1d", "1wk"], index=0)
period_map = {"1d": "6mo", "1wk": "1y"}
period = period_map[timeframe]

st.sidebar.header("🎯 Risk Yönetimi")
capital = st.sidebar.number_input("💰 Sermaye ($)", 1000, 1000000, 5000, step=1000)
risk_percent = st.sidebar.slider("📉 İşlem Risk %", 1.0, 5.0, 2.0, 0.1)
max_position = st.sidebar.slider("📊 Maks. Pozisyon %", 10.0, 50.0, 25.0, 5.0)

st.sidebar.header("🔧 Strateji Parametreleri")
rsi_oversold = st.sidebar.slider("📊 RSI Aşırı Satım", 20, 40, 30, 1)
rsi_overbought = st.sidebar.slider("📈 RSI Aşırı Alım", 60, 80, 70, 1)
atr_multiplier = st.sidebar.slider("🎯 ATR Çarpanı", 1.0, 3.0, 1.5, 0.1)

# =========================
# ANA UYGULAMA
# =========================
try:
    with st.spinner(f"🔄 {ticker_input} verileri çekiliyor..."):
        data = yf.download(ticker_input, period=period, interval=timeframe, progress=False)
    
    if data.empty:
        st.error("❌ Veri çekilemedi - Sembolü ve internet bağlantınızı kontrol edin")
        st.stop()
    
    data['RSI'] = calculate_rsi(data['Close'])
    data['EMA_20'] = calculate_ema(data['Close'], 20)
    data['EMA_50'] = calculate_ema(data['Close'], 50)
    data['EMA_200'] = calculate_ema(data['Close'], 200)
    data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = calculate_macd(data['Close'])
    data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = calculate_bollinger_bands(data['Close'])
    data['ATR'] = calculate_atr(data['High'], data['Low'], data['Close'])
    
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
    
    buy_signals = 0
    sell_signals = 0
    
    if rsi < rsi_oversold: buy_signals += 1
    if current_price > ema_20 and ema_20 > ema_50: buy_signals += 1
    if ema_50 > ema_200: buy_signals += 1
    if macd > macd_signal and macd_prev <= macd_signal_prev: buy_signals += 1
    if current_price < bb_lower: buy_signals += 1
    
    if rsi > rsi_overbought: sell_signals += 1
    if current_price < ema_20 and ema_20 < ema_50: sell_signals += 1
    if ema_50 < ema_200: sell_signals += 1
    if macd < macd_signal and macd_prev >= macd_signal_prev: sell_signals += 1
    if current_price > bb_upper: sell_signals += 1
    
    tab_analiz, tab_rehber = st.tabs(["📈 Analiz", "📚 Rehber"])
    
    with tab_analiz:
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
        
        st.markdown("### 📈 Mini Göstergeler - Satır 1")
        col1, col2, col3, col4 = st.columns([2, 2, 1, 2])
        
        with col1:
            st.write("**Bollinger Yakınlık**")
            bb_chart = create_bollinger_mini_chart(data)
            if bb_chart:
                st.pyplot(bb_chart)
                plt.close()
            else:
                st.info("⏳")
        
        with col2:
            st.write("**RSI Momentum**")
            rsi_chart = create_rsi_mini_chart(data)
            if rsi_chart:
                st.pyplot(rsi_chart)
                plt.close()
            else:
                st.info("⏳")
        
        with col3:
            st.write("**MACD Histogram**")
            macd_bars = create_macd_bars(data)
            if macd_bars:
                st.markdown(f'<div style="text-align: center; padding: 20px 0;">{macd_bars}</div>', unsafe_allow_html=True)
                current_hist = data['MACD_Hist'].iloc[-1]
                direction = "🟢 Pozitif" if current_hist > 0 else "🔴 Negatif"
                st.write(f"**{direction}**")
            else:
                st.info("⏳")
        
        with col4:
            st.write("**Fiyat & EMA**")
            price_chart = create_price_mini_chart(data)
            if price_chart:
                st.pyplot(price_chart)
                plt.close()
            else:
                st.info("⏳")
        
        st.markdown("### 📊 Mini Göstergeler - Satır 2")
        col5, col6, col7, col8 = st.columns([2, 2, 1, 2])
        
        with col5:
            st.write("**Fibonacci Seviyeleri**")
            fib_chart = create_fib_mini_chart(data)
            if fib_chart:
                st.pyplot(fib_chart)
                plt.close()
            else:
                st.info("⏳")
        
        with col6:
            st.write("**Volatilite (ATR)**")
            st.metric("ATR Değeri", f"${atr:.2f}")
            vol_ratio = atr / current_price * 100
            st.metric("Volatilite %", f"%{vol_ratio:.1f}")
        
        with col7:
            st.write("**Sinyal Özeti**")
            st.write(f"Al: **{buy_signals}**/5")
            st.write(f"Sat: **{sell_signals}**/5")
        
        with col8:
            st.write("**Bollinger Durumu**")
            if current_price < bb_lower:
                st.info("Alt Bantta")
            elif current_price > bb_upper:
                st.warning("Üst Bantta")
            else:
                st.success("Bant İçinde")
        
        st.markdown("---")
        
        risk_score = min(100, abs(buy_signals - sell_signals) * 20)
        
        if buy_signals >= 4:
            st.success(f"🎯 **GÜÇLÜ AL SİNYALİ** - Al: {buy_signals}/5 | Sat: {sell_signals}/5")
            recommendation = "AL"
        elif sell_signals >= 4:
            st.error(f"🎯 **GÜÇLÜ SAT SİNYALİ** - Al: {buy_signals}/5 | Sat: {sell_signals}/5")
            recommendation = "SAT"
        elif buy_signals > sell_signals:
            st.warning(f"🎯 **ZAYIF AL SİNYALİ** - Al: {buy_signals}/5 | Sat: {sell_signals}/5")
            recommendation = "AL"
        elif sell_signals > buy_signals:
            st.warning(f"🎯 **ZAYIF SAT SİNYALİ** - Al: {buy_signals}/5 | Sat: {sell_signals}/5")
            recommendation = "SAT"
        else:
            st.info(f"🎯 **NÖTR SİNYAL** - Al: {buy_signals}/5 | Sat: {sell_signals}/5")
            recommendation = "BEKLE"
        
        st.markdown("---")
        
        if recommendation in ["AL", "SAT"]:
            st.subheader("🎯 Detaylı İşlem Stratejisi")
            
            if recommendation == "AL":
                stop_loss = current_price - (atr * atr_multiplier)
                risk_per_coin = current_price - stop_loss
                
                tp1 = current_price + (risk_per_coin * 1.0)
                tp2 = current_price + (risk_per_coin * 2.0)
                tp3 = current_price + (risk_per_coin * 3.0)
                
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
                    st.write(f"- 🎯 TP1 (1R): `${tp1:.2f}`")
                    st.write(f"- 🎯 TP2 (2R): `${tp2:.2f}`")
                    st.write(f"- 🎯 TP3 (3R): `${tp3:.2f}`")
                    
                with col2:
                    st.write("**💰 Pozisyon Bilgileri:**")
                    st.write(f"- 📊 Pozisyon: `{final_position_size:.4f} {ticker_input.split('-')[0]}`")
                    st.write(f"- 💰 Değer: `${position_value:.2f}`")
                    st.write(f"- 📉 Risk: `${risk_amount:.2f}`")
                    st.write(f"- ⚖️ Risk/Reward: `1:3`")
                    st.write(f"- 🎯 Başarı Şansı: `%{min(80, risk_score + 30):.0f}`")
            
            else:
                base_level = max(float(data['Low'].tail(20).min()), bb_lower)
                reentry_low = base_level - (atr * 0.5)
                reentry_high = base_level + (atr * 0.5)
                
                st.write("**📉 SAT Stratejisi - Yeniden Alım Bölgesi:**")
                st.write(f"- 🎯 Taban Seviye: `${base_level:.2f}`")
                st.write(f"- 📥 Alım Bölgesi: `${reentry_low:.2f}` - `${reentry_high:.2f}`")
                st.write(f"- 📊 Kademeli Alım:**")
                st.write(f"  - %50: `${reentry_low:.2f}` - `${base_level:.2f}`")
                st.write(f"  - %30: `${base_level:.2f}`")
                st.write(f"  - %20: `${base_level:.2f}` - `${reentry_high:.2f}`")
                st.write(f"- 🛑 Stop Loss: `${reentry_low - atr:.2f}`")
        
        st.markdown("---")
        
        st.subheader("🧠 Detaylı Sinyal Gerekçeleri")
        
        line(f"EMA 20 (${ema_20:.2f}) > EMA 50 (${ema_50:.2f}) > EMA 200 (${ema_200:.2f})", 
             "pos" if ema_20 > ema_50 > ema_200 else "neg" if ema_20 < ema_50 < ema_200 else "neutral")
        
        line(f"Fiyat EMA 20'nin {'üstünde' if current_price > ema_20 else 'altında'}", 
             "pos" if current_price > ema_20 else "neg")
        
        if rsi < 30:
            line(f"RSI {rsi:.1f} - Aşırı satım bölgesi (AL sinyali)", "pos")
        elif rsi > 70:
            line(f"RSI {rsi:.1f} - Aşırı alım bölgesi (SAT sinyali)", "neg")
        else:
            line(f"RSI {rsi:.1f} - Nötr bölge", "neutral")
        
        if macd > macd_signal:
            line(f"MACD ({macd:.4f}) > Sinyal ({macd_signal:.4f}) - Pozitif momentum", "pos")
        else:
            line(f"MACD ({macd:.4f}) < Sinyal ({macd_signal:.4f}) - Negatif momentum", "neg")
        
        if current_price < bb_lower:
            line(f"Fiyat Bollinger alt bandında - Potansiyel dip", "pos")
        elif current_price > bb_upper:
            line(f"Fiyat Bollinger üst bandında - Potansiyel tepe", "neg")
        else:
            line(f"Fiyat Bollinger bantları içinde - Nötr", "neutral")
        
        vol_ratio = atr / current_price * 100
        if vol_ratio > 5:
            line(f"Yüksek volatilite (%{vol_ratio:.1f}) - Dikkatli pozisyon", "neg")
        elif vol_ratio < 2:
            line(f"Düşük volatilite (%{vol_ratio:.1f}) - Sakin piyasa", "pos")
        else:
            line(f"Orta volatilite (%{vol_ratio:.1f}) - Normal risk", "neutral")
    
    with tab_rehber:
        st.subheader("📚 Teknik Analiz Rehberi")
        
        st.markdown("""
        ### 📊 RSI (Relative Strength Index)
        - **14 periyot** standarttır
        - **<30**: Aşırı satım - Potansiyel alım fırsatı
        - **>70**: Aşırı alım - Potansiyel satım sinyali
        - **30-70**: Nötr bölge - Trend takibi önemli
        
        ### 📈 MACD (Moving Average Convergence Divergence)
        - **MACD > Sinyal**: Yukarı momentum
        - **MACD < Sinyal**: Aşağı momentum  
        - **Kesişimler**: Trend değişim sinyali
        - **Histogram**: Momentum gücü
        
        ### 🎯 EMA (Exponential Moving Average)
        - **EMA 20**: Kısa vade trend
        - **EMA 50**: Orta vade trend
        - **EMA 200**: Uzun vade trend
        - **Sıralama**: EMA20 > EMA50 > EMA200 = Güçlü yükseliş
        
        ### 📉 Bollinger Bantları
        - **Üst/Alt bant**: Volatilite göstergesi
        - **Daralma**: Volatilite düşüşü, kırılım yakın
        - **Genişleme**: Volatilite artışı
        - **Alt bant testi**: Potansiyel alım
        - **Üst bant testi**: Potansiyel satım
        
        ### ⚖️ Risk/Reward (R:R) Oranı
        - **R = Fiyat - Stop Loss** (Risk birimi)
        - **TP1 = 1R, TP2 = 2R, TP3 = 3R** (Take Profit)
        - **Minimum 1:2 R:R** önerilir
        - **Formül**: Beklenen Getiri = (Kazanç Oranı × R:R) - Kayıp Oranı
        
        ### 💰 Kademeli Alım Stratejisi
        1. **İlk giriş** %50 - Mevcut fiyat
        2. **İkinci giriş** %30 - Dip seviyelerde  
        3. **Son giriş** %20 - Trend onayında
        4. **Stop loss** tüm pozisyon için ortak
        
        ### 🛑 Risk Yönetimi
        - **Pozisyon büyüklüğü** = (Sermaye × Risk%) / R
        - **Maksimum pozisyon** %25'i geçmemeli
        - **Stop loss** olmadan işlem yapılmaz
        - **Emotional trading**'den kaçının
        """)
        
except Exception as e:
    st.error(f"❌ Sistem hatası: {str(e)}")
    st.info("Lütfen internet bağlantınızı kontrol edin ve tekrar deneyin")

st.markdown("---")
st.caption("🤖 Crypto AI Pro - Gelişmiş Algoritmik Analiz Sistemi | V2.0")