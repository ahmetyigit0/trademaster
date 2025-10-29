import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import time
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sayfa ayarı
st.set_page_config(
    page_title="🚀 Crypto Trading Dashboard",
    page_icon="📊",
    layout="wide"
)

# Başlık
st.title("🚀 Crypto Trading Dashboard")
st.markdown("---")

# Geri sayım ve veri yenileme
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()

# Real-time fiyatları getiren fonksiyon
@st.cache_data(ttl=10)  # 10 saniye cache
def get_real_time_prices(symbols):
    prices = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d', interval='1m')
            if not data.empty:
                current_price = data['Close'].iloc[-1]
                change = ((current_price - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
                prices[symbol] = {
                    'price': current_price,
                    'change': change
                }
        except:
            prices[symbol] = {'price': 0, 'change': 0}
    return prices

# Üstte real-time fiyatlar
st.subheader("📈 Real-Time Crypto Prices")

# Crypto sembolleri
crypto_symbols = {
    'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum', 
    'BNB-USD': 'Binance Coin',
    'XRP-USD': 'XRP',
    'THETA-USD': 'Theta',
    'AVAX-USD': 'Avalanche'
}

# Fiyatları göster
try:
    prices = get_real_time_prices(list(crypto_symbols.keys()))
    
    # 6 kolon oluştur
    cols = st.columns(6)
    
    for idx, (symbol, name) in enumerate(crypto_symbols.items()):
        with cols[idx]:
            if symbol in prices:
                price_data = prices[symbol]
                change_color = "green" if price_data['change'] >= 0 else "red"
                st.metric(
                    label=name,
                    value=f"${price_data['price']:.2f}",
                    delta=f"{price_data['change']:+.2f}%"
                )
            else:
                st.metric(label=name, value="N/A")
    
    # Geri sayım
    current_time = time.time()
    time_since_update = current_time - st.session_state.last_update
    countdown = max(0, 10 - int(time_since_update))
    
    st.caption(f"🔄 Veriler {countdown} saniye içinde yenilenecek...")
    
    if countdown == 0:
        st.session_state.last_update = current_time
        st.rerun()
        
except Exception as e:
    st.error("Fiyat verileri yüklenirken hata oluştu")

st.markdown("---")

# Sol sidebar - Sinyal analizi
st.sidebar.header("🔍 Crypto Signal Analysis")

# Kripto seçimi
crypto_options = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD", 
    "Binance Coin (BNB-USD)": "BNB-USD",
    "XRP (XRP-USD)": "XRP-USD",
    "Cardano (ADA-USD)": "ADA-USD",
    "Solana (SOL-USD)": "SOL-USD",
    "Polkadot (DOT-USD)": "DOT-USD",
    "Dogecoin (DOGE-USD)": "DOGE-USD",
    "Avalanche (AVAX-USD)": "AVAX-USD",
    "Polygon (MATIC-USD)": "MATIC-USD",
    "Litecoin (LTC-USD)": "LTC-USD",
    "Chainlink (LINK-USD)": "LINK-USD"
}

selected_crypto = st.sidebar.selectbox("Select Crypto:", list(crypto_options.keys()))
symbol = crypto_options[selected_crypto]

# Zaman ayarları
st.sidebar.subheader("⚡ Time Settings")
timeframe = st.sidebar.selectbox("Timeframe:", ["15m", "1h", "4h", "1d", "1wk"], index=2)
period_days = st.sidebar.slider("Data Period (Days):", 30, 365, 90)

# Gelişmiş gösterge sınıfı
class AdvancedTechnicalAnalysis:
    def __init__(self):
        pass
    
    def calculate_rsi(self, prices, window=14):
        """RSI hesapla"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rs = rs.fillna(1)
        return 100 - (100 / (1 + rs))
    
    def calculate_ema(self, prices, periods):
        """EMA hesapla"""
        return {f'EMA_{period}': prices.ewm(span=period).mean() for period in periods}
    
    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Bollinger Bands hesapla"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        return {
            'BB_Upper': upper_band,
            'BB_Middle': sma,
            'BB_Lower': lower_band,
            'BB_Width': (upper_band - lower_band) / sma  # Bant genişliği
        }
    
    def calculate_fibonacci_levels(self, high, low):
        """Fibonacci seviyelerini hesapla"""
        diff = high - low
        return {
            'Fib_0.236': high - diff * 0.236,
            'Fib_0.382': high - diff * 0.382,
            'Fib_0.5': high - diff * 0.5,
            'Fib_0.618': high - diff * 0.618,
            'Fib_0.786': high - diff * 0.786
        }
    
    def calculate_atr(self, high, low, close, period=14):
        """Average True Range hesapla"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        return tr.rolling(period).mean()
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """MACD hesapla"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return {
            'MACD': macd,
            'MACD_Signal': signal_line,
            'MACD_Histogram': histogram
        }
    
    def calculate_stochastic(self, high, low, close, k_period=14, d_period=3):
        """Stochastic Oscillator hesapla"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_period).mean()
        
        return {
            'Stoch_K': k,
            'Stoch_D': d
        }
    
    def get_all_indicators(self, df):
        """Tüm göstergeleri hesapla"""
        df = df.copy()
        
        # RSI
        df['RSI_14'] = self.calculate_rsi(df['Close'], 14)
        df['RSI_7'] = self.calculate_rsi(df['Close'], 7)
        
        # EMA'lar
        ema_periods = [8, 21, 50, 100, 200]
        emas = self.calculate_ema(df['Close'], ema_periods)
        for key, value in emas.items():
            df[key] = value
        
        # Bollinger Bands
        bb = self.calculate_bollinger_bands(df['Close'])
        for key, value in bb.items():
            df[key] = value
        
        # MACD
        macd = self.calculate_macd(df['Close'])
        for key, value in macd.items():
            df[key] = value
        
        # ATR
        df['ATR'] = self.calculate_atr(df['High'], df['Low'], df['Close'])
        
        # Stochastic
        stoch = self.calculate_stochastic(df['High'], df['Low'], df['Close'])
        for key, value in stoch.items():
            df[key] = value
        
        # Fibonacci (son 20 gün için)
        recent_high = df['High'].tail(20).max()
        recent_low = df['Low'].tail(20).min()
        fib_levels = self.calculate_fibonacci_levels(recent_high, recent_low)
        
        # Volume göstergeleri
        df['Volume_SMA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Momentum
        df['Momentum_5'] = df['Close'].pct_change(5)
        df['Momentum_10'] = df['Close'].pct_change(10)
        
        return df.fillna(0), fib_levels

# Veri yükleme
@st.cache_data
def load_crypto_data(symbol, period_days, timeframe):
    try:
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=period_days)
        data = yf.download(symbol, start=start_date, end=end_date, interval=timeframe, progress=False)
        return data
    except Exception as e:
        st.error(f"Data loading error: {e}")
        return None

# Sinyal analizini göster
def display_signal_analysis(df, fib_levels):
    if df is None or df.empty:
        st.error("No data available for analysis")
        return
    
    ta = AdvancedTechnicalAnalysis()
    current_data = df.iloc[-1]
    
    st.subheader(f"📊 Technical Analysis for {selected_crypto}")
    
    # Ana metrikler
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = current_data['Close']
        st.metric("Current Price", f"${current_price:.4f}")
    
    with col2:
        rsi = current_data['RSI_14']
        rsi_status = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
        st.metric("RSI (14)", f"{rsi:.1f}", rsi_status)
    
    with col3:
        bb_position = (current_price - current_data['BB_Lower']) / (current_data['BB_Upper'] - current_data['BB_Lower'])
        bb_status = "Upper Band" if bb_position > 0.8 else "Lower Band" if bb_position < 0.2 else "Middle"
        st.metric("Bollinger Position", f"{bb_position:.1%}", bb_status)
    
    with col4:
        atr = current_data['ATR']
        atr_percent = (atr / current_price) * 100
        st.metric("ATR", f"{atr_percent:.2f}%")
    
    st.markdown("---")
    
    # Detaylı göstergeler
    st.subheader("🔍 Detailed Indicators")
    
    # ROW 1: Trend göstergeleri
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.write("**Trend Analysis**")
        ema_8 = current_data['EMA_8']
        ema_21 = current_data['EMA_21']
        ema_50 = current_data['EMA_50']
        
        trend_score = 0
        if current_price > ema_8 > ema_21 > ema_50:
            trend = "🟢 Strong Uptrend"
            trend_score = 3
        elif current_price > ema_21 > ema_50:
            trend = "🟡 Moderate Uptrend"
            trend_score = 2
        elif current_price > ema_50:
            trend = "🟠 Weak Uptrend"
            trend_score = 1
        elif current_price < ema_8 < ema_21 < ema_50:
            trend = "🔴 Strong Downtrend"
            trend_score = -3
        elif current_price < ema_21 < ema_50:
            trend = "🟣 Moderate Downtrend" 
            trend_score = -2
        elif current_price < ema_50:
            trend = "🔵 Weak Downtrend"
            trend_score = -1
        else:
            trend = "⚪ Sideways"
        
        st.write(trend)
        st.progress((trend_score + 3) / 6)
    
    with col2:
        st.write("**MACD Signal**")
        macd = current_data['MACD']
        macd_signal = current_data['MACD_Signal']
        macd_hist = current_data['MACD_Histogram']
        
        if macd > macd_signal and macd_hist > 0:
            macd_signal_text = "🟢 Bullish"
        elif macd < macd_signal and macd_hist < 0:
            macd_signal_text = "🔴 Bearish"
        else:
            macd_signal_text = "🟡 Neutral"
            
        st.write(macd_signal_text)
        st.write(f"MACD: {macd:.4f}")
        st.write(f"Signal: {macd_signal:.4f}")
    
    with col3:
        st.write("**Stochastic**")
        stoch_k = current_data['Stoch_K']
        stoch_d = current_data['Stoch_D']
        
        if stoch_k < 20 and stoch_d < 20:
            stoch_signal = "🟢 Oversold"
        elif stoch_k > 80 and stoch_d > 80:
            stoch_signal = "🔴 Overbought"
        elif stoch_k > stoch_d:
            stoch_signal = "🟡 Bullish Cross"
        else:
            stoch_signal = "⚪ Neutral"
            
        st.write(stoch_signal)
        st.write(f"K: {stoch_k:.1f}")
        st.write(f"D: {stoch_d:.1f}")
    
    with col4:
        st.write("**Volume Analysis**")
        volume_ratio = current_data['Volume_Ratio']
        if volume_ratio > 2:
            volume_signal = "🟢 High Volume"
        elif volume_ratio > 1.2:
            volume_signal = "🟡 Above Average"
        else:
            volume_signal = "🔴 Low Volume"
            
        st.write(volume_signal)
        st.write(f"Volume Ratio: {volume_ratio:.2f}")
    
    # ROW 2: Fibonacci ve destek/direnç
    st.subheader("📊 Support & Resistance Levels")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Fibonacci Levels**")
        for level, value in fib_levels.items():
            diff_percent = ((current_price - value) / value) * 100
            st.write(f"{level}: ${value:.4f} ({diff_percent:+.2f}%)")
    
    with col2:
        st.write("**Bollinger Bands**")
        st.write(f"Upper: ${current_data['BB_Upper']:.4f}")
        st.write(f"Middle: ${current_data['BB_Middle']:.4f}")
        st.write(f"Lower: ${current_data['BB_Lower']:.4f}")
        st.write(f"Width: {(current_data['BB_Width']*100):.2f}%")
    
    # Sinyal özeti
    st.markdown("---")
    st.subheader("🎯 Trading Signal Summary")
    
    # Sinyal puanı hesapla
    signal_score = 0
    
    # RSI sinyali
    if rsi < 30:
        signal_score += 2
    elif rsi > 70:
        signal_score -= 2
    
    # Bollinger Bands sinyali
    if bb_position < 0.2:
        signal_score += 1
    elif bb_position > 0.8:
        signal_score -= 1
    
    # Trend sinyali
    signal_score += trend_score
    
    # MACD sinyali
    if macd > macd_signal:
        signal_score += 1
    else:
        signal_score -= 1
    
    # Volume sinyali
    if volume_ratio > 1.5:
        signal_score += 1
    
    # Sonuç
    if signal_score >= 3:
        signal = "🟢 STRONG BUY"
        explanation = "Multiple indicators suggest strong bullish momentum"
    elif signal_score >= 1:
        signal = "🟡 MODERATE BUY"
        explanation = "Some bullish signals present"
    elif signal_score <= -3:
        signal = "🔴 STRONG SELL"
        explanation = "Multiple indicators suggest strong bearish momentum"
    elif signal_score <= -1:
        signal = "🟣 MODERATE SELL"
        explanation = "Some bearish signals present"
    else:
        signal = "⚪ NEUTRAL"
        explanation = "Mixed signals - wait for clearer direction"
    
    st.success(f"**Overall Signal: {signal}**")
    st.write(f"**Signal Score:** {signal_score}/8")
    st.write(f"**Explanation:** {explanation}")
    
    # Detaylı açıklama
    with st.expander("📋 Detailed Signal Breakdown"):
        st.write(f"- RSI Signal: {'Bullish' if rsi < 30 else 'Bearish' if rsi > 70 else 'Neutral'}")
        st.write(f"- Trend Signal: {trend}")
        st.write(f"- Bollinger Position: {bb_status}")
        st.write(f"- MACD: {'Bullish' if macd > macd_signal else 'Bearish'}")
        st.write(f"- Volume: {volume_signal}")

# Ana uygulama
def main():
    # Verileri yükle
    with st.spinner("Loading data and calculating indicators..."):
        data = load_crypto_data(symbol, period_days, timeframe)
        
        if data is not None and not data.empty:
            ta = AdvancedTechnicalAnalysis()
            data_with_indicators, fib_levels = ta.get_all_indicators(data)
            
            # Sinyal analizini göster
            display_signal_analysis(data_with_indicators, fib_levels)
            
            # Son 10 mumun gösterge değerleri
            st.markdown("---")
            st.subheader("📈 Recent Indicator Values")
            recent_data = data_with_indicators.tail(10)[['Close', 'RSI_14', 'EMA_21', 'BB_Upper', 'BB_Lower', 'MACD', 'ATR']]
            st.dataframe(recent_data.style.format({
                'Close': '${:.4f}',
                'RSI_14': '{:.1f}',
                'EMA_21': '${:.4f}',
                'BB_Upper': '${:.4f}',
                'BB_Lower': '${:.4f}',
                'MACD': '{:.4f}',
                'ATR': '{:.4f}'
            }))
        else:
            st.error("Could not load data for the selected cryptocurrency")

if __name__ == "__main__":
    main()

st.markdown("---")
st.info("""
**📖 Indicator Guide:**
- **RSI**: Overbought (>70), Oversold (<30)
- **Bollinger Bands**: Price near upper band = overbought, near lower band = oversold
- **MACD**: Bullish when MACD > Signal line
- **ATR**: Higher values = more volatility
- **Fibonacci**: Key support/resistance levels
- **Volume**: Confirms price movements
""")
