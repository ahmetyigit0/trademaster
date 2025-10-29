import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import time
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

# Session state for countdown
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()
if 'countdown' not in st.session_state:
    st.session_state.countdown = 10

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
    'BTC-USD': 'BTC',
    'ETH-USD': 'ETH', 
    'BNB-USD': 'BNB',
    'XRP-USD': 'XRP',
    'THETA-USD': 'THETA',
    'AVAX-USD': 'AVAX'
}

# Countdown güncelleme
current_time = time.time()
elapsed = current_time - st.session_state.last_update
st.session_state.countdown = max(0, 10 - int(elapsed))

# Fiyatları göster
try:
    prices = get_real_time_prices(list(crypto_symbols.keys()))
    
    # 6 kolon oluştur
    cols = st.columns(6)
    
    for idx, (symbol, name) in enumerate(crypto_symbols.items()):
        with cols[idx]:
            if symbol in prices:
                price_data = prices[symbol]
                # Fiyat formatını küçült - daha kompakt gösterim
                if price_data['price'] > 1000:
                    price_str = f"${price_data['price']:,.0f}"
                elif price_data['price'] > 1:
                    price_str = f"${price_data['price']:.2f}"
                else:
                    price_str = f"${price_data['price']:.4f}"
                
                st.metric(
                    label=name,
                    value=price_str,
                    delta=f"{price_data['change']:+.2f}%"
                )
            else:
                st.metric(label=name, value="N/A")
    
    # Geri sayım
    countdown_display = st.session_state.countdown
    st.caption(f"🔄 Veriler {countdown_display} saniye içinde yenilenecek...")
    
    if st.session_state.countdown == 0:
        st.session_state.last_update = current_time
        st.session_state.countdown = 10
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

# Gelişmiş gösterge sınıfı - HATA DÜZELTMELİ
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
        emas = {}
        for period in periods:
            emas[f'EMA_{period}'] = prices.ewm(span=period).mean()
        return emas
    
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
            'BB_Width': (upper_band - lower_band) / sma.replace(0, 1)  # Zero division fix
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
        
        # Zero division fix
        denominator = (highest_high - lowest_low).replace(0, 1)
        k = 100 * (close - lowest_low) / denominator
        d = k.rolling(window=d_period).mean()
        
        return {
            'Stoch_K': k,
            'Stoch_D': d
        }
    
    def get_all_indicators(self, df):
        """Tüm göstergeleri hesapla - HATA DÜZELTMELİ"""
        df = df.copy()
        
        # Volume SMA hesapla (NaN'larla başa çık)
        df['Volume_SMA'] = df['Volume'].rolling(20, min_periods=1).mean()
        
        # Volume Ratio - GÜVENLİ HESAPLAMA
        volume_sma_safe = df['Volume_SMA'].replace(0, 1)  # Sıfır bölme hatasını önle
        df['Volume_Ratio'] = df['Volume'] / volume_sma_safe
        
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
        
        # Momentum
        df['Momentum_5'] = df['Close'].pct_change(5)
        df['Momentum_10'] = df['Close'].pct_change(10)
        
        # Tüm NaN değerleri temizle
        df = df.fillna(method='bfill').fillna(0)
        
        return df, fib_levels

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

# Fiyat formatı - küçük punto için
def format_price(price):
    if price > 1000:
        return f"${price:,.0f}"
    elif price > 1:
        return f"${price:.2f}"
    elif price > 0.01:
        return f"${price:.4f}"
    else:
        return f"${price:.6f}"

# Sinyal analizini göster
def display_signal_analysis(df, fib_levels):
    if df is None or df.empty:
        st.error("No data available for analysis")
        return
    
    ta = AdvancedTechnicalAnalysis()
    current_data = df.iloc[-1]
    
    st.subheader(f"📊 Technical Analysis for {selected_crypto}")
    
    # Ana metrikler - DAHA KOMPAKT
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = current_data['Close']
        st.metric("Current Price", format_price(current_price))
    
    with col2:
        rsi = current_data['RSI_14']
        rsi_status = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
        st.metric("RSI (14)", f"{rsi:.1f}", rsi_status)
    
    with col3:
        bb_upper = current_data['BB_Upper']
        bb_lower = current_data['BB_Lower']
        if bb_upper != bb_lower:  # Zero division protection
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
            bb_status = "Upper" if bb_position > 0.8 else "Lower" if bb_position < 0.2 else "Middle"
            st.metric("Bollinger Position", f"{bb_position:.1%}", bb_status)
        else:
            st.metric("Bollinger Position", "N/A")
    
    with col4:
        atr = current_data['ATR']
        atr_percent = (atr / current_price) * 100 if current_price > 0 else 0
        st.metric("ATR", f"{atr_percent:.2f}%")
    
    st.markdown("---")
    
    # Detaylı göstergeler - DAHA KOMPAKT
    st.subheader("🔍 Detailed Indicators")
    
    # ROW 1: Trend göstergeleri
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Trend Analysis**")
        ema_8 = current_data['EMA_8']
        ema_21 = current_data['EMA_21']
        ema_50 = current_data['EMA_50']
        
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
            trend_score = 0
        
        st.write(trend)
    
    with col2:
        st.write("**MACD Signal**")
        macd = current_data['MACD']
        macd_signal = current_data['MACD_Signal']
        
        if macd > macd_signal:
            macd_signal_text = "🟢 Bullish"
        else:
            macd_signal_text = "🔴 Bearish"
            
        st.write(macd_signal_text)
        st.write(f"Value: {macd:.4f}")
    
    with col3:
        st.write("**Volume**")
        volume_ratio = current_data['Volume_Ratio']
        if volume_ratio > 2:
            volume_signal = "🟢 High"
        elif volume_ratio > 1.2:
            volume_signal = "🟡 Average"
        else:
            volume_signal = "🔴 Low"
            
        st.write(volume_signal)
        st.write(f"Ratio: {volume_ratio:.1f}x")
    
    # ROW 2: Fibonacci
    st.subheader("📊 Fibonacci Levels")
    
    col1, col2 = st.columns(2)
    
    with col1:
        current_price = current_data['Close']
        for level, value in fib_levels.items():
            diff_percent = ((current_price - value) / value) * 100
            st.write(f"{level}: {format_price(value)} ({diff_percent:+.1f}%)")
    
    with col2:
        st.write("**Key Levels**")
        st.write(f"EMA 21: {format_price(current_data['EMA_21'])}")
        st.write(f"EMA 50: {format_price(current_data['EMA_50'])}")
        st.write(f"BB Middle: {format_price(current_data['BB_Middle'])}")
    
    # Sinyal özeti
    st.markdown("---")
    st.subheader("🎯 Trading Signal")
    
    # Basit sinyal hesapla
    rsi = current_data['RSI_14']
    macd = current_data['MACD']
    macd_signal = current_data['MACD_Signal']
    ema_21 = current_data['EMA_21']
    
    bullish_signals = 0
    bearish_signals = 0
    
    if rsi < 35:
        bullish_signals += 1
    elif rsi > 65:
        bearish_signals += 1
        
    if macd > macd_signal:
        bullish_signals += 1
    else:
        bearish_signals += 1
        
    if current_price > ema_21:
        bullish_signals += 1
    else:
        bearish_signals += 1
    
    if bullish_signals >= 2:
        signal = "🟢 BUY"
        reasoning = "Multiple bullish signals"
    elif bearish_signals >= 2:
        signal = "🔴 SELL" 
        reasoning = "Multiple bearish signals"
    else:
        signal = "🟡 HOLD"
        reasoning = "Mixed signals - wait for confirmation"
    
    st.success(f"**Signal: {signal}**")
    st.write(f"**Reason:** {reasoning}")
    st.write(f"Bullish: {bullish_signals}/3, Bearish: {bearish_signals}/3")

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
            
        else:
            st.error("Could not load data for the selected cryptocurrency")

# Uygulamayı çalıştır
if __name__ == "__main__":
    main()

st.markdown("---")
st.info("""
**📖 Quick Guide:**
- **RSI < 30**: Oversold (Buy), **> 70**: Overbought (Sell)
- **MACD > Signal**: Bullish, **< Signal**: Bearish  
- **Price > EMA 21**: Uptrend, **< EMA 21**: Downtrend
- **Volume > 1.2x**: Confirms movement
""")
