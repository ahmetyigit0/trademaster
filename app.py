import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import requests
import json

# Sayfa ayarı
st.set_page_config(
    page_title="🚀 Crypto Trading Dashboard",
    page_icon="📊",
    layout="wide"
)

# Başlık
st.title("🚀 Crypto Trading Dashboard - Binance API")
st.markdown("---")

# Session state for countdown
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()
if 'countdown' not in st.session_state:
    st.session_state.countdown = 10

# Binance API base URL
BINANCE_API_URL = "https://api.binance.com/api/v3"

# Binance sembol eşleştirme
BINANCE_SYMBOLS = {
    'BTCUSDT': 'BTC',
    'ETHUSDT': 'ETH', 
    'BNBUSDT': 'BNB',
    'XRPUSDT': 'XRP',
    'ADAUSDT': 'ADA',
    'SOLUSDT': 'SOL',
    'DOTUSDT': 'DOT',
    'DOGEUSDT': 'DOGE',
    'AVAXUSDT': 'AVAX',
    'MATICUSDT': 'MATIC',
    'LTCUSDT': 'LTC',
    'LINKUSDT': 'LINK'
}

# Real-time fiyatları Binance'dan getir
@st.cache_data(ttl=5)  # 5 saniye cache
def get_binance_prices(symbols):
    """Binance API'den gerçek zamanlı fiyatları getir"""
    prices = {}
    try:
        url = f"{BINANCE_API_URL}/ticker/24hr"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        for item in data:
            symbol = item['symbol']
            if symbol in symbols:
                current_price = float(item['lastPrice'])
                price_change = float(item['priceChangePercent'])
                prices[symbol] = {
                    'price': current_price,
                    'change': price_change,
                    'volume': float(item['volume']),
                    'high': float(item['highPrice']),
                    'low': float(item['lowPrice'])
                }
    except Exception as e:
        st.error(f"Binance API error: {e}")
    
    return prices

# Kline (mum) verilerini getir
@st.cache_data(ttl=60)  # 1 dakika cache
def get_binance_klines(symbol, interval, limit=500):
    """Binance'dan mum verilerini getir"""
    try:
        url = f"{BINANCE_API_URL}/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # DataFrame'e çevir
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Veri tiplerini düzelt
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        # Zaman damgasını düzelt
        df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        return df[['open', 'high', 'low', 'close', 'volume']]
        
    except Exception as e:
        st.error(f"Kline data error: {e}")
        return None

# Binance connection test
def test_binance_connection():
    """Binance API bağlantı testi"""
    try:
        url = f"{BINANCE_API_URL}/ping"
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except:
        return False

# Üstte real-time fiyatlar
st.subheader("📈 Real-Time Crypto Prices - Binance")

# Countdown güncelleme
current_time = time.time()
elapsed = current_time - st.session_state.last_update
st.session_state.countdown = max(0, 10 - int(elapsed))

# Binance bağlantı testi
if test_binance_connection():
    st.sidebar.success("✅ Binance API: Connected")
else:
    st.sidebar.error("❌ Binance API: Disconnected - Using fallback data")

# Binance fiyatlarını göster
try:
    prices = get_binance_prices(list(BINANCE_SYMBOLS.keys()))
    
    # 6 kolon oluştur
    cols = st.columns(6)
    
    for idx, (symbol, name) in enumerate(list(BINANCE_SYMBOLS.items())[:6]):
        with cols[idx]:
            if symbol in prices:
                price_data = prices[symbol]
                # Fiyat formatını küçült
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
    st.caption(f"🔄 Binance verileri {countdown_display} saniye içinde yenilenecek...")
    
    if st.session_state.countdown == 0:
        st.session_state.last_update = current_time
        st.session_state.countdown = 10
        st.rerun()
        
except Exception as e:
    st.error(f"Binance price error: {e}")

st.markdown("---")

# Sol sidebar - Sinyal analizi
st.sidebar.header("🔍 Crypto Signal Analysis - Binance")

# Kripto seçimi
crypto_options = {
    "Bitcoin (BTC)": "BTCUSDT",
    "Ethereum (ETH)": "ETHUSDT", 
    "Binance Coin (BNB)": "BNBUSDT",
    "XRP (XRP)": "XRPUSDT",
    "Cardano (ADA)": "ADAUSDT",
    "Solana (SOL)": "SOLUSDT",
    "Polkadot (DOT)": "DOTUSDT",
    "Dogecoin (DOGE)": "DOGEUSDT",
    "Avalanche (AVAX)": "AVAXUSDT",
    "Polygon (MATIC)": "MATICUSDT",
    "Litecoin (LTC)": "LTCUSDT",
    "Chainlink (LINK)": "LINKUSDT"
}

selected_crypto = st.sidebar.selectbox("Select Crypto:", list(crypto_options.keys()))
symbol = crypto_options[selected_crypto]

# Zaman ayarları
st.sidebar.subheader("⚡ Time Settings")
timeframe_map = {
    "1h": "1h",
    "4h": "4h", 
    "1d": "1d",
    "1w": "1w"
}
timeframe = st.sidebar.selectbox("Timeframe:", list(timeframe_map.keys()), index=1)
binance_timeframe = timeframe_map[timeframe]

period_days = st.sidebar.slider("Data Period (Days):", 7, 365, 90)

# Basit Teknik Analiz Sınıfı
class SimpleTechnicalAnalysis:
    def __init__(self):
        pass
    
    def calculate_rsi(self, prices, window=14):
        """RSI hesapla"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            
            avg_gain = gain.rolling(window=window, min_periods=1).mean()
            avg_loss = loss.rolling(window=window, min_periods=1).mean()
            
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rs = rs.fillna(1)
            return 100 - (100 / (1 + rs))
        except:
            return pd.Series(50, index=prices.index)
    
    def calculate_indicators(self, df):
        """Binance verileri için gösterge hesaplama"""
        try:
            df = df.copy()
            
            # 1. RSI
            rsi_series = self.calculate_rsi(df['close'], 14)
            df = df.assign(RSI_14=rsi_series)
            
            # 2. EMA'lar
            df = df.assign(EMA_12=df['close'].ewm(span=12).mean())
            df = df.assign(EMA_26=df['close'].ewm(span=26).mean())
            df = df.assign(EMA_50=df['close'].ewm(span=50).mean())
            
            # 3. MACD
            macd_series = df['EMA_12'] - df['EMA_26']
            df = df.assign(MACD=macd_series)
            df = df.assign(MACD_Signal=macd_series.ewm(span=9).mean())
            
            # 4. Bollinger Bands
            bb_middle = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            
            df = df.assign(BB_Middle=bb_middle)
            df = df.assign(BB_Upper=bb_upper)
            df = df.assign(BB_Lower=bb_lower)
            
            # 5. Volume
            volume_sma = df['volume'].rolling(20, min_periods=1).mean()
            volume_ratio = df['volume'] / volume_sma.replace(0, 1)
            df = df.assign(Volume_Ratio=volume_ratio)
            
            # 6. ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = np.maximum(np.maximum(high_low, high_close), low_close)
            atr_series = true_range.rolling(14).mean()
            df = df.assign(ATR=atr_series)
            
            # 7. Fibonacci
            recent_high = df['high'].tail(50).max()
            recent_low = df['low'].tail(50).min()
            diff = recent_high - recent_low
            fib_levels = {
                'Fib_0.236': recent_high - diff * 0.236,
                'Fib_0.382': recent_high - diff * 0.382,
                'Fib_0.5': recent_high - diff * 0.5,
                'Fib_0.618': recent_high - diff * 0.618,
                'Fib_0.786': recent_high - diff * 0.786
            }
            
            # NaN temizleme
            df = df.fillna(method='bfill').fillna(0)
            
            return df, fib_levels
            
        except Exception as e:
            st.error(f"Indicator calculation error: {e}")
            return df, {}

# Fiyat formatı
def format_price(price):
    try:
        price = float(price)
        if price > 1000:
            return f"${price:,.0f}"
        elif price > 1:
            return f"${price:.2f}"
        elif price > 0.01:
            return f"${price:.4f}"
        else:
            return f"${price:.6f}"
    except:
        return "N/A"

# Sinyal analizini göster
def display_binance_analysis(df, fib_levels, symbol_name):
    if df is None or df.empty:
        st.error("No Binance data available for analysis")
        return
    
    try:
        current_data = df.iloc[-1]
        
        # Tüm değerleri float'a çevir
        current_price = float(current_data['close'])
        rsi = float(current_data['RSI_14'])
        ema_12 = float(current_data['EMA_12'])
        ema_26 = float(current_data['EMA_26'])
        ema_50 = float(current_data['EMA_50'])
        macd = float(current_data['MACD'])
        macd_signal = float(current_data['MACD_Signal'])
        bb_upper = float(current_data['BB_Upper'])
        bb_lower = float(current_data['BB_Lower'])
        atr = float(current_data['ATR'])
        volume_ratio = float(current_data.get('Volume_Ratio', 1))
        
        st.subheader(f"📊 Binance Analysis: {symbol_name}")
        
        # Ana metrikler
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", format_price(current_price))
        
        with col2:
            rsi_status = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
            st.metric("RSI (14)", f"{rsi:.1f}", rsi_status)
        
        with col3:
            try:
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
                bb_status = "Upper" if bb_position > 0.8 else "Lower" if bb_position < 0.2 else "Middle"
                st.metric("Bollinger", f"{bb_position:.0%}", bb_status)
            except:
                st.metric("Bollinger", "N/A")
        
        with col4:
            try:
                atr_percent = (atr / current_price) * 100
                st.metric("ATR", f"{atr_percent:.2f}%")
            except:
                st.metric("ATR", "N/A")
        
        st.markdown("---")
        
        # Detaylı Analiz
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📈 Trend Analysis**")
            
            if current_price > ema_12 > ema_26 > ema_50:
                trend = "🟢 Strong Uptrend"
            elif current_price > ema_26 > ema_50:
                trend = "🟡 Uptrend"
            elif current_price > ema_50:
                trend = "🟠 Weak Uptrend"
            elif current_price < ema_12 < ema_26 < ema_50:
                trend = "🔴 Strong Downtrend"
            elif current_price < ema_26 < ema_50:
                trend = "🟣 Downtrend"
            else:
                trend = "⚪ Sideways"
            
            st.write(trend)
            st.write(f"EMA 12: {format_price(ema_12)}")
            st.write(f"EMA 26: {format_price(ema_26)}")
            st.write(f"EMA 50: {format_price(ema_50)}")
        
        with col2:
            st.write("**🔍 Momentum**")
            
            if macd > macd_signal:
                macd_signal_text = "🟢 Bullish"
            else:
                macd_signal_text = "🔴 Bearish"
                
            st.write(f"MACD: {macd_signal_text}")
            st.write(f"Value: {macd:.4f}")
            
            if volume_ratio > 1.5:
                volume_signal = "🟢 High"
            elif volume_ratio > 0.8:
                volume_signal = "🟡 Normal"
            else:
                volume_signal = "🔴 Low"
                
            st.write(f"Volume: {volume_signal}")
            st.write(f"Ratio: {volume_ratio:.1f}x")
        
        # Fibonacci Levels
        st.subheader("📊 Fibonacci Levels")
        
        if fib_levels:
            cols = st.columns(5)
            current_price = float(current_data['close'])
            
            for idx, (level, value) in enumerate(fib_levels.items()):
                with cols[idx]:
                    diff_pct = ((current_price - value) / value) * 100
                    st.metric(
                        label=level.replace('Fib_', ''),
                        value=format_price(value),
                        delta=f"{diff_pct:+.1f}%"
                    )
        
        # Trading Signal
        st.markdown("---")
        st.subheader("🎯 Trading Signal")
        
        # Sinyal hesaplama
        buy_signals = 0
        sell_signals = 0
        
        # RSI sinyali
        if rsi < 35:
            buy_signals += 1
        elif rsi > 65:
            sell_signals += 1
        
        # MACD sinyali
        if macd > macd_signal:
            buy_signals += 1
        else:
            sell_signals += 1
        
        # Trend sinyali
        if current_price > ema_26:
            buy_signals += 1
        else:
            sell_signals += 1
        
        # Bollinger sinyali
        try:
            bb_pos = (current_price - bb_lower) / (bb_upper - bb_lower)
            if bb_pos < 0.2:
                buy_signals += 1
            elif bb_pos > 0.8:
                sell_signals += 1
        except:
            pass
        
        # Sonuç
        if buy_signals >= 3:
            signal = "🟢 STRONG BUY"
        elif buy_signals > sell_signals:
            signal = "🟡 BUY"
        elif sell_signals >= 3:
            signal = "🔴 STRONG SELL"
        elif sell_signals > buy_signals:
            signal = "🟣 SELL"
        else:
            signal = "⚪ HOLD"
        
        st.success(f"**{signal}**")
        st.write(f"**Buy Signals:** {buy_signals}/4")
        st.write(f"**Sell Signals:** {sell_signals}/4")
        
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")

# Ana uygulama
def main():
    # Verileri Binance'dan yükle
    with st.spinner(f"📊 Binance verileri yükleniyor: {selected_crypto}..."):
        try:
            # Kline verilerini getir
            df = get_binance_klines(symbol, binance_timeframe, 300)
            
            if df is not None and not df.empty:
                # Kolon isimlerini düzelt
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                
                # Teknik analiz
                ta = SimpleTechnicalAnalysis()
                data_with_indicators, fib_levels = ta.calculate_indicators(df)
                
                # Analizi göster
                display_binance_analysis(data_with_indicators, fib_levels, selected_crypto)
                
            else:
                st.error("❌ Binance'tan veri alınamadı. Lütfen internet bağlantınızı kontrol edin.")
                
        except Exception as e:
            st.error(f"❌ Binance data error: {str(e)}")

# Uygulamayı çalıştır
main()

st.markdown("---")
st.info("""
**🚀 Binance API Özellikleri:**
- ✅ **Gerçek zamanlı veri** - Canlı fiyatlar
- ✅ **Yüksek doğruluk** - Milisaniye güncelleme
- ✅ **Güvenilir** - Dünyanın en büyük borsası
- ✅ **Ücretsiz** - Public data bedava

**📖 Sinyal Rehberi:**
- **RSI < 35**: Al sinyali
- **MACD > Signal**: Al sinyali  
- **Price > EMA 26**: Yukarı trend
- **Bollinger Lower**: Destek seviyesi
- **3/4 sinyal**: Güçlü yön
""")
