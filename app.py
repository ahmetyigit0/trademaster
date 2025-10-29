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
st.title("🚀 Crypto Trading Dashboard - Multi API")
st.markdown("---")

# Session state for countdown
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()
if 'countdown' not in st.session_state:
    st.session_state.countdown = 10
if 'api_status' not in st.session_state:
    st.session_state.api_status = "checking"

# Multiple API endpoints
API_ENDPOINTS = {
    "CoinGecko": "https://api.coingecko.com/api/v3",
    "Yahoo Finance": "https://query1.finance.yahoo.com/v8/finance/chart/",
    "CoinCap": "https://api.coincap.io/v2",
    "Binance": "https://api.binance.com/api/v3"
}

# Crypto symbols mapping
CRYPTO_SYMBOLS = {
    'BTC': {'coingecko': 'bitcoin', 'yahoo': 'BTC-USD', 'coincap': 'bitcoin', 'binance': 'BTCUSDT'},
    'ETH': {'coingecko': 'ethereum', 'yahoo': 'ETH-USD', 'coincap': 'ethereum', 'binance': 'ETHUSDT'},
    'BNB': {'coingecko': 'binancecoin', 'yahoo': 'BNB-USD', 'coincap': 'binance-coin', 'binance': 'BNBUSDT'},
    'XRP': {'coingecko': 'ripple', 'yahoo': 'XRP-USD', 'coincap': 'ripple', 'binance': 'XRPUSDT'},
    'ADA': {'coingecko': 'cardano', 'yahoo': 'ADA-USD', 'coincap': 'cardano', 'binance': 'ADAUSDT'},
    'SOL': {'coingecko': 'solana', 'yahoo': 'SOL-USD', 'coincap': 'solana', 'binance': 'SOLUSDT'}
}

# Test all APIs and find the working one
def find_working_api():
    """Çalışan API'yi bul"""
    test_symbol = 'BTC'
    
    # Test CoinGecko
    try:
        url = f"{API_ENDPOINTS['CoinGecko']}/simple/price"
        params = {'ids': CRYPTO_SYMBOLS[test_symbol]['coingecko'], 'vs_currencies': 'usd', 'include_24hr_change': 'true'}
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            return "CoinGecko"
    except:
        pass
    
    # Test Yahoo Finance
    try:
        url = f"{API_ENDPOINTS['Yahoo Finance']}{CRYPTO_SYMBOLS[test_symbol]['yahoo']}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return "Yahoo Finance"
    except:
        pass
    
    # Test CoinCap
    try:
        url = f"{API_ENDPOINTS['CoinCap']}/assets/{CRYPTO_SYMBOLS[test_symbol]['coincap']}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return "CoinCap"
    except:
        pass
    
    return "None"

# Get prices from working API
@st.cache_data(ttl=10)
def get_crypto_prices(api_source):
    """Çalışan API'den fiyatları getir"""
    prices = {}
    
    if api_source == "CoinGecko":
        try:
            symbols = [CRYPTO_SYMBOLS[sym]['coingecko'] for sym in CRYPTO_SYMBOLS.keys()]
            url = f"{API_ENDPOINTS['CoinGecko']}/simple/price"
            params = {
                'ids': ','.join(symbols),
                'vs_currencies': 'usd', 
                'include_24hr_change': 'true'
            }
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            for display_symbol, api_data in CRYPTO_SYMBOLS.items():
                if api_data['coingecko'] in data:
                    coin_data = data[api_data['coingecko']]
                    prices[display_symbol] = {
                        'price': coin_data['usd'],
                        'change': coin_data.get('usd_24h_change', 0),
                        'source': 'CoinGecko'
                    }
        except Exception as e:
            st.error(f"CoinGecko error: {e}")
    
    elif api_source == "Yahoo Finance":
        try:
            for display_symbol, api_data in CRYPTO_SYMBOLS.items():
                url = f"{API_ENDPOINTS['Yahoo Finance']}{api_data['yahoo']}"
                response = requests.get(url, timeout=10)
                data = response.json()
                
                if 'chart' in data and 'result' in data['chart']:
                    result = data['chart']['result'][0]
                    current_price = result['meta']['regularMarketPrice']
                    previous_price = result['meta']['previousClose']
                    change = ((current_price - previous_price) / previous_price) * 100
                    
                    prices[display_symbol] = {
                        'price': current_price,
                        'change': change,
                        'source': 'Yahoo Finance'
                    }
        except Exception as e:
            st.error(f"Yahoo Finance error: {e}")
    
    elif api_source == "CoinCap":
        try:
            for display_symbol, api_data in CRYPTO_SYMBOLS.items():
                url = f"{API_ENDPOINTS['CoinCap']}/assets/{api_data['coincap']}"
                response = requests.get(url, timeout=10)
                data = response.json()
                
                if 'data' in data:
                    coin_data = data['data']
                    prices[display_symbol] = {
                        'price': float(coin_data['priceUsd']),
                        'change': float(coin_data['changePercent24Hr']),
                        'source': 'CoinCap'
                    }
        except Exception as e:
            st.error(f"CoinCap error: {e}")
    
    return prices

# Get historical data
@st.cache_data(ttl=300)  # 5 dakika cache
def get_historical_data(symbol, days=90):
    """Geçmiş verileri getir (CoinGecko)"""
    try:
        crypto_id = CRYPTO_SYMBOLS[symbol]['coingecko']
        url = f"{API_ENDPOINTS['CoinGecko']}/coins/{crypto_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'daily'
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        # DataFrame'e çevir
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        df = df[['price']]
        df.columns = ['close']
        
        # High, Low, Open hesapla (basit)
        df['high'] = df['close'] * 1.02  # Yaklaşık değer
        df['low'] = df['close'] * 0.98   # Yaklaşık değer
        df['open'] = df['close'].shift(1)
        df['volume'] = 1000000  # Varsayılan volume
        
        return df.fillna(method='bfill')
        
    except Exception as e:
        st.error(f"Historical data error: {e}")
        return None

# Üstte real-time fiyatlar
st.subheader("📈 Real-Time Crypto Prices")

# API kontrolü
if st.session_state.api_status == "checking":
    working_api = find_working_api()
    st.session_state.api_status = working_api
    st.session_state.working_api = working_api

working_api = st.session_state.get('working_api', 'CoinGecko')

if working_api != "None":
    st.sidebar.success(f"✅ Connected: {working_api}")
else:
    st.sidebar.error("❌ All APIs disconnected")

# Countdown güncelleme
current_time = time.time()
elapsed = current_time - st.session_state.last_update
st.session_state.countdown = max(0, 10 - int(elapsed))

# Fiyatları göster
try:
    if working_api != "None":
        prices = get_crypto_prices(working_api)
        
        # 6 kolon oluştur
        cols = st.columns(6)
        
        for idx, symbol in enumerate(list(CRYPTO_SYMBOLS.keys())[:6]):
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
                        label=symbol,
                        value=price_str,
                        delta=f"{price_data['change']:+.2f}%"
                    )
                    st.caption(f"via {price_data['source']}")
                else:
                    st.metric(label=symbol, value="N/A")
        
        # Geri sayım
        countdown_display = st.session_state.countdown
        st.caption(f"🔄 {working_api} verileri {countdown_display} saniye içinde yenilenecek...")
        
        if st.session_state.countdown == 0:
            st.session_state.last_update = current_time
            st.session_state.countdown = 10
            st.rerun()
    else:
        st.error("❌ No working API found. Please check your internet connection.")
        
except Exception as e:
    st.error(f"Price error: {e}")

st.markdown("---")

# Sol sidebar - Sinyal analizi
st.sidebar.header("🔍 Crypto Signal Analysis")

# Kripto seçimi
crypto_options = {
    "Bitcoin (BTC)": "BTC",
    "Ethereum (ETH)": "ETH", 
    "Binance Coin (BNB)": "BNB",
    "XRP (XRP)": "XRP",
    "Cardano (ADA)": "ADA",
    "Solana (SOL)": "SOL"
}

selected_crypto = st.sidebar.selectbox("Select Crypto:", list(crypto_options.keys()))
symbol = crypto_options[selected_crypto]

# Zaman ayarları
st.sidebar.subheader("⚡ Analysis Settings")
timeframe = st.sidebar.selectbox("Timeframe:", ["Daily", "4H", "1H"], index=0)
period_days = st.sidebar.slider("Data Period (Days):", 30, 365, 90)

# Teknik Analiz Sınıfı
class TechnicalAnalysis:
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
        """Tüm göstergeleri hesapla"""
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
            df = df.assign(MACD_Histogram=df['MACD'] - df['MACD_Signal'])
            
            # 4. Bollinger Bands
            bb_middle = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            
            df = df.assign(BB_Middle=bb_middle)
            df = df.assign(BB_Upper=bb_upper)
            df = df.assign(BB_Lower=bb_lower)
            df = df.assign(BB_Width=(bb_upper - bb_lower) / bb_middle)
            
            # 5. Support & Resistance
            df['Resistance'] = df['high'].rolling(20).max()
            df['Support'] = df['low'].rolling(20).min()
            
            # 6. Fibonacci
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
def display_technical_analysis(df, fib_levels, symbol_name):
    if df is None or df.empty:
        st.error("No data available for analysis")
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
        bb_middle = float(current_data['BB_Middle'])
        
        st.subheader(f"📊 Technical Analysis: {symbol_name}")
        
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
                volatility = float(current_data['BB_Width']) * 100
                st.metric("Volatility", f"{volatility:.1f}%")
            except:
                st.metric("Volatility", "N/A")
        
        st.markdown("---")
        
        # Detaylı Analiz
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📈 Trend Analysis**")
            
            if current_price > ema_12 > ema_26 > ema_50:
                trend = "🟢 Strong Uptrend"
                trend_score = 3
            elif current_price > ema_26 > ema_50:
                trend = "🟡 Uptrend"
                trend_score = 2
            elif current_price > ema_50:
                trend = "🟠 Weak Uptrend"
                trend_score = 1
            elif current_price < ema_12 < ema_26 < ema_50:
                trend = "🔴 Strong Downtrend"
                trend_score = -3
            elif current_price < ema_26 < ema_50:
                trend = "🟣 Downtrend"
                trend_score = -2
            else:
                trend = "⚪ Sideways"
                trend_score = 0
            
            st.write(trend)
            st.write(f"EMA 12: {format_price(ema_12)}")
            st.write(f"EMA 26: {format_price(ema_26)}")
            st.write(f"EMA 50: {format_price(ema_50)}")
        
        with col2:
            st.write("**🔍 Momentum & Volume**")
            
            if macd > macd_signal:
                macd_signal_text = "🟢 Bullish"
                macd_score = 1
            else:
                macd_signal_text = "🔴 Bearish"
                macd_score = -1
                
            st.write(f"MACD: {macd_signal_text}")
            st.write(f"Histogram: {float(current_data['MACD_Histogram']):.4f}")
            
            # Support/Resistance
            support = float(current_data['Support'])
            resistance = float(current_data['Resistance'])
            st.write(f"Support: {format_price(support)}")
            st.write(f"Resistance: {format_price(resistance)}")
        
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
        
        # Detaylı sinyal hesaplama
        signals = {
            'RSI': 1 if rsi < 35 else -1 if rsi > 65 else 0,
            'MACD': 1 if macd > macd_signal else -1,
            'Trend': 1 if current_price > ema_50 else -1,
            'Bollinger': 1 if (current_price - bb_lower) / (bb_upper - bb_lower) < 0.2 else 
                        -1 if (current_price - bb_lower) / (bb_upper - bb_lower) > 0.8 else 0,
            'Support': 1 if (current_price - support) / current_price < 0.03 else 0,
            'EMA_Alignment': 1 if ema_12 > ema_26 > ema_50 else -1 if ema_12 < ema_26 < ema_50 else 0
        }
        
        total_score = sum(signals.values())
        
        if total_score >= 4:
            signal = "🟢 STRONG BUY"
            color = "green"
        elif total_score >= 2:
            signal = "🟡 BUY"
            color = "blue"
        elif total_score <= -4:
            signal = "🔴 STRONG SELL"
            color = "red"
        elif total_score <= -2:
            signal = "🟣 SELL"
            color = "purple"
        else:
            signal = "⚪ HOLD"
            color = "gray"
        
        st.success(f"**{signal}**")
        st.write(f"**Signal Score:** {total_score}/6")
        
        # Sinyal detayları
        with st.expander("📋 Signal Details"):
            for indicator, score in signals.items():
                st.write(f"{indicator}: {'+' if score > 0 else ''}{score}")
        
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")

# Ana uygulama
def main():
    # Geçmiş verileri yükle
    with st.spinner(f"📊 Loading historical data for {selected_crypto}..."):
        try:
            df = get_historical_data(symbol, period_days)
            
            if df is not None and not df.empty:
                # Teknik analiz
                ta = TechnicalAnalysis()
                data_with_indicators, fib_levels = ta.calculate_indicators(df)
                
                # Analizi göster
                display_technical_analysis(data_with_indicators, fib_levels, selected_crypto)
                
                # Son 10 günün verileri
                with st.expander("📈 Recent Price Data"):
                    recent_data = data_with_indicators.tail(10)[['close', 'RSI_14', 'EMA_26', 'BB_Upper', 'BB_Lower']]
                    st.dataframe(recent_data.style.format({
                        'close': '${:.2f}',
                        'RSI_14': '{:.1f}',
                        'EMA_26': '${:.2f}',
                        'BB_Upper': '${:.2f}',
                        'BB_Lower': '${:.2f}'
                    }))
                
            else:
                st.error("❌ Could not load historical data")
                
        except Exception as e:
            st.error(f"❌ Data loading error: {str(e)}")

# Uygulamayı çalıştır
main()

st.markdown("---")
st.info("""
**🚀 Multi-API System:**
- ✅ **CoinGecko** - Primary API
- ✅ **Yahoo Finance** - Fallback 1  
- ✅ **CoinCap** - Fallback 2
- ✅ **Binance** - Fallback 3

**📖 Trading Signals:**
- **RSI < 35 + MACD Bullish** = Strong Buy
- **Price > All EMAs** = Uptrend Confirmation
- **Bollinger Lower Band** = Oversold Bounce
- **Fibonacci Support** = Key Levels
- **4+ Signals** = High Conviction
""")

# API durumu
st.sidebar.markdown("---")
st.sidebar.write("**🌐 API Status:**")
st.sidebar.write(f"Active: {working_api}")
st.sidebar.write("Backups: CoinGecko, Yahoo, CoinCap")

# Manual refresh
if st.sidebar.button("🔄 Refresh APIs"):
    st.session_state.api_status = "checking"
    st.rerun()
