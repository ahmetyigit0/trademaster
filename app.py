import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import requests
import json
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Sayfa ayarı
st.set_page_config(
    page_title="🚀 Crypto Trading Dashboard",
    page_icon="📊",
    layout="wide"
)

# Başlık
st.title("🚀 Crypto Trading Dashboard - Binance API")
st.markdown("---")

# Binance API configuration
@st.cache_resource
def init_binance_client():
    """Binance client initializer - Public data için API key gerekmez"""
    try:
        # Public data için API key gerekmez
        client = Client()
        return client
    except Exception as e:
        st.error(f"Binance connection error: {e}")
        return None

# Session state for countdown
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()
if 'countdown' not in st.session_state:
    st.session_state.countdown = 10

# Binance sembol eşleştirme
BINANCE_SYMBOLS = {
    'BTC-USD': 'BTCUSDT',
    'ETH-USD': 'ETHUSDT', 
    'BNB-USD': 'BNBUSDT',
    'XRP-USD': 'XRPUSDT',
    'ADA-USD': 'ADAUSDT',
    'SOL-USD': 'SOLUSDT',
    'DOT-USD': 'DOTUSDT',
    'DOGE-USD': 'DOGEUSDT',
    'AVAX-USD': 'AVAXUSDT',
    'MATIC-USD': 'MATICUSDT',
    'LTC-USD': 'LTCUSDT',
    'LINK-USD': 'LINKUSDT'
}

REVERSE_SYMBOLS = {v: k for k, v in BINANCE_SYMBOLS.items()}

# Real-time fiyatları Binance'dan getir
@st.cache_data(ttl=5)  # 5 saniye cache
def get_binance_prices(symbols):
    """Binance API'den gerçek zamanlı fiyatları getir"""
    prices = {}
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url)
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
        url = f"https://api.binance.com/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        response = requests.get(url, params=params)
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

# Üstte real-time fiyatlar
st.subheader("📈 Real-Time Crypto Prices - Binance")

# Crypto sembolleri
crypto_symbols = {
    'BTCUSDT': 'BTC',
    'ETHUSDT': 'ETH', 
    'BNBUSDT': 'BNB',
    'XRPUSDT': 'XRP',
    'ADAUSDT': 'ADA',
    'SOLUSDT': 'SOL'
}

# Countdown güncelleme
current_time = time.time()
elapsed = current_time - st.session_state.last_update
st.session_state.countdown = max(0, 10 - int(elapsed))

# Binance fiyatlarını göster
try:
    prices = get_binance_prices(list(crypto_symbols.keys()))
    
    # 6 kolon oluştur
    cols = st.columns(6)
    
    for idx, (symbol, name) in enumerate(crypto_symbols.items()):
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
                
                # Mini info
                with st.expander("ℹ️"):
                    st.write(f"24h High: ${price_data['high']:.2f}")
                    st.write(f"24h Low: ${price_data['low']:.2f}")
                    st.write(f"Volume: {price_data['volume']:,.0f}")
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
    "1h": Client.KLINE_INTERVAL_1HOUR,
    "4h": Client.KLINE_INTERVAL_4HOUR,
    "1d": Client.KLINE_INTERVAL_1DAY,
    "1w": Client.KLINE_INTERVAL_1WEEK
}
timeframe = st.sidebar.selectbox("Timeframe:", list(timeframe_map.keys()), index=1)
binance_timeframe = timeframe_map[timeframe]

period_days = st.sidebar.slider("Data Period (Days):", 7, 365, 90)

# Gelişmiş Teknik Analiz Sınıfı - Binance için optimize
class BinanceTechnicalAnalysis:
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
            df = df.assign(EMA_200=df['close'].ewm(span=200).mean())
            
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
            
            # 5. Volume indicators
            volume_sma = df['volume'].rolling(20, min_periods=1).mean()
            volume_ratio = df['volume'] / volume_sma.replace(0, 1)
            df = df.assign(Volume_Ratio=volume_ratio)
            df = df.assign(Volume_SMA=volume_sma)
            
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
                'Fib_0.786': recent_high - diff * 0.786,
                'Fib_1.0': recent_high
            }
            
            # Support & Resistance
            df['Resistance'] = df['high'].rolling(20).max()
            df['Support'] = df['low'].rolling(20).min()
            
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
        ema_200 = float(current_data['EMA_200'])
        macd = float(current_data['MACD'])
        macd_signal = float(current_data['MACD_Signal'])
        bb_upper = float(current_data['BB_Upper'])
        bb_lower = float(current_data['BB_Lower'])
        bb_middle = float(current_data['BB_Middle'])
        atr = float(current_data['ATR'])
        volume_ratio = float(current_data.get('Volume_Ratio', 1))
        
        st.subheader(f"📊 Binance Analysis: {symbol_name}")
        
        # Price and basic metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", format_price(current_price))
            st.metric("24h Change", 
                     f"{((current_price - float(df['close'].iloc[-2])) / float(df['close'].iloc[-2]) * 100):+.2f}%")
        
        with col2:
            rsi_status = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
            st.metric("RSI (14)", f"{rsi:.1f}", rsi_status)
            st.metric("Trend", "Bullish" if current_price > ema_200 else "Bearish")
        
        with col3:
            try:
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
                bb_status = "Upper" if bb_position > 0.8 else "Lower" if bb_position < 0.2 else "Middle"
                st.metric("Bollinger", f"{bb_position:.0%}", bb_status)
                st.metric("Volatility", f"{(float(current_data['BB_Width']) * 100):.1f}%")
            except:
                st.metric("Bollinger", "N/A")
        
        with col4:
            try:
                atr_percent = (atr / current_price) * 100
                st.metric("ATR", f"{atr_percent:.2f}%")
                st.metric("Volume", f"{volume_ratio:.1f}x")
            except:
                st.metric("ATR", "N/A")
        
        st.markdown("---")
        
        # Detailed Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Moving Averages")
            
            # EMA Analysis
            ema_data = {
                'EMA': ['EMA 12', 'EMA 26', 'EMA 50', 'EMA 200'],
                'Value': [ema_12, ema_26, ema_50, ema_200],
                'Distance %': [
                    ((current_price - ema_12) / ema_12 * 100),
                    ((current_price - ema_26) / ema_26 * 100),
                    ((current_price - ema_50) / ema_50 * 100),
                    ((current_price - ema_200) / ema_200 * 100)
                ]
            }
            ema_df = pd.DataFrame(ema_data)
            st.dataframe(ema_df.style.format({
                'Value': '${:.2f}',
                'Distance %': '{:+.2f}%'
            }))
            
            # Trend Strength
            above_emas = sum([current_price > ema_12, current_price > ema_26, 
                            current_price > ema_50, current_price > ema_200])
            trend_strength = "Strong Bull" if above_emas == 4 else \
                           "Bullish" if above_emas >= 2 else \
                           "Strong Bear" if above_emas == 0 else "Bearish"
            
            st.metric("Trend Strength", trend_strength)
        
        with col2:
            st.subheader("🔍 Oscillators")
            
            # RSI Analysis
            rsi_status = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
            st.progress(rsi/100)
            st.write(f"RSI: {rsi:.1f} ({rsi_status})")
            
            # MACD Analysis
            macd_signal = "Bullish" if macd > 0 else "Bearish"
            macd_histogram = float(current_data['MACD_Histogram'])
            st.write(f"MACD: {macd:.4f} ({macd_signal})")
            st.write(f"Histogram: {macd_histogram:.4f}")
            
            # Volume Analysis
            volume_status = "High" if volume_ratio > 1.5 else "Normal" if volume_ratio > 0.8 else "Low"
            st.write(f"Volume: {volume_ratio:.1f}x ({volume_status})")
        
        # Fibonacci Levels
        st.subheader("📊 Fibonacci & Key Levels")
        
        if fib_levels:
            cols = st.columns(6)
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
        st.subheader("🎯 Binance Trading Signal")
        
        # Advanced signal calculation
        signals = {
            'RSI': 1 if rsi < 35 else -1 if rsi > 65 else 0,
            'MACD': 1 if macd > 0 and macd > float(current_data['MACD_Signal']) else -1,
            'Trend': 1 if current_price > ema_50 else -1,
            'Bollinger': 1 if (current_price - bb_lower) / (bb_upper - bb_lower) < 0.2 else 
                        -1 if (current_price - bb_lower) / (bb_upper - bb_lower) > 0.8 else 0,
            'Volume': 1 if volume_ratio > 1.2 else 0,
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
        
        st.success(f"**{signal}** (Score: {total_score}/6)")
        
        # Signal breakdown
        with st.expander("📋 Signal Details"):
            for indicator, score in signals.items():
                st.write(f"{indicator}: {'+' if score > 0 else ''}{score}")
        
        # Additional market data
        st.subheader("📊 Market Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Support", format_price(float(current_data['Support'])))
        with col2:
            st.metric("Resistance", format_price(float(current_data['Resistance'])))
        with col3:
            st.metric("ATR Value", f"{atr:.4f}")
        
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")

# Ana uygulama
def main():
    # Binance client'ı başlat
    client = init_binance_client()
    
    if client is None:
        st.error("❌ Binance API'ye bağlanılamıyor. Lütfen internet bağlantınızı kontrol edin.")
        return
    
    # Verileri Binance'dan yükle
    with st.spinner(f"📊 Binance verileri yükleniyor: {selected_crypto}..."):
        try:
            # Kline verilerini getir
            df = get_binance_klines(symbol, binance_timeframe, 500)
            
            if df is not None and not df.empty:
                # Kolon isimlerini düzelt
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                
                # Teknik analiz
                ta = BinanceTechnicalAnalysis()
                data_with_indicators, fib_levels = ta.calculate_indicators(df)
                
                # Analizi göster
                display_binance_analysis(data_with_indicators, fib_levels, selected_crypto)
                
                # Son 5 mumu göster
                with st.expander("📈 Recent Price Data"):
                    recent_data = data_with_indicators.tail()[['close', 'volume', 'RSI_14', 'EMA_26', 'BB_Upper', 'BB_Lower']]
                    st.dataframe(recent_data.style.format({
                        'close': '${:.4f}',
                        'volume': '{:,.0f}',
                        'RSI_14': '{:.1f}',
                        'EMA_26': '${:.4f}',
                        'BB_Upper': '${:.4f}',
                        'BB_Lower': '${:.4f}'
                    }))
                
            else:
                st.error("❌ Binance'tan veri alınamadı")
                
        except Exception as e:
            st.error(f"❌ Binance data error: {str(e)}")

# Uygulamayı çalıştır
main()

st.markdown("---")
st.info("""
**🚀 Binance API Advantages:**
- ✅ **Real-time data** - Milisaniye doğruluk
- ✅ **High reliability** - Dünyanın en büyük borsası
- ✅ **More indicators** - Volume, order book data
- ✅ **Better performance** - Hızlı response
- ✅ **Free** - Public data ücretsiz

**📖 Trading Signals:**
- **RSI < 30 + MACD Bullish + Volume High** = Strong Buy
- **Price > All EMAs** = Strong Uptrend
- **Bollinger Lower Band** = Potential Bounce
- **Fibonacci Support** = Key Levels
""")

# Binance connection status
try:
    client = init_binance_client()
    if client:
        st.sidebar.success("✅ Binance API: Connected")
        ping = client.get_server_time()
        st.sidebar.info(f"🕒 Server Time: {datetime.datetime.fromtimestamp(ping['serverTime']/1000).strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.sidebar.error("❌ Binance API: Disconnected")
except:
    st.sidebar.error("❌ Binance API: Connection Failed")
