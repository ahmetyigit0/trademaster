import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import time

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
timeframe = st.sidebar.selectbox("Timeframe:", ["1h", "4h", "1d", "1wk"], index=1)
period_days = st.sidebar.slider("Data Period (Days):", 30, 365, 90)

# ÇOK BASİT ve GÜVENLİ Teknik Analiz Sınıfı
class SimpleTechnicalAnalysis:
    def __init__(self):
        pass
    
    def calculate_rsi(self, prices, window=14):
        """Basit RSI hesapla"""
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
    
    def calculate_simple_indicators(self, df):
        """BASİT ve GÜVENLİ gösterge hesaplama"""
        try:
            df = df.copy()
            
            # 1. RSI - Series olarak
            rsi_series = self.calculate_rsi(df['Close'], 14)
            df = df.assign(RSI_14=rsi_series)
            
            # 2. EMA'lar - tek tek assign
            df = df.assign(EMA_12=df['Close'].ewm(span=12).mean())
            df = df.assign(EMA_26=df['Close'].ewm(span=26).mean())
            df = df.assign(EMA_50=df['Close'].ewm(span=50).mean())
            
            # 3. MACD
            macd_series = df['EMA_12'] - df['EMA_26']
            df = df.assign(MACD=macd_series)
            df = df.assign(MACD_Signal=macd_series.ewm(span=9).mean())
            
            # 4. Bollinger Bands - Series olarak hesapla
            bb_middle = df['Close'].rolling(20).mean()
            bb_std = df['Close'].rolling(20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            
            df = df.assign(BB_Middle=bb_middle)
            df = df.assign(BB_Upper=bb_upper)
            df = df.assign(BB_Lower=bb_lower)
            
            # 5. Volume - Series olarak
            volume_sma = df['Volume'].rolling(20, min_periods=1).mean()
            volume_ratio = df['Volume'] / volume_sma.replace(0, 1)
            df = df.assign(Volume_Ratio=volume_ratio)
            
            # 6. ATR
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            true_range = np.maximum(np.maximum(high_low, high_close), low_close)
            atr_series = true_range.rolling(14).mean()
            df = df.assign(ATR=atr_series)
            
            # 7. Fibonacci
            recent_high = df['High'].tail(20).max()
            recent_low = df['Low'].tail(20).min()
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
            st.error(f"Indicator error: {str(e)}")
            return df, {}

# Veri yükleme
def load_crypto_data(symbol, period_days, timeframe):
    try:
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=period_days)
        data = yf.download(symbol, start=start_date, end=end_date, interval=timeframe, progress=False)
        return data
    except Exception as e:
        st.error(f"Data loading error: {e}")
        return None

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

# GÜVENLİ Sinyal analizi - pandas Series hatası düzeltildi
def display_signal_analysis(df, fib_levels):
    if df is None or df.empty:
        st.error("No data available for analysis")
        return
    
    try:
        current_data = df.iloc[-1]
        
        # Tüm değerleri float'a çevir - PANDAS SERIES HATASI ÇÖZÜMÜ
        current_price = float(current_data['Close'])
        rsi = float(current_data['RSI_14'])
        ema_12 = float(current_data['EMA_12'])
        ema_26 = float(current_data['EMA_26'])
        ema_50 = float(current_data['EMA_50'])
        macd = float(current_data['MACD'])
        macd_signal = float(current_data['MACD_Signal'])
        bb_upper = float(current_data['BB_Upper'])
        bb_lower = float(current_data['BB_Lower'])
        bb_middle = float(current_data['BB_Middle'])
        atr = float(current_data['ATR'])
        volume_ratio = float(current_data.get('Volume_Ratio', 1))
        
        st.subheader(f"📊 Technical Analysis for {selected_crypto}")
        
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
        
        # Trend Analizi
        st.subheader("🔍 Trend Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Moving Averages**")
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
        
        with col2:
            st.write("**MACD Signal**")
            if macd > macd_signal:
                signal = "🟢 Bullish"
            else:
                signal = "🔴 Bearish"
                
            st.write(signal)
            st.write(f"MACD: {macd:.4f}")
            st.write(f"Signal: {macd_signal:.4f}")
        
        with col3:
            st.write("**Volume & Momentum**")
            if volume_ratio > 1.5:
                vol_signal = "🟢 High"
            elif volume_ratio > 0.8:
                vol_signal = "🟡 Normal"
            else:
                vol_signal = "🔴 Low"
                
            st.write(f"Volume: {vol_signal}")
            st.write(f"Ratio: {volume_ratio:.1f}x")
        
        # Fibonacci Levels
        st.subheader("📊 Fibonacci Levels")
        
        if fib_levels:
            cols = st.columns(5)
            
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
        
        # Basit sinyal hesaplama - TÜM DEĞERLER FLOAT
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
    # Verileri yükle
    with st.spinner("Loading data..."):
        data = load_crypto_data(symbol, period_days, timeframe)
        
        if data is not None and not data.empty:
            ta = SimpleTechnicalAnalysis()
            data_with_indicators, fib_levels = ta.calculate_simple_indicators(data)
            
            # Sinyal analizini göster
            display_signal_analysis(data_with_indicators, fib_levels)
            
        else:
            st.error("Could not load data for analysis")

# Uygulamayı çalıştır
main()

st.markdown("---")
st.info("""
**📖 Trading Guide:**
- **RSI < 35**: Buy signal, **> 65**: Sell signal
- **MACD > Signal**: Buy, **< Signal**: Sell
- **Price > EMA 26**: Uptrend, **< EMA 26**: Downtrend  
- **Bollinger Lower**: Buy, **Upper**: Sell
- **4/4 signals**: Strong conviction
""")
