import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import datetime

# Fonksiyon: CoinGecko API'den fiyat verilerini çek
def fetch_crypto_data(coin_id, vs_currency='usd', days=365):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency={vs_currency}&days={days}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # OHLC verisi için basit bir yaklaşım (CoinGecko tam OHLC vermiyor, sadece close kullanıyoruz)
        # Gerçek OHLC için ccxt kütüphanesini kullanabilirsiniz, ancak burada basit tutuyoruz
        df['open'] = df['close'].shift(1)  # Yaklaşık değerler
        df['high'] = df['close'].rolling(window=2).max()
        df['low'] = df['close'].rolling(window=2).min()
        df.dropna(inplace=True)
        return df
    else:
        st.error("Veri çekme hatası!")
        return None

# Destek ve direnç noktalarını hesapla (basit pivot point yöntemi)
def calculate_support_resistance(df):
    high = df['high'].max()
    low = df['low'].min()
    close = df['close'].iloc[-1]
    
    pivot = (high + low + close) / 3
    support1 = (2 * pivot) - high
    resistance1 = (2 * pivot) - low
    support2 = pivot - (high - low)
    resistance2 = pivot + (high - low)
    
    return {
        'support1': support1,
        'resistance1': resistance1,
        'support2': support2,
        'resistance2': resistance2
    }

# Göstergeleri hesapla
def calculate_indicators(df):
    # EMA (Exponential Moving Average) - 12 ve 26 periyot
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # RSI (Relative Strength Index) - 14 periyot
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands - 20 periyot, 2 std
    df['bb_mid'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * 2)
    
    # Fibonacci Retracement seviyeleri (son yüksek ve düşük baz alınarak)
    high = df['high'].max()
    low = df['low'].min()
    diff = high - low
    fib_levels = {
        'fib_0': high,
        'fib_236': high - (0.236 * diff),
        'fib_382': high - (0.382 * diff),
        'fib_500': high - (0.5 * diff),
        'fib_618': high - (0.618 * diff),
        'fib_100': low
    }
    
    return df, fib_levels

# Stratejilere göre al/sat sinyali üret
def generate_signals(df):
    signals = []
    
    # EMA Crossover Stratejisi
    if df['ema12'].iloc[-1] > df['ema26'].iloc[-1] and df['ema12'].iloc[-2] <= df['ema26'].iloc[-2]:
        signals.append("EMA: AL Sinyali (Crossover)")
    elif df['ema12'].iloc[-1] < df['ema26'].iloc[-1] and df['ema12'].iloc[-2] >= df['ema26'].iloc[-2]:
        signals.append("EMA: SAT Sinyali (Crossover)")
    
    # RSI Stratejisi
    if df['rsi'].iloc[-1] < 30:
        signals.append("RSI: AL Sinyali (Aşırı Satım)")
    elif df['rsi'].iloc[-1] > 70:
        signals.append("RSI: SAT Sinyali (Aşırı Alım)")
    
    # Bollinger Bands Stratejisi
    if df['close'].iloc[-1] < df['bb_lower'].iloc[-1]:
        signals.append("Bollinger: AL Sinyali (Alt Bant Kırılımı)")
    elif df['close'].iloc[-1] > df['bb_upper'].iloc[-1]:
        signals.append("Bollinger: SAT Sinyali (Üst Bant Kırılımı)")
    
    # Fibonacci Stratejisi (Fiyat fib seviyelerine yakınsa)
    current_price = df['close'].iloc[-1]
    fib_levels = calculate_indicators(df)[1]  # Fib seviyelerini yeniden al
    for level, value in fib_levels.items():
        if abs(current_price - value) / current_price < 0.01:  # %1 yakınlık
            if 'fib_618' in level or 'fib_500' in level:
                signals.append(f"FIB: AL Sinyali ({level} seviyesi)")
            elif 'fib_236' in level or 'fib_382' in level:
                signals.append(f"FIB: SAT Sinyali ({level} seviyesi)")
    
    if not signals:
        signals.append("Şu an sinyal yok.")
    
    return signals

# Grafik oluştur
def plot_chart(df, sr_levels, fib_levels):
    fig = go.Figure()
    
    # Candlestick grafik
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['open'],
                                 high=df['high'],
                                 low=df['low'],
                                 close=df['close'],
                                 name='Fiyat'))
    
    # EMA'lar
    fig.add_trace(go.Scatter(x=df.index, y=df['ema12'], name='EMA12', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['ema26'], name='EMA26', line=dict(color='orange')))
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['bb_upper'], name='BB Upper', line=dict(color='gray', dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=df['bb_lower'], name='BB Lower', line=dict(color='gray', dash='dash')))
    
    # Destek/Direnç seviyeleri
    fig.add_hline(y=sr_levels['support1'], line_dash="dot", annotation_text="S1", line_color="green")
    fig.add_hline(y=sr_levels['resistance1'], line_dash="dot", annotation_text="R1", line_color="red")
    fig.add_hline(y=sr_levels['support2'], line_dash="dot", annotation_text="S2", line_color="green")
    fig.add_hline(y=sr_levels['resistance2'], line_dash="dot", annotation_text="R2", line_color="red")
    
    # Fibonacci seviyeleri
    for level, value in fib_levels.items():
        fig.add_hline(y=value, line_dash="dash", annotation_text=level, line_color="purple")
    
    fig.update_layout(title='Altcoin Fiyat Grafiği', xaxis_title='Tarih', yaxis_title='Fiyat (USD)', height=600)
    return fig

# Streamlit Uygulaması
st.title("Altcoin Analiz Uygulaması")

# Kullanıcı girdileri
coin_id = st.text_input("Altcoin ID (örneğin: ethereum, solana)", "ethereum")
days = st.slider("Veri Gün Sayısı", 30, 365, 180)

if st.button("Analiz Et"):
    df = fetch_crypto_data(coin_id, days=days)
    if df is not None:
        df, fib_levels = calculate_indicators(df)
        sr_levels = calculate_support_resistance(df)
        signals = generate_signals(df)
        
        # Grafiği göster
        fig = plot_chart(df, sr_levels, fib_levels)
        st.plotly_chart(fig)
        
        # RSI grafiği (ayrı)
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI'))
        rsi_fig.add_hline(y=30, line_dash="dot", line_color="green")
        rsi_fig.add_hline(y=70, line_dash="dot", line_color="red")
        rsi_fig.update_layout(title='RSI Grafiği', height=300)
        st.plotly_chart(rsi_fig)
        
        # Sinyalleri göster
        st.subheader("Al/Sat Sinyalleri")
        for signal in signals:
            st.write(signal)
