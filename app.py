import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
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
        st.markdown("""
            <style>
                .main {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    height: 100vh;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }
                .login-box {
                    background: white;
                    padding: 3rem;
                    border-radius: 15px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                    text-align: center;
                }
            </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        st.title("🔐 Crypto AI Pro")
        st.markdown("**Profesyonel Algoritmik Analiz Platformu**")
        st.text_input("Şifre", type="password", on_change=password_entered, key="password", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        return False
    return True

if not check_password():
    st.stop()

# =========================
# PAGE CONFIG - MODERN
# =========================
st.set_page_config(
    page_title="Crypto AI Pro", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .signal-buy {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
    }
    .signal-sell {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
    }
    .signal-neutral {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
    }
    .mini-chart {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

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

def create_pro_chart(data):
    """Profesyonel TradingView benzeri grafik"""
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="Price"
    ))

    # EMA'lar
    fig.add_trace(go.Scatter(
        x=data.index, y=data['EMA_20'],
        name='EMA 20',
        line=dict(color='orange', width=1)
    ))
    fig.add_trace(go.Scatter(
        x=data.index, y=data['EMA_50'],
        name='EMA 50',
        line=dict(color='red', width=1)
    ))

    # Bollinger Bantları
    fig.add_trace(go.Scatter(
        x=data.index, y=data['BB_Upper'],
        name='BB Upper',
        line=dict(color='gray', width=1, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=data.index, y=data['BB_Lower'],
        name='BB Lower',
        line=dict(color='gray', width=1, dash='dash'),
        fill='tonexty'
    ))

    fig.update_layout(
        title=f'Profesyonel Price Chart',
        height=400,
        xaxis_rangeslider_visible=False,
        template='plotly_white'
    )
    return fig

def create_advanced_rsi_chart(data):
    """Gelişmiş RSI grafiği"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data.index, y=data['RSI'],
        name='RSI',
        line=dict(color='purple', width=2)
    ))

    fig.add_hline(y=30, line_dash="dash", line_color="green")
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=50, line_dash="dot", line_color="gray")

    fig.update_layout(
        title='RSI Momentum',
        height=200,
        yaxis_range=[0, 100],
        template='plotly_white'
    )
    return fig

def create_macd_chart(data):
    """MACD grafiği"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data.index, y=data['MACD'],
        name='MACD',
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=data.index, y=data['MACD_Signal'],
        name='Signal',
        line=dict(color='red', width=1)
    ))

    # Histogram
    colors = ['green' if x >= 0 else 'red' for x in data['MACD_Hist']]
    fig.add_trace(go.Bar(
        x=data.index, y=data['MACD_Hist'],
        name='Histogram',
        marker_color=colors,
        opacity=0.5
    ))

    fig.update_layout(
        title='MACD',
        height=200,
        template='plotly_white'
    )
    return fig

# =========================
# MODERN ARAYÜZ
# =========================
st.markdown('<h1 class="main-header">🚀 Crypto AI Pro</h1>', unsafe_allow_html=True)
st.markdown("**Profesyonel Algoritmik Analiz Platformu**")

# Hızlı ayar çubuğu
col1, col2, col3, col4 = st.columns(4)
with col1:
    ticker_input = st.selectbox("🎯 Asset", ["BTC-USD", "ETH-USD", "ADA-USD", "BNB-USD", "XRP-USD", "SOL-USD"], index=0)
with col2:
    timeframe = st.selectbox("⏰ Timeframe", ["1d", "1wk"], index=0)
with col3:
    capital = st.number_input("💰 Capital", 1000, 1000000, 10000, step=1000)
with col4:
    risk_percent = st.slider("📉 Risk %", 1.0, 5.0, 2.0, 0.1)

# Ana içerik
try:
    with st.spinner(f"🔄 Loading {ticker_input} data..."):
        period_map = {"1d": "6mo", "1wk": "1y"}
        period = period_map[timeframe]
        data = yf.download(ticker_input, period=period, interval=timeframe, progress=False)
    
    if data.empty:
        st.error("❌ Data loading failed")
        st.stop()

    # Hesaplamalar
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
    atr = float(data['ATR'].iloc[-1])

    # Sinyal hesaplama
    buy_signals = sum([
        rsi < 35,
        current_price > ema_20 > ema_50,
        ema_50 > ema_200,
        macd > macd_signal,
        current_price < data['BB_Lower'].iloc[-1]
    ])
    
    sell_signals = sum([
        rsi > 65,
        current_price < ema_20 < ema_50,
        ema_50 < ema_200,
        macd < macd_signal,
        current_price > data['BB_Upper'].iloc[-1]
    ])

    # Ana metrikler
    st.markdown("### 📊 Real-Time Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        price_change = ((current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100
        st.metric("Current Price", f"${current_price:.2f}", f"{price_change:+.2f}%")
    
    with col2:
        st.metric("RSI", f"{rsi:.1f}", "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral")
    
    with col3:
        trend = "Bullish" if ema_20 > ema_50 > ema_200 else "Bearish" if ema_20 < ema_50 < ema_200 else "Neutral"
        st.metric("Trend", trend)
    
    with col4:
        st.metric("Volatility", f"${atr:.2f}")

    # Sinyal gösterimi
    st.markdown("### 🎯 Trading Signal")
    if buy_signals >= 4:
        st.markdown(f'<div class="signal-buy"><h2>🚀 STRONG BUY SIGNAL</h2><p>Confidence: {buy_signals}/5 indicators bullish</p></div>', unsafe_allow_html=True)
        recommendation = "BUY"
    elif sell_signals >= 4:
        st.markdown(f'<div class="signal-sell"><h2>🔻 STRONG SELL SIGNAL</h2><p>Confidence: {sell_signals}/5 indicators bearish</p></div>', unsafe_allow_html=True)
        recommendation = "SELL"
    else:
        st.markdown(f'<div class="signal-neutral"><h2>⚡ NEUTRAL SIGNAL</h2><p>Market consolidating - Wait for confirmation</p></div>', unsafe_allow_html=True)
        recommendation = "HOLD"

    # Profesyonel grafikler
    st.markdown("### 📈 Advanced Charts")
    
    tab1, tab2, tab3 = st.tabs(["Price Action", "RSI Momentum", "MACD Analysis"])
    
    with tab1:
        st.plotly_chart(create_pro_chart(data.tail(100)), use_container_width=True)
    
    with tab2:
        st.plotly_chart(create_advanced_rsi_chart(data.tail(100)), use_container_width=True)
    
    with tab3:
        st.plotly_chart(create_macd_chart(data.tail(100)), use_container_width=True)

    # Trading stratejisi
    st.markdown("### 💡 Trading Strategy")
    
    if recommendation == "BUY":
        stop_loss = current_price - (atr * 1.5)
        risk_amount = capital * (risk_percent / 100)
        risk_per_coin = current_price - stop_loss
        position_size = risk_amount / risk_per_coin
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Entry Strategy:**
            - Market Buy: 60% at current price
            - Limit Buy: 40% at 2% below
            """)
            
            st.metric("Stop Loss", f"${stop_loss:.2f}")
            st.metric("Position Size", f"{position_size:.4f} {ticker_input.split('-')[0]}")
        
        with col2:
            st.markdown("""
            **Take Profit Levels:**
            - TP1 (1:1): +{:.2f}%
            - TP2 (1:2): +{:.2f}%
            - TP3 (1:3): +{:.2f}%
            """.format(
                (risk_per_coin/current_price*100),
                (risk_per_coin*2/current_price*100),
                (risk_per_coin*3/current_price*100)
            ))

    elif recommendation == "SELL":
        st.markdown("""
        **Short Strategy:**
        - Consider short positions or wait for better entry
        - Key resistance: ${:.2f}
        - Support level: ${:.2f}
        """.format(data['High'].tail(20).max(), data['Low'].tail(20).min()))

    # Market insights
    st.markdown("### 🔍 Market Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Trend Analysis**")
        st.write(f"EMA Alignment: {'✅ Bullish' if ema_20 > ema_50 > ema_200 else '❌ Bearish' if ema_20 < ema_50 < ema_200 else '⚡ Mixed'}")
        st.write(f"Price vs EMA20: {'Above' if current_price > ema_20 else 'Below'}")
    
    with col2:
        st.markdown("**Momentum**")
        st.write(f"RSI: {'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'}")
        st.write(f"MACD: {'Bullish' if macd > macd_signal else 'Bearish'}")
    
    with col3:
        st.markdown("**Risk Assessment**")
        vol_ratio = (atr / current_price * 100)
        st.write(f"Volatility: {'High' if vol_ratio > 5 else 'Low' if vol_ratio < 2 else 'Medium'}")
        st.write(f"Signal Strength: {max(buy_signals, sell_signals)}/5")

except Exception as e:
    st.error(f"❌ System error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>⚠️ Risk Disclaimer:</strong> This is for educational purposes only. Cryptocurrency trading involves substantial risk.</p>
    <p>Crypto AI Pro v3.0 | Professional Algorithmic Analysis</p>
</div>
""", unsafe_allow_html=True)
