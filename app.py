import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from datetime import datetime, timedelta

st.set_page_config(page_title="TradingView Clone", layout="wide")

# Şifre koruması
def check_password():
    def password_entered():
        if st.session_state["password"] == "efe":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Şifre", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Şifre", type="password", on_change=password_entered, key="password")
        st.error("❌ Şifre yanlış!")
        return False
    else:
        return True

if not check_password():
    st.stop()

# TradingView CSS
st.markdown("""
<style>
    .tv-header {
        background: #1e222d;
        padding: 8px 15px;
        border-bottom: 1px solid #363c4e;
        color: white;
    }
    .symbol-display {
        font-size: 20px;
        font-weight: bold;
        color: #ececec;
    }
    .price-display {
        font-size: 24px;
        font-weight: bold;
    }
    .price-up { color: #00b15d; }
    .price-down { color: #ff5b5a; }
    .indicator-panel {
        background: #1e222d;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
    .timeframe-btn {
        background: #2a2e39;
        border: 1px solid #363c4e;
        border-radius: 3px;
        padding: 5px 10px;
        margin: 2px;
        color: #b2b5be;
        cursor: pointer;
    }
    .timeframe-btn.active {
        background: #2962ff;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# TradingView benzeri header
col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
with col1:
    st.markdown('<div class="tv-header"><div class="symbol-display">BTC/USD</div></div>', unsafe_allow_html=True)
with col2:
    current_price = 43250.50
    price_change = 2.45
    st.markdown(f'<div class="tv-header price-display price-up">${current_price:,.2f}</div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="tv-header price-up">+{price_change}%</div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="tv-header">24H Volume: $28.5B</div>', unsafe_allow_html=True)
with col5:
    st.markdown('<div class="tv-header">Market Cap: $845B</div>', unsafe_allow_html=True)

# Ana layout
main_col, side_col = st.columns([4, 1])

with main_col:
    # Timeframe butonları
    timeframes = ["1m", "5m", "15m", "1H", "4H", "1D", "1W", "1M"]
    cols = st.columns(len(timeframes))
    for i, tf in enumerate(timeframes):
        with cols[i]:
            if st.button(tf, key=f"tf_{tf}"):
                st.session_state.selected_tf = tf
    
    # Grafik container
    st.markdown("""
    <div style="background: #131722; border-radius: 5px; padding: 10px; margin: 10px 0;">
    """, unsafe_allow_html=True)
    
    # TradingView benzeri grafik
    def create_tradingview_chart():
        # Örnek veri oluştur
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        np.random.seed(42)
        prices = []
        current_price = 40000
        for _ in range(len(dates)):
            change = np.random.normal(0, 0.02)
            current_price *= (1 + change)
            prices.append(current_price)
        
        df = pd.DataFrame({
            'Date': dates,
            'Open': [p * (1 + np.random.normal(0, 0.01)) for p in prices],
            'High': [p * (1 + abs(np.random.normal(0, 0.015))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.015))) for p in prices],
            'Close': prices,
            'Volume': [np.random.randint(1000000, 5000000) for _ in range(len(dates))]
        })
        
        # Ana grafik
        fig = sp.make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price Chart', 'Volume'),
            row_width=[0.7, 0.3]
        )
        
        # Mum grafiği
        fig.add_trace(go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ), row=1, col=1)
        
        # EMA'lar
        df['EMA_20'] = df['Close'].ewm(span=20).mean()
        df['EMA_50'] = df['Close'].ewm(span=50).mean()
        
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['EMA_20'],
            name='EMA 20',
            line=dict(color='#ff6b6b', width=1.5)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['EMA_50'],
            name='EMA 50',
            line=dict(color='#4ecdc4', width=1.5)
        ), row=1, col=1)
        
        # Volume
        colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' 
                 for i in range(len(df))]
        
        fig.add_trace(go.Bar(
            x=df['Date'],
            y=df['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ), row=2, col=1)
        
        # Layout güncellemeleri
        fig.update_layout(
            height=600,
            plot_bgcolor='#131722',
            paper_bgcolor='#131722',
            font=dict(color='#d1d4dc'),
            xaxis=dict(
                gridcolor='#2a2e39',
                rangeslider=dict(visible=False)
            ),
            yaxis=dict(gridcolor='#2a2e39'),
            xaxis2=dict(gridcolor='#2a2e39'),
            yaxis2=dict(gridcolor='#2a2e39'),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Grafik stilleri
        fig.update_xaxes(showline=True, linewidth=1, linecolor='#363c4e')
        fig.update_yaxes(showline=True, linewidth=1, linecolor='#363c4e')
        
        return fig
    
    # Grafiği göster
    chart_fig = create_tradingview_chart()
    st.plotly_chart(chart_fig, use_container_width=True, config={
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawcircle', 'drawrect', 'eraseshape'],
        'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
    })

with side_col:
    st.markdown("""
    <div style="background: #1e222d; border-radius: 5px; padding: 15px; margin: 10px 0;">
    """, unsafe_allow_html=True)
    
    # Göstergeler paneli
    st.subheader("📊 Göstergeler")
    
    # RSI
    st.markdown('<div class="indicator-panel">', unsafe_allow_html=True)
    st.metric("RSI (14)", "56.7", "2.3")
    st.progress(56.7/100)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # MACD
    st.markdown('<div class="indicator-panel">', unsafe_allow_html=True)
    st.write("**MACD**")
    st.write("Histogram: 12.5")
    st.write("Signal: 8.2")
    st.write("MACD: 4.3")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Bollinger Bands
    st.markdown('<div class="indicator-panel">', unsafe_allow_html=True)
    st.write("**Bollinger Bands**")
    st.write("Upper: $44,230")
    st.write("Middle: $42,150")
    st.write("Lower: $40,070")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Trading Sinyalleri
    st.markdown("""
    <div style="background: #1e222d; border-radius: 5px; padding: 15px; margin: 10px 0;">
        <h4>🎯 Sinyaller</h4>
        <div style="color: #00b15d; margin: 5px 0;">✓ EMA 20 > EMA 50</div>
        <div style="color: #ff5b5a; margin: 5px 0;">✗ RSI > 70 (Aşırı Alım)</div>
        <div style="color: #00b15d; margin: 5px 0;">✓ Volume Artışı</div>
    </div>
    """, unsafe_allow_html=True)

# Alt panel - Teknik analiz
st.markdown("---")
st.subheader("📈 Teknik Analiz")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style="background: #1e222d; padding: 15px; border-radius: 5px;">
        <h4>Trend Analizi</h4>
        <div>Kısa Vade: ↗️ Yükseliş</div>
        <div>Orta Vade: ➡️ Yatay</div>
        <div>Uzun Vade: ↗️ Yükseliş</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background: #1e222d; padding: 15px; border-radius: 5px;">
        <h4>Destek/Direnç</h4>
        <div>Destek 1: $41,200</div>
        <div>Destek 2: $40,500</div>
        <div>Direnç 1: $43,800</div>
        <div>Direnç 2: $45,200</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background: #1e222d; padding: 15px; border-radius: 5px;">
        <h4>Volatilite</h4>
        <div>ATR: 850</div>
        <div>Volatilite: Orta</div>
        <div>Average Volume: 2.4M</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style="background: #1e222d; padding: 15px; border-radius: 5px;">
        <h4>Öneriler</h4>
        <div>Risk/Reward: 1:2.5</div>
        <div>Stop Loss: $40,800</div>
        <div>Take Profit: $45,500</div>
    </div>
    """, unsafe_allow_html=True)

# Çizim araçları
st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Çizim Araçları")
drawing_tools = st.sidebar.multiselect(
    "Araçlar",
    ["Yatay Çizgi", "Dikey Çizgi", "Trend Çizgi", "Fibonacci", Dikdörtgen", "Daire"],
    default=["Yatay Çizgi", "Trend Çizgi"]
)

# TradingView benzeri kısayollar
st.sidebar.markdown("---")
st.sidebar.subheader("⌨️ Kısayollar")
st.sidebar.write("**1-9** - Timeframe")
st.sidebar.write("**Alt + 1-5** - Grafik tipi")
st.sidebar.write("**Ctrl + Z** - Geri al")
st.sidebar.write("**Space** - Hareket aracı")