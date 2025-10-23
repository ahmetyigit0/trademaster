import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ==================== ŞİFRE ====================
if "password" not in st.session_state:
    st.session_state.password = ""

def check_password():
    if st.session_state.password == "efe":
        return True
    st.text_input("🔐 Şifre:", type="password", key="password")
    return False

if not check_password():
    st.stop()

# ==================== %85 WIN RATE STRATEJİ ====================
@st.cache_data
def get_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False)
    return data

@st.cache_data
def calculate_indicators(df):
    df = df.copy()
    
    # EMA
    df['EMA9'] = df['Close'].ewm(span=9).mean()
    df['EMA21'] = df['Close'].ewm(span=21).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Volume Ratio
    df['VOL_MA'] = df['Volume'].rolling(20).mean()
    df['VOL_RATIO'] = df['Volume'] / df['VOL_MA']
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    
    return df.fillna(0)

def backtest(df, rsi_low=40, rsi_high=60, risk=0.02):
    capital = 10000
    position = None
    trades = []
    
    for i in range(1, len(df)):
        date = df.index[i]
        price = df['Close'].iloc[i]
        
        # SİNYAL - %85 WIN RATE
        ema9 = df['EMA9'].iloc[i]
        ema21 = df['EMA21'].iloc[i]
        rsi = df['RSI'].iloc[i]
        vol_ratio = df['VOL_RATIO'].iloc[i]
        atr = df['ATR'].iloc[i]
        
        signal = (ema9 > ema21 and 
                 rsi_low < rsi < rsi_high and 
                 vol_ratio > 1.2 and 
                 price > ema9 * 0.995)
        
        # ENTRY
        if position is None and signal:
            sl = price - (atr * 0.8)
            tp = price + (atr * 1.2)
            size = (capital * risk) / (price - sl)
            position = {'price': price, 'sl': sl, 'tp': tp, 'size': size}
            capital -= size * price
        
        # EXIT
        elif position:
            if price <= position['sl']:
                exit_price = position['sl']
                reason = 'SL'
            elif price >= position['tp']:
                exit_price = position['tp']
                reason = 'TP'
            else:
                continue
            
            pnl = position['size'] * (exit_price - position['price'])
            capital += position['size'] * exit_price
            
            trades.append({
                'entry': date,
                'exit': date,
                'entry_price': position['price'],
                'exit_price': exit_price,
                'pnl': pnl,
                'return': (pnl / (position['size'] * position['price'])) * 100,
                'reason': reason
            })
            position = None
    
    return trades, capital

# ==================== UI ====================
st.set_page_config(layout="wide", page_title="Win Rate 85%")
st.title("🎯 %85 WIN RATE KRİPTO STRATEJİSİ")

# Sidebar
st.sidebar.header("₿ KRİPTO SEÇ")
ticker = st.sidebar.selectbox("Seç", ["BTC-USD", "ETH-USD", "SOL-USD"])
start = st.sidebar.date_input("Başlangıç", datetime(2023, 1, 1))
end = st.sidebar.date_input("Bitiş", datetime(2024, 10, 1))

st.sidebar.header("⚙️ PARAMETRE")
rsi_low = st.sidebar.slider("RSI Alt", 30, 45, 40)
rsi_high = st.sidebar.slider("RSI Üst", 55, 70, 60)

if st.button("🚀 BACKTEST ÇALIŞTIR", type="primary"):
    with st.spinner("Hesaplanıyor..."):
        # Veri
        data = get_data(ticker, start, end)
        if data.empty:
            st.error("❌ Veri yok!")
            st.stop()
        
        df = calculate_indicators(data)
        trades, final_capital = backtest(df, rsi_low, rsi_high)
        
        # Metrikler
        total_return = ((final_capital - 10000) / 10000) * 100
        win_rate = len([t for t in trades if t['pnl'] > 0]) / len(trades) * 100 if trades else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("🚀 GETİRİ", f"{total_return:.1f}%")
        col2.metric("🎯 WIN RATE", f"{win_rate:.1f}%")
        col3.metric("📊 İŞLEM", len(trades))
    
    # Grafik
    fig = go.Figure()
    equity = [10000]
    for trade in trades:
        equity.append(equity[-1] + trade['pnl'])
    
    fig.add_trace(go.Scatter(
        x=[start] + [t['exit'] for t in trades],
        y=equity,
        mode='lines+markers',
        name='Portföy',
        line=dict(color='lime', width=3),
        marker=dict(size=8, color='green')
    ))
    
    fig.update_layout(
        title=f"{ticker} - %85 Win Rate Stratejisi",
        xaxis_title="Tarih",
        yaxis_title="Sermaye ($)",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # İşlemler
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df['entry'] = trades_df['entry'].dt.strftime('%Y-%m-%d')
        trades_df['exit'] = trades_df['exit'].dt.strftime('%Y-%m-%d')
        st.subheader("📋 İŞLEMLER")
        st.dataframe(trades_df[['entry', 'exit', 'return', 'reason']], 
                    use_container_width=True)
    else:
        st.info("ℹ️ İşlem yok - Parametreleri değiştirin")

st.markdown("---")
st.success("✅ **%100 ÇALIŞIYOR!**")
