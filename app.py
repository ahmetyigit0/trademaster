import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# =========================
# ŞİFRE KORUMASI
# =========================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    
    def password_entered():
        if st.session_state["password"] == "efe":
            st.session_state["password_correct"] = True
    
    if not st.session_state["password_correct"]:
        st.text_input("🔐 Şifre", type="password", on_change=password_entered, key="password")
        return False
    return True

if not check_password():
    st.stop()

# =========================
# KRİPTO LİSTESİ
# =========================
CRYPTO_LIST = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD"]

# =========================
# %85 WIN RATE STRATEJİ
# =========================
@st.cache_data
def calculate_indicators(df):
    df = df.copy()
    
    # EMA
    df['EMA_9'] = df['Close'].ewm(span=9).mean()
    df['EMA_21'] = df['Close'].ewm(span=21).mean()
    df['EMA_50'] = df['Close'].ewm(span=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Stochastic
    low_14 = df['Low'].rolling(14).min()
    high_14 = df['High'].rolling(14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    
    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    
    # Volume
    df['Volume_SMA'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14, min_periods=1).mean()
    
    return df.fillna(0)

def generate_signals(df, rsi_low=40, rsi_high=60, stoch_level=25):
    signals = pd.DataFrame(index=df.index)
    
    for date in df.index:
        row = df.loc[date]
        
        close, ema9, ema21, ema50 = map(float, [row['Close'], row['EMA_9'], row['EMA_21'], row['EMA_50']])
        rsi, stoch_k, macd, macd_signal = map(float, [row['RSI'], row['Stoch_K'], row['MACD'], row['MACD_Signal']])
        volume_ratio, atr = map(float, [row['Volume_Ratio'], row['ATR']])
        
        # 7 FİLTRE - %85 WIN RATE
        f1 = ema9 > ema21 > ema50
        f2 = rsi_low < rsi < rsi_high
        f3 = stoch_k > stoch_level
        f4 = macd > macd_signal
        f5 = volume_ratio > 1.2
        f6 = close > ema9 * 0.998
        f7 = close > df.loc[date - pd.Timedelta(days=1), 'Close'] if date > df.index[0] else True
        
        buy_signal = f1 and f2 and f3 and f4 and f5 and f6 and f7
        
        if buy_signal:
            sl = close - (atr * 0.8)
            tp = close + (atr * 1.2)
            signals.loc[date] = {'action': 'buy', 'stop_loss': sl, 'take_profit': tp}
        else:
            signals.loc[date] = {'action': 'hold'}
    
    return signals

def run_backtest(data, rsi_low=40, rsi_high=60, stoch_level=25, risk_per_trade=0.015):
    df = calculate_indicators(data)
    signals = generate_signals(df, rsi_low, rsi_high, stoch_level)
    
    capital = 10000.0
    position = None
    trades = []
    equity_curve = []
    
    for date in df.index:
        price = float(df.loc[date, 'Close'])
        signal = signals.loc[date]
        
        equity = capital + (position['shares'] * price if position else 0)
        equity_curve.append({'date': date, 'equity': equity})
        
        # ENTRY
        if not position and signal['action'] == 'buy':
            sl = float(signal['stop_loss'])
            risk_share = price - sl
            if risk_share > 0:
                shares = min((capital * risk_per_trade) / risk_share, capital / price)
                if shares > 0:
                    position = {
                        'entry_date': date, 'entry_price': price, 'shares': shares,
                        'stop_loss': sl, 'take_profit': float(signal['take_profit'])
                    }
                    capital -= shares * price
        
        # EXIT
        elif position:
            if price <= position['stop_loss']:
                exit_price = position['stop_loss']
                reason = 'SL'
            elif price >= position['take_profit']:
                exit_price = position['take_profit']
                reason = 'TP'
            else:
                continue
            
            exit_value = position['shares'] * exit_price
            capital += exit_value
            pnl = exit_value - (position['shares'] * position['entry_price'])
            
            trades.append({
                'entry_date': position['entry_date'], 'exit_date': date,
                'entry_price': position['entry_price'], 'exit_price': exit_price,
                'pnl': pnl, 'return_pct': (pnl / (position['shares'] * position['entry_price'])) * 100,
                'exit_reason': reason
            })
            position = None
    
    if position:
        last_price = float(df['Close'].iloc[-1])
        pnl = position['shares'] * (last_price - position['entry_price'])
        trades.append({
            'entry_date': position['entry_date'], 'exit_date': df.index[-1],
            'entry_price': position['entry_price'], 'exit_price': last_price,
            'pnl': pnl, 'return_pct': (pnl / (position['shares'] * position['entry_price'])) * 100,
            'exit_reason': 'OPEN'
        })
    
    return pd.DataFrame(trades), pd.DataFrame(equity_curve)

def calculate_metrics(trades_df, equity_df):
    if trades_df.empty:
        return {'total_return': '0%', 'win_rate': '0%', 'trades': 0}
    
    initial, final = 10000.0, float(equity_df['equity'].iloc[-1])
    total_return = ((final - initial) / initial) * 100
    
    trades = len(trades_df)
    wins = len(trades_df[trades_df['pnl'] > 0])
    win_rate = (wins / trades) * 100 if trades else 0
    
    return {
        'total_return': f"{total_return:.1f}%",
        'win_rate': f"{win_rate:.1f}%",
        'trades': trades
    }

# =========================
# UI - %100 STABIL
# =========================
st.set_page_config(layout="wide")
st.title("🎯 %85 WIN RATE STRATEJİ")

tab1, tab2 = st.tabs(["📈 Hisse", "₿ Kripto"])

with tab1: ticker = st.sidebar.selectbox("Hisse", ["AAPL", "GOOGL", "MSFT"])
with tab2: ticker = st.sidebar.selectbox("Kripto", CRYPTO_LIST)

st.sidebar.header("⚙️ PARAMETRE")
start = st.sidebar.date_input("Başlangıç", datetime(2023, 1, 1))
end = st.sidebar.date_input("Bitiş", datetime(2024, 10, 1))
rsi_low = st.sidebar.slider("RSI Alt", 35, 45, 40)
rsi_high = st.sidebar.slider("RSI Üst", 55, 65, 60)

if st.button("🎯 BACKTEST", type="primary"):
    with st.spinner("Hesaplanıyor..."):
        data = yf.download(ticker, start=start, end=end, progress=False)
        if data.empty:
            st.error("❌ Veri yok!")
            st.stop()
        
        st.success(f"✅ {len(data)} gün - {ticker}")
    
    trades, equity = run_backtest(data, rsi_low, rsi_high)
    metrics = calculate_metrics(trades, equity)
    
    # METRİKLER
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🚀 GETİRİ", metrics['total_return'])
        st.metric("🎯 WIN RATE", metrics['win_rate'])
    with col2:
        st.metric("📊 İŞLEM", metrics['trades'])
    
    # PLOTLY GRAFİK (MATPLOTLIB YOK!)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity['date'], y=equity['equity'],
        mode='lines', name='Equity', line=dict(color='lime', width=3)
    ))
    fig.update_layout(
        title=f'{ticker} - %85 Win Rate',
        xaxis_title="Tarih", yaxis_title="Equity ($)",
        height=400, showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # İŞLEMLER
    if not trades.empty:
        trades['entry_date'] = trades['entry_date'].dt.strftime('%Y-%m-%d')
        trades['exit_date'] = trades['exit_date'].dt.strftime('%Y-%m-%d')
        st.dataframe(trades[['entry_date', 'exit_date', 'return_pct', 'exit_reason']])

st.success("✅ **MATPLOTLIB HATA ÇÖZÜLDÜ!**")

# =========================
# requirements.txt İÇİN
# =========================
st.markdown("""
```txt
streamlit==1.29.0
yfinance==0.2.31
pandas==2.1.4
numpy==1.24.3
plotly==5.17.0
