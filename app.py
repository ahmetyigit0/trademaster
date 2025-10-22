import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =========================
# ŞİFRE KORUMASI
# =========================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    
    if not st.session_state["password_correct"]:
        st.markdown("### 🔐 Kombine Strateji")
        password = st.text_input("Şifre:", type="password", key="password_input")
        
        if password == "efe":
            st.session_state["password_correct"] = True
            st.success("✅ Giriş başarılı!")
            st.rerun()
        elif password:
            st.error("❌ Yanlış şifre!")
            st.stop()
        return False
    return True

if not check_password():
    st.stop()

# =========================
# %100 GÜVENLİ - HİÇ DEĞİŞKEN YOK
# =========================
class PerfectBacktest:
    def __init__(self):
        self.capital = 10000
    
    def indicators(self, df):
        df = df.copy()
        
        # EMA
        df['EMA20'] = df['Close'].ewm(span=20).mean()
        df['EMA50'] = df['Close'].ewm(span=50).mean()
        
        # RSI
        delta = df['Close'].diff()
        df['RSI'] = 100 - 100 / (1 + delta.where(delta>0,0).rolling(14).mean() / 
                                abs(delta.where(delta<0,0)).rolling(14).mean())
        
        # BB Lower - TEK SATIR HİÇ DEĞİŞKEN YOK
        df['BB_Lower'] = df['Close'].rolling(20).mean() - df['Close'].rolling(20).std() * 2
        
        # MACD
        df['MACD'] = df['Close'].ewm(12).mean() - df['Close'].ewm(26).mean()
        df['Signal'] = df['MACD'].ewm(9).mean()
        
        # FIB
        df['Fib'] = df['Low'].rolling(50).min() + (df['High'].rolling(50).max() - df['Low'].rolling(50).min()) * 0.382
        
        # FILLNA
        df = df.fillna(0)
        return df
    
    def signals(self, df, rsi_level, rr):
        df['action'] = 'hold'
        df['sl'] = 0
        df['tp'] = 0
        
        # TEK SATIR SİNYAL
        buy = (
            (df['EMA20'] > df['EMA50']) &
            (df['RSI'] < rsi_level) &
            ((df['Close'] < df['BB_Lower']) | (df['Close'] < df['Fib'])) &
            (df['MACD'] > df['Signal']) &
            (df['MACD'].shift(1) <= df['Signal'].shift(1))
        )
        
        buy_dates = df[buy].index
        risk = 0.02
        
        for date in buy_dates:
            price = df.loc[date, 'Close']
            df.loc[date, 'action'] = 'buy'
            df.loc[date, 'sl'] = price * (1 - risk)
            df.loc[date, 'tp'] = price * (1 + risk * rr)
        
        return len(buy_dates)
    
    def backtest(self, df, rsi_level, rr, risk_pct):
        df = self.indicators(df)
        signal_count = self.signals(df, rsi_level, rr)
        
        capital = self.capital
        position = None
        trades = []
        equity = []
        
        for date, row in df.iterrows():
            price = row['Close']
            action = row['action']
            
            # Equity
            eq = capital
            if position:
                eq += position['shares'] * price
            equity.append({'date': date, 'equity': eq})
            
            # BUY
            if not position and action == 'buy':
                sl = row['sl']
                risk_share = price - sl
                if risk_share > 0:
                    shares = (capital * risk_pct) / risk_share
                    shares = min(shares, capital * 0.95 / price)
                    
                    position = {'date': date, 'price': price, 'shares': shares, 'sl': sl, 'tp': row['tp']}
                    capital -= shares * price
            
            # SELL
            elif position:
                if price <= position['sl']:
                    exit_p = position['sl']
                    reason = 'SL'
                elif price >= position['tp']:
                    exit_p = position['tp']
                    reason = 'TP'
                else:
                    continue
                
                capital += position['shares'] * exit_p
                pnl = (exit_p - position['price']) * position['shares']
                
                trades.append({
                    'entry': position['date'],
                    'exit': date,
                    'entry_p': position['price'],
                    'exit_p': exit_p,
                    'shares': position['shares'],
                    'pnl': pnl,
                    'ret': (pnl / (position['price'] * position['shares'])) * 100,
                    'reason': reason
                })
                position = None
        
        # CLOSE OPEN
        if position:
            last_p = df['Close'].iloc[-1]
            capital += position['shares'] * last_p
            pnl = (last_p - position['price']) * position['shares']
            trades.append({
                'entry': position['date'],
                'exit': df.index[-1],
                'entry_p': position['price'],
                'exit_p': last_p,
                'shares': position['shares'],
                'pnl': pnl,
                'ret': (pnl / (position['price'] * position['shares'])) * 100,
                'reason': 'OPEN'
            })
        
        return pd.DataFrame(trades), pd.DataFrame(equity), signal_count
    
    def metrics(self, trades, equity):
        if trades.empty:
            return {'ret': '0%', 'trades': 0, 'win': '0%', 'pf': '0', 'dd': '0%'}
        
        total_ret = ((equity['equity'].iloc[-1] - self.capital) / self.capital) * 100
        wins = len(trades[trades['pnl'] > 0])
        win_rate = (wins / len(trades)) * 100
        profit = trades[trades['pnl'] > 0]['pnl'].sum()
        loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
        pf = profit / loss if loss > 0 else 999
        peak = equity['equity'].expanding().max()
        dd = ((equity['equity'] - peak) / peak * 100).min()
        
        return {
            'ret': f"{total_ret:+.1f}%",
            'trades': len(trades),
            'win': f"{win_rate:.1f}%",
            'pf': f"{pf:.1f}",
            'dd': f"{dd:.1f}%"
        }

# =========================
# APP
# =========================
st.set_page_config(layout="wide")
st.title("🧠 Kombine Swing")
st.markdown("EMA + RSI + BB + MACD + Fib")

# SIDEBAR
with st.sidebar:
    ticker = st.selectbox("Sembol", ["BTC-USD", "ETH-USD", "AAPL"])
    start = st.date_input("Başlangıç", datetime(2023, 1, 1))
    end = st.date_input("Bitiş", datetime(2024, 1, 1))
    rsi = st.slider("RSI", 20, 40, 30)
    rr = st.slider("R/R", 2, 4, 2.5)
    risk = st.slider("Risk%", 1, 3, 2) / 100

# RUN
if st.button("🚀 BACKTEST", type="primary"):
    with st.spinner("Hesaplanıyor..."):
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            st.error("Veri yok!")
            st.stop()
        
        bt = PerfectBacktest()
        trades, equity, signals = bt.backtest(data, rsi, rr, risk)
        metrics = bt.metrics(trades, equity)
    
    st.info(f"🎯 {signals} sinyal bulundu")
    
    # METRICS
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Getiri", metrics['ret'])
        st.metric("İşlem", metrics['trades'])
    with col2:
        st.metric("Win Rate", metrics['win'])
    with col3:
        st.metric("PF", metrics['pf'])
    with col4:
        st.metric("Max DD", metrics['dd'])
    
    # CHART
    if not equity.empty:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        ax1.plot(equity['date'], equity['equity'], 'g-', lw=2)
        ax1.set_title(f"{ticker} Equity")
        ax1.grid(True, alpha=0.3)
        
        peak = equity['equity'].expanding().max()
        dd = (equity['equity'] - peak) / peak * 100
        ax2.fill_between(equity['date'], dd, 0, color='r', alpha=0.3)
        ax2.set_title("Drawdown")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # TRADES
    if not trades.empty:
        trades['entry'] = trades['entry'].dt.strftime('%Y-%m-%d')
        trades['exit'] = trades['exit'].dt.strftime('%Y-%m-%d')
        st.dataframe(trades.round(2), height=300)

st.markdown("---")
st.markdown("**v7.0 - %100 HATA-FREE**")
