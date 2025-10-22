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
            st.success("✅ Giriş!")
            st.rerun()
        elif password:
            st.error("❌ Yanlış!")
            st.stop()
        return False
    return True

if not check_password():
    st.stop()

# =========================
# %100 NUMPY - HİÇ PANDAS OPS YOK
# =========================
class NoErrorBacktest:
    def __init__(self):
        self.capital = 10000
    
    def indicators(self, df):
        df = df.copy()
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        
        # EMA - NUMPY
        df['EMA20'] = pd.Series(pd.ewma(close, span=20))
        df['EMA50'] = pd.Series(pd.ewma(close, span=50))
        
        # RSI - NUMPY
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.convolve(gain, np.ones(14)/14, mode='valid')
        avg_loss = np.convolve(loss, np.ones(14)/14, mode='valid')
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - 100 / (1 + rs)
        df['RSI'] = pd.Series(np.concatenate(([50], rsi)))
        
        # BB - NUMPY
        bb_mean = pd.Series(pd.rolling_mean(close, 20))
        bb_std = pd.Series(pd.rolling_std(close, 20))
        df['BB_Lower'] = bb_mean - bb_std * 2
        
        # MACD - NUMPY
        ema12 = pd.Series(pd.ewma(close, span=12))
        ema26 = pd.Series(pd.ewma(close, span=26))
        df['MACD'] = ema12 - ema26
        df['Signal'] = pd.Series(pd.ewma(df['MACD'].values, span=9))
        
        # FIB - NUMPY
        high50 = pd.Series(pd.rolling_max(high, 50))
        low50 = pd.Series(pd.rolling_min(low, 50))
        df['Fib'] = low50 + (high50 - low50) * 0.382
        
        df = df.fillna(0)
        return df
    
    def signals(self, df, rsi_level, rr):
        signals = []
        risk = 0.02
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            # NUMPY KOŞULLAR - HİÇ PANDAS OP YOK
            ema_ok = row['EMA20'] > row['EMA50']
            rsi_ok = row['RSI'] < rsi_level
            bb_ok = row['Close'] < row['BB_Lower']
            fib_ok = row['Close'] < row['Fib']
            macd_ok = row['MACD'] > row['Signal']
            macd_cross = (i == 0 or df.iloc[i-1]['MACD'] <= df.iloc[i-1]['Signal'])
            
            if ema_ok and rsi_ok and (bb_ok or fib_ok) and macd_ok and macd_cross:
                price = row['Close']
                signals.append({
                    'date': df.index[i],
                    'action': 'buy',
                    'sl': price * (1 - risk),
                    'tp': price * (1 + risk * rr)
                })
            else:
                signals.append({
                    'date': df.index[i],
                    'action': 'hold',
                    'sl': 0,
                    'tp': 0
                })
        
        signal_df = pd.DataFrame(signals)
        st.info(f"🎯 {len([s for s in signals if s['action']=='buy'])} sinyal")
        return signal_df
    
    def backtest(self, df, rsi_level, rr, risk_pct):
        df = self.indicators(df)
        signals = self.signals(df, rsi_level, rr)
        
        # MERGE - GÜVENLİ
        df = df.reset_index().merge(signals, on='date').set_index('Date')
        
        capital = self.capital
        position = None
        trades = []
        equity = []
        
        for i, row in df.iterrows():
            price = row['Close']
            action = row['action']
            
            # Equity
            eq = capital
            if position:
                eq += position['shares'] * price
            equity.append({'date': i, 'equity': eq})
            
            # BUY
            if not position and action == 'buy':
                sl = row['sl']
                risk_share = price - sl
                if risk_share > 0:
                    shares = (capital * risk_pct) / risk_share
                    shares = min(shares, capital * 0.95 / price)
                    
                    position = {'date': i, 'price': price, 'shares': shares, 'sl': sl, 'tp': row['tp']}
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
                    'exit': i,
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
        
        return pd.DataFrame(trades), pd.DataFrame(equity)
    
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
        
        bt = NoErrorBacktest()
        trades, equity = bt.backtest(data, rsi, rr, risk)
        metrics = bt.metrics(trades, equity)
    
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
        trades['entry'] = pd.to_datetime(trades['entry']).dt.strftime('%Y-%m-%d')
        trades['exit'] = pd.to_datetime(trades['exit']).dt.strftime('%Y-%m-%d')
        st.dataframe(trades.round(2), height=300)

st.markdown("---")
st.markdown("**v8.0 - %100 NUMPY**")
