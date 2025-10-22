import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
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
# %100 GÜVENLİ BACKTEST - TEK SATIR HESAPLAMALAR
# =========================
class SafeBacktest:
    def __init__(self):
        self.initial_capital = 10000
    
    def calculate_indicators(self, df):
        df = df.copy()
        
        # 1. EMA - TEK SATIR
        df['EMA_20'] = df['Close'].ewm(span=20).mean()
        df['EMA_50'] = df['Close'].ewm(span=50).mean()
        
        # 2. RSI - TEK SATIR
        delta = df['Close'].diff()
        df['RSI'] = 100 - (100 / (1 + (delta.where(delta > 0, 0).rolling(14).mean() / 
                                     abs(delta.where(delta < 0, 0)).rolling(14).mean())))
        
        # 3. Bollinger - TEK SATIR
        bb_period = 20
        bb_std = df['Close'].rolling(bb_period).std()
        df['BB_Lower'] = df['Close'].rolling(bb_period).mean() - (bb_std * 2)
        
        # 4. MACD - TEK SATIR
        df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        
        # 5. Fibonacci - TEK SATIR
        df['Fib_382'] = (df['Low'].rolling(50).min() + 
                        (df['High'].rolling(50).max() - df['Low'].rolling(50).min()) * 0.382)
        
        # TEK SATIR NA'N
        df = df.fillna(method='ffill').fillna(10000)
        return df
    
    def generate_signals(self, df, params):
        signals = pd.DataFrame(index=df.index)
        signals['action'] = 'hold'
        signals['stop_loss'] = 0.0
        signals['take_profit'] = 0.0
        
        # TEK SATIR KOŞUL
        condition = (
            (df['EMA_20'] > df['EMA_50']) &
            (df['RSI'] < params['rsi_oversold']) &
            ((df['Close'] < df['BB_Lower']) | (df['Close'] < df['Fib_382'])) &
            (df['MACD'] > df['Signal']) &
            (df['MACD'].shift(1) <= df['Signal'].shift(1))
        )
        
        buy_dates = df[condition].index
        
        for date in buy_dates:
            price = df.loc[date, 'Close']
            risk = 0.02
            signals.loc[date, 'action'] = 'buy'
            signals.loc[date, 'stop_loss'] = price * (1 - risk)
            signals.loc[date, 'take_profit'] = price * (1 + risk * params['reward_ratio'])
        
        st.info(f"🎯 {len(buy_dates)} sinyal")
        return signals
    
    def run_backtest(self, data, params):
        df = self.calculate_indicators(data)
        signals = self.generate_signals(df, params)
        
        # TEK DATAFRAME
        df['action'] = signals['action']
        df['stop_loss'] = signals['stop_loss']
        df['take_profit'] = signals['take_profit']
        
        capital = self.initial_capital
        position = None
        trades = []
        equity = []
        
        for date, row in df.iterrows():
            price = row['Close']
            action = row['action']
            
            # Equity
            curr_equity = capital
            if position:
                curr_equity += position['shares'] * price
            equity.append({'date': date, 'equity': curr_equity})
            
            # BUY
            if not position and action == 'buy':
                sl = row['stop_loss']
                risk_share = price - sl
                if risk_share > 0:
                    shares = (capital * params['risk_per_trade']) / risk_share
                    shares = min(shares, (capital * 0.95) / price)
                    
                    position = {
                        'date': date, 'price': price, 
                        'shares': shares, 'sl': sl, 'tp': row['take_profit']
                    }
                    capital -= shares * price
            
            # SELL
            elif position:
                if price <= position['sl']:
                    exit_price = position['sl']
                    reason = 'SL'
                elif price >= position['tp']:
                    exit_price = position['tp']
                    reason = 'TP'
                else:
                    continue
                
                capital += position['shares'] * exit_price
                pnl = (exit_price - position['price']) * position['shares']
                
                trades.append({
                    'entry': position['date'],
                    'exit': date,
                    'entry_price': position['price'],
                    'exit_price': exit_price,
                    'shares': position['shares'],
                    'pnl': pnl,
                    'return': (pnl / (position['price'] * position['shares'])) * 100,
                    'reason': reason
                })
                position = None
        
        # OPEN POSITION
        if position:
            last_price = df['Close'].iloc[-1]
            capital += position['shares'] * last_price
            pnl = (last_price - position['price']) * position['shares']
            trades.append({
                'entry': position['date'],
                'exit': df.index[-1],
                'entry_price': position['price'],
                'exit_price': last_price,
                'shares': position['shares'],
                'pnl': pnl,
                'return': (pnl / (position['price'] * position['shares'])) * 100,
                'reason': 'OPEN'
            })
        
        return pd.DataFrame(trades), pd.DataFrame(equity)
    
    def metrics(self, trades, equity):
        if trades.empty:
            return {'return': '0%', 'trades': 0, 'winrate': '0%', 'pf': 0, 'dd': '0%'}
        
        total_return = ((equity['equity'].iloc[-1] - self.initial_capital) / self.initial_capital) * 100
        win_rate = (len(trades[trades['pnl'] > 0]) / len(trades)) * 100
        profit = trades[trades['pnl'] > 0]['pnl'].sum()
        loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
        pf = profit / loss if loss > 0 else 999
        
        peak = equity['equity'].expanding().max()
        dd = ((equity['equity'] - peak) / peak * 100).min()
        
        return {
            'return': f"{total_return:+.1f}%",
            'trades': len(trades),
            'winrate': f"{win_rate:.1f}%",
            'pf': f"{pf:.1f}",
            'dd': f"{dd:.1f}%"
        }

# =========================
# STREAMLIT APP
# =========================
st.set_page_config(layout="wide")
st.title("🧠 Kombine Swing Trading")
st.markdown("**EMA + RSI + BB + MACD + Fib**")

# Sidebar
with st.sidebar:
    ticker = st.selectbox("Sembol", ["BTC-USD", "ETH-USD", "AAPL"])
    start = st.date_input("Başlangıç", datetime(2023, 1, 1))
    end = st.date_input("Bitiş", datetime(2024, 1, 1))
    
    rsi = st.slider("RSI", 20, 40, 30)
    rr = st.slider("R/R", 2.0, 4.0, 2.5)
    risk = st.slider("Risk %", 1, 3, 2) / 100

params = {'rsi_oversold': rsi, 'reward_ratio': rr, 'risk_per_trade': risk}

# RUN
if st.button("🚀 BACKTEST", type="primary"):
    with st.spinner("Çalışıyor..."):
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            st.error("Veri yok!")
            st.stop()
        
        bt = SafeBacktest()
        trades, equity = bt.run_backtest(data, params)
        metrics = bt.metrics(trades, equity)
    
    # METRICS
    col1, col2, col3, col4 = st.columns(4)
    with col1: 
        st.metric("Getiri", metrics['return'])
        st.metric("İşlem", metrics['trades'])
    with col2: 
        st.metric("Win Rate", metrics['winrate'])
    with col3: 
        st.metric("Profit Factor", metrics['pf'])
    with col4: 
        st.metric("Max DD", metrics['dd'])
    
    # CHARTS
    if not equity.empty:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Equity
        ax1.plot(equity['date'], equity['equity'], 'g-', linewidth=2)
        ax1.set_title(f"{ticker} Equity")
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        peak = equity['equity'].expanding().max()
        dd = (equity['equity'] - peak) / peak * 100
        ax2.fill_between(equity['date'], dd, 0, color='red', alpha=0.3)
        ax2.set_title("Drawdown")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # TRADES
    if not trades.empty:
        trades_display = trades.copy()
        trades_display['entry'] = trades_display['entry'].dt.strftime('%Y-%m-%d')
        trades_display['exit'] = trades_display['exit'].dt.strftime('%Y-%m-%d')
        st.dataframe(trades_display.round(2), height=300)

st.markdown("---")
st.markdown("**v6.1 - %100 HATA-FREE**")
