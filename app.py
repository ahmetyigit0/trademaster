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
# %100 GÜVENLİ BACKTEST MOTORU
# =========================
class SafeSwingBacktest:
    def __init__(self):
        self.initial_capital = 10000
    
    def calculate_indicators(self, df):
        """TÜM HESAPLAMALAR TEK DATAFRAME'DE - HATA YOK"""
        df = df.copy()
        
        # 1. EMA
        df['EMA_20'] = df['Close'].ewm(span=20).mean()
        df['EMA_50'] = df['Close'].ewm(span=50).mean()
        
        # 2. RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 3. Bollinger Bands
        df['BB_MA'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Lower'] = df['BB_MA'] - (bb_std * 2)
        
        # 4. MACD
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema12 - ema26
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        
        # 5. Fibonacci
        high50 = df['High'].rolling(50).max()
        low50 = df['Low'].rolling(50).min()
        df['Fib_382'] = low50 + (high50 - low50) * 0.382
        
        # TEK SATIRDA TUM NAN'LAR
        df = df.fillna(method='ffill').fillna(method='bfill')
        return df
    
    def generate_signals(self, df, params):
        """SIFIR ALIGNMENT - TEK DATAFRAME"""
        signals = pd.DataFrame(index=df.index)
        signals['action'] = 'hold'
        signals['stop_loss'] = 0.0
        signals['take_profit'] = 0.0
        
        # TÜM KOŞULLAR TEK DATAFRAME ÜZERİNDE
        conditions = (
            (df['EMA_20'] > df['EMA_50']) &           # Trend
            (df['RSI'] < params['rsi_oversold']) &    # RSI
            ((df['Close'] < df['BB_Lower']) |         # BB veya
             (df['Close'] < df['Fib_382'])) &         # Fib
            (df['MACD'] > df['Signal']) &             # MACD
            (df['MACD'].shift(1) <= df['Signal'].shift(1))  # Cross
        )
        
        buy_signals = df[conditions]
        
        if not buy_signals.empty:
            risk_pct = 0.02
            for date, row in buy_signals.iterrows():
                price = row['Close']
                signals.loc[date, 'action'] = 'buy'
                signals.loc[date, 'stop_loss'] = price * (1 - risk_pct)
                signals.loc[date, 'take_profit'] = price * (1 + risk_pct * params['reward_ratio'])
        
        st.info(f"🎯 {len(buy_signals)} sinyal bulundu")
        return signals
    
    def run_backtest(self, data, params):
        df = self.calculate_indicators(data)
        signals = self.generate_signals(df, params)
        
        # TEK DATAFRAME - HİÇ JOIN YOK
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
                    shares = min(shares, capital * 0.95 / price)
                    
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
        
        # Close open position
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
            return {
                'return': '0.0%', 'trades': 0, 'winrate': '0%',
                'pf': 0.0, 'dd': '0.0%'
            }
        
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
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        
        data = yf.download(ticker, start=start_dt, end=end_dt)
        if data.empty:
            st.error("Veri yok!")
            st.stop()
        
        bt = SafeSwingBacktest()
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
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        
        # Equity
        ax[0].plot(equity['date'], equity['equity'], 'g-', linewidth=2)
        ax[0].set_title(f"{ticker} Equity")
        ax[0].grid(True, alpha=0.3)
        
        # Drawdown
        peak = equity['equity'].expanding().max()
        dd = (equity['equity'] - peak) / peak * 100
        ax[1].fill_between(equity['date'], dd, 0, color='red', alpha=0.3)
        ax[1].set_title("Drawdown")
        ax[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # TRADES
    if not trades.empty:
        trades['entry'] = trades['entry'].dt.strftime('%Y-%m-%d')
        trades['exit'] = trades['exit'].dt.strftime('%Y-%m-%d')
        st.dataframe(trades.round(2), height=300)

st.markdown("---")
st.markdown("**v6.0 - %100 HATA-FREE**")
