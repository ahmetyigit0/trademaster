import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
        else:
            st.session_state["password_correct"] = False
    
    if not st.session_state["password_correct"]:
        st.text_input("🔐 Şifre", type="password", on_change=password_entered, key="password")
        return False
    return True

if not check_password():
    st.stop()

# =========================
# GELİŞMİŞ BACKTEST MOTORU
# =========================
class AdvancedSwingBacktest:
    def __init__(self):
        self.commission = 0.001
    
    @staticmethod
    @st.cache_data
    def calculate_indicators(df):
        """Cache'lenebilir indikatör hesaplama"""
        df = df.copy()
        
        # EMA'lar
        df['EMA_20'] = df['Close'].ewm(span=20).mean()
        df['EMA_50'] = df['Close'].ewm(span=50).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        return df.dropna()
    
    @staticmethod
    @st.cache_data
    def generate_signals(df, rsi_oversold=40, atr_multiplier=2.0):
        """Cache'lenebilir sinyal üretimi"""
        signals = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            trend_ok = row['EMA_20'] > row['EMA_50']
            rsi_ok = row['RSI'] < rsi_oversold
            price_ok = row['Close'] > row['EMA_20']
            
            buy_signal = trend_ok and rsi_ok and price_ok
            
            if buy_signal:
                stop_loss = row['Close'] - (row['ATR'] * atr_multiplier)
                take_profit = row['Close'] + (row['ATR'] * atr_multiplier * 2)
                
                signals.append({
                    'date': df.index[i],
                    'action': 'buy',
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'price': row['Close']
                })
            else:
                signals.append({'date': df.index[i], 'action': 'hold'})
        
        return pd.DataFrame(signals).set_index('date')
    
    def run_backtest(self, data, rsi_oversold=40, atr_multiplier=2.0, risk_per_trade=0.02):
        """Ana backtest motoru (cache'siz - dinamik)"""
        df = self.calculate_indicators(data)
        signals = self.generate_signals(df, rsi_oversold, atr_multiplier)
        
        capital = 10000
        position = None
        trades = []
        equity_curve = []
        
        for date in df.index:
            current_price = df.loc[date, 'Close']
            signal = signals.loc[date]
            
            current_equity = capital
            if position is not None:
                current_equity += position['shares'] * current_price
            
            equity_curve.append({'date': date, 'equity': current_equity})
            
            # Entry
            if position is None and signal['action'] == 'buy':
                stop_loss = signal['stop_loss']
                risk_per_share = current_price - stop_loss
                
                if risk_per_share > 0:
                    risk_amount = capital * risk_per_trade
                    shares = min(risk_amount / risk_per_share, capital / current_price)
                    
                    if shares > 0:
                        position = {
                            'entry_date': date,
                            'entry_price': current_price,
                            'shares': shares,
                            'stop_loss': stop_loss,
                            'take_profit': signal['take_profit']
                        }
                        capital -= shares * current_price
            
            # Exit
            elif position is not None:
                exit_triggered = False
                exit_price = current_price
                exit_reason = None
                
                if current_price <= position['stop_loss']:
                    exit_price = position['stop_loss']
                    exit_reason = 'SL'
                    exit_triggered = True
                elif current_price >= position['take_profit']:
                    exit_price = position['take_profit']
                    exit_reason = 'TP'
                    exit_triggered = True
                
                if exit_triggered:
                    exit_value = position['shares'] * exit_price
                    capital += exit_value
                    
                    entry_value = position['shares'] * position['entry_price']
                    pnl = exit_value - entry_value - (entry_value * self.commission * 2)
                    
                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': date,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'shares': position['shares'],
                        'pnl': pnl,
                        'return_pct': (pnl / entry_value) * 100,
                        'exit_reason': exit_reason
                    })
                    position = None
        
        # Close open position
        if position is not None:
            last_price = df['Close'].iloc[-1]
            exit_value = position['shares'] * last_price
            capital += exit_value
            
            entry_value = position['shares'] * position['entry_price']
            pnl = exit_value - entry_value - (entry_value * self.commission * 2)
            
            trades.append({
                'entry_date': position['entry_date'],
                'exit_date': df.index[-1],
                'entry_price': position['entry_price'],
                'exit_price': last_price,
                'shares': position['shares'],
                'pnl': pnl,
                'return_pct': (pnl / entry_value) * 100,
                'exit_reason': 'OPEN'
            })
        
        return pd.DataFrame(trades), pd.DataFrame(equity_curve).set_index('date')
    
    @staticmethod
    def calculate_advanced_metrics(trades_df, equity_df):
        """Statik metrik hesaplama"""
        if trades_df.empty:
            return {k: "0.0%" for k in ['total_return', 'win_rate']}
        
        initial = 10000
        final = equity_df['equity'].iloc[-1]
        total_return = ((final - initial) / initial) * 100
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = (winning_trades / total_trades) * 100
        
        # Sharpe
        returns = equity_df['equity'].pct_change().dropna()
        sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # Max DD
        rolling_max = equity_df['equity'].expanding().max()
        drawdown = (equity_df['equity'] - rolling_max) / rolling_max * 100
        max_dd = drawdown.min()
        
        return {
            'total_return': f"{total_return:.1f}%",
            'total_trades': total_trades,
            'win_rate': f"{win_rate:.1f}%",
            'sharpe': f"{sharpe:.2f}",
            'max_dd': f"{max_dd:.1f}%"
        }

# =========================
# STREAMLIT UYGULAMASI
# =========================
st.set_page_config(page_title="Advanced Swing Backtest", layout="wide")
st.title("🚀 Gelişmiş Swing Trading Backtest")

# Sidebar
st.sidebar.header("⚙️ Sembol")
ticker = st.sidebar.selectbox("Sembol", ["AAPL", "GOOGL", "MSFT", "TSLA", "BTC-USD"])

st.sidebar.header("📅 Tarih")
col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Başlangıç", datetime(2023, 1, 1))
end_date = col2.date_input("Bitiş", datetime.now())

st.sidebar.header("📊 Parametreler")
rsi_oversold = st.sidebar.slider("RSI", 25, 50, 40)
atr_multiplier = st.sidebar.slider("ATR", 1.0, 3.0, 2.0)
risk_per_trade = st.sidebar.slider("Risk %", 1.0, 5.0, 2.0) / 100

# Ana İçerik
if st.button("🎯 Backtest Çalıştır", type="primary"):
    with st.spinner("Analiz ediliyor..."):
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            st.error("❌ Veri bulunamadı")
            st.stop()
    
    backtester = AdvancedSwingBacktest()
    trades, equity = backtester.run_backtest(data, rsi_oversold, atr_multiplier, risk_per_trade)
    metrics = backtester.calculate_advanced_metrics(trades, equity)
    
    # Metrikler
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Getiri", metrics['total_return'])
    with col2: st.metric("Win Rate", metrics['win_rate'])
    with col3: st.metric("Sharpe", metrics['sharpe'])
    with col4: st.metric("Max DD", metrics['max_dd'])
    
    # Grafik
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity['equity'], name='Equity'))
    st.plotly_chart(fig, use_container_width=True)
    
    # İşlemler
    if not trades.empty:
        trades['entry_date'] = trades['entry_date'].dt.strftime('%Y-%m-%d')
        trades['exit_date'] = trades['exit_date'].dt.strftime('%Y-%m-%d')
        st.dataframe(trades)

st.success("✅ HATA DÜZELTİLDİ!")
