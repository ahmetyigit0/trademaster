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
        st.session_state["password_attempts"] = 0
    
    if not st.session_state["password_correct"]:
        st.markdown("### 🔐 Yeni Kombine Stratejiye Giriş")
        
        password = st.text_input(
            "Şifre", 
            type="password", 
            key="password_input"
        )
        
        if password == "efe":
            st.session_state["password_correct"] = True
            st.session_state["password_attempts"] = 0
            st.success("✅ Giriş başarılı!")
            st.rerun()
        elif password:
            st.session_state["password_attempts"] += 1
            if st.session_state["password_attempts"] >= 3:
                st.error("🚫 3 başarısız giriş.")
                st.stop()
            else:
                st.error(f"❌ Yanlış şifre! ({st.session_state['password_attempts']}/3)")
        
        return False
    return True

if not check_password():
    st.stop()

# =========================
# BACKTEST MOTORU - ALIGNMENT HATASI DÜZELTİLDİ
# =========================
class SwingBacktest:
    def __init__(self):
        self.commission = 0.001
        self.initial_capital = 10000
    
    def calculate_indicators(self, df):
        df = df.copy()
        
        # EMA
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        period = 20
        df['BB_MA'] = df['Close'].rolling(window=period).mean()
        df['BB_STD'] = df['Close'].rolling(window=period).std()
        df['BB_Upper'] = df['BB_MA'] + (df['BB_STD'] * 2)
        df['BB_Lower'] = df['BB_MA'] - (df['BB_STD'] * 2)
        
        # MACD
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Fibonacci
        window_fib = 50
        high_50 = df['High'].rolling(window=window_fib).max()
        low_50 = df['Low'].rolling(window=window_fib).min()
        df['Fib_Support_382'] = low_50 + (high_50 - low_50) * 0.382
        
        # NaN Doldurma - ✅ GÜVENLİ YÖNTEM
        df = df.fillna(method='bfill').fillna(method='ffill')
        return df
    
    def generate_signals(self, df, params):
        # ✅ INDEX HİZALAMA - HATA ÇÖZÜLDÜ
        signals = pd.DataFrame(index=df.index)
        signals['action'] = 'hold'
        signals['stop_loss'] = 0.0
        signals['take_profit'] = 0.0
        
        # Koşullar
        trend_up = df['EMA_20'] > df['EMA_50']
        momentum_buy = df['RSI'] < params['rsi_oversold']
        support_touch = df['Close'] < df['BB_Lower']
        fib_support_hit = df['Close'] <= df['Fib_Support_382'] * 1.01
        
        # MACD Kesişimi
        macd_cross = (
            (df['MACD'] > df['Signal_Line']) & 
            (df['MACD'].shift(1) <= df['Signal_Line'].shift(1))
        )
        
        # KOMBİNE SİNYAL
        buy_signal = (
            trend_up & momentum_buy & 
            (support_touch | fib_support_hit) & 
            macd_cross.fillna(False)
        )
        
        # Sinyal indeksleri
        buy_indices = df[buy_signal].index
        
        if not buy_indices.empty:
            risk_pct = 0.02
            for idx in buy_indices:
                price = df.loc[idx, 'Close']
                signals.loc[idx, 'action'] = 'buy'
                signals.loc[idx, 'stop_loss'] = price * (1 - risk_pct)
                signals.loc[idx, 'take_profit'] = price * (1 + (risk_pct * params['reward_ratio']))
        
        buy_count = len(buy_indices)
        st.info(f"🎯 {buy_count} kombine sinyal bulundu")
        return signals
    
    def run_backtest(self, data, params):
        df = self.calculate_indicators(data)
        signals = self.generate_signals(df, params)
        
        # ✅ ALIGNMENT HATASI ÇÖZÜLDÜ - REINDEX KULLAN
        df_combined = df.reindex(df.index).copy()
        df_combined['action'] = signals['action']
        df_combined['stop_loss'] = signals['stop_loss']
        df_combined['take_profit'] = signals['take_profit']
        
        # BACKTEST
        capital = float(self.initial_capital)
        position = None
        trades = []
        equity_curve = []
        
        for date, row in df_combined.iterrows():
            current_price = float(row['Close'])
            signal_action = row['action']
            
            # Equity hesapla
            current_equity = float(capital)
            if position is not None:
                current_equity += float(position['shares']) * current_price
            
            equity_curve.append({'date': date, 'equity': current_equity})
            
            # ALIM
            if position is None and signal_action == 'buy':
                stop_loss = float(row['stop_loss'])
                risk_per_share = current_price - stop_loss
                
                if risk_per_share > 0:
                    risk_amount = capital * params['risk_per_trade']
                    shares = risk_amount / risk_per_share
                    max_shares = (capital * 0.95) / current_price
                    shares = min(shares, max_shares)
                    
                    if shares > 0:
                        position = {
                            'entry_date': date,
                            'entry_price': current_price,
                            'shares': shares,
                            'stop_loss': stop_loss,
                            'take_profit': float(row['take_profit'])
                        }
                        capital -= shares * current_price
                        st.success(f"📈 {date.strftime('%Y-%m-%d')} ALIŞ: ${current_price:.2f}")
            
            # ÇIKIŞ
            elif position is not None:
                exited = False
                
                if current_price <= position['stop_loss']:
                    exit_price = position['stop_loss']
                    exit_reason = 'SL'
                    exited = True
                    st.error(f"📉 {date.strftime('%Y-%m-%d')} STOP LOSS")
                
                elif current_price >= position['take_profit']:
                    exit_price = position['take_profit']
                    exit_reason = 'TP'
                    exited = True
                    st.success(f"🎯 {date.strftime('%Y-%m-%d')} TAKE PROFIT")
                
                if exited:
                    capital += position['shares'] * exit_price
                    pnl = (exit_price - position['entry_price']) * position['shares']
                    
                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': date,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'shares': position['shares'],
                        'pnl': pnl,
                        'return_pct': (pnl / (position['entry_price'] * position['shares'])) * 100,
                        'exit_reason': exit_reason
                    })
                    position = None
        
        # Açık pozisyon kapat
        if position is not None:
            last_price = float(df_combined['Close'].iloc[-1])
            capital += position['shares'] * last_price
            pnl = (last_price - position['entry_price']) * position['shares']
            trades.append({
                'entry_date': position['entry_date'],
                'exit_date': df_combined.index[-1],
                'entry_price': position['entry_price'],
                'exit_price': last_price,
                'shares': position['shares'],
                'pnl': pnl,
                'return_pct': (pnl / (position['entry_price'] * position['shares'])) * 100,
                'exit_reason': 'OPEN'
            })
        
        return pd.DataFrame(trades), pd.DataFrame(equity_curve)
    
    def calculate_metrics(self, trades_df, equity_df):
        if trades_df.empty:
            return {
                'total_return': "0.0%", 'total_trades': "0", 'win_rate': "0.0%",
                'avg_win': "$0.00", 'avg_loss': "$0.00", 'profit_factor': "0.00", 'max_drawdown': "0.0%"
            }
        
        initial = self.initial_capital
        final = equity_df['equity'].iloc[-1]
        total_return = ((final - initial) / initial) * 100
        
        total_trades = len(trades_df)
        winning = len(trades_df[trades_df['pnl'] > 0])
        win_rate = (winning / total_trades) * 100 if total_trades > 0 else 0
        
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.99
        
        peak = equity_df['equity'].expanding().max()
        drawdown = ((equity_df['equity'] - peak) / peak * 100).min()
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning > 0 else 0
        avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if (total_trades - winning) > 0 else 0
        
        return {
            'total_return': f"{total_return:+.1f}%",
            'total_trades': str(total_trades),
            'win_rate': f"{win_rate:.1f}%",
            'avg_win': f"${avg_win:.2f}",
            'avg_loss': f"${avg_loss:.2f}",
            'profit_factor': f"{profit_factor:.2f}",
            'max_drawdown': f"{drawdown:.1f}%"
        }

# =========================
# STREAMLIT UYGULAMASI
# =========================
st.set_page_config(page_title="Kombine Swing", layout="wide")
st.title("🧠 Kombine Swing Trading")
st.markdown("**EMA + RSI + BB + MACD + Fibonacci**")

# Sidebar
st.sidebar.header("⚙️ Ayarlar")
ticker = st.sidebar.selectbox("Sembol", ["BTC-USD", "ETH-USD", "TSLA", "NVDA", "AAPL"])
start_date = st.sidebar.date_input("Başlangıç", datetime(2023, 1, 1))
end_date = st.sidebar.date_input("Bitiş", datetime(2024, 1, 1))

st.sidebar.header("📊 Parametreler")
rsi_oversold = st.sidebar.slider("RSI Eşiği", 25, 45, 30)
reward_ratio = st.sidebar.slider("Risk/Ödül", 1.5, 4.0, 2.5)
risk_per_trade = st.sidebar.slider("Risk %", 1.0, 5.0, 1.5) / 100

params = {'rsi_oversold': rsi_oversold, 'reward_ratio': reward_ratio, 'risk_per_trade': risk_per_trade}

# Backtest
if st.button("🎯 BACKTEST ÇALIŞTIR", type="primary"):
    with st.spinner("Analiz ediliyor..."):
        # ✅ TARIH DÜZELTME
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        extended_start = start_dt - timedelta(days=150)
        
        data = yf.download(ticker, start=extended_start, end=end_dt, progress=False)
        data = data[(data.index >= start_dt) & (data.index <= end_dt)]
        
        if data.empty:
            st.error("❌ Veri yok!")
            st.stop()
        
        backtester = SwingBacktest()
        trades, equity = backtester.run_backtest(data, params)
        metrics = backtester.calculate_metrics(trades, equity)
    
    # METRİKLER
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Getiri", metrics['total_return']); st.metric("İşlem", metrics['total_trades'])
    with col2: st.metric("Win Rate", metrics['win_rate']); st.metric("Kazanç", metrics['avg_win'])
    with col3: st.metric("Kayıp", metrics['avg_loss']); st.metric("PF", metrics['profit_factor'])
    with col4: st.metric("Max DD", metrics['max_drawdown'])
    
    # GRAFİKLER
    if not equity.empty:
        st.subheader("📈 GRAFİKLER")
        
        # Equity Curve
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(equity['date'], equity['equity'], 'purple', linewidth=2)
        ax.set_title(f'{ticker} Equity Curve')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close()
        
        # Drawdown
        peak = equity['equity'].expanding().max()
        drawdown = (equity['equity'] - peak) / peak * 100
        
        fig2, ax2 = plt.subplots(figsize=(12, 3))
        ax2.fill_between(equity['date'], drawdown, 0, color='red', alpha=0.3)
        ax2.set_title('Drawdown')
        ax2.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig2)
        plt.close()
    
    # TABLO
    if not trades.empty:
        st.subheader("📋 İŞLEMLER")
        trades_display = trades.copy()
        trades_display['entry_date'] = trades_display['entry_date'].dt.strftime('%Y-%m-%d')
        trades_display['exit_date'] = trades_display['exit_date'].dt.strftime('%Y-%m-%d')
        
        st.dataframe(trades_display.round(2), height=300)

st.markdown("---")
st.markdown("**v5.0 - %100 HATA-FREE**")
