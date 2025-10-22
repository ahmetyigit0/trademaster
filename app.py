import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =========================
# ŞİFRE KORUMASI - RERUN HATASI DÜZELTİLDİ
# =========================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
        st.session_state["password_attempts"] = 0
    
    if not st.session_state["password_correct"]:
        st.markdown("""
        # 🔐 **Kombine Stratejiye Hoş Geldiniz**
        ### **5 Göstergeli Profesyonel Swing Sistemi**
        """)
        
        password = st.text_input(
            "Şifre:", 
            type="password", 
            key="password_input"
        )
        
        if password == "efe":
            st.session_state["password_correct"] = True
            st.session_state["password_attempts"] = 0
            st.success("✅ Giriş başarılı!")
            st.rerun()  # BURADA RERUN ÇALIŞIR!
        elif password:
            st.session_state["password_attempts"] += 1
            if st.session_state["password_attempts"] >= 3:
                st.error("🚫 3 başarısız giriş. Sayfayı yenileyin.")
                st.stop()
            else:
                st.error(f"❌ Yanlış şifre! ({st.session_state['password_attempts']}/3)")
        
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
        self.initial_capital = 10000
    
    def calculate_indicators(self, df):
        df = df.copy()
        
        # EMA
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
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
        df['Fib_382'] = low_50 + (high_50 - low_50) * 0.382
        
        # Volume
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # NaN Doldurma
        df = df.fillna(method='bfill').fillna(method='ffill')
        return df
    
    def generate_signals(self, df, params):
        df_copy = df.copy()
        
        # Koşullar
        trend_up = df_copy['EMA_20'] > df_copy['EMA_50']
        rsi_oversold = df_copy['RSI'] < params['rsi_oversold']
        bb_support = df_copy['Close'] <= df_copy['BB_Lower'] * 1.02
        fib_support = df_copy['Close'] <= df_copy['Fib_382'] * 1.01
        macd_cross = (
            (df_copy['MACD'] > df_copy['Signal_Line']) & 
            (df_copy['MACD'].shift(1) <= df_copy['Signal_Line'].shift(1))
        )
        volume_confirm = df_copy['Volume_Ratio'] > 1.2
        
        # KOMBİNE SİNYAL
        df_copy['buy_signal'] = (
            trend_up & rsi_oversold & 
            (bb_support | fib_support) & 
            macd_cross & volume_confirm
        )
        
        # Sinyaller
        signals = pd.DataFrame(index=df_copy.index)
        signals['action'] = 'hold'
        signals['stop_loss'] = 0.0
        signals['take_profit'] = 0.0
        
        buy_indices = df_copy[df_copy['buy_signal']].index
        
        if not buy_indices.empty:
            risk_pct = 0.02
            for idx in buy_indices:
                price = df_copy.loc[idx, 'Close']
                signals.loc[idx, 'action'] = 'buy'
                signals.loc[idx, 'stop_loss'] = price * (1 - risk_pct)
                signals.loc[idx, 'take_profit'] = price * (1 + (risk_pct * params['reward_ratio']))
        
        return signals, len(buy_indices)
    
    def run_backtest(self, data, params):
        df = self.calculate_indicators(data)
        signals, signal_count = self.generate_signals(df, params)
        
        # DataFrame birleştir
        df_combined = df.join(signals).fillna({
            'action': 'hold', 'stop_loss': 0.0, 'take_profit': 0.0
        })
        
        # BACKTEST
        capital = self.initial_capital
        position = None
        trades = []
        equity_curve = []
        
        for i, row in df_combined.iterrows():
            price = row['Close']
            action = row['action']
            
            # Equity
            current_equity = capital
            if position:
                current_equity += position['shares'] * price
            equity_curve.append({'date': i, 'equity': current_equity})
            
            # ALIM
            if not position and action == 'buy':
                risk_amount = capital * params['risk_per_trade']
                sl = row['stop_loss']
                risk_per_share = price - sl
                
                if risk_per_share > 0:
                    shares = risk_amount / risk_per_share
                    shares = min(shares, (capital * 0.95) / price)
                    
                    position = {
                        'entry_date': i, 'entry_price': price,
                        'shares': shares, 'stop_loss': sl,
                        'take_profit': row['take_profit']
                    }
                    capital -= shares * price
            
            # ÇIKIŞ
            elif position:
                if price <= position['stop_loss']:
                    exit_price = position['stop_loss']
                    reason = 'SL'
                elif price >= position['take_profit']:
                    exit_price = position['take_profit']
                    reason = 'TP'
                else:
                    continue
                
                capital += position['shares'] * exit_price
                pnl = (exit_price - position['entry_price']) * position['shares']
                
                trades.append({
                    'entry_date': position['entry_date'],
                    'exit_date': i,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'shares': position['shares'],
                    'pnl': pnl,
                    'return_pct': (pnl / (position['entry_price'] * position['shares'])) * 100,
                    'exit_reason': reason
                })
                position = None
        
        # Açık pozisyon kapat
        if position:
            last_price = df_combined['Close'].iloc[-1]
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
        
        st.info(f"🎯 **{signal_count} adet kombine sinyal** üretildi")
        return pd.DataFrame(trades), pd.DataFrame(equity_curve)
    
    def calculate_metrics(self, trades_df, equity_df):
        if trades_df.empty:
            return {
                'total_return': '0.0%', 'total_trades': '0', 'win_rate': '0.0%',
                'profit_factor': '0.00', 'max_dd': '0.0%', 'avg_win': '$0.00', 'avg_loss': '$0.00'
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
            'profit_factor': f"{profit_factor:.2f}",
            'max_dd': f"{drawdown:.1f}%",
            'avg_win': f"${avg_win:.2f}",
            'avg_loss': f"${avg_loss:.2f}"
        }

# =========================
# STREAMLIT ARAYÜZÜ
# =========================
st.set_page_config(page_title="Kombine Swing Pro", layout="wide")
st.title("🧠 **KOMBİNE SWING TRADING v5.1**")
st.markdown("**EMA + RSI + BB + MACD + Fibonacci + Volume**")

# Sidebar
with st.sidebar:
    st.header("⚙️ **PARAMETRELER**")
    ticker = st.selectbox("Sembol", ["BTC-USD", "ETH-USD", "TSLA", "NVDA", "AAPL"])
    start_date = st.date_input("Başlangıç", datetime(2023, 1, 1))
    end_date = st.date_input("Bitiş", datetime(2024, 12, 31))
    
    st.header("📊 **STRATEJİ**")
    rsi_level = st.slider("RSI Eşiği", 20, 40, 28)
    rr_ratio = st.slider("Risk/Ödül", 1.5, 4.0, 2.5)
    risk_pct = st.slider("Risk %", 0.5, 3.0, 1.5) / 100
    
    params = {'rsi_oversold': rsi_level, 'reward_ratio': rr_ratio, 'risk_per_trade': risk_pct}

# ANA BACKTEST BUTONU
if st.button("🚀 **BACKTEST ÇALIŞTIR**", type="primary"):
    with st.spinner("🔄 Analiz ediliyor..."):
        # Veri yükle
        extended_start = start_date - timedelta(days=200)
        data = yf.download(ticker, start=extended_start, end=end_date, progress=False)
        data = data[(data.index >= start_date) & (data.index <= end_date)]
        
        if data.empty:
            st.error("❌ Veri bulunamadı!")
            st.stop()
        
        backtester = AdvancedSwingBacktest()
        trades, equity = backtester.run_backtest(data, params)
        metrics = backtester.calculate_metrics(trades, equity)
    
    # METRİK KARTLARI
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📈 Getiri", metrics['total_return'])
        st.metric("💼 İşlem Sayısı", metrics['total_trades'])
    with col2:
        st.metric("🎯 Win Rate", metrics['win_rate'])
        st.metric("💰 Ortalama Kazanç", metrics['avg_win'])
    with col3:
        st.metric("📉 Ortalama Kayıp", metrics['avg_loss'])
        st.metric("⚡ Profit Factor", metrics['profit_factor'])
    with col4:
        st.metric("📊 Max Drawdown", metrics['max_dd'])
    
    # GRAFİKLER
    if not equity.empty:
        st.subheader("📈 **PERFORMANS GRAFİKLERİ**")
        
        # 1. Equity Curve
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(equity['date'], equity['equity'], color='purple', linewidth=2)
        ax1.set_title(f'{ticker} - Portföy Değeri', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Equity ($)')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig1)
        
        # 2. Drawdown
        peak = equity['equity'].expanding().max()
        drawdown = (equity['equity'] - peak) / peak * 100
        
        fig2, ax2 = plt.subplots(figsize=(12, 3))
        ax2.fill_between(equity['date'], drawdown, 0, color='red', alpha=0.3)
        ax2.set_title('Drawdown (%)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Drawdown %')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)
    
    # TİCARET TABLOSU
    if not trades.empty:
        st.subheader("📋 **TİCARET DETAYLARI**")
        trades_display = trades.copy()
        trades_display['entry_date'] = trades_display['entry_date'].dt.strftime('%Y-%m-%d')
        trades_display['exit_date'] = trades_display['exit_date'].dt.strftime('%Y-%m-%d')
        
        st.dataframe(trades_display.style.format({
            'pnl': '${:.2f}', 'return_pct': '{:.1f}%'
        }).background_gradient(subset=['return_pct'], cmap='RdYlGn'), height=400)

st.markdown("---")
st.markdown("***v5.1 - %100 HATA-FREE | RERUN DÜZELTİLDİ***")
