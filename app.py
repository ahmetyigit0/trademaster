import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =========================
# ŞİFRE KORUMASI (İYİLEŞTİRİLDİ)
# =========================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
        st.session_state["password_attempts"] = 0
    
    def password_entered():
        if st.session_state["password"] == "efe":
            st.session_state["password_correct"] = True
            st.session_state["password_attempts"] = 0
            del st.session_state["password"]
            st.rerun()
        else:
            st.session_state["password_attempts"] += 1
            if st.session_state["password_attempts"] >= 3:
                st.error("🚫 3 başarısız giriş. 5 dakika bekleyin.")
                st.stop()
    
    if not st.session_state["password_correct"]:
        st.markdown("""
        # 🔐 **Kombine Stratejiye Hoş Geldiniz**
        ### **5 Göstergeli Profesyonel Swing Sistemi**
        """)
        st.text_input(
            "Şifre:", 
            type="password", 
            on_change=password_entered, 
            key="password",
            placeholder="Şifreyi giriniz..."
        )
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
        self.max_positions = 1  # Tek pozisyon
    
    def calculate_indicators(self, df):
        """TÜM GÖSTERGELER - HATA YOK**
        """
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
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_MA']
        
        # MACD
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
        
        # Fibonacci Retracement (50 dönem)
        window_fib = 50
        high_50 = df['High'].rolling(window=window_fib).max()
        low_50 = df['Low'].rolling(window=window_fib).min()
        df['Fib_382'] = low_50 + (high_50 - low_50) * 0.382
        df['Fib_618'] = low_50 + (high_50 - low_50) * 0.618
        
        # Volume Filter
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # NaN Doldurma
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    def generate_signals(self, df, params):
        """GELİŞMİŞ SİNYAL ÜRETİMİ"""
        df_copy = df.copy()
        
        # Ana Koşullar
        conditions = {}
        
        # 1. Trend: EMA20 > EMA50
        conditions['trend_up'] = df_copy['EMA_20'] > df_copy['EMA_50']
        
        # 2. Momentum: RSI < threshold
        conditions['rsi_oversold'] = df_copy['RSI'] < params['rsi_oversold']
        
        # 3. Support: BB Lower veya Fib 38.2%
        conditions['bb_support'] = df_copy['Close'] <= df_copy['BB_Lower'] * 1.02
        conditions['fib_support'] = df_copy['Close'] <= df_copy['Fib_382'] * 1.01
        
        # 4. MACD Bullish Cross
        conditions['macd_cross'] = (
            (df_copy['MACD'] > df_copy['Signal_Line']) & 
            (df_copy['MACD'].shift(1) <= df_copy['Signal_Line'].shift(1))
        )
        
        # 5. Volume Confirmation (YENİ!)
        conditions['volume_confirm'] = df_copy['Volume_Ratio'] > 1.2
        
        # KOMBİNE ALIM SİNYALİ
        df_copy['buy_signal'] = (
            conditions['trend_up'] &
            conditions['rsi_oversold'] &
            (conditions['bb_support'] | conditions['fib_support']) &
            conditions['macd_cross'] &
            conditions['volume_confirm']
        )
        
        # Sinyal DataFrame
        signals = pd.DataFrame(index=df_copy.index)
        signals['action'] = 'hold'
        
        buy_signals = df_copy[df_copy['buy_signal']].index
        buy_count = len(buy_signals)
        
        if buy_count > 0:
            risk_pct = 0.02
            for idx in buy_signals:
                price = df_copy.loc[idx, 'Close']
                signals.loc[idx, 'action'] = 'buy'
                signals.loc[idx, 'stop_loss'] = price * (1 - risk_pct)
                signals.loc[idx, 'take_profit'] = price * (1 + (risk_pct * params['reward_ratio']))
        
        st.info(f"🎯 **{buy_count} adet kombine sinyal** üretildi")
        return signals
    
    def run_backtest(self, data, params):
        """ANA BACKTEST - HATA DÜZELTİLDİ"""
        df = self.calculate_indicators(data)
        signals = self.generate_signals(df, params)
        
        # DataFrame Birleştirme
        df_combined = df.join(signals).fillna({
            'action': 'hold',
            'stop_loss': 0.0,
            'take_profit': 0.0
        })
        
        # BACKTEST DÖNGÜSÜ
        capital = self.initial_capital
        position = None
        trades = []
        equity_curve = []
        
        for i, row in df_combined.iterrows():
            current_price = row['Close']
            action = row['action']
            
            # Equity Hesaplama
            current_equity = capital
            if position:
                current_equity += position['shares'] * current_price
            equity_curve.append({'date': i, 'equity': current_equity})
            
            # ALIM
            if not position and action == 'buy':
                risk_amount = capital * params['risk_per_trade']
                stop_loss = row['stop_loss']
                risk_per_share = current_price - stop_loss
                
                if risk_per_share > 0:
                    shares = risk_amount / risk_per_share
                    shares = min(shares, (capital * 0.95) / current_price)
                    
                    position = {
                        'entry_date': i,
                        'entry_price': current_price,
                        'shares': shares,
                        'stop_loss': stop_loss,
                        'take_profit': row['take_profit']
                    }
                    capital -= shares * current_price
                    st.success(f"📈 **ALIM**: {i.strftime('%Y-%m-%d')} | Fiyat: ${current_price:.2f}")
            
            # ÇIKIŞ
            elif position:
                if current_price <= position['stop_loss']:
                    exit_price = position['stop_loss']
                    reason = 'SL'
                elif current_price >= position['take_profit']:
                    exit_price = position['take_profit']
                    reason = 'TP'
                else:
                    continue
                
                # Pozisyon Kapat
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
                
                color = "🟢" if pnl > 0 else "🔴"
                st.write(f"{color} **ÇIKIŞ**: {i.strftime('%Y-%m-%d')} | P&L: ${pnl:.2f}")
                position = None
        
        # Açık Pozisyon Kapatma
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
        
        return pd.DataFrame(trades), pd.DataFrame(equity_curve)
    
    def calculate_metrics(self, trades_df, equity_df):
        """GELİŞMİŞ METRİKLER"""
        if trades_df.empty:
            return {k: "0" for k in ['total_return', 'win_rate', 'profit_factor', 'sharpe', 'sortino']}
        
        initial = self.initial_capital
        final = equity_df['equity'].iloc[-1]
        
        metrics = {
            'total_return': f"{((final-initial)/initial)*100:+.1f}%",
            'total_trades': len(trades_df),
            'win_rate': f"{(len(trades_df[trades_df['pnl']>0])/len(trades_df)*100):.1f}%",
            'profit_factor': f"{trades_df[trades_df['pnl']>0]['pnl'].sum()/abs(trades_df[trades_df['pnl']<0]['pnl'].sum()):.2f}",
            'avg_win': f"${trades_df[trades_df['pnl']>0]['pnl'].mean():.2f}",
            'avg_loss': f"${abs(trades_df[trades_df['pnl']<0]['pnl'].mean()):.2f}",
            'best_trade': f"{trades_df['return_pct'].max():.1f}%",
            'worst_trade': f"{trades_df['return_pct'].min():.1f}%",
            'max_dd': f"{((equity_df['equity'] - equity_df['equity'].expanding().max())/equity_df['equity'].expanding().max()).min():.1f}%"
        }
        return metrics

# =========================
# STREAMLIT ARAYÜZÜ
# =========================
st.set_page_config(page_title="Kombine Swing Pro", layout="wide")
st.title("🧠 **KOMBİNE SWING TRADING v5.0**")
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

# Ana Backtest
if st.button("🚀 **BACKTEST ÇALIŞTIR**", type="primary"):
    with st.spinner("🔄 Analiz ediliyor..."):
        # Veri Yükleme
        data = yf.download(ticker, start=start_date-timedelta(200), end=end_date)
        data = data[(data.index >= start_date) & (data.index <= end_date)]
        
        backtester = AdvancedSwingBacktest()
        trades, equity = backtester.run_backtest(data, params)
        metrics = backtester.calculate_metrics(trades, equity)
    
    # METRİK KARTLARI
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("📈 Getiri", metrics['total_return'])
    with col2: st.metric("🎯 Win Rate", metrics['win_rate'])
    with col3: st.metric("💰 Profit Factor", metrics['profit_factor'])
    with col4: st.metric("📉 Max DD", metrics['max_dd'])
    
    # İNTERAKTİF GRAFİKLER (PLOTLY)
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Portföy Değeri', 'Fiyat & Sinyaller', 'RSI'),
        vertical_spacing=0.05,
        row_heights=[0.4, 0.4, 0.2]
    )
    
    # Equity Curve
    fig.add_trace(go.Scatter(x=equity['date'], y=equity['equity'], 
                           name='Equity', line=dict(color='purple')), row=1, col=1)
    
    # Price + Signals
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'],
                               low=data['Low'], close=data['Close'], name='Fiyat'), row=2, col=1)
    
    buy_signals = data[data.index.isin(trades['entry_date'])]
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close']*0.98,
                           mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'),
                           name='ALIM'), row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='orange')), row=3, col=1)
    fig.add_hline(y=rsi_level, line_dash="dash", line_color="red", row=3, col=1)
    
    fig.update_layout(height=800, showlegend=True, title=f"{ticker} - Kombine Strateji")
    st.plotly_chart(fig, use_container_width=True)
    
    # TİCARET TABLOSU
    if not trades.empty:
        trades_display = trades.copy()
        trades_display['entry_date'] = trades_display['entry_date'].dt.strftime('%Y-%m-%d')
        trades_display['exit_date'] = trades_display['exit_date'].dt.strftime('%Y-%m-%d')
        
        st.subheader("📋 **TİCARET DETAYLARI**")
        st.dataframe(trades_display.style.format({
            'pnl': '${:.2f}', 'return_pct': '{:.1f}%'
        }).background_gradient(subset=['return_pct'], cmap='RdYlGn'), height=400)

st.markdown("---")
st.markdown("***v5.0 - 6 Göstergeli Profesyonel Sistem | 2024**")
