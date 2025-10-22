import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

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
# BACKTEST MOTORU - MACD EKLENDİ
# =========================
class SwingBacktest:
    def __init__(self):
        self.commission = 0.001
    
    def calculate_indicators(self, df):
        df = df.copy()
        
        # EMA'lar
        df['EMA_20'] = df['Close'].ewm(span=20, min_periods=1).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, min_periods=1).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD EKLENDİ
        exp1 = df['Close'].ewm(span=12, min_periods=1).mean()
        exp2 = df['Close'].ewm(span=26, min_periods=1).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, min_periods=1).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift(1))
        low_close = np.abs(df['Low'] - df['Close'].shift(1))
        
        true_range_values = []
        for i in range(len(df)):
            if i == 0:
                true_range_values.append(float(high_low.iloc[i]))
            else:
                tr = max(float(high_low.iloc[i]), float(high_close.iloc[i]), float(low_close.iloc[i]))
                true_range_values.append(tr)
        
        df['ATR'] = pd.Series(true_range_values, index=df.index).rolling(window=14, min_periods=1).mean()
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    def generate_signals(self, df, rsi_oversold=40, atr_multiplier=2.0, use_macd=True):
        signals = []
        buy_count = 0
        
        for i in range(len(df)):
            try:
                row = df.iloc[i]
                
                close_val = float(row['Close'])
                ema_20_val = float(row['EMA_20'])
                ema_50_val = float(row['EMA_50'])
                rsi_val = float(row['RSI'])
                atr_val = float(row['ATR'])
                macd_hist_val = float(row['MACD_Hist'])
                
                # TEMEL KOŞULLAR
                trend_ok = ema_20_val > ema_50_val
                rsi_ok = rsi_val < rsi_oversold
                price_ok = close_val > ema_20_val
                
                # MACD KOŞULLARI
                macd_bullish = macd_hist_val > 0
                macd_turning = macd_hist_val > df['MACD_Hist'].iloc[i-1] if i > 0 else False
                
                # STRATEJİ 1: RSI + Trend (Orjinal)
                strategy1 = trend_ok and rsi_ok and price_ok
                
                # STRATEJİ 2: MACD + Trend (Yeni)
                strategy2 = trend_ok and macd_bullish and price_ok
                
                # STRATEJİ 3: RSI + MACD Kombinasyonu
                strategy3 = trend_ok and rsi_ok and macd_bullish
                
                # Sinyal seçimi
                if use_macd:
                    buy_signal = strategy1 or strategy2 or strategy3
                else:
                    buy_signal = strategy1
                
                if buy_signal:
                    stop_loss = close_val - (atr_val * atr_multiplier)
                    take_profit = close_val + (atr_val * atr_multiplier * 2)
                    
                    signals.append({
                        'date': df.index[i],
                        'action': 'buy',
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'strategy': 'RSI+MACD' if strategy3 else ('MACD' if strategy2 else 'RSI')
                    })
                    buy_count += 1
                else:
                    signals.append({
                        'date': df.index[i],
                        'action': 'hold'
                    })
                    
            except:
                signals.append({
                    'date': df.index[i],
                    'action': 'hold'
                })
        
        signals_df = pd.DataFrame(signals)
        if not signals_df.empty:
            signals_df = signals_df.set_index('date')
        
        st.info(f"🎯 {buy_count} alış sinyali bulundu (MACD aktif)")
        return signals_df
    
    def run_backtest(self, data, rsi_oversold=40, atr_multiplier=2.0, risk_per_trade=0.02, use_macd=True):
        df = self.calculate_indicators(data)
        signals = self.generate_signals(df, rsi_oversold, atr_multiplier, use_macd)
        
        capital = 10000
        position = None
        trades = []
        equity_curve = []
        
        for date in df.index:
            current_price = float(df.loc[date, 'Close'])
            signal = signals.loc[date]
            
            current_equity = capital
            if position is not None:
                current_equity += position['shares'] * current_price
            
            equity_curve.append({'date': date, 'equity': current_equity})
            
            if position is None and signal['action'] == 'buy':
                stop_loss = float(signal['stop_loss'])
                risk_per_share = current_price - stop_loss
                
                if risk_per_share > 0:
                    risk_amount = capital * risk_per_trade
                    shares = risk_amount / risk_per_share
                    
                    if shares > 0:
                        position = {
                            'entry_date': date,
                            'entry_price': current_price,
                            'shares': shares,
                            'stop_loss': stop_loss,
                            'take_profit': float(signal['take_profit']),
                            'strategy': signal.get('strategy', 'RSI')
                        }
                        capital -= shares * current_price
            
            elif position is not None:
                if current_price <= position['stop_loss']:
                    exit_price = position['stop_loss']
                    exit_value = position['shares'] * exit_price
                    capital += exit_value
                    
                    entry_value = position['shares'] * position['entry_price']
                    pnl = exit_value - entry_value
                    
                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': date,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'return_pct': (pnl / entry_value) * 100,
                        'exit_reason': 'SL',
                        'strategy': position['strategy']
                    })
                    position = None
                
                elif current_price >= position['take_profit']:
                    exit_price = position['take_profit']
                    exit_value = position['shares'] * exit_price
                    capital += exit_value
                    
                    entry_value = position['shares'] * position['entry_price']
                    pnl = exit_value - entry_value
                    
                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': date,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'return_pct': (pnl / entry_value) * 100,
                        'exit_reason': 'TP',
                        'strategy': position['strategy']
                    })
                    position = None
        
        if position is not None:
            last_price = float(df['Close'].iloc[-1])
            exit_value = position['shares'] * last_price
            capital += exit_value
            
            entry_value = position['shares'] * position['entry_price']
            pnl = exit_value - entry_value
            
            trades.append({
                'entry_date': position['entry_date'],
                'exit_date': df.index[-1],
                'entry_price': position['entry_price'],
                'exit_price': last_price,
                'pnl': pnl,
                'return_pct': (pnl / entry_value) * 100,
                'exit_reason': 'OPEN',
                'strategy': position['strategy']
            })
        
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        equity_df = pd.DataFrame(equity_curve)
        
        return trades_df, equity_df
    
    def calculate_metrics(self, trades_df, equity_df):
        if trades_df.empty:
            return {
                'total_return': "0.0%",
                'total_trades': "0",
                'win_rate': "0.0%",
                'avg_win': "$0.00",
                'avg_loss': "$0.00"
            }
        
        try:
            initial_equity = 10000.0
            final_equity = float(equity_df['equity'].iloc[-1])
            total_return = (final_equity - initial_equity) / initial_equity * 100.0
            
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            win_rate = (winning_trades / total_trades) * 100.0 if total_trades > 0 else 0.0
            
            avg_win = float(trades_df[trades_df['pnl'] > 0]['pnl'].mean()) if winning_trades > 0 else 0.0
            avg_loss = float(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if (total_trades - winning_trades) > 0 else 0.0
            
            return {
                'total_return': f"{round(total_return, 2)}%",
                'total_trades': str(total_trades),
                'win_rate': f"{round(win_rate, 1)}%",
                'avg_win': f"${round(avg_win, 2)}",
                'avg_loss': f"${round(avg_loss, 2)}"
            }
            
        except:
            return {
                'total_return': "0.0%",
                'total_trades': "0",
                'win_rate': "0.0%",
                'avg_win': "$0.00",
                'avg_loss': "$0.00"
            }

# =========================
# STREAMLIT UYGULAMASI
# =========================
st.set_page_config(page_title="Swing Backtest", layout="wide")
st.title("🚀 Swing Trading Backtest - MACD Eklendi")

# Sidebar
st.sidebar.header("⚙️ Ayarlar")
ticker = st.sidebar.selectbox("Sembol", ["AAPL", "GOOGL", "MSFT", "TSLA", "BTC-USD", "ETH-USD", "NVDA", "AMZN"])
start_date = st.sidebar.date_input("Başlangıç", datetime(2023, 1, 1))
end_date = st.sidebar.date_input("Bitiş", datetime(2023, 12, 31))

st.sidebar.header("📊 Parametreler")
rsi_oversold = st.sidebar.slider("RSI Aşırı Satım", 25, 50, 40)
atr_multiplier = st.sidebar.slider("ATR Çarpanı", 1.0, 3.0, 2.0)
risk_per_trade = st.sidebar.slider("Risk %", 1.0, 5.0, 2.0) / 100
use_macd = st.sidebar.checkbox("MACD Stratejisini Kullan", value=True)

st.sidebar.info("""
**Yeni Stratejiler:**
- RSI + Trend (Orjinal)
- MACD + Trend (Yeni)
- RSI + MACD Kombinasyonu
""")

# Ana içerik
if st.button("🎯 Backtest Çalıştır"):
    try:
        with st.spinner("Veri yükleniyor..."):
            extended_start = start_date - timedelta(days=100)
            data = yf.download(ticker, start=extended_start, end=end_date, progress=False)
            
            if data.empty:
                st.error("❌ Veri bulunamadı")
                st.stop()
            
            data = data[data.index >= pd.to_datetime(start_date)]
            data = data[data.index <= pd.to_datetime(end_date)]
            
            st.success(f"✅ {len(data)} günlük veri yüklendi")
            st.info(f"📈 Fiyat aralığı: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
        
        backtester = SwingBacktest()
        
        with st.spinner("Backtest çalıştırılıyor..."):
            trades, equity = backtester.run_backtest(data, rsi_oversold, atr_multiplier, risk_per_trade, use_macd)
            metrics = backtester.calculate_metrics(trades, equity)
        
        st.subheader("📊 Performans Özeti")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Toplam Getiri", metrics['total_return'])
            st.metric("Toplam İşlem", metrics['total_trades'])
        
        with col2:
            st.metric("Win Rate", metrics['win_rate'])
            st.metric("Ort. Kazanç", metrics['avg_win'])
        
        with col3:
            st.metric("Ort. Kayıp", metrics['avg_loss'])
        
        if not trades.empty:
            # Strateji Dağılımı
            strategy_counts = trades['strategy'].value_counts()
            st.info(f"**Strateji Dağılımı:** {dict(strategy_counts)}")
            
            st.subheader("📈 Performans Grafikleri")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(equity['date'], equity['equity'], color='green', linewidth=2)
            ax.set_title('Portföy Değeri')
            ax.set_ylabel('Equity ($)')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            st.subheader("📋 İşlem Listesi")
            display_trades = trades.copy()
            display_trades['entry_date'] = display_trades['entry_date'].dt.strftime('%Y-%m-%d')
            display_trades['exit_date'] = display_trades['exit_date'].dt.strftime('%Y-%m-%d')
            st.dataframe(display_trades)
            
        else:
            st.warning("""
            **🤔 Hala işlem yok! Şunları deneyin:**
            - RSI değerini 45-50'ye çıkarın
            - BTC-USD veya TSLA gibi volatil semboller deneyin
            - ATR çarpanını 1.5'e düşürün
            - MACD'yi aktif edin
            """)
            
    except Exception as e:
        st.error(f"❌ Hata: {str(e)}")

st.markdown("---")
st.markdown("**Backtest Sistemi v2.0 | MACD Eklendi**")