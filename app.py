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
# BACKTEST MOTORU
# =========================
class SwingBacktest:
    def __init__(self):
        self.commission = 0.001
    
    def calculate_indicators(self, df):
        """Teknik göstergeleri hesapla"""
        try:
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
            
            # NaN değerleri temizle
            df = df.fillna(method='bfill').fillna(method='ffill')
            
            return df
            
        except Exception as e:
            st.error(f"İndikatör hatası: {e}")
            return df
    
    def generate_signals(self, df, rsi_oversold=40, atr_multiplier=2.0):
        """Sinyal üret - DAHA GEVŞEK PARAMETRELER"""
        try:
            signals = []
            
            for i in range(len(df)):
                try:
                    row = df.iloc[i]
                    
                    close_val = float(row['Close'])
                    ema_20_val = float(row['EMA_20'])
                    ema_50_val = float(row['EMA_50'])
                    rsi_val = float(row['RSI'])
                    atr_val = float(row['ATR'])
                    
                    # DAHA GEVŞEK KOŞULLAR:
                    # 1. Sadece trend koşulu (EMA20 > EMA50)
                    # 2. RSI 40'tan küçük (daha gevşek)
                    # 3. Fiyat EMA20'nin üstünde
                    
                    trend_ok = ema_20_val > ema_50_val
                    rsi_ok = rsi_val < rsi_oversold  # 30 yerine 40
                    price_ok = close_val > ema_20_val
                    
                    buy_signal = trend_ok and rsi_ok and price_ok
                    
                    if buy_signal:
                        stop_loss = close_val - (atr_val * atr_multiplier)
                        take_profit = close_val + (atr_val * atr_multiplier * 2)
                        
                        signals.append({
                            'date': df.index[i],
                            'action': 'buy',
                            'price': close_val,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit
                        })
                    else:
                        signals.append({
                            'date': df.index[i],
                            'action': 'hold',
                            'price': close_val,
                            'stop_loss': 0,
                            'take_profit': 0
                        })
                        
                except:
                    signals.append({
                        'date': df.index[i],
                        'action': 'hold',
                        'price': float(df.iloc[i]['Close']),
                        'stop_loss': 0,
                        'take_profit': 0
                    })
            
            signals_df = pd.DataFrame(signals)
            if not signals_df.empty:
                signals_df = signals_df.set_index('date')
            
            st.info(f"📊 {len([s for s in signals if s['action'] == 'buy'])} alış sinyali bulundu")
            return signals_df
            
        except Exception as e:
            st.error(f"Sinyal hatası: {e}")
            return pd.DataFrame()
    
    def run_backtest(self, data, rsi_oversold=40, atr_multiplier=2.0, risk_per_trade=0.02):
        """Backtest çalıştır"""
        try:
            df = self.calculate_indicators(data)
            
            if df.empty:
                return pd.DataFrame(), pd.DataFrame()
            
            signals = self.generate_signals(df, rsi_oversold, atr_multiplier)
            
            if signals.empty:
                return pd.DataFrame(), pd.DataFrame()
            
            capital = 10000
            position = None
            trades = []
            equity_curve = []
            
            for date in df.index:
                try:
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
                                    'take_profit': float(signal['take_profit'])
                                }
                                capital -= shares * current_price
                    
                    elif position is not None:
                        exit_reason = None
                        exit_price = None
                        
                        if current_price <= position['stop_loss']:
                            exit_reason = 'SL'
                            exit_price = position['stop_loss']
                        elif current_price >= position['take_profit']:
                            exit_reason = 'TP'
                            exit_price = position['take_profit']
                        
                        if exit_reason:
                            exit_value = position['shares'] * exit_price
                            capital += exit_value
                            
                            entry_value = position['shares'] * position['entry_price']
                            pnl = exit_value - entry_value
                            
                            trades.append({
                                'entry_date': position['entry_date'],
                                'exit_date': date,
                                'entry_price': position['entry_price'],
                                'exit_price': exit_price,
                                'shares': position['shares'],
                                'pnl': pnl,
                                'return_pct': (pnl / entry_value) * 100,
                                'exit_reason': exit_reason,
                                'hold_days': (date - position['entry_date']).days
                            })
                            
                            position = None
                            
                except:
                    continue
            
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
                    'shares': position['shares'],
                    'pnl': pnl,
                    'return_pct': (pnl / entry_value) * 100,
                    'exit_reason': 'OPEN',
                    'hold_days': (df.index[-1] - position['entry_date']).days
                })
            
            trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
            equity_df = pd.DataFrame(equity_curve)
            
            return trades_df, equity_df
            
        except Exception as e:
            st.error(f"Backtest hatası: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def calculate_metrics(self, trades_df, equity_df):
        """Performans metrikleri"""
        if trades_df.empty or equity_df.empty:
            return {
                'total_return_%': 0,
                'total_trades': 0,
                'win_rate_%': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'max_drawdown_%': 0
            }
        
        try:
            initial_equity = 10000
            final_equity = equity_df['equity'].iloc[-1]
            total_return = (final_equity - initial_equity) / initial_equity * 100
            
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if (total_trades - winning_trades) > 0 else 0
            
            equity_series = equity_df.set_index('date')['equity']
            rolling_max = equity_series.expanding().max()
            drawdown = (equity_series - rolling_max) / rolling_max * 100
            max_drawdown = drawdown.min()
            
            return {
                'total_return_%': round(total_return, 2),
                'total_trades': total_trades,
                'win_rate_%': round(win_rate, 1),
                'avg_win': round(avg_win, 2),
                'avg_loss': round(avg_loss, 2),
                'max_drawdown_%': round(max_drawdown, 2)
            }
            
        except:
            return {
                'total_return_%': 0,
                'total_trades': 0,
                'win_rate_%': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'max_drawdown_%': 0
            }

# =========================
# STREAMLIT UYGULAMASI
# =========================
st.set_page_config(page_title="Swing Backtest", layout="wide")
st.title("🚀 Swing Trading Backtest")
st.markdown("**Daha Gevşek Strateji ile**")

# Sidebar
st.sidebar.header("⚙️ Ayarlar")
ticker = st.sidebar.selectbox("Sembol", ["AAPL", "GOOGL", "MSFT", "TSLA", "BTC-USD", "ETH-USD", "NVDA", "AMZN"])
start_date = st.sidebar.date_input("Başlangıç", datetime(2022, 1, 1))  # Daha uzun tarih
end_date = st.sidebar.date_input("Bitiş", datetime(2023, 12, 31))

st.sidebar.header("📊 Parametreler")
rsi_oversold = st.sidebar.slider("RSI Aşırı Satım", 25, 50, 40)  # 40'a çıkarıldı
atr_multiplier = st.sidebar.slider("ATR Çarpanı", 1.0, 3.0, 2.0)
risk_per_trade = st.sidebar.slider("Risk %", 1.0, 5.0, 2.0) / 100

st.sidebar.info("""
**Strateji:**
- EMA20 > EMA50 (Trend)
- RSI < Seçilen Değer (Aşırı Satım)
- Fiyat > EMA20
""")

# Ana içerik
if st.button("🎯 Backtest Çalıştır", type="primary"):
    try:
        with st.spinner("Veri yükleniyor..."):
            # Daha uzun tarih aralığı
            extended_start = start_date - timedelta(days=100)
            data = yf.download(ticker, start=extended_start, end=end_date, progress=False)
            
            if data.empty:
                st.error("❌ Veri bulunamadı")
                st.stop()
            
            # Sadece istenen tarih aralığını kullan
            data = data[data.index >= pd.to_datetime(start_date)]
            data = data[data.index <= pd.to_datetime(end_date)]
            
            st.success(f"✅ {len(data)} günlük veri yüklendi")
            st.info(f"📈 Fiyat aralığı: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
        
        backtester = SwingBacktest()
        
        with st.spinner("Backtest çalıştırılıyor..."):
            trades, equity = backtester.run_backtest(data, rsi_oversold, atr_multiplier, risk_per_trade)
            metrics = backtester.calculate_metrics(trades, equity)
        
        st.subheader("📊 Performans Özeti")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Toplam Getiri", f"{metrics['total_return_%']}%")
            st.metric("Toplam İşlem", f"{metrics['total_trades']}")
        
        with col2:
            st.metric("Win Rate", f"{metrics['win_rate_%']}%")
            st.metric("Ort. Kazanç", f"${metrics['avg_win']:.2f}")
        
        with col3:
            st.metric("Ort. Kayıp", f"${metrics['avg_loss']:.2f}")
            st.metric("Max Drawdown", f"{metrics['max_drawdown_%']}%")
        
        if not trades.empty:
            st.subheader("📈 Performans Grafikleri")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            ax1.plot(equity['date'], equity['equity'], color='green', linewidth=2)
            ax1.set_title('Portföy Değeri')
            ax1.set_ylabel('Equity ($)')
            ax1.grid(True, alpha=0.3)
            
            equity_series = equity.set_index('date')['equity']
            rolling_max = equity_series.expanding().max()
            drawdown = (equity_series - rolling_max) / rolling_max * 100
            
            ax2.fill_between(equity['date'], drawdown.values, 0, alpha=0.3, color='red')
            ax2.set_title('Drawdown')
            ax2.set_ylabel('Drawdown %')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
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
            - Farklı sembol deneyin (BTC-USD, TSLA daha volatil)
            - Tarih aralığını genişletin
            - ATR çarpanını 1.5'e düşürün
            """)
            
    except Exception as e:
        st.error(f"❌ Hata: {str(e)}")

st.markdown("---")
st.markdown("**Swing Backtest v2.0 | Gevşek Strateji**")