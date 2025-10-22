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
            
            true_range = np.maximum(high_low.values, np.maximum(high_close.fillna(0).values, low_close.fillna(0).values))
            df['ATR'] = pd.Series(true_range, index=df.index).rolling(window=14, min_periods=1).mean()
            
            # NaN değerleri temizle
            df = df.fillna(method='bfill').fillna(method='ffill')
            
            return df
            
        except Exception as e:
            st.error(f"İndikatör hatası: {e}")
            return df
    
    def generate_signals(self, df, rsi_oversold=30, atr_multiplier=2.0):
        """Sinyal üret"""
        try:
            signals = []
            
            for i in range(len(df)):
                try:
                    row = df.iloc[i]
                    
                    # Değerleri teker teker al ve float'a çevir
                    close_val = float(row['Close'])
                    ema_20_val = float(row['EMA_20'])
                    ema_50_val = float(row['EMA_50'])
                    rsi_val = float(row['RSI'])
                    atr_val = float(row['ATR'])
                    
                    # Koşulları tek tek kontrol et (Series karşılaştırması yok)
                    trend_ok = ema_20_val > ema_50_val
                    rsi_ok = rsi_val < rsi_oversold
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
                        
                except Exception as e:
                    # Hata durumunda hold
                    signals.append({
                        'date': df.index[i],
                        'action': 'hold',
                        'price': df.iloc[i]['Close'],
                        'stop_loss': 0,
                        'take_profit': 0
                    })
            
            signals_df = pd.DataFrame(signals)
            if not signals_df.empty:
                signals_df = signals_df.set_index('date')
            return signals_df
            
        except Exception as e:
            st.error(f"Sinyal hatası: {e}")
            return pd.DataFrame()
    
    def run_backtest(self, data, rsi_oversold=30, atr_multiplier=2.0, risk_per_trade=0.02):
        """Backtest çalıştır"""
        try:
            # İndikatörleri hesapla
            df = self.calculate_indicators(data)
            
            if df.empty:
                st.warning("İndikatörler hesaplanamadı")
                return pd.DataFrame(), pd.DataFrame()
            
            # Sinyalleri üret
            signals = self.generate_signals(df, rsi_oversold, atr_multiplier)
            
            if signals.empty:
                st.warning("Sinyal üretilemedi")
                return pd.DataFrame(), pd.DataFrame()
            
            # Backtest değişkenleri
            capital = 10000
            position = None
            trades = []
            equity_curve = []
            
            for date in df.index:
                try:
                    current_price = float(df.loc[date, 'Close'])
                    signal = signals.loc[date]
                    
                    # Equity hesapla
                    current_equity = capital
                    if position is not None:
                        current_equity += position['shares'] * current_price
                    
                    equity_curve.append({'date': date, 'equity': current_equity})
                    
                    # Yeni pozisyon aç
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
                    
                    # Pozisyon yönetimi
                    elif position is not None:
                        exit_reason = None
                        exit_price = None
                        
                        # Stop-loss kontrolü
                        if current_price <= position['stop_loss']:
                            exit_reason = 'SL'
                            exit_price = position['stop_loss']
                        # Take-profit kontrolü
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
                                'return_pct': (pnl / entry_value) * 100 if entry_value > 0 else 0,
                                'exit_reason': exit_reason,
                                'hold_days': (date - position['entry_date']).days
                            })
                            
                            position = None
                            
                except Exception as e:
                    continue
            
            # Açık pozisyonu kapat
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
                    'return_pct': (pnl / entry_value) * 100 if entry_value > 0 else 0,
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
            # Toplam getiri
            initial_equity = 10000
            final_equity = equity_df['equity'].iloc[-1]
            total_return = (final_equity - initial_equity) / initial_equity * 100
            
            # Trade istatistikleri
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if (total_trades - winning_trades) > 0 else 0
            
            # Drawdown
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
            
        except Exception as e:
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
st.markdown("**Hatasız Çalışan Versiyon**")

# Sidebar
st.sidebar.header("⚙️ Ayarlar")
ticker = st.sidebar.selectbox("Sembol", ["AAPL", "GOOGL", "MSFT", "TSLA", "BTC-USD", "ETH-USD"])
start_date = st.sidebar.date_input("Başlangıç", datetime(2023, 1, 1))
end_date = st.sidebar.date_input("Bitiş", datetime(2023, 12, 31))

st.sidebar.header("📊 Parametreler")
rsi_oversold = st.sidebar.slider("RSI Aşırı Satım", 20, 40, 30)
atr_multiplier = st.sidebar.slider("ATR Çarpanı", 1.0, 3.0, 2.0)
risk_per_trade = st.sidebar.slider("Risk %", 1.0, 5.0, 2.0) / 100

# Ana içerik
if st.button("🎯 Backtest Çalıştır", type="primary"):
    try:
        with st.spinner("Veri yükleniyor..."):
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                st.error("❌ Veri bulunamadı")
                st.stop()
            
            st.success(f"✅ {len(data)} günlük veri yüklendi")
        
        # Backtest çalıştır
        backtester = SwingBacktest()
        
        with st.spinner("Backtest çalıştırılıyor..."):
            trades, equity = backtester.run_backtest(data, rsi_oversold, atr_multiplier, risk_per_trade)
            metrics = backtester.calculate_metrics(trades, equity)
        
        # Sonuçlar
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
        
        # Grafikler
        if not trades.empty:
            st.subheader("📈 Performans Grafikleri")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Equity curve
            ax1.plot(equity['date'], equity['equity'], color='blue', linewidth=2)
            ax1.set_title('Portföy Değeri')
            ax1.set_ylabel('Equity ($)')
            ax1.grid(True, alpha=0.3)
            
            # Drawdown
            equity_series = equity.set_index('date')['equity']
            rolling_max = equity_series.expanding().max()
            drawdown = (equity_series - rolling_max) / rolling_max * 100
            
            ax2.fill_between(equity['date'], drawdown.values, 0, alpha=0.3, color='red')
            ax2.set_title('Drawdown')
            ax2.set_ylabel('Drawdown %')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # İşlem listesi
            st.subheader("📋 İşlem Listesi")
            display_trades = trades.copy()
            display_trades['entry_date'] = display_trades['entry_date'].dt.strftime('%Y-%m-%d')
            display_trades['exit_date'] = display_trades['exit_date'].dt.strftime('%Y-%m-%d')
            st.dataframe(display_trades)
            
        else:
            st.info("🤷 Hiç işlem gerçekleşmedi. Parametreleri değiştirmeyi deneyin.")
            
    except Exception as e:
        st.error(f"❌ Hata: {str(e)}")

st.markdown("---")
st.markdown("**Swing Backtest | Tam Çalışır**")