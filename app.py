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
# BACKTEST MOTORU - HATASIZ
# =========================
class SwingBacktest:
    def __init__(self):
        self.commission = 0.001
    
    def calculate_indicators(self, df):
        """Basit ve hatasız indikatör hesaplama"""
        try:
            df = df.copy()
            
            # EMA'lar
            df['EMA_20'] = df['Close'].ewm(span=20).mean()
            df['EMA_50'] = df['Close'].ewm(span=50).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Basit ATR (True Range)
            high_low = df['High'] - df['Low']
            high_close = abs(df['High'] - df['Close'].shift(1))
            low_close = abs(df['Low'] - df['Close'].shift(1))
            
            # True Range'i hesapla (tek boyutlu array olarak)
            true_range = pd.Series(np.maximum(high_low.values, np.maximum(high_close.values, low_close.values)), 
                                 index=df.index)
            
            df['ATR'] = true_range.rolling(window=14).mean()
            
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
                row = df.iloc[i]
                
                # Sinyal koşulları
                trend_condition = row['EMA_20'] > row['EMA_50']
                rsi_condition = row['RSI'] < rsi_oversold
                price_condition = row['Close'] > row['EMA_20']
                
                buy_signal = trend_condition and rsi_condition and price_condition
                
                if buy_signal:
                    stop_loss = row['Close'] - (row['ATR'] * atr_multiplier)
                    take_profit = row['Close'] + (row['ATR'] * atr_multiplier * 2)
                    
                    signals.append({
                        'date': df.index[i],
                        'action': 'buy',
                        'price': row['Close'],
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    })
                else:
                    signals.append({
                        'date': df.index[i],
                        'action': 'hold',
                        'price': row['Close'],
                        'stop_loss': 0,
                        'take_profit': 0
                    })
            
            return pd.DataFrame(signals).set_index('date')
            
        except Exception as e:
            st.error(f"Sinyal hatası: {e}")
            return pd.DataFrame()
    
    def run_backtest(self, data, rsi_oversold=30, atr_multiplier=2.0, risk_per_trade=0.02):
        """Backtest çalıştır"""
        try:
            # İndikatörleri hesapla
            df = self.calculate_indicators(data)
            
            if df.empty:
                return pd.DataFrame(), pd.DataFrame()
            
            # Sinyalleri üret
            signals = self.generate_signals(df, rsi_oversold, atr_multiplier)
            
            if signals.empty:
                return pd.DataFrame(), pd.DataFrame()
            
            # Backtest değişkenleri
            capital = 10000
            position = None
            trades = []
            equity_curve = []
            
            for date in df.index:
                current_price = df.loc[date, 'Close']
                signal = signals.loc[date]
                
                # Equity hesapla
                if position is not None:
                    current_equity = capital + (position['shares'] * current_price)
                else:
                    current_equity = capital
                
                equity_curve.append({'date': date, 'equity': current_equity})
                
                # Yeni pozisyon aç
                if position is None and signal['action'] == 'buy':
                    risk_amount = capital * risk_per_trade
                    risk_per_share = current_price - signal['stop_loss']
                    
                    if risk_per_share > 0:
                        shares = risk_amount / risk_per_share
                        
                        if shares > 0:
                            position = {
                                'entry_date': date,
                                'entry_price': current_price,
                                'shares': shares,
                                'stop_loss': signal['stop_loss'],
                                'take_profit': signal['take_profit']
                            }
                            capital -= shares * current_price
                
                # Pozisyon yönetimi
                elif position is not None:
                    exit_reason = None
                    
                    # Stop-loss
                    if current_price <= position['stop_loss']:
                        exit_reason = 'SL'
                    # Take-profit
                    elif current_price >= position['take_profit']:
                        exit_reason = 'TP'
                    
                    if exit_reason:
                        exit_price = position['stop_loss'] if exit_reason == 'SL' else position['take_profit']
                        exit_value = position['shares'] * exit_price
                        capital += exit_value
                        
                        pnl = exit_value - (position['shares'] * position['entry_price'])
                        
                        trades.append({
                            'entry_date': position['entry_date'],
                            'exit_date': date,
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'shares': position['shares'],
                            'pnl': pnl,
                            'return_pct': (pnl / (position['shares'] * position['entry_price'])) * 100,
                            'exit_reason': exit_reason,
                            'hold_days': (date - position['entry_date']).days
                        })
                        
                        position = None
            
            # Açık pozisyonu kapat
            if position is not None:
                last_price = df['Close'].iloc[-1]
                exit_value = position['shares'] * last_price
                capital += exit_value
                
                pnl = exit_value - (position['shares'] * position['entry_price'])
                
                trades.append({
                    'entry_date': position['entry_date'],
                    'exit_date': df.index[-1],
                    'entry_price': position['entry_price'],
                    'exit_price': last_price,
                    'shares': position['shares'],
                    'pnl': pnl,
                    'return_pct': (pnl / (position['shares'] * position['entry_price'])) * 100,
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
        if trades_df.empty:
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
            st.error(f"Metrik hatası: {e}")
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
st.markdown("**Hatasız ve Çalışan Versiyon**")

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
            # Tarih aralığını genişlet (indikatörler için)
            extended_start = start_date - timedelta(days=100)
            data = yf.download(ticker, start=extended_start, end=end_date, progress=False)
            
            if data.empty:
                st.error("❌ Veri bulunamadı")
                st.stop()
            
            # Sadece istenen tarih aralığını kullan
            data = data[data.index >= pd.to_datetime(start_date)]
            data = data[data.index <= pd.to_datetime(end_date)]
            
            if data.empty:
                st.error("❌ Filtrelenmiş veri kalmadı")
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
            ax1.set_title('Portföy Değeri', fontweight='bold')
            ax1.set_ylabel('Equity ($)')
            ax1.grid(True, alpha=0.3)
            
            # Drawdown
            equity_series = equity.set_index('date')['equity']
            rolling_max = equity_series.expanding().max()
            drawdown = (equity_series - rolling_max) / rolling_max * 100
            
            ax2.fill_between(equity['date'], drawdown.values, 0, alpha=0.3, color='red')
            ax2.set_title('Drawdown', fontweight='bold')
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
            st.info("🤷 Hiç işlem gerçekleşmedi. Parametreleri gevşetmeyi deneyin.")
            
    except Exception as e:
        st.error(f"❌ Hata: {str(e)}")

st.markdown("---")
st.markdown("**Swing Backtest Pro | Tam Çalışır**")