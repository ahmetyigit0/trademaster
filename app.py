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
# BASİT BACKTEST MOTORU
# =========================
class SimpleBacktest:
    def __init__(self):
        self.commission = 0.001
        
    def calculate_rsi(self, prices, window=14):
        """RSI hesapla"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def calculate_ema(self, prices, window):
        """EMA hesapla"""
        return prices.ewm(span=window, min_periods=1).mean()
    
    def run_backtest(self, data, rsi_oversold=30, atr_multiplier=2.0, risk_per_trade=0.02):
        """Basit backtest çalıştır"""
        df = data.copy()
        
        # Teknik göstergeleri hesapla
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['EMA_20'] = self.calculate_ema(df['Close'], 20)
        df['EMA_50'] = self.calculate_ema(df['Close'], 50)
        
        # ATR hesapla
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift(1))
        low_close = np.abs(df['Low'] - df['Close'].shift(1))
        true_range = np.maximum(np.maximum(high_low, high_close), low_close)
        df['ATR'] = true_range.rolling(window=14, min_periods=1).mean()
        
        # Sinyal hesapla
        df['Signal'] = 0
        df['Position'] = 0
        
        for i in range(1, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Alış sinyali
            if (current['EMA_20'] > current['EMA_50'] and 
                current['RSI'] < rsi_oversold and 
                current['Close'] > current['EMA_20']):
                df.iloc[i, df.columns.get_loc('Signal')] = 1
        
        # Pozisyon yönetimi
        position = 0
        entry_price = 0
        trades = []
        equity = [10000]  # Başlangıç sermayesi
        
        for i in range(1, len(df)):
            current_signal = df.iloc[i]['Signal']
            current_price = df.iloc[i]['Close']
            current_atr = df.iloc[i]['ATR']
            
            # Yeni pozisyon aç
            if position == 0 and current_signal == 1:
                position = 1
                entry_price = current_price
                stop_loss = current_price - (current_atr * atr_multiplier)
                take_profit = current_price + (current_atr * atr_multiplier * 2)
                
            # Pozisyon yönetimi
            elif position == 1:
                # Stop-loss kontrolü
                if current_price <= stop_loss:
                    pnl = (current_price - entry_price) / entry_price
                    trades.append({
                        'entry_date': df.index[i-1],
                        'exit_date': df.index[i],
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl': pnl * 100,
                        'return_pct': pnl * 100,
                        'exit_reason': 'SL'
                    })
                    position = 0
                
                # Take-profit kontrolü
                elif current_price >= take_profit:
                    pnl = (current_price - entry_price) / entry_price
                    trades.append({
                        'entry_date': df.index[i-1],
                        'exit_date': df.index[i],
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl': pnl * 100,
                        'return_pct': pnl * 100,
                        'exit_reason': 'TP'
                    })
                    position = 0
            
            # Equity hesapla
            if position == 1:
                current_equity = equity[-1] * (1 + (current_price - entry_price) / entry_price)
            else:
                current_equity = equity[-1]
            
            equity.append(current_equity)
        
        # Açık pozisyonu kapat
        if position == 1:
            last_price = df.iloc[-1]['Close']
            pnl = (last_price - entry_price) / entry_price
            trades.append({
                'entry_date': df.index[-2],
                'exit_date': df.index[-1],
                'entry_price': entry_price,
                'exit_price': last_price,
                'pnl': pnl * 100,
                'return_pct': pnl * 100,
                'exit_reason': 'OPEN'
            })
        
        # Equity curve
        equity_df = pd.DataFrame({
            'date': df.index[:len(equity)],
            'equity': equity
        })
        
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        return trades_df, equity_df
    
    def calculate_metrics(self, trades_df, equity_df, initial_capital=10000):
        """Performans metriklerini hesapla"""
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
            final_equity = equity_df['equity'].iloc[-1]
            total_return = (final_equity - initial_capital) / initial_capital * 100
            
            # Trade istatistikleri
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            win_rate = (winning_trades / total_trades) * 100
            
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean()
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean()
            
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
st.title("📈 Swing Trading Backtest")
st.markdown("Basit ve hatasız backtest sistemi")

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
            data = yf.download(ticker, start=start_date, end=end_date)
            
            if data.empty:
                st.error("Veri bulunamadı")
                st.stop()
            
            st.success(f"{len(data)} günlük veri yüklendi")
        
        # Backtest çalıştır
        backtester = SimpleBacktest()
        trades, equity = backtester.run_backtest(data, rsi_oversold, atr_multiplier, risk_per_trade)
        metrics = backtester.calculate_metrics(trades, equity)
        
        # Sonuçlar
        st.subheader("📊 Performans")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Toplam Getiri", f"{metrics['total_return_%']}%")
            st.metric("Toplam İşlem", metrics['total_trades'])
        
        with col2:
            st.metric("Win Rate", f"{metrics['win_rate_%']}%")
            st.metric("Ort. Kazanç", f"{metrics['avg_win']:.2f}%")
        
        with col3:
            st.metric("Ort. Kayıp", f"{metrics['avg_loss']:.2f}%")
            st.metric("Max Drawdown", f"{metrics['max_drawdown_%']}%")
        
        # Grafik
        if not trades.empty:
            st.subheader("📈 Grafikler")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Equity curve
            ax1.plot(equity['date'], equity['equity'], color='green', linewidth=2)
            ax1.set_title('Portföy Değeri')
            ax1.grid(True, alpha=0.3)
            
            # Drawdown
            equity_series = equity.set_index('date')['equity']
            rolling_max = equity_series.expanding().max()
            drawdown = (equity_series - rolling_max) / rolling_max * 100
            
            ax2.fill_between(equity['date'], drawdown.values, 0, alpha=0.3, color='red')
            ax2.set_title('Drawdown')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # İşlemler
            st.subheader("📋 İşlemler")
            display_trades = trades.copy()
            display_trades['entry_date'] = display_trades['entry_date'].dt.strftime('%Y-%m-%d')
            display_trades['exit_date'] = display_trades['exit_date'].dt.strftime('%Y-%m-%d')
            st.dataframe(display_trades)
        else:
            st.info("İşlem bulunamadı")
            
    except Exception as e:
        st.error(f"Hata: {str(e)}")

st.markdown("---")
st.markdown("Simple Backtest v1.0")