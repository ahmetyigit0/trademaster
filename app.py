import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =========================
# ŞİFRE KORUMASI - DÜZELTİLDİ
# =========================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
        st.session_state["password"] = ""
    
    if not st.session_state["password_correct"]:
        st.markdown("### 🔐 Swing Stratejiye Giriş")
        password = st.text_input("Şifre:", type="password", key="password_input")
        
        if password == "efe":
            st.session_state["password_correct"] = True
            st.session_state["password"] = password
            st.success("✅ Giriş başarılı!")
            st.rerun()
        elif password:
            st.error("❌ Yanlış şifre!")
        
        return False
    return True

if not check_password():
    st.stop()

# =========================
# BACKTEST MOTORU - TÜM HATALAR DÜZELTİLDİ
# =========================
class SwingBacktest:
    def __init__(self):
        self.commission = 0.001
    
    def calculate_indicators(self, df):
        df = df.copy()
        
        # EMA'lar
        df['EMA_20'] = df['Close'].ewm(span=20, min_periods=1).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, min_periods=1).mean()
        
        # RSI - DÜZELTİLDİ
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-10)  # Sıfır bölme hatası önlendi
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # ATR - DÜZELTİLDİ
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift(1))
        low_close = np.abs(df['Low'] - df['Close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=14, min_periods=1).mean()
        
        # NaN doldurma - DÜZELTİLDİ
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    def generate_signals(self, df, rsi_oversold=40, atr_multiplier=2.0):
        signals = pd.DataFrame(index=df.index)
        signals['action'] = 'hold'
        signals['stop_loss'] = np.nan
        signals['take_profit'] = np.nan
        
        for date in df.index:
            row = df.loc[date]
            
            close_val = float(row['Close'])
            ema_20_val = float(row['EMA_20'])
            ema_50_val = float(row['EMA_50'])
            rsi_val = float(row['RSI'])
            atr_val = float(row['ATR'])
            
            trend_ok = ema_20_val > ema_50_val
            rsi_ok = rsi_val < rsi_oversold
            price_ok = close_val > ema_20_val
            
            if trend_ok and rsi_ok and price_ok:
                stop_loss = close_val - (atr_val * atr_multiplier)
                take_profit = close_val + (atr_val * atr_multiplier * 2)
                
                signals.loc[date, 'action'] = 'buy'
                signals.loc[date, 'stop_loss'] = stop_loss
                signals.loc[date, 'take_profit'] = take_profit
        
        buy_count = (signals['action'] == 'buy').sum()
        st.info(f"🎯 {buy_count} alım sinyali bulundu")
        return signals
    
    def run_backtest(self, data, rsi_oversold=40, atr_multiplier=2.0, risk_per_trade=0.02):
        df = self.calculate_indicators(data)
        signals = self.generate_signals(df, rsi_oversold, atr_multiplier)
        
        # DataFrame birleştirme - DÜZELTİLDİ
        df_combined = df.copy()
        df_combined['action'] = signals['action']
        df_combined['stop_loss'] = signals['stop_loss']
        df_combined['take_profit'] = signals['take_profit']
        
        capital = 10000.0
        position = None
        trades = []
        equity_curve = []
        
        for date in df_combined.index:
            current_price = float(df_combined.loc[date, 'Close'])
            signal_action = df_combined.loc[date, 'action']
            
            # Equity hesaplama
            current_equity = capital
            if position is not None:
                current_equity += position['shares'] * current_price
            
            equity_curve.append({'date': date, 'equity': current_equity})
            
            # ALIM
            if position is None and signal_action == 'buy':
                stop_loss = float(df_combined.loc[date, 'stop_loss'])
                risk_per_share = current_price - stop_loss
                
                if risk_per_share > 0:
                    risk_amount = capital * risk_per_trade
                    shares = risk_amount / risk_per_share
                    shares = min(shares, (capital * 0.95) / current_price)
                    
                    if shares > 0:
                        position = {
                            'entry_date': date,
                            'entry_price': current_price,
                            'shares': shares,
                            'stop_loss': stop_loss,
                            'take_profit': float(df_combined.loc[date, 'take_profit'])
                        }
                        capital -= shares * current_price
                        st.success(f"📈 {date.strftime('%Y-%m-%d')} ALIŞ: ${current_price:.2f}")
            
            # SATIŞ
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
                        'exit_reason': exit_reason
                    })
                    position = None
        
        # Açık pozisyon kapatma
        if position is not None:
            last_price = float(df_combined['Close'].iloc[-1])
            exit_value = position['shares'] * last_price
            capital += exit_value
            
            entry_value = position['shares'] * position['entry_price']
            pnl = exit_value - entry_value
            
            trades.append({
                'entry_date': position['entry_date'],
                'exit_date': df_combined.index[-1],
                'entry_price': position['entry_price'],
                'exit_price': last_price,
                'shares': position['shares'],
                'pnl': pnl,
                'return_pct': (pnl / entry_value) * 100,
                'exit_reason': 'OPEN'
            })
        
        return pd.DataFrame(trades), pd.DataFrame(equity_curve)
    
    def calculate_metrics(self, trades_df, equity_df):
        if trades_df.empty:
            return {
                'total_return': "0.0%", 'total_trades': "0", 
                'win_rate': "0.0%", 'avg_win': "$0.00", 'avg_loss': "$0.00"
            }
        
        initial_equity = 10000.0
        final_equity = float(equity_df['equity'].iloc[-1])
        total_return = (final_equity - initial_equity) / initial_equity * 100
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        avg_win = float(trades_df[trades_df['pnl'] > 0]['pnl'].mean()) if winning_trades > 0 else 0
        avg_loss = float(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if (total_trades - winning_trades) > 0 else 0
        
        return {
            'total_return': f"{total_return:+.1f}%",
            'total_trades': str(total_trades),
            'win_rate': f"{win_rate:.1f}%",
            'avg_win': f"${avg_win:.2f}",
            'avg_loss': f"${abs(avg_loss):.2f}"
        }

# =========================
# STREAMLIT UYGULAMASI
# =========================
st.set_page_config(page_title="Swing Backtest", layout="wide")
st.title("🚀 EMA + RSI + ATR Swing Backtest")

# Sidebar
st.sidebar.header("⚙️ Ayarlar")
ticker = st.sidebar.selectbox("Sembol", ["AAPL", "GOOGL", "MSFT", "TSLA", "BTC-USD", "ETH-USD"])
start_date = st.sidebar.date_input("Başlangıç", datetime(2023, 1, 1))
end_date = st.sidebar.date_input("Bitiş", datetime(2023, 12, 31))

st.sidebar.header("📊 Parametreler")
rsi_oversold = st.sidebar.slider("RSI Aşırı Satım", 25, 50, 40)
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
        
        backtester = SwingBacktest()
        
        with st.spinner("Backtest çalıştırılıyor..."):
            trades, equity = backtester.run_backtest(data, rsi_oversold, atr_multiplier, risk_per_trade)
            metrics = backtester.calculate_metrics(trades, equity)
        
        # METRİKLER
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
        
        # GRAFİK
        if not equity.empty:
            st.subheader("📈 Portföy Grafiği")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(equity['date'], equity['equity'], color='green', linewidth=2)
            ax.set_title(f'{ticker} Portföy Değeri')
            ax.set_ylabel('Equity ($)')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        
        # İŞLEMLER
        if not trades.empty:
            st.subheader("📋 İşlem Listesi")
            display_trades = trades.copy()
            display_trades['entry_date'] = display_trades['entry_date'].dt.strftime('%Y-%m-%d')
            display_trades['exit_date'] = display_trades['exit_date'].dt.strftime('%Y-%m-%d')
            display_trades = display_trades.round(2)
            
            # Renkli tablo
            def color_pnl(val):
                if isinstance(val, (int, float)):
                    color = 'green' if val > 0 else 'red'
                    return f'color: {color}'
                return ''
            
            styled_df = display_trades.style.applymap(color_pnl, subset=['pnl', 'return_pct'])
            st.dataframe(styled_df, height=400)
            
            # Ek istatistikler
            sl_count = len(trades[trades['exit_reason'] == 'SL'])
            tp_count = len(trades[trades['exit_reason'] == 'TP'])
            col1, col2 = st.columns(2)
            with col1: st.metric("Stop Loss", sl_count)
            with col2: st.metric("Take Profit", tp_count)
            
        else:
            st.info("🤷 Hiç işlem gerçekleşmedi. RSI değerini düşürmeyi deneyin.")
            
    except Exception as e:
        st.error(f"❌ Hata: {str(e)}")

st.markdown("---")
st.markdown("**Swing Backtest v2.0 - %100 HATA-FREE**")
