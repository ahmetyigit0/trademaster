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
            st.session_state["password_attempts"] = 0
            del st.session_state["password"]
        else:
            st.session_state["password_attempts"] = st.session_state.get("password_attempts", 0) + 1
            st.session_state["password_correct"] = False
            if st.session_state["password_attempts"] >= 3:
                st.error("🚫 3 başarısız giriş. Lütfen daha sonra tekrar deneyin.")
                st.stop()
    
    if not st.session_state["password_correct"]:
        st.markdown("### 🔐 Swing Backtest Sistemine Giriş")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.text_input(
                "Şifre", 
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
# BACKTEST MOTORU
# =========================
class SwingBacktest:
    def __init__(self):
        self.commission = 0.001
        self.initial_capital = 10000
    
    def calculate_indicators(self, df):
        df = df.copy()
        
        try:
            # EMA'lar
            df['EMA_20'] = df['Close'].ewm(span=20, min_periods=1).mean()
            df['EMA_50'] = df['Close'].ewm(span=50, min_periods=1).mean()
            df['EMA_200'] = df['Close'].ewm(span=200, min_periods=1).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            avg_gain = gain.rolling(window=14, min_periods=1).mean()
            avg_loss = loss.rolling(window=14, min_periods=1).mean()
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['Close'].ewm(span=12, min_periods=1).mean()
            exp2 = df['Close'].ewm(span=26, min_periods=1).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9, min_periods=1).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20, min_periods=1).mean()
            bb_std = df['Close'].rolling(window=20, min_periods=1).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            
            # Stochastic
            low_14 = df['Low'].rolling(window=14, min_periods=1).min()
            high_14 = df['High'].rolling(window=14, min_periods=1).max()
            df['STOCH_K'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
            df['STOCH_D'] = df['STOCH_K'].rolling(window=3, min_periods=1).mean()
            
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
            
            # Volume SMA
            df['Volume_SMA'] = df['Volume'].rolling(window=20, min_periods=1).mean()
            
            df = df.fillna(method='bfill').fillna(method='ffill')
            
            return df
            
        except Exception as e:
            st.error(f"Gösterge hesaplama hatası: {e}")
            return df
    
    def generate_signals(self, df, params):
        signals = []
        
        for i in range(len(df)):
            try:
                if i < 50:
                    signals.append({'date': df.index[i], 'action': 'hold'})
                    continue
                    
                row = df.iloc[i]
                prev_row = df.iloc[i-1]
                
                close_val = float(row['Close'])
                ema_20_val = float(row['EMA_20'])
                ema_50_val = float(row['EMA_50'])
                rsi_val = float(row['RSI'])
                atr_val = float(row['ATR'])
                macd_hist_val = float(row['MACD_Hist'])
                bb_lower_val = float(row['BB_Lower'])
                stoch_k_val = float(row['STOCH_K'])
                stoch_d_val = float(row['STOCH_D'])
                volume_val = float(row['Volume'])
                volume_sma_val = float(row['Volume_SMA'])
                
                # Trend ve momentum koşulları
                trend_ok = ema_20_val > ema_50_val
                rsi_ok = rsi_val < params['rsi_oversold']
                macd_ok = macd_hist_val > 0
                stoch_ok = stoch_k_val < 30 and stoch_d_val < 30
                volume_ok = volume_val > volume_sma_val * 1.1
                near_bb_lower = close_val <= bb_lower_val * 1.02
                
                # Stratejiler
                strategy1 = trend_ok and rsi_ok and near_bb_lower
                strategy2 = trend_ok and stoch_ok and volume_ok
                strategy3 = trend_ok and macd_ok and rsi_ok
                
                buy_signals = [strategy1, strategy2, strategy3]
                confirmed_signals = sum(buy_signals)
                
                buy_signal = confirmed_signals >= params['min_signal_strength']
                
                if buy_signal:
                    stop_loss = close_val - (atr_val * params['atr_multiplier'])
                    take_profit = close_val + (atr_val * params['atr_multiplier'] * params['reward_ratio'])
                    
                    signals.append({
                        'date': df.index[i],
                        'action': 'buy',
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    })
                else:
                    signals.append({
                        'date': df.index[i],
                        'action': 'hold'
                    })
                    
            except Exception as e:
                signals.append({
                    'date': df.index[i],
                    'action': 'hold'
                })
        
        signals_df = pd.DataFrame(signals)
        if not signals_df.empty:
            signals_df = signals_df.set_index('date')
        
        buy_count = len([s for s in signals if s.get('action') == 'buy'])
        st.info(f"🎯 {buy_count} alış sinyali bulundu")
        return signals_df
    
    def run_backtest(self, data, params):
        df = self.calculate_indicators(data)
        signals = self.generate_signals(df, params)
        
        capital = self.initial_capital
        position = None
        trades = []
        equity_curve = []
        max_drawdown = 0
        peak_equity = capital
        
        for date in df.index:
            if date not in signals.index:
                continue
                
            current_price = float(df.loc[date, 'Close'])
            signal = signals.loc[date]
            
            current_equity = capital
            if position is not None:
                current_equity += position['shares'] * current_price
            
            # Drawdown hesaplama
            if current_equity > peak_equity:
                peak_equity = current_equity
            drawdown = (peak_equity - current_equity) / peak_equity * 100
            max_drawdown = max(max_drawdown, drawdown)
            
            equity_curve.append({'date': date, 'equity': current_equity, 'drawdown': drawdown})
            
            if position is None and signal['action'] == 'buy':
                stop_loss = float(signal['stop_loss'])
                risk_per_share = current_price - stop_loss
                
                if risk_per_share > 0:
                    risk_amount = capital * params['risk_per_trade']
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
                        'exit_reason': 'SL'
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
                        'exit_reason': 'TP'
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
                'exit_reason': 'OPEN'
            })
        
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        equity_df = pd.DataFrame(equity_curve)
        
        return trades_df, equity_df, max_drawdown
    
    def calculate_metrics(self, trades_df, equity_df, max_drawdown):
        if trades_df.empty:
            return {
                'total_return': "0.0%",
                'total_trades': "0",
                'win_rate': "0.0%",
                'avg_win': "$0.00",
                'avg_loss': "$0.00",
                'max_drawdown': "0.0%",
                'best_trade': "0.0%",
                'worst_trade': "0.0%"
            }
        
        try:
            initial_equity = self.initial_capital
            final_equity = float(equity_df['equity'].iloc[-1])
            total_return = (final_equity - initial_equity) / initial_equity * 100.0
            
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            win_rate = (winning_trades / total_trades) * 100.0 if total_trades > 0 else 0.0
            
            avg_win = float(trades_df[trades_df['pnl'] > 0]['pnl'].mean()) if winning_trades > 0 else 0.0
            avg_loss = float(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if (total_trades - winning_trades) > 0 else 0.0
            
            best_trade = float(trades_df['return_pct'].max()) if not trades_df.empty else 0.0
            worst_trade = float(trades_df['return_pct'].min()) if not trades_df.empty else 0.0
            
            return {
                'total_return': f"{total_return:+.2f}%",
                'total_trades': str(total_trades),
                'win_rate': f"{win_rate:.1f}%",
                'avg_win': f"${avg_win:.2f}",
                'avg_loss': f"${avg_loss:.2f}",
                'max_drawdown': f"{max_drawdown:.2f}%",
                'best_trade': f"{best_trade:.2f}%",
                'worst_trade': f"{worst_trade:.2f}%"
            }
            
        except:
            return {
                'total_return': "0.0%",
                'total_trades': "0",
                'win_rate': "0.0%",
                'avg_win': "$0.00",
                'avg_loss': "$0.00",
                'max_drawdown': "0.0%",
                'best_trade': "0.0%",
                'worst_trade': "0.0%"
            }

# =========================
# STREAMLIT UYGULAMASI
# =========================
st.set_page_config(page_title="Swing Backtest", layout="wide")
st.title("🚀 Swing Trading Backtest")
st.markdown("**Profesyonel Swing Trading Stratejisi**")

# Sidebar
st.sidebar.header("⚙️ Ayarlar")
ticker = st.sidebar.selectbox("Sembol", ["BTC-USD", "ETH-USD", "TSLA", "NVDA", "AAPL", "GOOGL", "MSFT"])
start_date = st.sidebar.date_input("Başlangıç", datetime(2023, 1, 1))
end_date = st.sidebar.date_input("Bitiş", datetime(2024, 1, 1))

st.sidebar.header("📊 Parametreler")
rsi_oversold = st.sidebar.slider("RSI Aşırı Satım", 25, 45, 35)
atr_multiplier = st.sidebar.slider("ATR Çarpanı", 1.0, 3.0, 1.5)
reward_ratio = st.sidebar.slider("Risk/Ödül Oranı", 1.5, 3.0, 2.0)
risk_per_trade = st.sidebar.slider("Risk %", 1.0, 5.0, 2.0) / 100
min_signal_strength = st.sidebar.slider("Min Sinyal Gücü", 1, 3, 2)

params = {
    'rsi_oversold': rsi_oversold,
    'atr_multiplier': atr_multiplier,
    'reward_ratio': reward_ratio,
    'risk_per_trade': risk_per_trade,
    'min_signal_strength': min_signal_strength
}

# Ana içerik
if st.button("🎯 Backtest Çalıştır", type="primary"):
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
        
        backtester = SwingBacktest()
        
        with st.spinner("Backtest çalıştırılıyor..."):
            trades, equity, max_drawdown = backtester.run_backtest(data, params)
            metrics = backtester.calculate_metrics(trades, equity, max_drawdown)
        
        st.subheader("📊 Performans Özeti")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Toplam Getiri", metrics['total_return'])
            st.metric("Toplam İşlem", metrics['total_trades'])
        
        with col2:
            st.metric("Win Rate", metrics['win_rate'])
            st.metric("Ort. Kazanç", metrics['avg_win'])
        
        with col3:
            st.metric("Ort. Kayıp", metrics['avg_loss'])
            st.metric("Max Drawdown", metrics['max_drawdown'])
        
        with col4:
            st.metric("En İyi İşlem", metrics['best_trade'])
            st.metric("En Kötü İşlem", metrics['worst_trade'])
        
        if not trades.empty:
            st.subheader("📈 Performans Grafikleri")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Equity curve
            ax1.plot(equity['ity['date'], equity['equitydate'], equity['equity'], color='green', line'], color='green', linewidthwidth=2)
            ax1.set=2)
            ax1.set_title('Portföy Değeri')
            ax1.set_ylabel('Equity ($)')
            ax1.grid(_title('Portföy Değeri')
            ax1.set_ylabel('Equity ($)')
            ax1.grid(TrueTrue, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # Drawdown
            ax2.fill, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # Drawdown
            ax2.fill_between(equity['date'], equity['drawdown'], color='red', alpha=0.3)
            ax2.plot(equity['date'], equity['drawdown'], color='red', linewidth=1)
            ax2.set_title('Drawdown')
_between(equity['date'], equity['drawdown'], color='red', alpha=0.3)
            ax2.plot(equity['date'], equity['drawdown'], color='red', linewidth=1)
            ax2.set_title('Draw            ax2.set_ylabel('Drawdown (%)')
            ax2.set_xlabel('Taridown')
            ax2.set_ylabel('Drawdown (%)')
            ax2.set_xlabel('h')
            ax2.grid(True, alpha=0.3)
            ax2.tickTarih')
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x',_params(axis='x', rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.subheader("📋 İşlem rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st Listesi")
            display_trades = trades.copy()
            display_trades['entry_date'] = display_trades['entry_date'].dt.subheader("📋 İşlem Listesi")
            display_trades = trades.copy()
            display_trades['entry_date'] = display_trades['entry_date'].dt.strftime('%Y-%m-%d')
            display_trades['exit_date'] = display_t.strftime('%Y-%m-%d')
            display_trades['exit_date'] = display_trades['exit_date'].dt.strftime('%Y-%m-%d')
            display_trades['pnlrades['exit_date'].dt.strftime('%Y-%m-%d'] = display_trades['pnl'].round(2)
            display_trades['return_pct'] = display_trades')
            display_trades['pnl'] = display_trades['pnl'].round(2)
            display_trades['return_pct'] = display_trades['return_pct'].round(2)
            st.dataframe(display_t['return_pct'].round(2)
            st.dataframe(display_trades)
            
        else:
            st.info("🤷 Hiç işlem gerçekrades)
            
        else:
            st.info("🤷 Hiç işlem gerçekleşmedi.")
            
    except Exception as e:
        st.errorleşmedi.")
            
    except Exception as e:
        st.error(f"❌ Hata: {str(e)}")

st.markdown("---")
st.markdown("**Swing Backtest(f"❌ Hata: {str(e)}")

st.markdown("---")
st.markdown("**Swing Backtest S Sistemi v3.0 | Profesyonel Strateji**")