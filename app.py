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
# BACKTEST MOTORU - SADELEŞTİRİLMİŞ
# =========================
class SwingBacktest:
    def __init__(self):
        self.commission = 0.001
        self.initial_capital = 10000
    
    def calculate_indicators(self, df):
        df = df.copy()
        
        try:
            # SADECE 3 TEMEL GÖSTERGE - HEPsi GÜVENLİ
            
            # 1. EMA'lar - Çok güvenilir
            df['EMA_20'] = df['Close'].ewm(span=20).mean()
            df['EMA_50'] = df['Close'].ewm(span=50).mean()
            
            # 2. RSI - Basit ve etkili
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # 3. Basit Price Channels (En basit hali)
            df['Channel_High'] = df['High'].rolling(window=20).max()
            df['Channel_Low'] = df['Low'].rolling(window=20).min()
            
            # NaN değerleri temizle
            df = df.fillna(method='bfill').fillna(method='ffill')
            
            return df
            
        except Exception as e:
            st.error(f"Gösterge hesaplama hatası: {e}")
            # Acil durum göstergeleri
            df['EMA_20'] = df['Close']
            df['EMA_50'] = df['Close']
            df['RSI'] = 50
            df['Channel_High'] = df['Close'] * 1.1
            df['Channel_Low'] = df['Close'] * 0.9
            return df
    
    def generate_signals(self, df, params):
        signals = []
        
        for i in range(len(df)):
            try:
                if i < 20:
                    signals.append({'date': df.index[i], 'action': 'hold'})
                    continue
                    
                row = df.iloc[i]
                
                # Basit değer atamaları
                close = row['Close']
                ema_20 = row['EMA_20']
                ema_50 = row['EMA_50']
                rsi = row['RSI']
                channel_low = row['Channel_Low']
                
                # BASİT ve ETKİLİ SİNYAL KOŞULLARI
                
                # 1. Trend koşulu
                trend_up = ema_20 > ema_50
                
                # 2. Momentum koşulu
                rsi_oversold = rsi < params['rsi_oversold']
                
                # 3. Destek seviyesi
                near_support = close <= channel_low * 1.02
                
                # ÇOK BASİT STRATEJİ: Trend + Oversold + Destek
                buy_signal = trend_up and rsi_oversold and near_support
                
                if buy_signal:
                    # Basit risk yönetimi (ATR yerine yüzde bazlı)
                    risk_pct = 0.02  %2 risk
                    stop_loss = close * (1 - risk_pct)
                    take_profit = close * (1 + (risk_pct * params['reward_ratio']))
                    
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
        
        for date in df.index:
            if date not in signals.index:
                # Sinyal yoksa equity'i güncelle
                current_price = df.loc[date, 'Close']
                current_equity = capital
                if position is not None:
                    current_equity += position['shares'] * current_price
                equity_curve.append({'date': date, 'equity': current_equity})
                continue
                
            current_price = df.loc[date, 'Close']
            signal = signals.loc[date]
            
            current_equity = capital
            if position is not None:
                current_equity += position['shares'] * current_price
            
            equity_curve.append({'date': date, 'equity': current_equity})
            
            if position is None and signal['action'] == 'buy':
                stop_loss = signal['stop_loss']
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
                            'take_profit': signal['take_profit']
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
            last_price = df['Close'].iloc[-1]
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
        
        return trades_df, equity_df
    
    def calculate_metrics(self, trades_df, equity_df):
        if trades_df.empty:
            return {
                'total_return': "0.0%",
                'total_trades': "0",
                'win_rate': "0.0%",
                'avg_win': "$0.00",
                'avg_loss': "$0.00",
                'best_trade': "0.0%",
                'worst_trade': "0.0%"
            }
        
        try:
            initial_equity = self.initial_capital
            final_equity = equity_df['equity'].iloc[-1]
            total_return = (final_equity - initial_equity) / initial_equity * 100
            
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if (total_trades - winning_trades) > 0 else 0
            
            best_trade = trades_df['return_pct'].max() if not trades_df.empty else 0
            worst_trade = trades_df['return_pct'].min() if not trades_df.empty else 0
            
            return {
                'total_return': f"{total_return:+.2f}%",
                'total_trades': str(total_trades),
                'win_rate': f"{win_rate:.1f}%",
                'avg_win': f"${avg_win:.2f}",
                'avg_loss': f"${avg_loss:.2f}",
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
                'best_trade': "0.0%",
                'worst_trade': "0.0%"
            }

# =========================
# STREAMLIT UYGULAMASI
# =========================
st.set_page_config(page_title="Swing Backtest", layout="wide")
st.title("🚀 Swing Trading Backtest")
st.markdown("**3 İndikatörlü Basit & Etkili Strateji - EMA, RSI, Price Channels**")

# Sidebar
st.sidebar.header("⚙️ Ayarlar")
ticker = st.sidebar.selectbox("Sembol", ["BTC-USD", "ETH-USD", "TSLA", "NVDA", "AAPL", "GOOGL", "MSFT"])
start_date = st.sidebar.date_input("Başlangıç", datetime(2023, 1, 1))
end_date = st.sidebar.date_input("Bitiş", datetime(2024, 1, 1))

st.sidebar.header("📊 Parametreler")
rsi_oversold = st.sidebar.slider("RSI Aşırı Satım", 25, 45, 35)
reward_ratio = st.sidebar.slider("Risk/Ödül Oranı", 1.5, 3.0, 2.0)
risk_per_trade = st.sidebar.slider("Risk %", 1.0, 5.0, 2.0) / 100

params = {
    'rsi_oversold': rsi_oversold,
    'reward_ratio': reward_ratio,
    'risk_per_trade': risk_per_trade
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
            trades, equity = backtester.run_backtest(data, params)
            metrics = backtester.calculate_metrics(trades, equity)
        
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
            st.metric("En İyi İşlem", metrics['best_trade'])
        
        with col4:
            st.metric("En Kötü İşlem", metrics['worst_trade'])
        
        if not trades.empty:
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
            displayd')
            display_trades_trades['pnl'] = display_trades['pnl'].round(2['pnl'] = display_trades['pnl'].round(2)
            display)
            display_trades['return_pct'] = display_trades['_trades['return_pct'] = display_trades['return_preturn_pct'].round(2)
            st.dataframe(display_trades)
            
        else:
            st.info("🤷 Hiç işlem gerçekleşmedi. Parametct'].round(2)
            st.dataframe(display_trades)
            
        else:
            st.info("🤷 Hiç işlem gerçekleşmedi. Parametreleri değiştirmreleri değiştirmeyieyi deneyin.")
            
    deneyin.")
            
    except Exception except Exception as e:
        as e:
        st.error(f" st.error(f"❌ H❌ Hata: {ata: {str(e)}")

str(e)}")

st.markdown("---")
st.markdown("st.markdown("---")
st.markdown("**Backtest Sistemi**Backtest Sistemi v3.0 v3.0 | 3 | 3 Güvenilir Güvenilir İndikat İndikatör**")
ör**")