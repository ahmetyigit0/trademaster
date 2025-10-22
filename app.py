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
                    risk_pct = 0.02  # %2 risk
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
            total_return['equity'].iloc[-1]
            total_return = (final_equity - = (final_equity - initial_equity) / initial initial_equity) / initial_equity * _equity * 100
            
            total_trades = len(t100
            
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] >rades_df)
            winning_trades = len(trades_df[trades 0])
            win_rate_df['pnl'] > 0])
            win_rate = ( = (winning_trades /winning_trades / total total_trades) * 100 if total_trades > _trades) * 100 if total_trades > 0 else0 else 0
            
            avg_win = trades_df[trades 0
            
            avg_win = trades_df[trades_df_df['pnl'] > ['pnl'] > 00]['pnl'].mean() if winning_trades]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss > 0 else 0
            avg_loss = trades_df[trades_df = trades_df[trades_df['pnl'] < 0['pnl'] < 0][']['pnl'].mean() if (total_tpnl'].mean() if (total_trades - winning_trades) > 0rades - winning_trades) > 0 else  else 0
            
            best_t0
            
            best_trade = tradesrade = trades_df['return_p_df['return_pct'].max() if not trades_df.empty else 0
            worstct'].max() if not trades_df.empty else 0
            worst_trade = trades_df['return_pct'].min() if not trades_df.empty else _trade = trades_df['return_pct'].min() if not trades_df.empty else 0
            
0
            
            return {
                           return {
                'total_return': f"{total_return 'total_return': f"{total_return:+.2f:+.2f}%",
                'total_trades': str}%",
                'total_trades': str(total_trades),
                'win_rate': f"{win_rate:.1f}%(total_trades),
                'win_rate': f"{win_rate:.1f}%",
               ",
                'avg_win': f 'avg_win': f"${"${avg_win:.avg_win:.2f2f}",
                '}",
                'avg_lossavg_loss': f"${': f"${avg_loss:.2f}",
               avg_loss:.2 'best_trade': f"{best_trade:.2f}%",
                'worst_trade': f"{worf}",
                'best_trade': f"{best_trade:.2f}%",
                'worst_trade': f"{worst_trade:.2st_trade:.2f}%f}%"
            }
            
       "
            }
            
        except:
            return {
 except:
            return {
                '                'total_return': "0total_return': "0.0.0%",
                'total%",
                'total_trades_trades': "0",
': "0",
                '                'win_ratewin_rate': "0.0%': "0.0%",
                'avg",
                'avg_win_win': "$0.': "$0.00",
00",
                'avg_loss                'avg_loss': "$0.00",
                'best_trade': "0.0%",
                'worst_trade': "$0.00",
                'best_trade': "0.0%",
                'worst_trade': "0.0%': "0.0%"
"
            }

# =========================
            }

# =========================
# STREAMLIT UYG# STREAMLIT UYGULAMASI
# =ULAMASI
# =========================
st.set_page_config(page========================
st.set_page_config(page_title_title="Swing Backtest",="Swing Backtest", layout="wide")
st.title("🚀 Swing Trading layout="wide")
st.title("🚀 Swing Trading Backtest")
st.markdown("**3 İndikatörlü Basit & Etkili Strateji - EMA, RSI, Price Channels**")

# Sidebar
st.sidebar.header("⚙️ Ayarlar")
ticker = st.sidebar.selectbox("Sembol", ["BTC-USD", "ETH-USD", "TSLA", "NVDA", "AAPL", "GOOGL", "MSFT"])
start_date = st.sidebar.date_input("Başlangıç", datetime(2023, 1, 1))
end_date = Backtest")
st.markdown("**3 İndikatörlü Basit & Etkili Strateji - EMA, RSI, Price Channels**")

# Sidebar
st.sidebar.header("⚙️ Ayarlar")
ticker = st.sidebar.selectbox("Sembol", ["BTC-USD", "ETH-USD", "TSLA", "NVDA", "AAPL", "GOOGL", "MSFT"])
start_date = st.sidebar.date_input("Başlangıç", datetime(2023, 1, 1))
end_date = st.sidebar.date st.sidebar.date_input("Biti_input("Bitiş", datetimeş", datetime(2024, 1, 1))

st(2024, 1, 1))

st.side.sidebar.header("bar.header("📊 Paramet📊 Parametrereler")
rsi_oversler")
rsi_oversold = st.sold = st.sideidebar.slider("RSbar.slider("RSI Aşırı SatI Aşırı Satım", 25, 45ım", 25, 45, 35)
reward_ratio =, 35)
reward_ratio = st.sidebar st.sidebar.slider(".slider("Risk/ÖdRisk/Ödül Oranül Oranı", 1.ı", 1.5,5, 3 3.0, .0, 2.0)
risk2.0)
risk_per_trade_per_trade = st.side = st.sidebar.slider("Risk %", bar.slider("Risk %", 1.0,1.0, 5. 5.0,0, 2.0) / 100

params = 2.0) / 100

params = {
    'rsi {
    'rsi_overs_oversold': rsiold': rsi_overs_oversold,
    'reward_ratio': reward_ratioold,
    'reward_ratio': reward_ratio,
    ',
    'risk_per_trade': risk_per_trade
}

# Ana içerik
ifrisk_per_trade': risk_per_trade
}

# Ana içerik
if st.button st.button("🎯 Back("🎯 Backtest Çalıştır", type="primarytest Çalıştır", type="primary"):
    try:
"):
    try:
        with        with st.spinner("Ver st.spinner("Veri yi yükleniyükleniyor..."):
            extendedor..."):
            extended_start = start_start = start_date - timedelta(days_date - timedelta(days=100)
            data = yf=100)
            data = yf.download(ticker, start=extended_start, end.download(ticker, start=extended_start, end=end_date, progress=False)
            
            if data.empty:
               =end_date, progress=False)
            
            if data.empty:
                st st.error("❌ Veri.error("❌ Veri bulunamadı")
                st.stop()
            
            data = data[ bulunamadı")
                st.stop()
            
            data = data[datadata.index >= pd.to_datetime.index >= pd.to_datetime(start_date(start_date)]
            data = data[data.index <= pd.to_dat)]
            data = data[data.index <= pd.to_datetimeetime(end_date)]
            
           (end_date)]
            
            st.success st.success(f"✅ {(f"✅ {len(datalen(data)} günlük)} günlük veri y veri yüklendi")
üklendi")
        
        backt        
        backtester = SwingBacktest()
        
        with st.spinner("Backtestester = SwingBacktest()
        
        with st.spinner("Backtest çalışt çalıştırıırılıyor..."):
lıyor..."):
            trades, equity            trades, equity = backtester.run_backtest(data, params = backtester.run_backtest(data, params)
            metrics = backt)
            metrics = backtester.calculate_ester.calculate_metrics(trades, equity)
        
        st.subheader("📊 Performansmetrics(trades, equity)
        
        st.subheader("📊 Performans Özeti")
 Özeti")
        col1        col1, col2, col2, col3,, col3, col4 col4 = st.columns( = st.columns(4)
        
       4)
        
        with col1 with col1:
            st.metric("Top:
            st.metric("Toplam Getiri",lam Getiri", metrics metrics['total_return'])
            st['total_return'])
            st.metric(".metric("Toplam İşlem", metrics['total_trades'])
        
       Toplam İşlem", metrics['total_trades'])
        
        with col2:
            with col2:
            st.metric("Win st.metric("Win Rate", Rate", metrics['win_rate metrics['win_rate'])
           '])
            st.metric("Ort st.metric("Ort. Kazanç", metrics['avg. Kazanç", metrics['avg_win'])
        
_win'])
        
        with        with col3:
            col3:
            st.metric st.metric("O("Ort. Kayırt. Kayıp", metrics['avgp", metrics['avg_loss'])
            st.metric_loss'])
            st.metric("En İyi İşlem",("En İyi İşlem", metrics['best_trade'])
        
        metrics['best_trade'])
        
        with col4:
            st with col4:
            st.m.metric("En Kötü İşlemetric("En Kötü İşlem", metrics['wor", metrics['worst_tst_trade'])
        
       rade'])
        
        if not if not trades.empty trades.empty:
:
            st.sub            st.subheader("📈 Performans Grafikleri")
            
            fig, axheader("📈 Performans Grafikleri")
            
            fig, ax = plt.subplots(figsize=(12 = plt.subplots(figsize=(12, 6))
, 6))
            ax            ax.plot(equity['date'], equity['equity'],.plot(equity['date'], equity['equity'], color='green', linewidth=2 color='green', linewidth=2)
            ax.set_title(')
            ax.set_title('PortPortföy Değeri')
            ax.set_ylabelföy Değeri')
            ax.set_ylabel('Equity ($)')
            ax.grid(('Equity ($)')
            axTrue, alpha=0.3)
            st.pyplot(fig.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            st.sub)
            
            st.subheader("📋 İşlem Listheader("📋 İşlem Listesi")
            display_tradesesi")
            display_trades = trades.copy()
            display_t = trades.copy()
            display_tradesrades['entry_date']['entry_date'] = display_trades['entry_date'].dt.str = display_trades['entry_date'].dt.strftime('%Y-%m-%d')
            display_tradesftime('%Y-%m-%d')
            display_trades['exit_date'] = display_t['exit_date'] = display_tradesrades['exit_date'].dt.str['exit_date'].dt.strftime('%Y-%m-%dftime('%Y-%m-%d')
            display_trades['pn')
            display_trades['pnl'] = display_tradesl'] = display_trades['pnl'].round(['pnl'].round(2)
            display_trades['return2)
            display_trades['return_pct'] = display_t_pct'] = display_trades['return_pct'].round(rades['return_pct'].round(2)
            st.dataframe2)
            st.dataframe(display_trades)
            
(display_trades)
            
        else:
            st.info("🤷 Hiç işlem        else:
            st.info(" gerçekleşmedi. Parametreleri değişt🤷 Hiç işlem gerçekleşmedi. Parametreleri değiştirmeyirmeyi deneyin.")
            
    except Exception as ei deneyin.")
            
    except Exception as e:
       :
        st.error(f" st.error(f"❌ Hata: {str(e)}")

st.markdown("---")
st.markdown("**Backtest Sistemi v3.0 | 3 Güvenilir İndikatör**")