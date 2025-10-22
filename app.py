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
# BACKTEST MOTORU
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
    
    def generate_signals(self, df, rsi_oversold=40, atr_multiplier=2.0):
        signals = []
        
        for i in range(len(df)):
            try:
                row = df.iloc[i]
                
                close_val = float(row['Close'])
                ema_20_val = float(row['EMA_20'])
                ema_50_val = float(row['EMA_50'])
                rsi_val = float(row['RSI'])
                atr_val = float(row['ATR'])
                
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
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    })
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
        
        return signals_df
    
    def run_backtest(self, data, rsi_oversold=40, atr_multiplier=2.0, risk_per_trade=0.02):
        df = self.calculate_indicators(data)
        signals = self.generate_signals(df, rsi_oversold, atr_multiplier)
        
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
        
        return trades_df, equity_df
    
    def calculate_metrics(self, trades_df, equity_df):
        if trades_df.empty:
            # Basit değerler döndür - HİÇ PANDAS SERIES YOK
            return {
                'total_return': "0.0%",
                'total_trades': "0",
                'win_rate': "0.0%",
                'avg_win': "$0.00",
                'avg_loss': "$0.00"
            }
        
        try:
            initial_equity = 10000.0
            final_equity = = float(equity_df[' float(equity_df['equity'].iloc[-equity'].iloc[-1])
            total_return =1])
            total_return = ( (final_equity -final_equity - initial_ initial_equity) /equity) / initial_ initial_equity * 100.equity * 100.0
            
            total_trades0
            
            total_trades = len(trades_df)
 = len(trades_df)
            winning_trades = len            winning_trades = len(t(trades_df[trades_df['rades_df[trades_df['pnpnl'] > l'] > 00])
            win_rate =])
            win_rate = ( (winwinning_trades / total_trades) *ning_trades / total_trades) * 100.0 if total_trades > 0 100.0 if total_trades > else  0 else 0.0
            
            avg_win0.0
            
            avg_win = float(trades = float(trades_df[trades_df_df[trades_df['pn['pnl'] > 0]['pnl'].mean()) if winning_trades >l'] > 0]['pnl'].mean()) if winning_trades > 0 else  0 else 0.0.0
            avg0
            avg_loss =_loss = float(trades_df float(trades_df[trades[trades_df['pnl'] < 0]['pnl'].mean()) if (_df['pnl'] < 0]['pnl'].mean()) if (total_trades - winning_tradestotal_trades - winning_trades) > 0 else ) > 0 else 0.0
            
            # H0.0
            
            # HATAATA Ç ÇÖZÖZÜMÜ: Tüm değÜMÜ: Tüm değerleri stringerleri string olarak döndür
            olarak döndür
            return {
 return {
                'total_return': f"{round                'total_return': f"{round(total_return, (total_return, 2)}%",
                'total2)}%",
                'total_t_trades': str(total_trades': str(total_trades),
rades),
                'win_rate                'win_rate': f"{round(win_rate': f"{round(win_rate,, 1)}%",
                1)}%",
                'avg_win': f 'avg_win': f"${"${round(avg_round(avg_win,win, 2)}",
 2)}",
                '                'avg_loss': favg_loss': f"${"${round(avg_lossround(avg_loss, , 2)}"
            }
2)}"
            }
            
            
        except:
        except:
            return            return {
                'total {
                'total_return':_return': "0.0 "0.0%",
%",
                'total_t                'total_trades':rades': "0",
                ' "0",
                'win_ratewin_rate':': "0.0 "0.0%",
                'avg_%",
                'avg_win':win': "$0.00 "$0.00",
               ",
                'avg_loss': "$ 'avg_loss': "$00.00"
            }

#.00"
            }

# =========================
# STREAML =========================
# STREAMLIT UYGIT UYGULAMASI
# =========================
st.set_page_config(page_titleULAMASI
# =========================
st.set_page_config(page_title="Swing Backtest",="Swing Backtest", layout=" layout="widewide")
st.title("🚀 Swing Trading")
st.title("🚀 Swing Trading Backtest")

 Backtest")

# Sidebar
st.sidebar.header("# Sidebar
st.sidebar.header("⚙️ Ay⚙️ Ayarlararlar")
ticker =")
ticker = st.sidebar.selectbox("Sembol", st.sidebar.selectbox("Sembol", ["AAPL", ["AAPL", "GOOG "GOOGL",L", "MSFT", " "MSFT", "TSTSLA", "BTC-USDLA", "BTC-USD", "ETH-USD"])
", "ETH-USD"])
start_datestart_date = st.side = st.sidebar.date_input("Başlangıbar.date_input("Başlangıç", datetime(202ç", datetime(2023, 3, 1, 11, 1))
end))
end_date =_date = st.s st.sidebar.date_inputidebar.date_input("Bitiş", datetime(202("Bitiş", datetime(2023,3, 12 12, 31))

, 31))

st.sst.sidebar.header("idebar.header("📊📊 Parametreler")
 Parametreler")
rsi_rsi_oversold =oversold = st st.sidebar.slider.sidebar.slider("RS("RSI AşırI Aşırı Satımı Satım", 25", 25, , 50,50, 40)
at 40)
atr_multr_multiplier = stiplier = st.side.sidebar.sliderbar.slider("ATR Ç("ATR Çarpanarpanı", 1.ı", 1.00, 3.0, 3.0, 2, 2.0)
risk.0)
risk_per_t_per_trade = st.srade = st.sidebaridebar.slider(".slider("Risk %", Risk %", 1.0, 5.1.0, 5.0,0, 2.0) / 2.0) / 100

 100

# Ana içerik
if# Ana içerik
if st.button st.button("🎯 Backtest Çal("🎯 Backtest Çalıştır"):
    try:
       ıştır"):
    try:
        with st with st.spinner("Ver.spinner("Veri yi yükleniyorükleniyor..."):
..."):
            data = yf.download(ticker            data = yf.download(ticker, start=, start=start_date, endstart_date, end=end=end_date, progress=False)
            
_date, progress=False)
            
                       if if data.empty data.empty:
                st.error("❌ Veri bulun:
                st.error("❌ Veriamadı")
                st.stop()
            
 bulunamadı")
                st.stop()
            
            st.success(f"✅ {len            st.success(f"✅ {len(data)} günlük(data)} günlük veri yü veri yüklendi")
        
        backtklendi")
        
        backtester = SwingBacktest()
ester = SwingBacktest()
        
        with st.sp        
        with st.spinner("Backtest çalıştinner("Backtest çalıştırılıyor..."):
ırılıyor..."):
            trades, equity = backt            trades, equity = backtester.run_backtest(data, rsi_oversold, atr_multiplier, riskester.run_backtest(data, rsi_oversold, atr_multiplier, risk_per_trade)
            metrics_per_trade)
            metrics = back = backtester.calculatetester.calculate_metrics_metrics(trades, equity(trades, equity)
        
)
        
        st.subheader("        st.subheader("📊 Performans Özeti📊 Performans Özeti")
")
        col1, col        col1, col2,2, col3 = st col3 = st.columns(.columns(3)
        
        with3)
        
        with col col1:
            # ARTIK SADECE STRING1:
            # ARTIK SADECE STRING DEĞERLER DEĞERLER
            st.metric("
            st.metric("Toplam Getiri", metricsToplam Getiri", metrics['['total_return'])
            st.mtotal_return'])
            st.metric("Toplametric("Toplam İ İşlem", metrics['şlem", metrics['total_tradestotal_trades'])
        
        with col'])
        
        with col2:
2:
            st.metric            st.metric("Win Rate",("Win Rate", metrics['win_rate metrics['win_rate'])
            st'])
            st.metric(".metric("Ort.Ort. Kazanç Kazanç", metrics['avg", metrics['avg_win_win'])
        
        with'])
        
        with col3 col3:
            st.m:
            st.metric("etric("Ort. KayOrt. Kayıpıp", metrics['avg_loss", metrics['avg_loss'])
        
       '])
        
        if not trades if not trades.empty.empty:
            st.subheader(":
            st.subheader("📈📈 Performans Grafik Performans Grafikleri")
leri")
            
            fig, ax            
            fig, ax = = plt.subplots(f plt.subplots(figsizeigsize=(12,=(12, 6 6))
))
            ax.plot            ax.plot(equ(equity['date'], equity['equity['date'], equity['equity'], color='green', linewidth=2)
ity'], color='green', linewidth=2            ax.set_title('Portfö)
            ax.set_title('Portföy Değeri')
           y Değeri')
            ax.set_ylabel('Equ ax.set_ylabel('Equity ($)')
            ax.grid(True, alpha=0.ity ($)')
            ax.grid(True, alpha=0.3)
            st.pyplot(f3)
            st.pyplot(fig)
            
            st.subig)
            
            st.subheader("📋 İşlemheader("📋 İşlem Listesi")
            display_trades Listesi")
            display_trades = trades.copy()
 = trades.copy()
            display_trades            display_trades['entry_date'] = display['entry_date'] = display_trades['entry_date'].dt.strftime_trades['entry_date'].dt('%Y-%m-%d')
            display_trades['exit.strftime('%Y-%m-%d')
            display_trades['exit_date']_date'] = display = display_trades['exit_date']._trades['exit_date'].dt.strftime('%Y-%m-%ddt.strftime('%Y-%m-%d')
            st.data')
            st.dataframe(frame(display_trades)
            
        else:
           display_trades)
            
        else:
            st.info st.info("🤷 Hi("🤷 Hiç iç işlem gerçşlem gerçekleşekleşmedi.")
            
medi.")
            
    except Exception as    except Exception as e:
        e:
        st.error(f st.error(f"❌ H"❌ Hata: {str(e)}")

stata: {str(e)}")

st.markdown.markdown("("---")
st.markdown("**Back---")
st.markdown("**Backtest Stest Sistemi**")