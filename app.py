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
# BACKTEST MOTORU - TAM ÇALIŞAN
# =========================
class SwingBacktest:
    def __init__(self):
        self.commission = 0.001
    
    def calculate_indicators(self, df):
        """Hatasız indikatör hesaplama"""
        try:
            df = df.copy()
            
            # EMA'lar
            df['EMA_20'] = df['Close'].ewm(span=20, min_periods=1).mean()
            df['EMA_50'] = df['Close'].ewm(span=50, min_periods=1).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # ATR - Basit ve güvenli versiyon
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift(1))
            low_close = np.abs(df['Low'] - df['Close'].shift(1))
            
            # Tek boyutlu array oluştur
            true_range_values = []
            for i in range(len(df)):
                if i == 0:
                    true_range_values.append(high_low.iloc[i])
                else:
                    tr = max(high_low.iloc[i], high_close.iloc[i], low_close.iloc[i])
                    true_range_values.append(tr)
            
            df['ATR'] = pd.Series(true_range_values, index=df.index).rolling(window=14, min_periods=1).mean()
            
            # NaN değerleri temizle
            df = df.fillna(method='bfill').fillna(method='ffill')
            
            return df
            
        except Exception as e:
            st.error(f"İndikatör hatası: {e}")
            return df
    
    def generate_signals(self, df, rsi_oversold=30, atr_multiplier=2.0):
        """Basit sinyal üretimi"""
        try:
            signals = []
            
            for i in range(len(df)):
                try:
                    row = df.iloc[i]
                    
                    # Değerleri güvenli şekilde al
                    ema_20 = float(row['EMA_20'])
                    ema_50 = float(row['EMA_50'])
                    rsi = float(row['RSI'])
                    close = float(row['Close'])
                    atr = float(row['ATR'])
                    
                    # Sinyal koşulları
                    trend_condition = ema_20 > ema_50
                    rsi_condition = rsi < rsi_oversold
                    price_condition = close > ema_20
                    
                    buy_signal = trend_condition and rsi_condition and price_condition
                    
                    if buy_signal:
                        stop_loss = close - (atr * atr_multiplier)
                        take_profit = close + (atr * atr_multiplier * 2)
                        
                        signals.append({
                            'date': df.index[i],
                            'action': 'buy',
                            'price': close,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit
                        })
                    else:
                        signals.append({
                            'date': df.index[i],
                            'action': 'hold',
                            'price': close,
                            'stop_loss': 0,
                            'take_profit': 0
                        })
                        
                except Exception as e:
                    # Hata durumunda hold sinyali
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
            
            # Backtest
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
                        
                        # Stop-loss
                        if current_price <= position['stop_loss']:
                            exit_reason = 'SL'
                            exit_price = position['stop_loss']
                        # Take-profit
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
                    'pnl': p['entry_price'],
                    'exit_price': last_price,
nl,
                    'return_p                    'shares': position['shares'],
                    'pnl': pnl,
                    'return_pct': (pnl /ct': (pnl / entry entry_value) * 100_value) * 100 if entry if entry_value > _value > 0 else 0 else 0,
                    '0,
                    'exitexit_reason': 'OPEN',
                    'hold_days_reason': 'OPEN',
                    'hold_days':': (df.index[-1 (df.index[-1] -] - position['entry_date position['entry_date']).']).days
days
                })
                })
            
            trades_df            
            trades_df = pd = pd.DataFrame(trades).DataFrame(trades) if trades if trades else pd.DataFrame()
 else pd.DataFrame()
            equity            equity_df =_df = pd.DataFrame(equity_curve)
            
 pd.DataFrame(equity_curve)
            
            return trades_df            return trades_df, equity_df
            
        except, equity_df
            
        except Exception Exception as e as e:
            st.error(f":
            st.error(f"Backtest hatasıBacktest hatası: {e}")
: {e}")
            return            return pd.DataFrame(), pd pd.DataFrame(), pd.DataFrame()
.DataFrame()
    
    
    def calculate    def calculate_metrics(self_metrics(self, trades_df, equity_df):
, trades_df, equity_df):
        """        """Performans metrikPerformans metriklerileri"""
        if"""
        if trades_df.empty or equity_df.empty:
 trades_df.empty or equity_df.empty:
            return            return {
                'total_return {
                'total_return_%_%': 0': 0,
               ,
                'total_trades': 0,
                'win_rate_%': 'total_trades': 0,
                'win_rate_%': 0 0,
                'avg,
                'avg_win_win': 0,
': 0,
                'avg                'avg_loss': _loss': 0,
0,
                'max_d                'max_drawdownrawdown_%_%': 0
           ': 0
            }
 }
        
        try:
                   
        try:
            # Top # Toplam getiri
lam getiri
            initial            initial_equity =_equity = 100 10000
            final00
            final_equity_equity = equity_df = equity_df['equity['equity'].il'].iloc[-1]
            totaloc[-1]
            total_return = (_return = (final_equity - initialfinal_equity - initial_equity_equity) / initial_equity) / initial_equity * 100
            
 * 100
            
            # Trade istatistikleri            # Trade istatistikleri
            total
            total_trades = len(trades_df)
            winning_trades = len(trades_df)
            winning_trades = len_trades = len(trades_df[trades_df['pn(trades_df[trades_df['pnl']l'] >  > 0])
            win0])
            win_rate =_rate = (win (winning_trades /ning_trades / total_trades) * 100 if total total_trades) * 100 if total_trades > 0 else _trades > 0 else 0
            
            avg_win = trades0
            
            avg_win = trades_df[trades_df_df[trades_df['pnl'] > 0]['pnl['pnl'] > 0]['pnl'].mean() if'].mean() if winning_t winning_trades > rades > 00 else else  0
           0
            avg_loss avg_loss = trades_df[trades_df['pn = trades_df[trades_df['pnl']l'] < 0]['pnl']. < 0]['pnl'].mean()mean() if (total_trades if (total_trades - - winning_trades) > winning_trades) >  0 else 0
            
           0 else 0
            
            # Drawdown
            # Drawdown
            equity equity_series = equity_df_series = equity_df.set.set_index('date')['equ_index('date')['equity']
            rolling_maxity']
            rolling_max = = equity_series.expanding(). equity_series.expanding().max()
            drawdownmax()
            drawdown = = (equity_series - (equity_series - rolling_max) / rolling_max rolling_max) / rolling_max * 100
            max * 100
            max_drawdown = drawdown_drawdown = drawdown.min()
            
            return {
                'total_return_%': round(total_return, .min()
            
            return {
2),
                'total_t                'total_return_%': round(total_return, 2),
                'total_trades': total_trades': total_trades,
rades,
                'win_rate                'win_rate_%_%': round(win_rate,': round(win_rate, 1),
                'avg 1),
                'avg_win': round(avg_win': round(avg__win, 2),
win, 2),
                               'avg_loss': round 'avg_loss': round((avg_loss, 2avg_loss, 2),
),
                'max                'max_drawdown_%': round(max_drawdown_%': round(max_d_drawdown, 2rawdown, 2)
)
            }
            
        except            }
            
        except Exception as e:
            return Exception as e:
            return {
 {
                'total_return_                'total_return_%': 0,
               %': 0,
                'total_trades':  'total_trades': 0,
0,
                'win_rate                'win_rate_%': _%': 0,
0,
                'avg_                'avg_win':win': 0,
 0,
                'avg                'avg_loss':_loss': 0,
                0,
                'max_d 'max_drawrawdown_%':down_%': 0
            0
            }

# = }

# =========================
#========================
# STREAMLIT STREAMLIT UYG UYGULAMASI
# =========================
st.set_page_config(page_title="Swing Backtest", layout="wide")
ULAMASI
# =========================
st.set_page_config(page_title="Swing Backtest", layout="wide")
stst.title(".title("🚀 Swing Trading🚀 Swing Trading Backtest")
st.markdown Backtest")
st.markdown("**Tam("**Tam Çalışan Versiyon Çalışan Versiyon**")

#**")

# Sidebar
st.side Sidebar
st.sidebar.headerbar.header("⚙️ Ayarlar")
("⚙️ Ayarlar")
ticker =ticker = st.sidebar.selectbox("S st.sidebar.selectbox("Sembol",embol", ["AAPL", "GOOGL ["AAPL", "GOOGL", "", "MSFT", "MSFT", "TSLATSLA", "BTC-", "BTC-USD",USD", "ETH-USD "ETH-USD"])
start"])
start_date = st.side_date = st.sidebar.date_input("Babar.date_input("Başlangşlangıç", datetimeıç", datetime(202(2023, 13, 1, , 1))
end_date1))
end_date = st = st.sidebar.date.sidebar.date_input("_input("Bitiş",Bitiş", datetime( datetime(2023, 12, 31))

st.sidebar.header("2023, 12, 31))

st.sidebar.header("📊 Parametre📊 Parametreler")
rsi_oversler")
rsi_oversold = st.sidebar.sold = st.sidebarlider(".slider("RSI AşırıRSI Aşır Satım", 20, 40ı Satım", 20, 40, 30)
atr_multipl, 30)
atr_multiplier = st.sidebarier = st.sidebar.slider(".slider("ATR ÇarATR Çarpanı", 1.panı", 1.0, 3.0, 2.0, 3.0, 2.0)
0)
risk_per_trade = st.sidebar.slider("Risk %", 1.0, risk_per_trade = st.sidebar.slider("Risk %", 1.0, 5.0, 2.0) /5.0, 2.0) / 100

 100# Ana içerik
if st.button("

# Ana içerik
if st.button("🎯 Backtest Çalıştır🎯 Backtest Çalışt", type="primary"):
    try:
       ır", type="primary"):
    try:
        with st.spinner("Veri with st.spinner("Veri yükleniyor yükleniyor..."):
            # Daha..."):
            # Daha k kısa tarihısa tarih ar aralığı ile başalığı ile başlala
            data =
            data = yf yf.download(t.download(ticker,icker, start=start_date, end start=start_date, end==end_date, progress=False)
            
            if dataend_date, progress=False)
            
            if data.empty:
                st.empty:
                st.error.error("❌ Veri bulun("❌ Veri bulunamadı")
                stamadı")
                st.stop()
.stop()
            
            st.success            
            st.success(f"(f"✅ {len(data✅ {len(data)} günlük)} günl veri yüklendi")
        
       ük veri yüklendi")
 # Backtest çalış        
        # Backtest çalıştır
       tır
        backt backtester = SwingBackester = SwingBacktest()
test()
        
        with st        
        with st.spinner.spinner("Backtest ç("Backtest çalışalıştırılıtırılıyor..."yor..."):
            trades,):
            trades, equity = equity = backtester.run backtester.run_backtest_backtest(data, rsi(data, rsi_overs_oversold, atr_multiplier, risk_perold, atr_multiplier, risk_per_trade)
           _trade)
            metrics = metrics = backtester.c backtester.calculate_alculate_metrics(trades,metrics(trades, equity)
 equity)
        
        # Sonu        
        # Sonuçlarçlar
        st.subheader
        st.subheader("("📊 Performans Ö📊 Performans Özetizeti")
        col1")
        col1, col, col2, col32, col3 = st = st.columns(3)
.columns(3)
        
               
        with col1:
 with col1:
                       st.metric st.metric("Top("Toplam Getiri",lam Getiri", f"{ f"{metrics['total_returnmetrics['total_return_%_%']}%")
']}%")
            st            st.metric("Toplam İşlem", f"{metrics['total_trades.metric("Toplam İşlem", f"{metrics['total_trades']}")
']}")
        
        with col        
        with col2:
2:
            st.metric            st.metric("Win("Win Rate", f"{ Rate", f"{metrics['win_rate_%metrics['win_rate_%']}%")
            st']}%")
            st.metric("Ort..metric("Ort. Kazanç", f Kazanç", f"${metrics['avg_win"${metrics['avg_win']:.2f}")
        
']:.2f}")
        
        with col3:
                   with col3:
            st.metric("Ort. Kayıp", f st.metric("Ort. Kayıp", f"${"${metricsmetrics['avg_loss']:.2f}")
['avg_loss']:.2f}")
            st.metric("Max Drawdown", f"{metrics            st.metric("Max Drawdown", f"{metrics['['max_drawdown_%']}%")
        
        #max_drawdown_%']}%")
        
        # Grafikler
        if not trades.empty Grafikler
        if not trades.empty:
            st.subheader(":
            st.subheader("📈 Performans Grafik📈 Performans Grafikleri")
            
            fig, (ax1,leri")
            
            fig, (ax1, ax2) = ax2) = plt.sub plt.subplots(2,plots(2, 1 1, figsize=(, figsize=(1212, 8, 8))
            
           ))
            
            # Equity curve
            # Equity curve
            ax1 ax1.plot(equity.plot(equity['date['date'], equity['equity'],'], equity['equity'], color='green', line color='green', linewidthwidth=2)
            ax1=2)
            ax1.set.set_title('Portföy_title('Portföy Değeri')
 Değeri')
            ax            ax1.set_ylabel1.set_ylabel('Equ('Equity ($)')
ity ($)')
                       ax1.grid(True ax1.grid(True, alpha, alpha=0=0.3)
            
.3)
            
            # Drawdown            # Drawdown
           
            equity_series = equity equity_series = equity.set_index('date')['equity']
            rolling_max =.set_index('date')['equity']
            rolling_max = equity_ equity_series.expanding().series.expanding().max()
max()
            draw            drawdown =down = (equity_ (equity_series - rollingseries - rolling_max) / rolling_max) / rolling_max * _max * 100
            
100
            
            ax2.fill            ax2.fill_between(_between(equity['equity['date'], drawdate'], drawdown.values, 0down.values, 0,, alpha=0 alpha=0.3,.3, color='red')
            color='red')
            ax ax2.set_title('2.set_title('DrawdownDrawdown')
            ax2')
            ax2.set.set_ylabel('Draw_ylabel('Drawdown %down %')
            ax2.grid(True, alpha=0')
            ax2.grid(True, alpha=0..3)
            
            plt.t3)
            
            plt.tight_layoutight_layout()
           ()
            st.pyplot(fig st.pyplot(fig)
            
)
            
            # İ            # İşlem listesi
şlem listesi
            st.sub            st.subheader("header("📋 İşlem📋 İşlem Listesi Listesi")
            display_t")
            display_trades = tradesrades = trades.copy.copy()
()
            display_trades['entry_date']            display_trades['entry_date'] = display_trades['entry_date'].dt.strftime = display_trades['entry_date'].dt.str('%Y-%m-%dftime('%Y-%m-%d')
            display_trades['exit_date']')
            display_trades['exit_date'] = display_trades['exit_date'].dt.strftime('% = display_trades['exit_date'].dt.strftime('%Y-%m-%d')
            st.dataY-%m-%d')
            st.dataframe(display_tframe(display_trades)
            
        elserades)
            
        else:
:
            st.info("🤷            st.info("🤷 Hiç iş Hiç işlem gerçekleşmedi.lem gerçekleşmedi. Parametreleri de Parametreleri değiğiştirmeyi deneyştirmeyi deneyin.")
            
    except Exceptionin.")
            
    except Exception as e as e:
        st.error:
        st.error(f"❌(f"❌ Hata Hata: {: {str(estr(e)}")

st.mark)}")

st.markdown("---")
stdown("---")
st.markdown(".markdown("**Swing Backtest**Swing Backtest | | Tam Çalışır** Tam Çalışır**")
")