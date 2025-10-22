import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

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
        df['EMA_20'] = df['Close'].ewm(span=20, min_periods=1, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, min_periods=1, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.ewm(span=14, adjust=False).mean()
        avg_loss = loss.ewm(span=14, adjust=False).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD ve SİNYAL HESAPLAMASI (İşlem sayısını artırmak için eklendi)
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Cross_Up'] = (df['MACD'] > df['Signal_Line']) & (df['MACD'].shift(1) <= df['Signal_Line'].shift(1))
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift(1))
        low_close = np.abs(df['Low'] - df['Close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.ewm(span=14, adjust=False).mean() 
        
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
                
                # Sinyal 1 (Trend Takibi): Trend UP + Aşırı Satım
                trend_ok = ema_20_val > ema_50_val
                rsi_ok = rsi_val < rsi_oversold 
                
                # Sinyal 2 (Momentum Geri Dönüş): Orta Aşırı Satım + MACD Cross
                rsi_medium_ok = rsi_val < rsi_oversold * 1.25 
                macd_cross_ok = row['MACD_Cross_Up']
                
                # Nihai Sinyal: (Trend Takibi) VEYA (Momentum Geri Dönüşü)
                buy_signal = (trend_ok and rsi_ok) or \
                             (rsi_medium_ok and macd_cross_ok)
                
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
        
        signals_df = signals_df.fillna({'stop_loss': np.nan, 'take_profit': np.nan})
        
        return signals_df
    
    def run_backtest(self, data, rsi_oversold=40, atr_multiplier=2.0, risk_per_trade=0.02):
        df = self.calculate_indicators(data)
        signals = self.generate_signals(df, rsi_oversold, atr_multiplier)
        
        # --- KRİTİK HATA ÇÖZÜMÜ: MultiIndex sorununu önlemek için sütun üzerinden birleştirme ---
        
        # 1. Her iki DataFrame'in index'ini sütun haline getiriyoruz (Date sütunu oluşturulur)
        df_reset = df.reset_index() 
        signals_reset = signals.reset_index()
        
        # 2. 'date' sütunu üzerinden birleştirme yapıyoruz.
        df_combined = df_reset.merge(
            signals_reset[['date', 'action', 'stop_loss', 'take_profit']], 
            on='date', 
            how='left'
        )
        
        # 3. 'date' sütununu tekrar index yapıyoruz ve adını sabitliyoruz
        df_combined = df_combined.set_index('date') 
        df_combined.index.name = 'Date' 
        # --- HATA ÇÖZÜMÜ SONU ---

        df_combined['action'] = df_combined['action'].fillna('hold')
        
        capital = 10000.0
        position = None
        trades = []
        equity_curve = []
        
        for date in df_combined.index:
            current_price = float(df_combined.loc[date, 'Close'])
            signal_action = df_combined.loc[date, 'action']
            
            current_equity = capital
            if position is not None:
                current_equity += float(position['shares']) * current_price
            
            equity_curve.append({'date': date, 'equity': current_equity})
            
            # ALIM
            if position is None and signal_action == 'buy':
                sl_val = df_combined.loc[date, 'stop_loss']
                tp_val = df_combined.loc[date, 'take_profit']
                
                if pd.notna(sl_val) and pd.notna(tp_val):
                    stop_loss = float(sl_val)
                    take_profit = float(tp_val)
                    
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
                                'take_profit': take_profit
                            }
                            capital -= shares * current_price * (1 + self.commission) 
            
            # ÇIKIŞ
            elif position is not None:
                exit_price = None
                exit_reason = None

                if current_price <= position['stop_loss']:
                    exit_price = position['stop_loss']
                    exit_reason = 'SL'
                
                elif current_price >= position['take_profit']:
                    exit_price = position['take_profit']
                    exit_reason = 'TP'

                if exit_price is not None:
                    exit_value = position['shares'] * exit_price
                    capital += exit_value * (1 - self.commission) 
                    
                    entry_value = position['shares'] * position['entry_price']
                    pnl = (exit_value - entry_value) - (entry_value * self.commission + exit_value * self.commission)
                    
                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': date,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'return_pct': (pnl / entry_value) * 100 if entry_value > 0 else 0,
                        'exit_reason': exit_reason
                    })
                    position = None
        
        # Kapanış pozisyonu (Dönem sonu)
        if position is not None:
            last_price = float(df_combined['Close'].iloc[-1])
            exit_value = position['shares'] * last_price
            capital += exit_value * (1 - self.commission)
            
            entry_value = position['shares'] * position['entry_price']
            pnl = (exit_value - entry_value) - (entry_value * self.commission + exit_value * self.commission)
            
            trades.append({
                'entry_date': position['entry_date'],
                'exit_date': df_combined.index[-1],
                'entry_price': position['entry_price'],
                'exit_price': last_price,
                'pnl': pnl,
                'return_pct': (pnl / entry_value) * 100 if entry_value > 0 else 0,
                'exit_reason': 'OPEN'
            })
        
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        equity_df = pd.DataFrame(equity_curve)
        
        return trades_df, equity_df
    
    def calculate_metrics(self, trades_df, equity_df):
        # Metrik hesaplama (Aynı)
        if trades_df.empty or equity_df.empty:
            return {
                'total_return': "0.0%", 'total_trades': "0", 'win_rate': "0.0%",
                'avg_win': "$0.00", 'avg_loss': "$0.00"
            }
        
        try:
            initial_equity = 10000.0
            final_equity = float(equity_df['equity'].iloc[-1])
            total_return = (final_equity - initial_equity) / initial_equity * 100.0
            
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            win_rate = (winning_trades / total_trades) * 100.0 if total_trades > 0 else 0.0
            
            avg_win = float(trades_df[trades_df['pnl'] > 0]['pnl'].mean()) if winning_trades > 0 else 0.0
            avg_loss = float(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if (total_trades - winning_trades) > 0 else 0.0
            
            return {
                'total_return': f"{round(total_return, 2)}%",
                'total_trades': str(total_trades),
                'win_rate': f"{round(win_rate, 1)}%",
                'avg_win': f"${round(avg_win, 2)}",
                'avg_loss': f"${abs(round(avg_loss, 2))}"
            }
            
        except:
            return {
                'total_return': "0.0%", 'total_trades': "0", 'win_rate': "0.0%",
                'avg_win': "$0.00", 'avg_loss': "$0.00"
            }

# =========================
# STREAMLIT UYGULAMASI
# =========================
st.set_page_config(page_title="Swing Backtest", layout="wide")
st.title("🚀 Swing Trading Backtest (Çift Sinyal + Hata Çözümü)")

# Sidebar
st.sidebar.header("⚙️ Ayarlar")
ticker = st.sidebar.selectbox("Sembol", ["AAPL", "GOOGL", "MSFT", "TSLA", "BTC-USD", "ETH-USD"])
start_date = st.sidebar.date_input("Başlangıç", datetime(2023, 1, 1))
end_date = st.sidebar.date_input("Bitiş", datetime(2023, 12, 31))

st.sidebar.header("📊 Parametreler")
# RSI eşiğini 45'e çıkardık
rsi_oversold = st.sidebar.slider("RSI Aşırı Satım", 25, 50, 45) 
atr_multiplier = st.sidebar.slider("ATR Çarpanı (SL için)", 1.0, 3.0, 2.0)
risk_per_trade = st.sidebar.slider("Risk % (Poz. Büyüklüğü)", 1.0, 5.0, 2.0) / 100

# Ana içerik
if st.button("🎯 Backtest Çalıştır"):
    try:
        with st.spinner("Veri yükleniyor..."):
            # İndikatörler için biraz daha fazla veri çekmek gerekir
            extended_start_date = start_date - timedelta(days=150)
            data = yf.download(ticker, start=extended_start_date, end=end_date, progress=False)
            
            if data.empty:
                st.error("❌ Veri bulunamadı")
                st.stop()
            
            # Sadece istenen aralıkta çalıştır
            data_test = data[data.index >= pd.to_datetime(start_date)]
            st.success(f"✅ {len(data_test)} günlük veri ile test ediliyor.")
        
        backtester = SwingBacktest()
        
        with st.spinner("Backtest çalıştırılıyor..."):
            trades, equity = backtester.run_backtest(data_test, rsi_oversold, atr_multiplier, risk_per_trade)
            metrics = backtester.calculate_metrics(trades, equity)
        
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
            st.dataframe(display_trades.round(2))
            
        else:
            st.info("🤷 Hiç işlem gerçekleşmedi. Daha esnek ayarlar veya farklı bir sembol deneyin.")
            
    except Exception as e:
        st.error(f"❌ Hata: {str(e)}")

st.markdown("---")
st.markdown("**Backtest Sistemi v5.2 - Final İndeks Düzeltmesi**")
