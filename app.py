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
# POPÜLER KRİPTO LİSTESİ
# =========================
CRYPTO_LIST = [
    "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD",
    "SOL-USD", "DOGE-USD", "AVAX-USD", "SHIB-USD", "DOT-USD"
]

# =========================
# YENİ DİPTEN ALIM STRATEJİSİ
# =========================
class DipBuyBacktest:
    def __init__(self):
        self.commission = 0.001
    
    def calculate_indicators(self, df):
        df = df.copy()
        
        # EMA'lar
        df['EMA_20'] = df['Close'].ewm(span=20, min_periods=1).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, min_periods=1).mean()
        
        # RSI (Düşüş için)
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
        
        # YENİ: DÜŞÜŞ TESPİTİ İÇİN
        df['Price_Change'] = df['Close'].pct_change()
        df['Recent_Low'] = df['Close'].rolling(5).min()
        df['Bounce'] = (df['Close'] > df['Recent_Low'] * 1.02)  # %2 bounce
        
        df = df.fillna(method='bfill').fillna(method='ffill')
        return df
    
    def generate_signals(self, df, rsi_oversold=35, atr_multiplier=1.5, bounce_threshold=0.02):
        signals = []
        
        for i in range(len(df)):
            try:
                row = df.iloc[i]
                
                close_val = float(row['Close'])
                ema_20_val = float(row['EMA_20'])
                ema_50_val = float(row['EMA_50'])
                rsi_val = float(row['RSI'])
                atr_val = float(row['ATR'])
                recent_low = float(row['Recent_Low'])
                bounce = bool(row['Bounce'])
                
                # 🟢 DİPTEN ALIM KOŞULLARI:
                # 1. RSI DÜŞÜK (Aşırı satım)
                # 2. Fiyat son 5 günün DİBİNDE
                # 3. %2 BOUNCE başladı
                # 4. EMA trend OK (uzun vadeli)
                
                rsi_ok = rsi_val < rsi_oversold
                dip_ok = close_val <= recent_low * 1.01  # Dipte
                bounce_ok = bounce  # Bounce başladı
                trend_ok = ema_20_val > ema_50_val  # Uzun vadeli trend up
                
                buy_signal = rsi_ok and dip_ok and bounce_ok and trend_ok
                
                if buy_signal:
                    # TIGHT STOP LOSS (dip için)
                    stop_loss = close_val - (atr_val * atr_multiplier)
                    # 1:3 Risk/Reward
                    take_profit = close_val + (atr_val * atr_multiplier * 3)
                    
                    signals.append({
                        'date': df.index[i],
                        'action': 'buy',
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'rsi': rsi_val,
                        'bounce': bounce
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
    
    def run_backtest(self, data, rsi_oversold=35, atr_multiplier=1.5, risk_per_trade=0.02):
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
            
            # DİPTEN ALIM
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
            
            # ÇIKIŞ (SL/TP)
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
                        'exit_reason': 'SL',
                        'rsi_entry': signals.loc[position['entry_date']]['rsi']
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
                        'exit_reason': 'TP',
                        'rsi_entry': signals.loc[position['entry_date']]['rsi']
                    })
                    position = None
        
        # OPEN pozisyon kapat
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
                'exit_reason': 'OPEN',
                'rsi_entry': signals.loc[position['entry_date']]['rsi']
            })
        
        return pd.DataFrame(trades) if trades else pd.DataFrame(), pd.DataFrame(equity_curve)
    
    def calculate_metrics(self, trades_df, equity_df):
        if trades_df.empty:
            return {
                'total_return': "0.0%", 'total_trades': "0",
                'win_rate': "0.0%", 'avg_win': "$0.00", 'avg_loss': "$0.00"
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
                'avg_loss': f"${round(avg_loss, 2)}"
            }
        except:
            return {
                'total_return': "0.0%", 'total_trades': "0",
                'win_rate': "0.0%", 'avg_win': "$0.00", 'avg_loss': "$0.00"
            }

# =========================
# STREAMLIT UYGULAMASI
# =========================
st.set_page_config(page_title="Dip Buy Backtest", layout="wide")
st.title("🚀 DİPTEN ALIM STRATEJİSİ")

tab1, tab2 = st.tabs(["📈 Hisse Senetleri", "₿ Kripto Para"])

with tab1:
    st.sidebar.header("⚙️ Hisse Ayarları")
    stock_tickers = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    ticker = st.sidebar.selectbox("Hisse Seç", stock_tickers)

with tab2:
    st.sidebar.header("₿ Kripto Ayarları")
    ticker = st.sidebar.selectbox("Kripto Seç", CRYPTO_LIST)

# DİPTEN ALIM PARAMETRELERİ
st.sidebar.header("🎯 DİP ALIM PARAMETRELERİ")
start_date = st.sidebar.date_input("Başlangıç", datetime(2023, 1, 1))
end_date = st.sidebar.date_input("Bitiş", datetime(2023, 12, 31))
rsi_oversold = st.sidebar.slider("RSI Dip Seviyesi", 25, 40, 35)
atr_multiplier = st.sidebar.slider("Stop Loss ATR", 1.0, 2.5, 1.5)
risk_per_trade = st.sidebar.slider("Risk %", 1.0, 5.0, 2.0) / 100

if st.button("🎯 DİPTEN ALIM BACKTEST"):
    try:
        with st.spinner("Veri yükleniyor..."):
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if data.empty:
                st.error("❌ Veri bulunamadı")
                st.stop()
            st.success(f"✅ {len(data)} günlük veri - {ticker}")
        
        backtester = DipBuyBacktest()
        with st.spinner("Dipten alım hesaplanıyor..."):
            trades, equity = backtester.run_backtest(data, rsi_oversold, atr_multiplier, risk_per_trade)
            metrics = backtester.calculate_metrics(trades, equity)
        
        st.subheader("📊 DİP ALIM SONUÇLARI")
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
            st.subheader(f"📈 {ticker} Portföy Grafiği")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(equity['date'], equity['equity'], color='green', linewidth=2)
            ax.set_title(f'Dipten Alım Stratejisi - {ticker}')
            ax.set_ylabel('Equity ($)')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            st.subheader("📋 DİP ALIM İŞLEMLERİ")
            display_trades = trades.copy()
            display_trades['entry_date'] = display_trades['entry_date'].dt.strftime('%Y-%m-%d')
            display_trades['exit_date'] = display_trades['exit_date'].dt.strftime('%Y-%m-%d')
            display_trades['pnl'] = display_trades['pnl'].round(2)
            display_trades['return_pct'] = display_trades['return_pct'].round(1)
            st.dataframe(display_trades)
            
        else:
            st.info("🤷 Hiç dip alım sinyali yok")
            
    except Exception as e:
        st.error(f"❌ Hata: {str(e)}")

st.markdown("---")
st.markdown("""
**🎯 DİPTEN ALIM STRATEJİSİ:**
- **RSI < 35:** Aşırı satım
- **Dip Tespiti:** Son 5 günün en düşüğü
- **%2 Bounce:** Yükseliş başladı
- **1:3 Risk/Reward:** Max kazanç
- **Komisyon:** %0.1 dahil
""")
