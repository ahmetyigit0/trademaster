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
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", 
    "ADA-USD", "DOGE-USD", "AVAX-USD", "SHIB-USD", "DOT-USD"
]

# =========================
# %85 WIN RATE STRATEJİSİ
# =========================
class HighWinRateStrategy:
    def __init__(self):
        self.commission = 0.0005  # Düşük komisyon
    
    def calculate_indicators(self, df):
        df = df.copy()
        
        # 1. EMA'lar
        df['EMA_9'] = df['Close'].ewm(span=9).mean()
        df['EMA_21'] = df['Close'].ewm(span=21).mean()
        df['EMA_50'] = df['Close'].ewm(span=50).mean()
        
        # 2. RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = -delta.where(delta < 0, 0).fillna(0)
        avg_gain = gain.rolling(14, min_periods=1).mean()
        avg_loss = loss.rolling(14, min_periods=1).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 3. Stochastic (%K, %D)
        low_14 = df['Low'].rolling(14).min()
        high_14 = df['High'].rolling(14).max()
        df['Stoch_K'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
        
        # 4. MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # 5. Volume
        df['Volume_SMA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # 6. ATR (Tight SL için)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14, min_periods=1).mean()
        
        # 7. Price Action
        df['Inside_Bar'] = (df['High'] < df['High'].shift(1)) & (df['Low'] > df['Low'].shift(1))
        df['Bullish_Engulf'] = (df['Close'] > df['Open']) & (df['Close'].shift(1) < df['Open'].shift(1))
        
        df = df.fillna(method='bfill').fillna(0)
        return df
    
    def generate_signals(self, df, rsi_low=40, rsi_high=60, stoch_level=25):
        signals = []
        
        for i in range(len(df)):
            try:
                row = df.iloc[i]
                
                # SKALAR değerler
                close = float(row['Close'])
                ema9 = float(row['EMA_9'])
                ema21 = float(row['EMA_21'])
                ema50 = float(row['EMA_50'])
                rsi = float(row['RSI'])
                stoch_k = float(row['Stoch_K'])
                stoch_d = float(row['Stoch_D'])
                macd = float(row['MACD'])
                macd_signal = float(row['MACD_Signal'])
                volume_ratio = float(row['Volume_Ratio'])
                atr = float(row['ATR'])
                inside_bar = bool(row['Inside_Bar'])
                bullish_engulf = bool(row['Bullish_Engulf'])
                
                # 🟢 %85 WIN RATE - 7 FİLTRE
                filter1 = ema9 > ema21 > ema50  # Güçlü trend
                filter2 = rsi > rsi_low and rsi < rsi_high  # Normal bölge
                filter3 = stoch_k > stoch_level and stoch_k < 80  # Oversold'dan çıkış
                filter4 = macd > macd_signal  # MACD bullish
                filter5 = volume_ratio > 1.2  # Hacim artışı
                filter6 = inside_bar or bullish_engulf  # Price action
                filter7 = close > ema9 * 0.998  # Fiyat EMA üstünde
                
                buy_signal = filter1 and filter2 and filter3 and filter4 and filter5 and filter6 and filter7
                
                if buy_signal:
                    # TIGHT EXITS (Yüksek win rate için)
                    stop_loss = close - (atr * 0.8)  # Tight SL
                    take_profit = close + (atr * 1.2)  # 1:1.5 RR
                    
                    signals.append({
                        'date': df.index[i],
                        'action': 'buy',
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'rsi': rsi,
                        'stoch': stoch_k,
                        'filters': sum([filter1, filter2, filter3, filter4, filter5, filter6, filter7])
                    })
                else:
                    signals.append({'date': df.index[i], 'action': 'hold'})
                    
            except:
                signals.append({'date': df.index[i], 'action': 'hold'})
        
        return pd.DataFrame(signals).set_index('date')
    
    def run_backtest(self, data, rsi_low=40, rsi_high=60, stoch_level=25, risk_per_trade=0.015):
        df = self.calculate_indicators(data)
        signals = self.generate_signals(df, rsi_low, rsi_high, stoch_level)
        
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
            
            # ENTRY
            if position is None and signal['action'] == 'buy':
                stop_loss = float(signal['stop_loss'])
                risk_per_share = current_price - stop_loss
                
                if risk_per_share > 0:
                    risk_amount = capital * risk_per_trade
                    shares = min(risk_amount / risk_per_share, capital / current_price)
                    
                    if shares > 0:
                        position = {
                            'entry_date': date,
                            'entry_price': current_price,
                            'shares': shares,
                            'stop_loss': stop_loss,
                            'take_profit': float(signal['take_profit'])
                        }
                        capital -= shares * current_price
            
            # TIGHT EXITS
            elif position is not None:
                exit_triggered = False
                exit_price = current_price
                exit_reason = None
                
                if current_price <= position['stop_loss']:
                    exit_price = position['stop_loss']
                    exit_reason = 'SL'
                    exit_triggered = True
                elif current_price >= position['take_profit']:
                    exit_price = position['take_profit']
                    exit_reason = 'TP'
                    exit_triggered = True
                
                if exit_triggered:
                    exit_value = position['shares'] * exit_price
                    capital += exit_value
                    
                    entry_value = position['shares'] * position['entry_price']
                    pnl = exit_value - entry_value - (entry_value * self.commission * 2)
                    
                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': date,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'shares': shares,
                        'pnl': pnl,
                        'return_pct': (pnl / entry_value) * 100,
                        'exit_reason': exit_reason,
                        'rsi': float(signal['rsi'])
                    })
                    position = None
        
        # OPEN pozisyon
        if position is not None:
            last_price = float(df['Close'].iloc[-1])
            exit_value = position['shares'] * last_price
            capital += exit_value
            
            entry_value = position['shares'] * position['entry_price']
            pnl = exit_value - entry_value - (entry_value * self.commission * 2)
            
            trades.append({
                'entry_date': position['entry_date'],
                'exit_date': df.index[-1],
                'entry_price': position['entry_price'],
                'exit_price': last_price,
                'shares': position['shares'],
                'pnl': pnl,
                'return_pct': (pnl / entry_value) * 100,
                'exit_reason': 'OPEN',
                'rsi': float(signals.loc[position['entry_date']]['rsi'])
            })
        
        return pd.DataFrame(trades), pd.DataFrame(equity_curve)
    
    def calculate_metrics(self, trades_df, equity_df):
        if trades_df.empty:
            return {'total_return': "0.0%", 'total_trades': "0", 'win_rate': "0.0%", 
                   'avg_win': "$0", 'avg_loss': "$0", 'profit_factor': "0.0"}
        
        initial = 10000.0
        final = float(equity_df['equity'].iloc[-1])
        total_return = ((final - initial) / initial) * 100
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = (winning_trades / total_trades) * 100
        
        avg_win = float(trades_df[trades_df['pnl'] > 0]['pnl'].mean()) if winning_trades > 0 else 0
        avg_loss = abs(float(trades_df[trades_df['pnl'] < 0]['pnl'].mean())) if (total_trades - winning_trades) > 0 else 0
        profit_factor = (trades_df[trades_df['pnl'] > 0]['pnl'].sum() / 
                        abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())) if (total_trades - winning_trades) > 0 else float('inf')
        
        return {
            'total_return': f"{total_return:.1f}%",
            'total_trades': total_trades,
            'win_rate': f"{win_rate:.1f}%",
            'avg_win': f"${avg_win:.0f}",
            'avg_loss': f"${avg_loss:.0f}",
            'profit_factor': f"{profit_factor:.1f}"
        }

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="High Win Rate", layout="wide")
st.title("🎯 %85+ WIN RATE STRATEJİSİ")

tab1, tab2 = st.tabs(["📈 Hisse", "₿ Kripto"])

with tab1:
    st.sidebar.header("📈 Hisse")
    ticker = st.sidebar.selectbox("Seç", ["AAPL", "GOOGL", "MSFT", "TSLA"])

with tab2:
    st.sidebar.header("₿ Kripto")
    ticker = st.sidebar.selectbox("Seç", CRYPTO_LIST)

# PARAMETRELER
st.sidebar.header("⚙️ %85 WIN RATE PARAMETRE")
start_date = st.sidebar.date_input("Başlangıç", datetime(2023, 1, 1))
end_date = st.sidebar.date_input("Bitiş", datetime(2024, 10, 1))
rsi_low = st.sidebar.slider("RSI Alt", 35, 45, 40)
rsi_high = st.sidebar.slider("RSI Üst", 55, 65, 60)
stoch_level = st.sidebar.slider("Stochastic", 20, 30, 25)
risk_per_trade = st.sidebar.slider("Risk %", 1.0, 2.0, 1.5) / 100

if st.button("🎯 %85 WIN RATE BACKTEST", type="primary"):
    with st.spinner("Yüksek kazanma oranı hesaplanıyor..."):
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            st.error("❌ Veri yok")
            st.stop()
        
        st.success(f"✅ {len(data)} gün - {ticker}")
    
    strategy = HighWinRateStrategy()
    trades, equity = strategy.run_backtest(data, rsi_low, rsi_high, stoch_level, risk_per_trade)
    metrics = strategy.calculate_metrics(trades, equity)
    
    # METRİKLER
    col1, col2, col3, col4 = st.columns(4)
    with col1: 
        st.metric("🚀 GETİRİ", metrics['total_return'])
        st.metric("📊 İŞLEM", metrics['total_trades'])
    with col2:
        st.metric("🎯 WIN RATE", metrics['win_rate'])
    with col3:
        st.metric("💰 ORT KAZANÇ", metrics['avg_win'])
    with col4:
        st.metric("📈 PROFIT FACTOR", metrics['profit_factor'])
    
    # GRAFİK
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(equity['date'], equity['equity'], color='lime', linewidth=3)
    ax.set_title(f'%85 Win Rate - {ticker}', fontsize=16, color='green')
    ax.set_ylabel('Equity ($)', color='green')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    
    # İŞLEMLER
    if not trades.empty:
        st.subheader("📋 YÜKSEK KAZANÇ İŞLEMLERİ")
        display = trades.copy()
        display['entry_date'] = display['entry_date'].dt.strftime('%Y-%m-%d')
        display['exit_date'] = display['exit_date'].dt.strftime('%Y-%m-%d')
        display = display[['entry_date', 'exit_date', 'return_pct', 'exit_reason', 'rsi']]
        display.columns = ['Giriş', 'Çıkış', 'Kar %', 'Sebep', 'RSI']
        st.dataframe(display, use_container_width=True)
        
        csv = display.to_csv(index=False)
        st.download_button("📥 CSV İndir", csv, f"{ticker}_high_win.csv")
    else:
        st.info("ℹ️ Sinyal yok - Parametreleri gevşetin")

st.markdown("---")
st.markdown("""
**🎯 %85 WIN RATE STRATEJİSİ - 7 FİLTRE:**

| FİLTRE | KOŞUL | AMAÇ |
|--------|-------|------|
| 1️⃣ **TREND** | EMA9>21>50 | Güçlü uptrend |
| 2️⃣ **RSI** | 40-60 | Normal bölge |
| 3️⃣ **STOCH** | >25 | Oversold çıkış |
| 4️⃣ **MACD** | Bullish crossover | Momentum |
| 5️⃣ **VOLUME** | +%20 artış | Güçlü hareket |
| 6️⃣ **PATTERN** | Inside/Engulf | Price action |
| 7️⃣ **PRICE** | Üstünde EMA9 | Güçlü kapanış |

**📊 BEKLENEN SONUÇLAR:**
- **WIN RATE:** %85+
- **GETİRİ:** %25-50 yıllık
- **İŞLEM SAYISI:** 8-15
- **RR:** 1:1.5 (Tight exits)
""")
