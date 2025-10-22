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
# BACKTEST MOTORU - PROFESYONEL STRATEJİ
# =========================
class SwingBacktest:
    def __init__(self):
        self.commission = 0.001
    
    def calculate_indicators(self, df):
        df = df.copy()
        
        # EMA'lar - Trend tespiti
        df['EMA_20'] = df['Close'].ewm(span=20, min_periods=1).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, min_periods=1).mean()
        df['EMA_100'] = df['Close'].ewm(span=100, min_periods=1).mean()
        
        # RSI - Momentum
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD - Momentum
        exp1 = df['Close'].ewm(span=12, min_periods=1).mean()
        exp2 = df['Close'].ewm(span=26, min_periods=1).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, min_periods=1).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands - Volatilite
        df['BB_Middle'] = df['Close'].rolling(window=20, min_periods=1).mean()
        bb_std = df['Close'].rolling(window=20, min_periods=1).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # Fibonacci Seviyeleri (Son 20 günün high/low'u)
        df['Recent_High'] = df['High'].rolling(window=20, min_periods=1).max()
        df['Recent_Low'] = df['Low'].rolling(window=20, min_periods=1).min()
        range_high_low = df['Recent_High'] - df['Recent_Low']
        df['Fib_382'] = df['Recent_High'] - (range_high_low * 0.382)
        df['Fib_618'] = df['Recent_High'] - (range_high_low * 0.618)
        
        # ATR - Volatilite için Stop Loss
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
    
    def generate_signals(self, df, rsi_oversold=35, atr_multiplier=1.5):
        signals = []
        
        for i in range(len(df)):
            try:
                if i < 20:  # İlk 20 gün yeterli veri yok
                    signals.append({'date': df.index[i], 'action': 'hold'})
                    continue
                    
                row = df.iloc[i]
                
                close_val = float(row['Close'])
                ema_20_val = float(row['EMA_20'])
                ema_50_val = float(row['EMA_50'])
                ema_100_val = float(row['EMA_100'])
                rsi_val = float(row['RSI'])
                atr_val = float(row['ATR'])
                macd_hist_val = float(row['MACD_Hist'])
                bb_lower_val = float(row['BB_Lower'])
                bb_upper_val = float(row['BB_Upper'])
                fib_382_val = float(row['Fib_382'])
                fib_618_val = float(row['Fib_618'])
                
                # TREND KOŞULLARI (ÇOK ÖNEMLİ)
                strong_uptrend = (ema_20_val > ema_50_val > ema_100_val)
                weak_uptrend = (ema_20_val > ema_50_val)
                
                # MOMENTUM KOŞULLARI
                rsi_oversold_ok = rsi_val < rsi_oversold
                rsi_bullish = (rsi_val > 30) and (rsi_val < 70)  # Neutral bölge
                macd_bullish = macd_hist_val > 0
                macd_turning = macd_hist_val > df['MACD_Hist'].iloc[i-1] if i > 0 else False
                
                # FİYAT KONUMU
                near_bb_lower = close_val <= bb_lower_val * 1.02  # Bollinger alt bandına yakın
                near_fib_618 = abs(close_val - fib_618_val) / fib_618_val < 0.02  # Fib 0.618'e yakın
                near_fib_382 = abs(close_val - fib_382_val) / fib_382_val < 0.02  # Fib 0.382'ye yakın
                above_ema20 = close_val > ema_20_val
                
                # VOLATİLITE
                low_volatility = row['BB_Width'] < 0.05  # Düşük volatilite
                
                # STRATEJİ 1: GÜÇLÜ TREND + DİP ALIŞ
                strategy1 = (strong_uptrend and 
                           rsi_oversold_ok and 
                           (near_bb_lower or near_fib_618))
                
                # STRATEJİ 2: TREND + MOMENTUM DÖNÜŞÜMÜ
                strategy2 = (weak_uptrend and 
                           rsi_bullish and 
                           macd_bullish and 
                           macd_turning and 
                           above_ema20)
                
                # STRATEJİ 3: FİBONACCİ DESTEK + BOLLINGER
                strategy3 = (weak_uptrend and 
                           (near_fib_382 or near_fib_618) and 
                           near_bb_lower and 
                           rsi_val < 45)
                
                # STRATEJİ 4: DÜŞÜK VOLATİLİTE + TREND
                strategy4 = (strong_uptrend and 
                           low_volatility and 
                           rsi_bullish and 
                           macd_bullish)
                
                # ANA SİNYAL - EN AZ 2 STRATEJİ ONAY VERMELİ
                buy_signals = [strategy1, strategy2, strategy3, strategy4]
                confirmed_signals = sum(buy_signals)
                
                buy_signal = confirmed_signals >= 2  # En az 2 strateji onay vermeli
                
                if buy_signal:
                    # AKILLI STOP LOSS SEVİYESİ
                    support_levels = [bb_lower_val, fib_618_val, fib_382_val, ema_50_val]
                    valid_supports = [s for s in support_levels if s < close_val]
                    smart_stop_loss = max(valid_supports) if valid_supports else close_val - (atr_val * atr_multiplier)
                    
                    # TP: Fib 0.382 veya BB Upper
                    take_profit = min(fib_382_val, bb_upper_val)
                    
                    # Risk/Reward kontrolü
                    risk = close_val - smart_stop_loss
                    reward = take_profit - close_val
                    risk_reward_ok = (reward / risk) >= 1.5 if risk > 0 else False
                    
                    if risk_reward_ok:
                        signals.append({
                            'date': df.index[i],
                            'action': 'buy',
                            'stop_loss': smart_stop_loss,
                            'take_profit': take_profit,
                            'strategy_count': confirmed_signals
                        })
                    else:
                        signals.append({
                            'date': df.index[i],
                            'action': 'hold'
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
        st.info(f"🎯 {buy_count} kaliteli alış sinyali bulundu")
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
                            'take_profit': float(signal['take_profit']),
                            'strategy_count': signal.get('strategy_count', 0)
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
                        'exit_reason': 'SL',
                        'strategy_count': position['strategy_count']
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
                        'strategy_count': position['strategy_count']
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
                'exit_reason': 'OPEN',
                'strategy_count': position['strategy_count']
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
                'profit_factor': "0.0",
                'avg_strategy_count': "0.0"
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
            
            total_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
            total_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            avg_strategy_count = float(trades_df['strategy_count'].mean()) if 'strategy_count' in trades_df.columns else 0.0
            
            return {
                'total_return': f"{round(total_return, 2)}%",
                'total_trades': str(total_trades),
                'win_rate': f"{round(win_rate, 1)}%",
                'avg_win': f"${round(avg_win, 2)}",
                'avg_loss': f"${round(avg_loss, 2)}",
                'profit_factor': f"{round(profit_factor, 2)}",
                'avg_strategy_count': f"{round(avg_strategy_count, 1)}"
            }
            
        except:
            return {
                'total_return': "0.0%",
                'total_trades': "0",
                'win_rate': "0.0%",
                'avg_win': "$0.00",
                'avg_loss': "$0.00",
                'profit_factor': "0.0",
                'avg_strategy_count': "0.0"
            }

# =========================
# STREAMLIT UYGULAMASI
# =========================
st.set_page_config(page_title="Pro Swing Backtest", layout="wide")
st.title("🚀 PROFESYONEL SWING BACKTEST")
st.markdown("**5 İndikatörlü Akıllı Strateji**")

# Sidebar
st.sidebar.header("⚙️ Ayarlar")
ticker = st.sidebar.selectbox("Sembol", ["BTC-USD", "ETH-USD", "TSLA", "NVDA", "AAPL", "GOOGL", "MSFT"])
start_date = st.sidebar.date_input("Başlangıç", datetime(2023, 1, 1))
end_date = st.sidebar.date_input("Bitiş", datetime(2023, 12, 31))

st.sidebar.header("🎯 Strateji Parametreleri")
rsi_oversold = st.sidebar.slider("RSI Aşırı Satım", 25, 40, 35)
atr_multiplier = st.sidebar.slider("ATR Çarpanı", 1.0, 2.5, 1.5)
risk_per_trade = st.sidebar.slider("Risk %", 1.0, 5.0, 2.0) / 100

st.sidebar.info("""
**🎯 STRATEJİ ÖZETİ:**
- EMA Trend Filtresi
- RSI + MACD Momentum
- Bollinger + Fibonacci Destek
- Çoklu Strateji Onayı
- Akıllı Stop Loss
""")

# Ana içerik
if st.button("🎯 BACKTEST ÇALIŞTIR", type="primary"):
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
            st.info(f"📈 Fiyat aralığı: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
        
        backtester = SwingBacktest()
        
        with st.spinner("Profesyonel backtest çalıştırılıyor..."):
            trades, equity = backtester.run_backtest(data, rsi_oversold, atr_multiplier, risk_per_trade)
            metrics = backtester.calculate_metrics(trades, equity)
        
        st.subheader("📊 DETAYLI PERFORMANS RAPORU")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Toplam Getiri", metrics['total_return'])
            st.metric("Toplam İşlem", metrics['total_trades'])
            st.metric("Win Rate", metrics['win_rate'])
        
        with col2:
            st.metric("Ort. Kazanç", metrics['avg_win'])
            st.metric("Ort. Kayıp", metrics['avg_loss'])
            st.metric("Profit Factor", metrics['profit_factor'])
        
        with col3:
            st.metric("Ort. Strateji Sayısı", metrics['avg_strategy_count'])
        
        if not trades.empty:
            # İstatistikler
            winning_trades = len(trades[trades['pnl'] > 0])
            total_trades = len(trades)
            win_rate = (winning_trades / total_trades) * 100
            
            st.success(f"**🎯 Başarı Oranı: {win_rate:.1f}%** ({winning_trades}/{total_trades} işlem)")
            
            st.subheader("📈 PERFORMANS GRAFİKLERİ")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Equity curve
            ax1.plot(equity['date'], equity['equity'], color='green', linewidth=2, label='Portföy Değeri')
            ax1.set_title('EQUITY CURVE', fontweight='bold', fontsize=14)
            ax1.set_ylabel('Portföy ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Drawdown
            equity_series = equity.set_index('date')['equity']
            rolling_max = equity_series.expanding().max()
            drawdown = (equity_series - rolling_max) / rolling_max * 100
            
            ax2.fill_between(equity['date'], drawdown.values, 0, alpha=0.3, color='red', label='Drawdown')
            ax2.set_title('DRAWDOWN', fontweight='bold', fontsize=14)
            ax2.set_ylabel('Drawdown %')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.subheader("📋 DETAYLI İŞLEM LİSTESİ")
            display_trades = trades.copy()
            display_trades['entry_date'] = display_trades['entry_date'].dt.strftime('%Y-%m-%d')
            display_trades['exit_date'] = display_trades['exit_date'].dt.strftime('%Y-%m-%d')
            st.dataframe(display_trades)
            
        else:
            st.warning("""
            **🤔 KALİTELİ SİNYAL BULUNAMADI!**
            
            **Çözüm Önerileri:**
            - RSI değerini 38-40'a çıkar
            - BTC-USD veya TSLA deneyin
            - Tarih aralığını genişletin
            - ATR çarpanını 1.2'ye düşürün
            """)
            
    except Exception as e:
        st.error(f"❌ Hata: {str(e)}")

st.markdown("---")
st.markdown("**🎯 PRO SWING STRATEJİ v3.0 | 5 İndikatörlü Akıllı Sistem**")