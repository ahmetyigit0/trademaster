import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats

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
# GELİŞMİŞ BACKTEST MOTORU
# =========================
class AdvancedSwingBacktest:
    def __init__(self):
        self.commission = 0.001
    
    def calculate_advanced_indicators(self, df):
        df = df.copy()
        
        # Trend Göstergeleri
        df['EMA_20'] = df['Close'].ewm(span=20, min_periods=1).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, min_periods=1).mean()
        df['EMA_100'] = df['Close'].ewm(span=100, min_periods=1).mean()
        df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
        
        # MACD
        exp1 = df['Close'].ewm(span=12, min_periods=1).mean()
        exp2 = df['Close'].ewm(span=26, min_periods=1).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, min_periods=1).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI (Gelişmiş)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Stokastik RSI
        rsi_min = df['RSI'].rolling(window=14, min_periods=1).min()
        rsi_max = df['RSI'].rolling(window=14, min_periods=1).max()
        df['Stoch_RSI'] = (df['RSI'] - rsi_min) / (rsi_max - rsi_min) * 100
        
        # Bollinger Bantları
        df['BB_Middle'] = df['Close'].rolling(window=20, min_periods=1).mean()
        bb_std = df['Close'].rolling(window=20, min_periods=1).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift(1))
        low_close = np.abs(df['Low'] - df['Close'].shift(1))
        
        true_range = np.maximum(np.maximum(high_low, high_close), low_close)
        df['ATR'] = true_range.rolling(window=14, min_periods=1).mean()
        
        # Volume Göstergeleri
        df['Volume_SMA'] = df['Volume'].rolling(window=20, min_periods=1).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Momentum Göstergeleri
        df['Momentum'] = df['Close'] - df['Close'].shift(5)
        df['Rate_of_Change'] = (df['Close'] / df['Close'].shift(10) - 1) * 100
        
        # Price Position
        df['Price_vs_EMA20'] = (df['Close'] / df['EMA_20'] - 1) * 100
        df['Price_vs_EMA50'] = (df['Close'] / df['EMA_50'] - 1) * 100
        
        # Support/Resistance Seviyeleri
        df['Resistance'] = df['High'].rolling(window=20, min_periods=1).max()
        df['Support'] = df['Low'].rolling(window=20, min_periods=1).min()
        df['Distance_to_Resistance'] = (df['Resistance'] - df['Close']) / df['Close'] * 100
        df['Distance_to_Support'] = (df['Close'] - df['Support']) / df['Close'] * 100
        
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    def calculate_buy_score(self, row, params):
        score = 0
        reasons = []
        
        try:
            # Trend Filtresi (30 puan)
            trend_score = 0
            if row['EMA_20'] > row['EMA_50']:
                trend_score += 10
                reasons.append("EMA20 > EMA50")
            if row['EMA_50'] > row['EMA_100']:
                trend_score += 10
                reasons.append("EMA50 > EMA100")
            if row['Close'] > row['EMA_20']:
                trend_score += 10
                reasons.append("Price > EMA20")
            
            score += trend_score
            
            # RSI ve Momentum (25 puan)
            momentum_score = 0
            if row['RSI'] < params['rsi_oversold']:
                momentum_score += 10
                reasons.append(f"RSI ({row['RSI']:.1f}) < {params['rsi_oversold']}")
            if row['Stoch_RSI'] < 20:
                momentum_score += 5
                reasons.append(f"StochRSI ({row['Stoch_RSI']:.1f}) oversold")
            if row['MACD_Histogram'] > 0:
                momentum_score += 5
                reasons.append("MACD Histogram positive")
            if row['Momentum'] > 0:
                momentum_score += 5
                reasons.append("5-day Momentum positive")
            
            score += momentum_score
            
            # Bollinger Bantları (20 puan)
            bb_score = 0
            if row['BB_Position'] < 0.2:  # Alt bantta
                bb_score += 10
                reasons.append(f"BB Position ({row['BB_Position']:.2f}) oversold")
            if row['BB_Width'] > params['min_bb_width']:
                bb_score += 10
                reasons.append(f"BB Width adequate ({row['BB_Width']:.3f})")
            
            score += bb_score
            
            # Volume ve Fiyat Hareketi (25 puan)
            volume_score = 0
            if row['Volume_Ratio'] > params['min_volume_ratio']:
                volume_score += 10
                reasons.append(f"Volume spike ({row['Volume_Ratio']:.1f}x)")
            if row['Distance_to_Resistance'] > params['min_resistance_distance']:
                volume_score += 10
                reasons.append(f"Good resistance distance ({row['Distance_to_Resistance']:.1f}%)")
            if row['Price_vs_EMA20'] > -2:  # EMA20'den çok uzak değil
                volume_score += 5
                reasons.append("Close to EMA20")
            
            score += volume_score
            
            # Ek Filtreler
            if row['Rate_of_Change'] > -5:  % 10 günlük düşüş -5%'ten az
                score += 5
                reasons.append("10-day RoC reasonable")
            
            return min(score, 100), reasons
            
        except Exception as e:
            return 0, [f"Error: {str(e)}"]
    
    def generate_advanced_signals(self, df, params):
        signals = []
        
        for i in range(1, len(df)):
            try:
                row = df.iloc[i]
                prev_row = df.iloc[i-1]
                
                # Alım koşullarını kontrol et
                buy_score, reasons = self.calculate_buy_score(row, params)
                
                if buy_score >= params['min_buy_score']:
                    # Risk yönetimi
                    atr_val = float(row['ATR'])
                    stop_loss = row['Close'] - (atr_val * params['atr_multiplier'])
                    take_profit = row['Close'] + (atr_val * params['atr_multiplier'] * params['reward_ratio'])
                    
                    # Pozisyon büyüklüğü
                    position_size = min(params['max_position_size'], 
                                      buy_score / 100.0 * params['position_size_factor'])
                    
                    signals.append({
                        'date': df.index[i],
                        'action': 'buy',
                        'score': buy_score,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'position_size': position_size,
                        'reasons': reasons,
                        'price': row['Close']
                    })
                else:
                    signals.append({
                        'date': df.index[i],
                        'action': 'hold',
                        'score': buy_score
                    })
                    
            except Exception as e:
                signals.append({
                    'date': df.index[i],
                    'action': 'hold',
                    'score': 0
                })
        
        signals_df = pd.DataFrame(signals)
        if not signals_df.empty:
            signals_df = signals_df.set_index('date')
        
        return signals_df
    
    def run_advanced_backtest(self, data, params):
        df = self.calculate_advanced_indicators(data)
        signals = self.generate_advanced_signals(df, params)
        
        capital = params['initial_capital']
        position = None
        trades = []
        equity_curve = []
        max_drawdown = 0
        peak_equity = capital
        
        for date in df.index:
            current_price = float(df.loc[date, 'Close'])
            signal = signals.loc[date] if date in signals.index else {'action': 'hold', 'score': 0}
            
            # Mevcut equity'yi hesapla
            current_equity = capital
            if position is not None:
                current_equity += position['shares'] * current_price
            
            # Drawdown hesapla
            if current_equity > peak_equity:
                peak_equity = current_equity
            drawdown = (peak_equity - current_equity) / peak_equity * 100
            max_drawdown = max(max_drawdown, drawdown)
            
            equity_curve.append({
                'date': date, 
                'equity': current_equity,
                'drawdown': drawdown
            })
            
            # Alım sinyali
            if position is None and signal['action'] == 'buy':
                stop_loss = float(signal['stop_loss'])
                risk_per_share = current_price - stop_loss
                
                if risk_per_share > 0:
                    risk_amount = capital * params['risk_per_trade'] * signal['position_size']
                    shares = risk_amount / risk_per_share
                    
                    max_shares_by_capital = (capital * signal['position_size']) / current_price
                    shares = min(shares, max_shares_by_capital)
                    
                    if shares > 0:
                        position = {
                            'entry_date': date,
                            'entry_price': current_price,
                            'shares': shares,
                            'stop_loss': stop_loss,
                            'take_profit': float(signal['take_profit']),
                            'score': signal['score'],
                            'reasons': signal['reasons']
                        }
                        capital -= shares * current_price
            
            # Pozisyon yönetimi
            elif position is not None:
                # Stop Loss
                if current_price <= position['stop_loss']:
                    exit_price = position['stop_loss']
                    self.close_position(capital, position, exit_price, date, 'SL', trades)
                    position = None
                
                # Take Profit
                elif current_price >= position['take_profit']:
                    exit_price = position['take_profit']
                    self.close_position(capital, position, exit_price, date, 'TP', trades)
                    position = None
                
                # Trailing Stop (isteğe bağlı)
                elif params['use_trailing_stop'] and current_price > position['entry_price']:
                    new_stop = current_price - (float(df.loc[date, 'ATR']) * params['trailing_atr_multiplier'])
                    if new_stop > position['stop_loss']:
                        position['stop_loss'] = new_stop
        
        # Açık pozisyonları kapat
        if position is not None:
            last_price = float(df['Close'].iloc[-1])
            self.close_position(capital, position, last_price, df.index[-1], 'OPEN', trades)
        
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        equity_df = pd.DataFrame(equity_curve)
        
        return trades_df, equity_df, max_drawdown
    
    def close_position(self, capital, position, exit_price, exit_date, exit_reason, trades):
        exit_value = position['shares'] * exit_price
        capital += exit_value
        
        entry_value = position['shares'] * position['entry_price']
        pnl = exit_value - entry_value
        pnl_pct = (pnl / entry_value) * 100
        
        trades.append({
            'entry_date': position['entry_date'],
            'exit_date': exit_date,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'pnl': pnl,
            'return_pct': pnl_pct,
            'exit_reason': exit_reason,
            'score': position['score'],
            'reasons': position['reasons'],
            'holding_days': (exit_date - position['entry_date']).days
        })
    
    def calculate_advanced_metrics(self, trades_df, equity_df, max_drawdown, initial_capital):
        if trades_df.empty:
            return {
                'total_return': "0.0%",
                'total_trades': "0",
                'win_rate': "0.0%",
                'avg_win': "$0.00",
                'avg_loss': "$0.00",
                'profit_factor': "0.00",
                'max_drawdown': "0.0%",
                'sharpe_ratio': "0.00",
                'avg_holding_days': "0"
            }
        
        try:
            initial_equity = float(initial_capital)
            final_equity = float(equity_df['equity'].iloc[-1])
            total_return = (final_equity - initial_equity) / initial_equity * 100.0
            
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            win_rate = (winning_trades / total_trades) * 100.0 if total_trades > 0 else 0.0
            
            avg_win = float(trades_df[trades_df['pnl'] > 0]['pnl'].mean()) if winning_trades > 0 else 0.0
            avg_loss = float(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if (total_trades - winning_trades) > 0 else 0.0
            
            total_gains = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
            total_losses = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
            profit_factor = total_gains / total_losses if total_losses > 0 else float('inf')
            
            # Sharpe Ratio (basitleştirilmiş)
            returns = trades_df['return_pct'] / 100.0
            if len(returns) > 1:
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            else:
                sharpe_ratio = 0
            
            avg_holding_days = trades_df['holding_days'].mean() if 'holding_days' in trades_df.columns else 0
            
            return {
                'total_return': f"{round(total_return, 2)}%",
                'total_trades': str(total_trades),
                'win_rate': f"{round(win_rate, 1)}%",
                'avg_win': f"${round(avg_win, 2)}",
                'avg_loss': f"${round(avg_loss, 2)}",
                'profit_factor': f"{round(profit_factor, 2)}" if profit_factor != float('inf') else "∞",
                'max_drawdown': f"{round(max_drawdown, 2)}%",
                'sharpe_ratio': f"{round(sharpe_ratio, 2)}",
                'avg_holding_days': f"{round(avg_holding_days, 1)}"
            }
            
        except Exception as e:
            return {
                'total_return': "0.0%",
                'total_trades': "0",
                'win_rate': "0.0%",
                'avg_win': "$0.00",
                'avg_loss': "$0.00",
                'profit_factor': "0.00",
                'max_drawdown': "0.0%",
                'sharpe_ratio': "0.00",
                'avg_holding_days': "0"
            }

# =========================
# GELİŞMİŞ STREAMLIT UYGULAMASI
# =========================
st.set_page_config(page_title="Advanced Swing Backtest", layout="wide")
st.title("🚀 Gelişmiş Swing Trading Backtest Sistemi")

# Sidebar - Gelişmiş Ayarlar
st.sidebar.header("⚙️ Temel Ayarlar")
ticker = st.sidebar.selectbox("Sembol", ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMD", "BTC-USD", "ETH-USD", "SPY", "QQQ"])
start_date = st.sidebar.date_input("Başlangıç", datetime(2023, 1, 1))
end_date = st.sidebar.date_input("Bitiş", datetime(2024, 1, 1))
initial_capital = st.sidebar.number_input("Başlangıç Sermayesi ($)", 1000, 1000000, 10000)

st.sidebar.header("🎯 Strateji Parametreleri")

# Skorlama Parametreleri
st.sidebar.subheader("Skorlama Sistemi")
min_buy_score = st.sidebar.slider("Minimum Alım Skoru", 50, 80, 65)
rsi_oversold = st.sidebar.slider("RSI Aşırı Satım", 25, 40, 32)
min_bb_width = st.sidebar.slider("Min Bollinger Genişliği", 0.01, 0.10, 0.03)
min_volume_ratio = st.sidebar.slider("Min Volume Çarpanı", 1.0, 3.0, 1.5)
min_resistance_distance = st.sidebar.slider("Min Direnç Mesafesi %", 2.0, 10.0, 5.0)

# Risk Yönetimi
st.sidebar.subheader("Risk Yönetimi")
risk_per_trade = st.sidebar.slider("Risk %", 0.5, 3.0, 1.5) / 100
atr_multiplier = st.sidebar.slider("ATR Çarpanı", 1.0, 3.0, 1.8)
reward_ratio = st.sidebar.slider("Ödül/Risk Oranı", 1.5, 4.0, 2.5)
max_position_size = st.sidebar.slider("Maks Pozisyon Büyüklüğü", 0.1, 1.0, 0.5)
position_size_factor = st.sidebar.slider("Pozisyon Büyüklük Çarpanı", 0.5, 2.0, 1.2)

# Gelişmiş Ayarlar
st.sidebar.subheader("Gelişmiş Ayarlar")
use_trailing_stop = st.sidebar.checkbox("Trailing Stop Kullan", value=True)
trailing_atr_multiplier = st.sidebar.slider("Trailing Stop ATR Çarpanı", 1.0, 3.0, 2.0) if use_trailing_stop else 2.0

# Ana içerik
if st.button("🎯 Gelişmiş Backtest Çalıştır"):
    try:
        with st.spinner("Veri yükleniyor ve göstergeler hesaplanıyor..."):
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                st.error("❌ Veri bulunamadı")
                st.stop()
            
            st.success(f"✅ {len(data)} günlük veri yüklendi")
        
        # Parametreleri hazırla
        params = {
            'rsi_oversold': rsi_oversold,
            'atr_multiplier': atr_multiplier,
            'risk_per_trade': risk_per_trade,
            'min_buy_score': min_buy_score,
            'min_bb_width': min_bb_width,
            'min_volume_ratio': min_volume_ratio,
            'min_resistance_distance': min_resistance_distance,
            'reward_ratio': reward_ratio,
            'max_position_size': max_position_size,
            'position_size_factor': position_size_factor,
            'use_trailing_stop': use_trailing_stop,
            'trailing_atr_multiplier': trailing_atr_multiplier,
            'initial_capital': initial_capital
        }
        
        backtester = AdvancedSwingBacktest()
        
        with st.spinner("Gelişmiş backtest çalıştırılıyor..."):
            trades, equity, max_drawdown = backtester.run_advanced_backtest(data, params)
            metrics = backtester.calculate_advanced_metrics(trades, equity, max_drawdown, initial_capital)
        
        # Sonuçları göster
        st.subheader("📊 Detaylı Performans Özeti")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Toplam Getiri", metrics['total_return'])
            st.metric("Toplam İşlem", metrics['total_trades'])
            st.metric("Win Rate", metrics['win_rate'])
        
        with col2:
            st.metric("Ort. Kazanç", metrics['avg_win'])
            st.metric("Ort. Kayıp", metrics['avg_loss'])
            st.metric("Profit Factor", metrics['profit_factor'])
        
        with col3:
            st.metric("Max Drawdown", metrics['max_drawdown'])
            st.metric("Sharpe Ratio", metrics['sharpe_ratio'])
            st.metric("Ort. Tutma Süresi", metrics['avg_holding_days'])
        
        with col4:
            final_equity = float(equity['equity'].iloc[-1])
            st.metric("Final Sermaye", f"${final_equity:,.2f}")
            net_profit = final_equity - initial_capital
            st.metric("Net Kar", f"${net_profit:,.2f}")
        
        if not trades.empty:
            # Grafikler
            st.subheader("📈 Detaylı Grafikler")
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Equity Curve
            ax1.plot(equity['date'], equity['equity'], color='green', linewidth=2)
            ax1.set_title('Portföy Değeri')
            ax1.set_ylabel('Equity ($)')
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # Drawdown
            ax2.fill_between(equity['date'], equity['drawdown'], 0, color='red', alpha=0.3)
            ax2.plot(equity['date'], equity['drawdown'], color='red', linewidth=1)
            ax2.set_title('Drawdown %')
            ax2.set_ylabel('Drawdown (%)')
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
            
            # Trade Returns Distribution
            returns = trades['return_pct']
            ax3.hist(returns, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
            ax3.axvline(returns.mean(), color='red', linestyle='--', label=f'Ortalama: {returns.mean():.2f}%')
            ax3.set_title('İşlem Getirileri Dağılımı')
            ax3.set_xlabel('Getiri (%)')
            ax3.set_ylabel('Frekans')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Exit Reasons
            exit_reasons = trades['exit_reason'].value_counts()
            ax4.pie(exit_reasons.values, labels=exit_reasons.index, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Çıkış Nedenleri')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Detaylı İşlem Listesi
            st.subheader("📋 Detaylı İşlem Listesi")
            
            display_trades = trades.copy()
            display_trades['entry_date'] = display_trades['entry_date'].dt.strftime('%Y-%m-%d')
            display_trades['exit_date'] = display_trades['exit_date'].dt.strftime('%Y-%m-%d')
            display_trades['entry_price'] = display_trades['entry_price'].round(2)
            display_trades['exit_price'] = display_trades['exit_price'].round(2)
            display_trades['pnl'] = display_trades['pnl'].round(2)
            display_trades['return_pct'] = display_trades['return_pct'].round(2)
            display_trades['score'] = display_trades['score'].round(1)
            
            # Renkli gösterim için
            def color_pnl(val):
                color = 'green' if val > 0 else 'red' if val < 0 else 'black'
                return f'color: {color}'
            
            st.dataframe(
                display_trades.style.applymap(color_pnl, subset=['pnl', 'return_pct']),
                height=400
            )
            
            # İşlem Detayları
            st.subheader("🔍 Örnek İşlem Analizi")
            if len(trades) > 0:
                best_trade = trades.loc[trades['return_pct'].idxmax()]
                worst_trade = trades.loc[trades['return_pct'].idxmin()]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("🎯 En İyi İşlem:")
                    st.write(f"Getiri: {best_trade['return_pct']:.2f}%")
                    st.write(f"Skor: {best_trade['score']:.1f}")
                    st.write("Nedenler:", best_trade['reasons'][:3])
                
                with col2:
                    st.write("💥 En Kötü İşlem:")
                    st.write(f"Getiri: {worst_trade['return_pct']:.2f}%")
                    st.write(f"Skor: {worst_trade['score']:.1f}")
                    st.write("Nedenler:", worst_trade['reasons'][:3])
            
        else:
            st.info("🤷 Hiç işlem gerçekleşmedi. Parametreleri gevşetmeyi deneyin.")
            
    except Exception as e:
        st.error(f"❌ Hata: {str(e)}")
        st.info("💡 İpucu: Tarih aralığını veya parametreleri değiştirmeyi deneyin")

# Strateji Açıklaması
with st.expander("ℹ️ Strateji Detayları"):
    st.markdown("""
    **Gelişmiş Swing Trading Stratejisi:**
    
    - **Trend Analizi:** EMA20 > EMA50 > EMA100
    - **Momentum:** RSI, StochRSI, MACD
    - **Volatilite:** Bollinger Bantları, ATR
    - **Volume:** Volume spike analizi
    - **Skorlama Sistemi:** Çoklu faktör puanlaması
    - **Risk Yönetimi:** Dinamik pozisyon büyüklüğü
    
    **Özellikler:**
    - Akıllı alım skorlama (0-100 puan)
    - Dinamik stop-loss ve take-profit
    - Trailing stop opsiyonu
    - Detaylı performans metrikleri
    """)

st.markdown("---")
st.markdown("**🚀 Gelişmiş Swing Trading Backtest Sistemi**")
