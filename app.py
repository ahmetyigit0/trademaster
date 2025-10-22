import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =========================
# ŞİFRE KORUMASI - GELİŞTİRİLMİŞ
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
# GELİŞMİŞ BACKTEST MOTORU - DÜZELTİLMİŞ
# =========================
class AdvancedSwingBacktest:
    def __init__(self):
        self.commission = 0.001
        self.initial_capital = 10000
    
    def calculate_advanced_indicators(self, df):
        df = df.copy()
        
        try:
            # Trend EMA'ları
            for period in [8, 13, 20, 50, 200]:
                df[f'EMA_{period}'] = df['Close'].ewm(span=period, min_periods=1).mean()
            
            # RSI - Düzeltilmiş hesaplama
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['Close'].ewm(span=12, min_periods=1).mean()
            exp2 = df['Close'].ewm(span=26, min_periods=1).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9, min_periods=1).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            # Bollinger Bands - DÜZELTİLDİ
            df['BB_Middle'] = df['Close'].rolling(window=20, min_periods=1).mean()
            bb_std = df['Close'].rolling(window=20, min_periods=1).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
            
            # Stochastic
            low_14 = df['Low'].rolling(window=14, min_periods=1).min()
            high_14 = df['High'].rolling(window=14, min_periods=1).max()
            df['STOCH_K'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
            df['STOCH_D'] = df['STOCH_K'].rolling(window=3, min_periods=1).mean()
            
            # ATR - Düzeltilmiş
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift(1))
            low_close = np.abs(df['Low'] - df['Close'].shift(1))
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = true_range.max(axis=1)
            df['ATR'] = true_range.rolling(window=14, min_periods=1).mean()
            
            # Volume SMA
            df['Volume_SMA'] = df['Volume'].rolling(window=20, min_periods=1).mean()
            
            # Support/Resistance
            df['Resistance'] = df['High'].rolling(window=20, min_periods=1).max()
            df['Support'] = df['Low'].rolling(window=20, min_periods=1).min()
            
            # Tüm NaN değerleri doldur
            df = df.fillna(method='bfill').fillna(method='ffill')
            
            return df
            
        except Exception as e:
            st.error(f"Gösterge hesaplama hatası: {e}")
            return df
    
    def generate_signals(self, df, params):
        signals = []
        in_position = False
        
        for i in range(len(df)):
            try:
                if i < 50:  # Yeterli veri kontrolü
                    signals.append({'date': df.index[i], 'action': 'hold', 'strength': 0})
                    continue
                
                row = df.iloc[i]
                prev_row = df.iloc[i-1] if i > 0 else row
                
                # Değerleri güvenli şekilde al
                close = float(row['Close'])
                ema_20 = float(row['EMA_20'])
                ema_50 = float(row['EMA_50'])
                ema_200 = float(row.get('EMA_200', ema_50))  # Fallback
                rsi = float(row['RSI'])
                macd_hist = float(row['MACD_Hist'])
                bb_lower = float(row['BB_Lower'])
                bb_upper = float(row['BB_Upper'])
                atr = float(row['ATR'])
                stoch_k = float(row['STOCH_K'])
                stoch_d = float(row['STOCH_D'])
                volume = float(row['Volume'])
                volume_sma = float(row['Volume_SMA'])
                
                # Trend analizi
                trend_up = ema_20 > ema_50
                price_above_ema20 = close > ema_20
                
                # Momentum göstergeleri
                rsi_oversold = rsi < params['rsi_oversold']
                rsi_overbought = rsi > params['rsi_overbought']
                stoch_oversold = stoch_k < 20 and stoch_d < 20
                stoch_bullish_cross = stoch_k > stoch_d and float(prev_row['STOCH_K']) <= float(prev_row['STOCH_D'])
                macd_bullish = macd_hist > 0 and float(prev_row['MACD_Hist']) <= 0
                
                # Volatilite ve konum
                near_bb_lower = close <= bb_lower * 1.01
                near_bb_upper = close >= bb_upper * 0.99
                high_volume = volume > volume_sma * 1.2
                
                # Sinyal gücü hesaplama
                buy_signals = []
                
                # ALIŞ sinyalleri
                if not in_position:
                    if trend_up and rsi_oversold and near_bb_lower:
                        buy_signals.append(3)  # Güçlü sinyal
                    if trend_up and stoch_oversold and stoch_bullish_cross:
                        buy_signals.append(2)
                    if trend_up and macd_bullish and high_volume:
                        buy_signals.append(2)
                    if trend_up and rsi_oversold and stoch_oversold:
                        buy_signals.append(1)
                    
                    if buy_signals and sum(buy_signals) >= params['min_signal_strength']:
                        stop_loss = close - (atr * params['atr_multiplier'])
                        take_profit = close + (atr * params['atr_multiplier'] * params['reward_ratio'])
                        
                        signals.append({
                            'date': df.index[i],
                            'action': 'buy',
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'strength': sum(buy_signals),
                            'rsi': rsi,
                            'volume_ratio': volume/volume_sma
                        })
                        in_position = True
                    else:
                        signals.append({'date': df.index[i], 'action': 'hold', 'strength': 0})
                
                # SATIŞ sinyalleri
                elif in_position:
                    sell_signals = []
                    
                    if rsi_overbought and near_bb_upper:
                        sell_signals.append(2)
                    if stoch_k > 80 and stoch_d > 80:
                        sell_signals.append(1)
                    if not trend_up:
                        sell_signals.append(1)
                    
                    if sell_signals and sum(sell_signals) >= 1:
                        signals.append({
                            'date': df.index[i], 
                            'action': 'sell', 
                            'strength': sum(sell_signals)
                        })
                        in_position = False
                    else:
                        signals.append({'date': df.index[i], 'action': 'hold', 'strength': 0})
                        
            except Exception as e:
                signals.append({'date': df.index[i], 'action': 'hold', 'strength': 0})
        
        signals_df = pd.DataFrame(signals)
        if not signals_df.empty:
            signals_df = signals_df.set_index('date')
        return signals_df
    
    def run_backtest(self, data, params):
        df = self.calculate_advanced_indicators(data)
        signals = self.generate_signals(df, params)
        
        capital = self.initial_capital
        position = None
        trades = []
        equity_curve = []
        max_drawdown = 0
        peak_equity = capital
        
        for i, date in enumerate(df.index):
            if date not in signals.index:
                # Sinyal yoksa hold olarak devam et
                current_price = float(df.loc[date, 'Close'])
                current_equity = capital
                if position is not None:
                    current_equity += position['shares'] * current_price
                
                equity_curve.append({
                    'date': date, 
                    'equity': current_equity,
                    'drawdown': 0
                })
                continue
                
            current_price = float(df.loc[date, 'Close'])
            signal = signals.loc[date]
            
            # Equity hesaplama
            current_equity = capital
            if position is not None:
                current_equity += position['shares'] * current_price
            
            # Drawdown hesaplama
            if current_equity > peak_equity:
                peak_equity = current_equity
            drawdown = (peak_equity - current_equity) / peak_equity * 100 if peak_equity > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
            
            equity_curve.append({
                'date': date, 
                'equity': current_equity,
                'drawdown': drawdown
            })
            
            # ALIŞ sinyali
            if position is None and signal['action'] == 'buy':
                stop_loss = float(signal['stop_loss'])
                risk_per_share = current_price - stop_loss
                
                if risk_per_share > 0:
                    risk_amount = capital * params['risk_per_trade']
                    shares = risk_amount / risk_per_share
                    shares = min(shares, capital / current_price)  # Margin kontrolü
                    
                    if shares > 0:
                        # Komisyon dahil
                        commission_cost = shares * current_price * self.commission
                        if capital >= (shares * current_price + commission_cost):
                            position = {
                                'entry_date': date,
                                'entry_price': current_price,
                                'shares': shares,
                                'stop_loss': stop_loss,
                                'take_profit': float(signal['take_profit']),
                                'signal_strength': signal['strength']
                            }
                            capital -= (shares * current_price + commission_cost)
            
            # SATIŞ veya ÇIKIŞ sinyali
            elif position is not None:
                exit_signal = False
                exit_reason = ""
                exit_price = current_price
                
                # Stop Loss
                if current_price <= position['stop_loss']:
                    exit_signal = True
                    exit_reason = "SL"
                    exit_price = position['stop_loss']
                
                # Take Profit
                elif current_price >= position['take_profit']:
                    exit_signal = True
                    exit_reason = "TP"
                    exit_price = position['take_profit']
                
                # Satış sinyali
                elif signal['action'] == 'sell':
                    exit_signal = True
                    exit_reason = "SELL_SIGNAL"
                
                if exit_signal:
                    # Komisyon dahil
                    commission_cost = position['shares'] * exit_price * self.commission
                    exit_value = position['shares'] * exit_price - commission_cost
                    capital += exit_value
                    
                    entry_value = position['shares'] * position['entry_price']
                    pnl = exit_value - entry_value
                    pnl_pct = (pnl / entry_value) * 100 if entry_value > 0 else 0
                    
                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': date,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'shares': position['shares'],
                        'pnl': pnl,
                        'return_pct': pnl_pct,
                        'exit_reason': exit_reason,
                        'signal_strength': position['signal_strength'],
                        'holding_days': (date - position['entry_date']).days
                    })
                    position = None
        
        # Açık pozisyonu kapat
        if position is not None:
            last_price = float(df['Close'].iloc[-1])
            commission_cost = position['shares'] * last_price * self.commission
            exit_value = position['shares'] * last_price - commission_cost
            capital += exit_value
            
            entry_value = position['shares'] * position['entry_price']
            pnl = exit_value - entry_value
            pnl_pct = (pnl / entry_value) * 100 if entry_value > 0 else 0
            
            trades.append({
                'entry_date': position['entry_date'],
                'exit_date': df.index[-1],
                'entry_price': position['entry_price'],
                'exit_price': last_price,
                'shares': position['shares'],
                'pnl': pnl,
                'return_pct': pnl_pct,
                'exit_reason': 'FORCE_CLOSE',
                'signal_strength': position['signal_strength'],
                'holding_days': (df.index[-1] - position['entry_date']).days
            })
        
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        equity_df = pd.DataFrame(equity_curve)
        
        return trades_df, equity_df, max_drawdown
    
    def calculate_advanced_metrics(self, trades_df, equity_df, max_drawdown):
        if trades_df.empty:
            return self._get_empty_metrics()
        
        try:
            initial_equity = self.initial_capital
            final_equity = float(equity_df['equity'].iloc[-1])
            total_return_pct = (final_equity - initial_equity) / initial_equity * 100
            
            # Temel metrikler
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            # Kazanç/kayıp oranları
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if (total_trades - winning_trades) > 0 else 0
            profit_factor = abs(avg_win * winning_trades) / abs(avg_loss * (total_trades - winning_trades)) if avg_loss != 0 else float('inf')
            
            # Getiri istatistikleri
            best_trade = trades_df['return_pct'].max() if not trades_df.empty else 0
            worst_trade = trades_df['return_pct'].min() if not trades_df.empty else 0
            avg_trade_return = trades_df['return_pct'].mean() if not trades_df.empty else 0
            
            # Sharpe Ratio (basitleştirilmiş)
            equity_returns = equity_df['equity'].pct_change().dropna()
            sharpe_ratio = equity_returns.mean() / equity_returns.std() * np.sqrt(252) if len(equity_returns) > 1 and equity_returns.std() > 0 else 0
            
            # Sinyal gücü analizi
            strong_signals = trades_df[trades_df['signal_strength'] >= 3]
            strong_win_rate = (len(strong_signals[strong_signals['pnl'] > 0]) / len(strong_signals) * 100) if len(strong_signals) > 0 else 0
            
            return {
                'total_return': f"{total_return_pct:+.2f}%",
                'total_trades': str(total_trades),
                'win_rate': f"{win_rate:.1f}%",
                'profit_factor': f"{profit_factor:.2f}",
                'avg_win': f"${avg_win:.2f}",
                'avg_loss': f"${avg_loss:.2f}",
                'best_trade': f"{best_trade:.2f}%",
                'worst_trade': f"{worst_trade:.2f}%",
                'avg_trade': f"{avg_trade_return:.2f}%",
                'max_drawdown': f"{max_drawdown:.2f}%",
                'sharpe_ratio': f"{sharpe_ratio:.2f}",
                'strong_signal_win_rate': f"{strong_win_rate:.1f}%"
            }
            
        except Exception as e:
            st.error(f"Metrik hesaplama hatası: {e}")
            return self._get_empty_metrics()
    
    def _get_empty_metrics(self):
        return {key: "0" for key in [
            'total_return', 'total_trades', 'win_rate', 'profit_factor',
            'avg_win', 'avg_loss', 'best_trade', 'worst_trade', 'avg_trade',
            'max_drawdown', 'sharpe_ratio', 'strong_signal_win_rate'
        ]}

# =========================
# STREAMLET UYGULAMASI
# =========================
st.set_page_config(page_title="Advanced Swing Backtest", layout="wide")

st.title("🚀 Advanced Swing Trading Backtest")
st.markdown("**Profesyonel Swing Trading Stratejisi | Risk Yönetimi | Detaylı Analiz**")

# Sidebar
st.sidebar.header("⚙️ Temel Ayarlar")
ticker = st.sidebar.selectbox("Sembol", ["BTC-USD", "ETH-USD", "ADA-USD", "TSLA", "NVDA", "AAPL", "GOOGL", "MSFT", "SPY", "QQQ"])
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Başlangıç", datetime(2023, 1, 1))
with col2:
    end_date = st.date_input("Bitiş", datetime(2024, 1, 1))

st.sidebar.header("📊 Strateji Parametreleri")
rsi_oversold = st.sidebar.slider("RSI Aşırı Satım", 20, 40, 30)
rsi_overbought = st.sidebar.slider("RSI Aşırı Alım", 60, 80, 70)
atr_multiplier = st.sidebar.slider("ATR Çarpanı (Stop Loss)", 1.0, 3.0, 1.5)
reward_ratio = st.sidebar.slider("Risk/Ödül Oranı", 1.5, 4.0, 2.0)
risk_per_trade = st.sidebar.slider("İşlem Başı Risk %", 0.5, 5.0, 2.0) / 100
min_signal_strength = st.sidebar.slider("Minimum Sinyal Gücü", 1, 5, 3)

params = {
    'rsi_oversold': rsi_oversold,
    'rsi_overbought': rsi_overbought,
    'atr_multiplier': atr_multiplier,
    'reward_ratio': reward_ratio,
    'risk_per_trade': risk_per_trade,
    'min_signal_strength': min_signal_strength
}

# Ana içerik
if st.button("🎯 Backtest Çalıştır", type="primary", use_container_width=True):
    try:
        with st.spinner("📊 Finansal veriler yükleniyor..."):
            # Daha uzun veri için
            extended_start = start_date - timedelta(days=100)
            data = yf.download(ticker, start=extended_start, end=end_date, progress=False)
            
            if data.empty:
                st.error("❌ Bu tarih aralığında veri bulunamadı")
                st.stop()
            
            # Filtreleme
            data = data[data.index >= pd.to_datetime(start_date)]
            data = data[data.index <= pd.to_datetime(end_date)]
            
            if len(data) < 50:
                st.error("❌ Yeterli veri yok (en az 50 gün gereklidir)")
                st.stop()
            
            st.success(f"✅ {len(data)} işlem günü yüklendi")
        
        # Backtest çalıştır
        backtester = AdvancedSwingBacktest()
        
        with st.spinner("🔍 Teknik analiz yapılıyor ve backtest çalıştırılıyor..."):
            trades, equity, max_drawdown = backtester.run_backtest(data, params)
            metrics = backtester.calculate_advanced_metrics(trades, equity, max_drawdown)
        
        # Sonuçları göster
        st.subheader("📈 Performans Özeti")
        
        # Ana metrikler
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Toplam Getiri", metrics['total_return'])
            st.metric("Toplam İşlem", metrics['total_trades'])
        
        with col2:
            st.metric("Win Rate", metrics['win_rate'])
            st.metric("Profit Factor", metrics['profit_factor'])
        
        with col3:
            st.metric("Ort. Kazanç", metrics['avg_win'])
            st.metric("Ort. Kayıp", metrics['avg_loss'])
        
        with col4:
            st.metric("Max Drawdown", metrics['max_drawdown'])
            st.metric("Sharpe Ratio", metrics['sharpe_ratio'])
        
        # Detaylı metrikler
        if not trades.empty:
            col5, col6.empty:
            col5, col6, col7 = st.columns(, col7 = st.columns(3)
            with col5:
3)
            with col5:
                st.metric("En                st.metric("En İyi İşlem", metrics[' İyi İşlem", metricsbest_trade'])
            with['best_trade'])
            with col6:
                st col6:
                st.m.metric("En Kötüetric("En Kötü İşlem", metrics['worst İşlem", metrics['worst_trade'])
           _trade'])
            with col with col7:
                st.metric7:
                st.metric("G("Güçlü Sinyüçlü Sinyal Win Rate", metrics['al Win Rate", metrics['strong_signal_win_rate'])
        
strong_signal_win_rate'])
        
        if not trades.empty:
        if not trades.empty:
            #            # Grafikler
            st Grafikler
            st.subheader("📊 Perform.subheader("📊 Performans Grafikleri")
            
ans Grafikleri")
            
                       tab1, tab2, tab3 = st.tabs(["Portföy Değeri", " tab1, tab2, tab3 = st.tabs(["Portföy Değeri", "Drawdown", "İşDrawdown", "İşlem Analizi"])
            
           lem Analizi"])
            
            with tab with tab1:
                fig, ax1:
                fig, ax = plt.subplots(fig = plt.subplots(figsize=(12,size=(12, 6))
                ax.plot(equity 6))
                ax.plot(equity['['date'], equity['equitydate'], equity['equity'], color='#00ff88','], color='#00ff linewidth=2)
               88', linewidth=2)
                ax.set_title('Portfö ax.set_title('Portföy Değer Geliy Değer Gelişimişimi', fontsize=14,', fontsize=14, fontweight='bold')
                fontweight='bold')
                ax.set ax.set_xlabel('Tarih')
                ax.set_ylabel_xlabel('Tarih')
                ax.set_ylabel('Portföy Değeri ($('Portföy Değeri ($)')
                ax.grid()')
                ax.grid(True, alpha=0.3)
True, alpha=0.3)
                ax.tick_params(                ax.tick_params(axis='x', rotation=axis='x', rotation=45)
                plt.t45)
                plt.tight_layout()
                st.pyplot(fig)
            
            with tab2:
ight_layout()
                st.pyplot(fig)
            
            with tab2:
                fig, ax = plt                fig, ax = plt.sub.subplots(figsize=(plots(figsize=(12,12, 6 6))
               ))
                ax.fill_between(equ ax.fill_between(equity['ity['date'], equity['drawdate'], equity['drawdowndown'], color='#ff444'], color='#ff4444',4', alpha=0.7 alpha=0.7)
)
                ax.plot(equ                ax.plot(equityity['date'], equity['draw['date'], equity['drawdowndown'], color='#ff4444', linewidth=1)
'], color='#ff4444', linewidth=1)
                ax.set_title('Port                ax.set_title('Portföyföy Drawdown', font Drawdown', fontsize=size=14, fontweight='bold14, fontweight='bold')
                ax.set_xlabel')
                ax.set_xlabel('Tarih')
                ax.set('Tarih')
                ax.set_ylabel('Draw_ylabel('Drawdown (%)')
                ax.grid(down (%)')
                ax.grid(TrueTrue, alpha=0.3, alpha=0.3)
                ax.tick_params)
                ax.tick_params(axis(axis='x', rotation=='x', rotation=4545)
                plt.tight)
                plt.tight_layout_layout()
                st.pyplot(fig()
                st.pyplot(fig)
            
            with tab)
            
            with tab3:
                # İ3:
                # İşlem sonuçları dağılımşlem sonuçları dağılımı
                figı
                fig, ax, ax = plt.subplots(f = plt.subplots(figsize=(12, 6igsize=(12, 6))
                colors =))
                colors = ['green' if x > 0 else ['green' if x > 0 else ' 'red' for x in tradesred' for x in trades['return_pct['return_pct']']]
                bars = ax.bar]
                bars = ax.bar(range(len(trades)), trades(range(len(trades)), trades['return_pct'], color['return_pct'], color=colors, alpha=0.7=colors, alpha=0.7)
                ax.set_title(')
                ax.set_title('İşlem Getirileriİşlem Getirileri Dağılımı', font Dağılımı', fontsize=14, fontweightsize=14, fontweight='bold')
                ax.set='bold')
                ax.set_xlabel('İ_xlabel('İşlem No')
şlem No')
                ax                ax.set_ylabel('Getiri.set_ylabel('Getiri (%)')
                ax.grid(True, (%)')
                ax.grid(True, alpha=0.3)
 alpha=0.3)
                plt                plt.tight.tight_layout()
                st.pyplot(fig)
            
            # İstatistikler
            st_layout()
                st.pyplot(fig)
            
            # İstatistikler
            st.subheader("📊 İş.subheader("📊 İşlem İstatlem İstatistikleriistikleri")
           ")
            col1 col1, col2, col2 = st.columns = st.columns(2)
            
(2)
            
            with col1            with col1:
                st:
                st.write("**.write("**Çıkış NÇıkış Nedenleriedenleri Dağılımı**")
 Dağılımı**")
                exit_reasons =                exit_reasons = trades[' trades['exit_reason'].exit_reason'].value_counts()
value_counts()
                fig,                fig, ax = ax = plt.subplots(figsize plt.subplots(figsize=(8=(8, 6, 6))
))
                ax                ax.pie(exit.pie(exit_reasons.values,_reasons.values, labels= labels=exit_reasons.indexexit_reasons.index, autopct='%1, autopct='%1.1f.1f%%', start%%', startangle=90angle=90)
                ax.set)
                ax.set_title('_title('Çıkış NÇıkış Nedenleri')
edenleri')
                st.pyplot(fig                st.pyplot(fig)
            
)
            
            with col2            with col2:
                st:
                st.write("**.write("**PozPozisyon Tutmaisyon Tutma Süresi (G Süresi (Gün)**")
               ün)**")
                holding_stats = trades holding_stats = trades['holding_days'].describe()
                st.data['holding_days'].describe()
                st.dataframe(holding_statsframe(holding_stats)
                
            # Detaylı i)
                
            # Detaylı işlem tablosu
           şlem tablosu
            st st.subheader("📋 İ.subheader("📋 İşlem Listesi")
şlem Listesi")
            
            
            display_trades = trades            display_trades = trades.copy()
            display_trades['entry_date'] = display_trades['entry_date']..copy()
            display_trades['entry_date'] = display_trades['entry_date'].dt.strftime('%Y-%dt.strftime('%Y-%mm-%d')
            display-%d')
            display_trades['exit_trades['exit_date'] = display_trades['exit_date'] = display_trades['exit_date'].dt.strftime('%Y-%_date'].dt.strftime('%Y-%m-%d')
            displaym-%d')
            display_trades['pnl'] = display_trades['pnl'] = display_trades['pn_trades['pnl'].round(2)
l'].round(2)
                       display_trades['return_pct display_trades['return_pct'] = display_t'] = display_trades['returnrades['return_pct'].round(_pct'].round(2)
            
            # DataFrame'i gö2)
            
            # DataFrame'i göster
            st.dataframester
            st.dataframe((display_trades, usedisplay_trades, use_container_width_container_width=True)
                
        else:
=True)
                
        else:
                       st.info("🤷 st.info("🤷 Hiç işlem ger Hiç işlem gerçekleşmedi. Paramçekleşmedi. Parametreleri değiştirmeyietreleri değiştirmeyi deneyin.")
            
    deneyin.")
            
    except Exception as e:
        except Exception as e:
        st.error(f"❌ st.error(f"❌ Sistem hatası: {str(e Sistem hatası: {str(e)})}")
        st.info("")
        st.info("💡 Lütfen tarih💡 Lütfen tarih aralığını veya aralığını veya paramet parametreleri değiştirip tekrar denreleri değiştirip tekrar deneyin.")

# Bilgieyin.")

# Bilgi paneli
st.sidebar.mark paneli
st.sidedown("---")
st.sbar.markdown("---")
st.sidebar.header("idebar.header("ℹ️ Sistem Bilgℹ️ Sistem Bilgisi")
st.sidebar.info("isi")
st.sidebar.info("""
**Kullanılan Gö""
**Kullanıstergeler:**
lan Göstergeler:**
- EMA (8- EMA (8, 13, 20,, 13, 20, 50, 200)
- 50, 200)
- RSI RSI & Stochastic
- MACD & Stochastic
- MACD & Boll & Bollinger Bands
inger Bands
- A- ATR & Volume AnalTR & Volume Analizi

**Risk Yönetimiizi

**Risk Yönetimi:**
- ATR tabanlı:**
- ATR tabanlı Stop Loss
- Stop Loss
- Dinamik Take Profit
- Position Sizing
 Dinamik Take Profit
- Position Sizing
- Komisyon hesabı
""")

st- Komisyon hesabı
""")

st.markdown("---")
st.mark.markdown("---")
st.markdown("**Advanced Swing Backtestdown("**Advanced Swing Backtest System v4.0 | System v4.0 | Profesyonel Profesyonel Strateji**")