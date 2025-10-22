import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
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
# GELİŞMİŞ BACKTEST MOTORU
# =========================
class AdvancedSwingBacktest:
    def __init__(self):
        self.commission = 0.001
        self.initial_capital = 10000
    
    def calculate_advanced_indicators(self, df):
        df = df.copy()
        
        # Trend EMA'ları
        for period in [8, 13, 20, 50, 200]:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, min_periods=1).mean()
        
        # RSI - Wilder'in yöntemi
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # Stochastic
        df['STOCH_K'] = 100 * (df['Close'] - df['Low'].rolling(14).min()) / (
            df['High'].rolling(14).max() - df['Low'].rolling(14).min())
        df['STOCH_D'] = df['STOCH_K'].rolling(3).mean()
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        # Volume SMA
        df['Volume_SMA'] = df['Volume'].rolling(20).mean()
        
        # Support/Resistance
        df['Resistance'] = df['High'].rolling(20).max()
        df['Support'] = df['Low'].rolling(20).min()
        
        return df.fillna(method='bfill')
    
    def generate_advanced_signals(self, df, params):
        signals = []
        in_position = False
        
        for i in range(2, len(df)):
            try:
                if i < 50:  # Daha fazla period bekleyelim
                    signals.append({'date': df.index[i], 'action': 'hold', 'strength': 0})
                    continue
                
                row = df.iloc[i]
                prev_row = df.iloc[i-1]
                
                # Değerleri al
                close = float(row['Close'])
                ema_20 = float(row['EMA_20'])
                ema_50 = float(row['EMA_50'])
                ema_200 = float(row['EMA_200'])
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
                trend_up = ema_20 > ema_50 > ema_200
                price_above_ema20 = close > ema_20
                
                # Momentum göstergeleri
                rsi_oversold = rsi < params['rsi_oversold']
                rsi_overbought = rsi > params['rsi_overbought']
                stoch_oversold = stoch_k < 20 and stoch_d < 20
                stoch_bullish_cross = stoch_k > stoch_d and prev_row['STOCH_K'] <= prev_row['STOCH_D']
                macd_bullish = macd_hist > 0 and prev_row['MACD_Hist'] <= 0
                
                # Volatilite ve konum
                near_bb_lower = close <= bb_lower * 1.01
                near_bb_upper = close >= bb_upper * 0.99
                high_volume = volume > volume_sma * 1.2
                
                # Sinyal gücü hesaplama
                signal_strength = 0
                
                # ALIŞ sinyalleri
                if not in_position:
                    buy_signals = []
                    
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
        
        return pd.DataFrame(signals).set_index('date')
    
    def run_backtest(self, data, params):
        df = self.calculate_advanced_indicators(data)
        signals = self.generate_advanced_signals(df, params)
        
        capital = self.initial_capital
        position = None
        trades = []
        equity_curve = []
        max_drawdown = 0
        peak_equity = capital
        
        for date in df.index:
            if date not in signals.index:
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
            drawdown = (peak_equity - current_equity) / peak_equity * 100
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
                    pnl_pct = (pnl / entry_value) * 100
                    
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
            
            trades.append({
                'entry_date': position['entry_date'],
                'exit_date': df.index[-1],
                'entry_price': position['entry_price'],
                'exit_price': last_price,
                'shares': position['shares'],
                'pnl': pnl,
                'return_pct': (pnl / entry_value) * 100,
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
            sharpe_ratio = equity_returns.mean() / equity_returns.std() * np.sqrt(252) if len(equity_returns) > 1 else 0
            
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
# GELİŞMİŞ STREAMLET UYGULAMASI
# =========================
st.set_page_config(page_title="Advanced Swing Backtest", layout="wide", page_icon="🚀")

st.title("🚀 Advanced Swing Trading Backtest")
st.markdown("**8+ İndikatörlü Profesyonel Strateji | Risk Yönetimi | Detaylı Analiz**")

# Sidebar - Gelişmiş ayarlar
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
if st.button("🎯 Gelişmiş Backtest Çalıştır", type="primary", use_container_width=True):
    try:
        with st.spinner("📊 Finansal veriler yükleniyor..."):
            # Daha uzun veri için
            extended_start = start_date - timedelta(days=200)
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
            
            st.success(f"✅ {len(data)} işlem günü yüklendi - {data.index[0].strftime('%d.%m.%Y')} - {data.index[-1].strftime('%d.%m.%Y')}")
        
        # Backtest çalıştır
        backtester = AdvancedSwingBacktest()
        
        with st.spinner("🔍 Teknik analiz yapılıyor ve backtest çalıştırılıyor..."):
            trades, equity, max_drawdown = backtester.run_backtest(data, params)
            metrics = backtester.calculate_advanced_metrics(trades, equity, max_drawdown)
        
        # Sonuçları göster
        st.subheader("📈 Detaylı Performans Analizi")
        
        # Ana metrikler
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            color = "green" if float(metrics['total_return'].replace('%', '').replace('+', '')) > 0 else "red"
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
        col5, col6, col7 = st.columns(3)
        with col5:
            st.metric("En İyi İşlem", metrics['best_trade'])
        with col6:
            st.metric("En Kötü İşlem", metrics['worst_trade'])
        with col7:
            st.metric("Güçlü Sinyal Win Rate", metrics['strong_signal_win_rate'])
        
        if not trades.empty:
            # Grafikler
            st.subheader("📊 Performans Grafikleri")
            
            tab1, tab2, tab3 = st.tabs(["Portföy Değeri", "Drawdown", "İşlem Analizi"])
            
            with tab1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=equity['date'], 
                    y=equity['equity'],
                    mode='lines',
                    name='Portföy Değeri',
                    line=dict(color='#00ff88', width=3)
                ))
                fig.update_layout(
                    title='Portföy Değer Gelişimi',
                    xaxis_title='Tarih',
                    yaxis_title='Portföy Değeri ($)',
                    template='plotly_dark'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=equity['date'], 
                    y=equity['drawdown'],
                    fill='tozeroy',
                    name='Drawdown',
                    line=dict(color='#ff4444', width=2)
                ))
                fig.update_layout(
                    title='Portföy Drawdown',
                    xaxis_title='Tarih',
                    yaxis_title='Drawdown (%)',
                    template='plotly_dark'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                # İşlem sonuçları dağılımı
                fig = go.Figure()
                colors = ['green' if x > 0 else 'red' for x in trades['return_pct']]
                fig.add_trace(go.Bar(
                    x=list(range(len(trades))),
                    y=trades['return_pct'],
                    marker_color=colors,
                    name='İşlem Getirisi'
                ))
                fig.update_layout(
                    title='İşlem Getirileri Dağılımı',
                    xaxis_title='İşlem No',
                    yaxis_title='Getiri (%)',
                    template='plotly_dark'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Detaylı işlem tablosu
            st.subheader("📋 Detaylı İşlem Listesi")
            
            display_trades = trades.copy()
            display_trades['entry_date'] = display_trades['entry_date'].dt.strftime('%Y-%m-%d')
            display_trades['exit_date'] = display_trades['exit_date'].dt.strftime('%Y-%m-%d')
            display_trades['pnl'] = display_trades['pnl'].round(2)
            display_trades['return_pct'] = display_trades['return_pct'].round(2)
            
            # Renkli gösterim
            def color_pnl(val):
                color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
                return f'color: {color}'
            
            styled_trades = display_trades.style.applymap(color_pnl, subset=['pnl', 'return_pct'])
            st.dataframe(styled_trades, use_container_width=True)
            
            # İstatistikler
            st.subheader("📊 İşlem İstatistikleri")
            col1, col2 = st.columns(2)
            
            with col1:
                exit_reasons = trades['exit_reason'].value_counts()
                st.plotly_chart(go.Figure(
                    data=[go.Pie(
                        labels=exit_reasons.index,
                        values=exit_reasons.values,
                        hole=.3
                    )],
                    layout=dict(title='Çıkış Nedenleri Dağılımı')
                ), use_container_width=True)
            
            with col2:
                holding_days = trades['holding_days'].describe()
                st.write("**Pozisyon Tutma Süresi (Gün)**")
                st.dataframe(holding_days)
                
        else:
            st.info("🤷 Hiç işlem gerçekleşmedi. Parametreleri değiştirmeyi deneyin.")
            
    except Exception as e:
        st.error(f"❌ Sistem hatası: {str(e)}")
        st.info("💡 Lütfen tarih aralığını veya parametreleri değiştirip tekrar deneyin.")

# Bilgi paneli
st.sidebar.markdown("---")
st.sidebar.header("ℹ️ Sistem Bilgisi")
st.sidebar.info("""
**Kullanılan Göstergeler:**
- EMA (8, 13, 20, 50, 200)
- RSI & Stochastic
- MACD & Bollinger Bands
- ATR & Volume Analizi
- Support/Resistance

**Risk Yönetimi:**
- ATR tabanlı Stop Loss
- Dinamik Take Profit
- Position Sizing
- Komisyon hesabı
""")

st.markdown("---")
st.markdown("**Advanced Swing Backtest System v4.0 | 8+ İndikatörlü Profesyonel Strateji**")