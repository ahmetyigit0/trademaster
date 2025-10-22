import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
# BACKTEST MOTORU - TAMAMEN YENİDEN YAZILDI
# =========================
class SwingBacktest:
    def __init__(self, commission=0.0005, slippage=0.0002):
        self.commission = commission
        self.slippage = slippage
    
    def calculate_indicators(self, df):
        """Teknik göstergeleri hesaplar - GÜVENLİ VERSİYON"""
        try:
            df = df.copy()
            
            # EMA'lar - basit hesaplama
            df['EMA_20'] = df['Close'].ewm(span=20, min_periods=1).mean()
            df['EMA_50'] = df['Close'].ewm(span=50, min_periods=1).mean()
            df['EMA_200'] = df['Close'].ewm(span=200, min_periods=1).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['Close'].ewm(span=12, min_periods=1).mean()
            exp2 = df['Close'].ewm(span=26, min_periods=1).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9, min_periods=1).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            # ATR
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.DataFrame({'high_low': high_low, 'high_close': high_close, 'low_close': low_close})
            true_range = ranges.max(axis=1)
            df['ATR'] = true_range.rolling(14, min_periods=1).mean()
            
            # NaN değerleri doldur
            df = df.fillna(method='bfill').fillna(method='ffill')
            
            return df
            
        except Exception as e:
            st.error(f"İndikatör hesaplama hatası: {e}")
            return df
    
    def swing_signal(self, df, params):
        """Swing trade sinyalleri üretir - ÇOK BASİT VERSİYON"""
        try:
            # İndikatörleri hesapla
            df_with_indicators = self.calculate_indicators(df)
            
            if df_with_indicators.empty:
                return pd.DataFrame()
            
            # Sadece gerekli kolonları seç ve kopyala
            df_clean = df_with_indicators[['Close', 'High', 'Low', 'EMA_20', 'EMA_50', 'RSI', 'MACD_Hist', 'ATR']].copy()
            
            # NaN kontrolü
            if df_clean.isna().any().any():
                df_clean = df_clean.fillna(method='bfill').fillna(method='ffill')
            
            # Sinyalleri hesapla - TEK TEK KONTROL
            signals_data = []
            
            for idx, row in df_clean.iterrows():
                try:
                    trend_up = row['EMA_20'] > row['EMA_50']
                    rsi_oversold = row['RSI'] < params.get('rsi_oversold', 35)
                    macd_bullish = row['MACD_Hist'] > params.get('macd_threshold', 0)
                    price_above_ema20 = row['Close'] > row['EMA_20']
                    
                    buy_signal = trend_up and rsi_oversold and macd_bullish and price_above_ema20
                    
                    # Stop ve TP seviyeleri
                    atr_multiplier = params.get('atr_multiplier', 1.5)
                    stop_loss = row['Close'] - (row['ATR'] * atr_multiplier)
                    risk_distance = row['ATR'] * atr_multiplier
                    tp1 = row['Close'] + risk_distance * 1.0
                    tp2 = row['Close'] + risk_distance * 2.0
                    tp3 = row['Close'] + risk_distance * 3.0
                    
                    signals_data.append({
                        'date': idx,
                        'action': 'buy' if buy_signal else 'hold',
                        'stop_loss': stop_loss,
                        'tp1': tp1,
                        'tp2': tp2,
                        'tp3': tp3
                    })
                    
                except Exception as e:
                    # Hata durumunda hold sinyali ver
                    signals_data.append({
                        'date': idx,
                        'action': 'hold',
                        'stop_loss': 0,
                        'tp1': 0,
                        'tp2': 0,
                        'tp3': 0
                    })
            
            signals_df = pd.DataFrame(signals_data)
            signals_df = signals_df.set_index('date')
            
            return signals_df
            
        except Exception as e:
            st.error(f"Sinyal oluşturma hatası: {e}")
            return pd.DataFrame()
    
    def backtest(self, df, params, initial_capital=10000):
        """Backtest yürütür - ÇOK BASİT VE GÜVENLİ"""
        try:
            st.info("🔄 Sinyaller hesaplanıyor...")
            signals = self.swing_signal(df, params)
            
            if signals.empty:
                st.warning("❌ Hiç sinyal oluşturulamadı")
                return pd.DataFrame(), pd.DataFrame()
            
            st.info(f"✅ {len(signals)} sinyal oluşturuldu")
            
            # Basit backtest
            capital = initial_capital
            position = None
            trades = []
            equity_curve = []
            
            for date in signals.index:
                try:
                    signal = signals.loc[date]
                    price_data = df.loc[date]
                    
                    current_price = price_data['Close']
                    current_high = price_data['High']
                    current_low = price_data['Low']
                    
                    # Equity güncelle
                    current_equity = capital
                    if position is not None:
                        current_equity += position['shares'] * current_price
                    
                    equity_curve.append({'date': date, 'equity': current_equity})
                    
                    # Yeni pozisyon
                    if position is None and signal['action'] == 'buy':
                        risk_per_share = current_price - signal['stop_loss']
                        if risk_per_share > 0:
                            risk_amount = capital * params.get('risk_per_trade', 0.02)
                            shares = risk_amount / risk_per_share
                            
                            # Basit pozisyon aç
                            if shares > 0:
                                position = {
                                    'entry_date': date,
                                    'entry_price': current_price,
                                    'shares': shares,
                                    'stop_loss': signal['stop_loss'],
                                    'tp1': signal['tp1']
                                }
                                capital -= shares * current_price
                    
                    # Pozisyon yönetimi
                    elif position is not None:
                        exit_reason = None
                        
                        # TP kontrolü
                        if current_high >= position['tp1']:
                            exit_reason = 'TP'
                        # SL kontrolü
                        elif current_low <= position['stop_loss']:
                            exit_reason = 'SL'
                        
                        if exit_reason:
                            exit_price = position['tp1'] if exit_reason == 'TP' else position['stop_loss']
                            exit_value = position['shares'] * exit_price
                            capital += exit_value
                            
                            pnl = exit_value - (position['shares'] * position['entry_price'])
                            
                            trades.append({
                                'entry_date': position['entry_date'],
                                'exit_date': date,
                                'entry_price': position['entry_price'],
                                'exit_price': exit_price,
                                'shares': position['shares'],
                                'pnl': pnl,
                                'return_pct': (pnl / (position['shares'] * position['entry_price'])) * 100,
                                'exit_reason': exit_reason,
                                'hold_days': (date - position['entry_date']).days
                            })
                            
                            position = None
                
                except Exception as e:
                    continue
            
            # Açık pozisyonu kapat
            if position is not None:
                last_price = df['Close'].iloc[-1]
                exit_value = position['shares'] * last_price
                capital += exit_value
                pnl = exit_value - (position['shares'] * position['entry_price'])
                
                trades.append({
                    'entry_date': position['entry_date'],
                    'exit_date': df.index[-1],
                    'entry_price': position['entry_price'],
                    'exit_price': last_price,
                    'shares': position['shares'],
                    'pnl': pnl,
                    'return_pct': (pnl / (position['shares'] * position['entry_price'])) * 100,
                    'exit_reason': 'OPEN',
                    'hold_days': (df.index[-1] - position['entry_date']).days
                })
            
            trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
            equity_df = pd.DataFrame(equity_curve)
            
            return trades_df, equity_df
            
        except Exception as e:
            st.error(f"Backtest hatası: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def calculate_metrics(self, trades_df, equity_df, initial_capital):
        """Basit metrik hesaplama"""
        if trades_df.empty:
            return {
                'total_return_%': 0,
                'total_trades': 0,
                'win_rate_%': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'max_drawdown_%': 0,
                'avg_hold_days': 0
            }
        
        try:
            final_equity = equity_df['equity'].iloc[-1]
            total_return = (final_equity - initial_capital) / initial_capital * 100
            
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if (total_trades - winning_trades) > 0 else 0
            
            # Basit drawdown
            equity_series = equity_df.set_index('date')['equity']
            rolling_max = equity_series.expanding().max()
            drawdown = (equity_series - rolling_max) / rolling_max * 100
            max_drawdown = drawdown.min()
            
            avg_hold_days = trades_df['hold_days'].mean() if not trades_df.empty else 0
            
            return {
                'total_return_%': round(total_return, 2),
                'total_trades': total_trades,
                'win_rate_%': round(win_rate, 1),
                'avg_win': round(avg_win, 2),
                'avg_loss': round(avg_loss, 2),
                'max_drawdown_%': round(max_drawdown, 2),
                'avg_hold_days': round(avg_hold_days, 1)
            }
            
        except Exception as e:
            st.error(f"Metrik hatası: {e}")
            return {
                'total_return_%': 0,
                'total_trades': 0,
                'win_rate_%': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'max_drawdown_%': 0,
                'avg_hold_days': 0
            }

# =========================
# STREAMLIT UYGULAMASI
# =========================
st.set_page_config(page_title="Swing Backtest Pro", layout="wide")
st.title("🚀 Swing Trade Backtest Sistemi")
st.markdown("**Basit ve Stabil Versiyon**")

# Sidebar
st.sidebar.header("⚙️ Ayarlar")
ticker = st.sidebar.selectbox("Sembol", ["AAPL", "GOOGL", "MSFT", "TSLA", "BTC-USD", "ETH-USD"])
start_date = st.sidebar.date_input("Başlangıç", datetime(2023, 1, 1))
end_date = st.sidebar.date_input("Bitiş", datetime(2023, 12, 31))

st.sidebar.header("📊 Parametreler")
rsi_oversold = st.sidebar.slider("RSI Aşırı Satım", 20, 40, 30)
atr_multiplier = st.sidebar.slider("ATR Çarpanı", 1.0, 3.0, 2.0)
risk_per_trade = st.sidebar.slider("Risk %", 1.0, 5.0, 2.0) / 100

# Ana içerik
if st.button("🎯 Backtest Başlat", type="primary"):
    try:
        with st.spinner("Veriler yükleniyor..."):
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                st.error("❌ Veri bulunamadı")
                st.stop()
            
            st.success(f"✅ {len(data)} günlük veri yüklendi")
        
        # Backtest
        backtester = SwingBacktest()
        params = {
            'rsi_oversold': rsi_oversold,
            'atr_multiplier': atr_multiplier,
            'risk_per_trade': risk_per_trade
        }
        
        trades, equity = backtester.backtest(data, params)
        metrics = backtester.calculate_metrics(trades, equity, 10000)
        
        # Sonuçlar
        st.subheader("📊 Sonuçlar")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Toplam Getiri", f"{metrics['total_return_%']}%")
            st.metric("Toplam İşlem", f"{metrics['total_trades']}")
            st.metric("Win Rate", f"{metrics['win_rate_%']}%")
        
        with col2:
            st.metric("Ort. Kazanç", f"${metrics['avg_win']:.2f}")
            st.metric("Ort. Kayıp", f"${metrics['avg_loss']:.2f}")
            st.metric("Max Drawdown", f"{metrics['max_drawdown_%']}%")
        
        # Grafik
        if not trades.empty:
            st.subheader("📈 Grafik")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(equity['date'], equity['equity'], linewidth=2, color='blue')
            ax.set_title('Portföy Değeri')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # İşlemler
            st.subheader("📋 İşlemler")
            display_trades = trades.copy()
            display_trades['entry_date'] = display_trades['entry_date'].dt.strftime('%Y-%m-%d')
            display_trades['exit_date'] = display_trades['exit_date'].dt.strftime('%Y-%m-%d')
            st.dataframe(display_trades)
        else:
            st.info("🤷 Hiç işlem yapılmadı")
            
    except Exception as e:
        st.error(f"❌ Hata: {e}")

st.markdown("---")
st.markdown("Swing Backtest v1.0 | Basit ve Stabil")
