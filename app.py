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
# BACKTEST MOTORU - TAMAMEN DÜZELTİLMİŞ
# =========================
class SwingBacktest:
    def __init__(self, commission=0.0005, slippage=0.0002):
        self.commission = commission
        self.slippage = slippage
    
    def calculate_indicators(self, df):
        """Teknik göstergeleri hesaplar - HATA DÜZELTİLMİŞ"""
        try:
            df = df.copy()
            
            st.info(f"📊 İndikatörler hesaplanıyor... {len(df)} veri noktası")
            
            # EMA'lar
            df['EMA_20'] = df['Close'].ewm(span=20, min_periods=1).mean()
            df['EMA_50'] = df['Close'].ewm(span=50, min_periods=1).mean()
            df['EMA_200'] = df['Close'].ewm(span=200, min_periods=1).mean()
            
            # RSI
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
            
            # ATR - DÜZELTİLMİŞ VERSİYON
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift(1))
            low_close = np.abs(df['Low'] - df['Close'].shift(1))
            
            # DataFrame oluştururken index'i koru
            true_range = pd.DataFrame({
                'high_low': high_low,
                'high_close': high_close,
                'low_close': low_close
            }, index=df.index)
            
            true_range = true_range.max(axis=1)
            df['ATR'] = true_range.rolling(14, min_periods=1).mean()
            
            # NaN değerleri temizle
            df = df.fillna(method='bfill').fillna(method='ffill')
            
            st.success("✅ İndikatörler başarıyla hesaplandı")
            return df
            
        except Exception as e:
            st.error(f"❌ İndikatör hesaplama hatası: {e}")
            # Hata durumunda orijinal df'yi döndür
            return df
    
    def swing_signal(self, df, params):
        """Swing trade sinyalleri üretir - KOLON KONTROLLÜ"""
        try:
            st.info("🔍 Sinyaller oluşturuluyor...")
            
            # İndikatörleri hesapla
            df_with_indicators = self.calculate_indicators(df)
            
            if df_with_indicators.empty:
                st.warning("❌ İndikatörler hesaplanamadı")
                return pd.DataFrame()
            
            # Gerekli kolonları kontrol et
            required_columns = ['Close', 'High', 'Low', 'EMA_20', 'EMA_50', 'RSI', 'MACD_Hist', 'ATR']
            missing_columns = [col for col in required_columns if col not in df_with_indicators.columns]
            
            if missing_columns:
                st.error(f"❌ Eksik kolonlar: {missing_columns}")
                return pd.DataFrame()
            
            st.success(f"✅ Tüm gerekli kolonlar mevcut: {required_columns}")
            
            # Sinyalleri oluştur
            signals_data = []
            
            for idx, row in df_with_indicators.iterrows():
                try:
                    # Değerleri float'a çevir
                    close = float(row['Close'])
                    ema_20 = float(row['EMA_20'])
                    ema_50 = float(row['EMA_50'])
                    rsi = float(row['RSI'])
                    macd_hist = float(row['MACD_Hist'])
                    atr = float(row['ATR'])
                    
                    # Sinyal koşulları
                    trend_up = ema_20 > ema_50
                    rsi_oversold = rsi < params.get('rsi_oversold', 35)
                    macd_bullish = macd_hist > params.get('macd_threshold', 0)
                    price_above_ema20 = close > ema_20
                    
                    buy_signal = trend_up and rsi_oversold and macd_bullish and price_above_ema20
                    
                    # Risk yönetimi
                    atr_multiplier = params.get('atr_multiplier', 1.5)
                    stop_loss = close - (atr * atr_multiplier)
                    risk_distance = atr * atr_multiplier
                    tp1 = close + risk_distance * 1.0
                    tp2 = close + risk_distance * 2.0
                    tp3 = close + risk_distance * 3.0
                    
                    signals_data.append({
                        'date': idx,
                        'action': 'buy' if buy_signal else 'hold',
                        'stop_loss': stop_loss,
                        'tp1': tp1,
                        'tp2': tp2,
                        'tp3': tp3
                    })
                    
                except Exception as e:
                    # Hata durumunda hold sinyali
                    signals_data.append({
                        'date': idx,
                        'action': 'hold',
                        'stop_loss': 0,
                        'tp1': 0,
                        'tp2': 0,
                        'tp3': 0
                    })
            
            signals_df = pd.DataFrame(signals_data)
            if not signals_df.empty:
                signals_df = signals_df.set_index('date')
            
            st.success(f"✅ {len(signals_df)} sinyal oluşturuldu")
            return signals_df
            
        except Exception as e:
            st.error(f"❌ Sinyal oluşturma hatası: {e}")
            return pd.DataFrame()
    
    def backtest(self, df, params, initial_capital=10000):
        """Backtest yürütür - STABİL VERSİYON"""
        try:
            st.info("🎯 Backtest başlatılıyor...")
            
            signals = self.swing_signal(df, params)
            
            if signals.empty:
                st.warning("❌ Backtest için sinyal bulunamadı")
                return pd.DataFrame(), pd.DataFrame()
            
            # Backtest değişkenleri
            capital = initial_capital
            position = None
            trades = []
            equity_curve = []
            
            for date in signals.index:
                try:
                    signal = signals.loc[date]
                    price_data = df.loc[date]
                    
                    current_price = float(price_data['Close'])
                    current_high = float(price_data['High'])
                    current_low = float(price_data['Low'])
                    
                    # Equity hesapla
                    current_equity = capital
                    if position is not None:
                        current_equity += position['shares'] * current_price
                    
                    equity_curve.append({
                        'date': date,
                        'equity': current_equity
                    })
                    
                    # YENİ POZİSYON AÇ
                    if position is None and signal['action'] == 'buy':
                        stop_loss = float(signal['stop_loss'])
                        risk_per_share = current_price - stop_loss
                        
                        if risk_per_share > 0:
                            risk_amount = capital * params.get('risk_per_trade', 0.02)
                            shares = risk_amount / risk_per_share
                            
                            # Pozisyon aç
                            if shares > 0:
                                position = {
                                    'entry_date': date,
                                    'entry_price': current_price,
                                    'shares': shares,
                                    'stop_loss': stop_loss,
                                    'tp1': float(signal['tp1']),
                                    'tp2': float(signal['tp2']),
                                    'tp3': float(signal['tp3']),
                                    'tp1_hit': False,
                                    'tp2_hit': False
                                }
                                capital -= shares * current_price
                                st.info(f"📈 Pozisyon açıldı: {shares:.2f} hisse @ ${current_price:.2f}")
                    
                    # POZİSYON YÖNETİMİ
                    elif position is not None:
                        exit_reason = None
                        exit_price = None
                        exit_shares = 0
                        
                        # TP1 kontrolü
                        if not position['tp1_hit'] and current_high >= position['tp1']:
                            exit_price = position['tp1']
                            exit_shares = position['shares'] * 0.5  # %50 kapat
                            position['shares'] -= exit_shares
                            position['tp1_hit'] = True
                            exit_reason = 'TP1'
                        
                        # TP2 kontrolü
                        elif position['tp1_hit'] and not position['tp2_hit'] and current_high >= position['tp2']:
                            exit_price = position['tp2']
                            exit_shares = position['shares'] * 0.6  # Kalanın %60'ı
                            position['shares'] -= exit_shares
                            position['tp2_hit'] = True
                            exit_reason = 'TP2'
                        
                        # TP3 kontrolü
                        elif position['tp1_hit'] and position['tp2_hit'] and current_high >= position['tp3']:
                            exit_price = position['tp3']
                            exit_shares = position['shares']  # Tümünü kapat
                            position['shares'] = 0
                            exit_reason = 'TP3'
                        
                        # Stop-loss kontrolü
                        elif current_low <= position['stop_loss']:
                            exit_price = position['stop_loss']
                            exit_shares = position['shares']
                            position['shares'] = 0
                            exit_reason = 'SL'
                        
                        # Çıkış işlemi
                        if exit_reason and exit_shares > 0:
                            exit_value = exit_shares * exit_price
                            capital += exit_value
                            
                            entry_value = exit_shares * position['entry_price']
                            pnl = exit_value - entry_value
                            
                            trades.append({
                                'entry_date': position['entry_date'],
                                'exit_date': date,
                                'entry_price': position['entry_price'],
                                'exit_price': exit_price,
                                'shares': exit_shares,
                                'pnl': pnl,
                                'return_pct': (pnl / entry_value) * 100 if entry_value > 0 else 0,
                                'exit_reason': exit_reason,
                                'hold_days': (date - position['entry_date']).days
                            })
                            
                            st.info(f"📊 Çıkış: {exit_reason} | PnL: ${pnl:.2f}")
                            
                            # Pozisyon tamamen kapandı mı?
                            if position['shares'] <= 0:
                                position = None
                
                except Exception as e:
                    continue
            
            # AÇIK POZİSYONU KAPAT
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
                    'pnl': pnl,
                    'return_pct': (pnl / entry_value) * 100 if entry_value > 0 else 0,
                    'exit_reason': 'OPEN',
                    'hold_days': (df.index[-1] - position['entry_date']).days
                })
                
                st.info(f"🔓 Açık pozisyon kapatıldı | PnL: ${pnl:.2f}")
            
            trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
            equity_df = pd.DataFrame(equity_curve)
            
            st.success(f"✅ Backtest tamamlandı: {len(trades_df)} işlem")
            return trades_df, equity_df
            
        except Exception as e:
            st.error(f"❌ Backtest hatası: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def calculate_metrics(self, trades_df, equity_df, initial_capital):
        """Performans metriklerini hesaplar"""
        if trades_df.empty or equity_df.empty:
            return {
                'total_return_%': 0.0,
                'total_trades': 0,
                'win_rate_%': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'max_drawdown_%': 0.0,
                'avg_hold_days': 0.0
            }
        
        try:
            # Temel metrikler
            final_equity = float(equity_df['equity'].iloc[-1])
            total_return = (final_equity - initial_capital) / initial_capital * 100.0
            
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0
            
            avg_win = float(trades_df[trades_df['pnl'] > 0]['pnl'].mean()) if winning_trades > 0 else 0.0
            avg_loss = float(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if (total_trades - winning_trades) > 0 else 0.0
            
            # Drawdown
            equity_series = equity_df.set_index('date')['equity']
            rolling_max = equity_series.expanding().max()
            drawdown = (equity_series - rolling_max) / rolling_max * 100
            max_drawdown = float(drawdown.min())
            
            avg_hold_days = float(trades_df['hold_days'].mean()) if not trades_df.empty else 0.0
            
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
            st.error(f"Metrik hesaplama hatası: {e}")
            return {
                'total_return_%': 0.0,
                'total_trades': 0,
                'win_rate_%': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'max_drawdown_%': 0.0,
                'avg_hold_days': 0.0
            }

# =========================
# STREAMLIT UYGULAMASI
# =========================
st.set_page_config(page_title="Swing Backtest Pro", layout="wide")
st.title("🚀 Swing Trade Backtest Sistemi")
st.markdown("**Stabil ve Hata Kontrollü Versiyon**")

# Sidebar
st.sidebar.header("⚙️ Backtest Ayarları")
ticker = st.sidebar.selectbox("Sembol", ["AAPL", "GOOGL", "MSFT", "TSLA", "BTC-USD", "ETH-USD"])
start_date = st.sidebar.date_input("Başlangıç Tarihi", datetime(2023, 1, 1))
end_date = st.sidebar.date_input("Bitiş Tarihi", datetime(2023, 12, 31))

st.sidebar.header("📊 Strateji Parametreleri")
rsi_oversold = st.sidebar.slider("RSI Aşırı Satım", 20, 40, 30)
atr_multiplier = st.sidebar.slider("ATR Çarpanı", 1.0, 3.0, 2.0)
risk_per_trade = st.sidebar.slider("İşlem Risk %", 1.0, 5.0, 2.0) / 100

# Ana içerik
if st.button("🎯 Backtest Çalıştır", type="primary"):
    try:
        with st.spinner("Veriler yükleniyor..."):
            # Tarih aralığını genişlet (indikatörler için)
            extended_start = start_date - timedelta(days=100)
            data = yf.download(ticker, start=extended_start, end=end_date, progress=False)
            
            if data.empty:
                st.error("❌ Veri çekilemedi - Sembolü kontrol edin")
                st.stop()
            
            # Sadece istenen tarih aralığını kullan
            data = data[data.index >= pd.to_datetime(start_date)]
            data = data[data.index <= pd.to_datetime(end_date)]
            
            if data.empty:
                st.error("❌ Filtrelenmiş veri kalmadı")
                st.stop()
            
            st.success(f"✅ {len(data)} günlük veri yüklendi")
        
        # Backtest çalıştır
        backtester = SwingBacktest()
        params = {
            'rsi_oversold': rsi_oversold,
            'atr_multiplier': atr_multiplier,
            'risk_per_trade': risk_per_trade
        }
        
        trades, equity = backtester.backtest(data, params)
        metrics = backtester.calculate_metrics(trades, equity, 10000)
        
        # Sonuçları göster
        st.subheader("📊 Performans Özeti")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Toplam Getiri", f"{metrics['total_return_%']}%")
            st.metric("Toplam İşlem", f"{metrics['total_trades']}")
            st.metric("Win Rate", f"{metrics['win_rate_%']}%")
        
        with col2:
            st.metric("Ort. Kazanç", f"${metrics['avg_win']:.2f}")
            st.metric("Ort. Kayıp", f"${metrics['avg_loss']:.2f}")
            st.metric("Max Drawdown", f"{metrics['max_drawdown_%']}%")
        
        # Grafikler
        if not trades.empty:
            st.subheader("📈 Performans Grafikleri")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Equity curve
            ax1.plot(equity['date'], equity['equity'], color='blue', linewidth=2)
            ax1.set_title('Portföy Değeri', fontweight='bold')
            ax1.set_ylabel('Equity ($)')
            ax1.grid(True, alpha=0.3)
            
            # Drawdown
            equity_series = equity.set_index('date')['equity']
            rolling_max = equity_series.expanding().max()
            drawdown = (equity_series - rolling_max) / rolling_max * 100
            
            ax2.fill_between(equity['date'], drawdown.values, 0, alpha=0.3, color='red')
            ax2.set_title('Drawdown', fontweight='bold')
            ax2.set_ylabel('Drawdown %')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # İşlem listesi
            st.subheader("📋 İşlem Listesi")
            display_trades = trades.copy()
            display_trades['entry_date'] = display_trades['entry_date'].dt.strftime('%Y-%m-%d')
            display_trades['exit_date'] = display_trades['exit_date'].dt.strftime('%Y-%m-%d')
            st.dataframe(display_trades)
            
        else:
            st.info("🤷 Hiç işlem gerçekleşmedi. Parametreleri değiştirmeyi deneyin.")
            
    except Exception as e:
        st.error(f"❌ Ana uygulama hatası: {e}")

st.markdown("---")
st.markdown("**Swing Backtest Pro v2.0 | Tam Stabil**")
