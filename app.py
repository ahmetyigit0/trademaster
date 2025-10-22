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
# BACKTEST MOTORU
# =========================
class SwingBacktest:
    def __init__(self, commission=0.0005, slippage=0.0002):
        self.commission = commission
        self.slippage = slippage
    
    def calculate_indicators(self, df):
        """Teknik göstergeleri hesaplar"""
        df = df.copy()
        
        # EMA'lar
        df['EMA_20'] = df['Close'].ewm(span=20).mean()
        df['EMA_50'] = df['Close'].ewm(span=50).mean()
        df['EMA_200'] = df['Close'].ewm(span=200).mean()
        
        # RSI
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
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        return df
    
    def swing_signal(self, df, params):
        """Swing trade sinyalleri üretir - YENİ YAKLAŞIM"""
        df = self.calculate_indicators(df)
        
        # NaN değerleri temizle
        df = df.dropna()
        
        if df.empty:
            return pd.DataFrame()
        
        # Sinyalleri doğrudan df içinde oluştur
        trend_up = df['EMA_20'] > df['EMA_50']
        rsi_oversold = df['RSI'] < params.get('rsi_oversold', 35)
        macd_bullish = df['MACD_Hist'] > params.get('macd_threshold', 0)
        price_above_ema20 = df['Close'] > df['EMA_20']
        
        # AL sinyali
        df['buy_signal'] = trend_up & rsi_oversold & macd_bullish & price_above_ema20
        
        # Stop ve TP seviyeleri
        atr_multiplier = params.get('atr_multiplier', 1.5)
        df['stop_loss'] = df['Close'] - (df['ATR'] * atr_multiplier)
        risk_distance = df['ATR'] * atr_multiplier
        df['tp1'] = df['Close'] + risk_distance * 1.0
        df['tp2'] = df['Close'] + risk_distance * 2.0
        df['tp3'] = df['Close'] + risk_distance * 3.0
        
        # Sadece gerekli kolonları seç
        signals = df[['buy_signal', 'stop_loss', 'tp1', 'tp2', 'tp3']].copy()
        signals['action'] = 'hold'
        signals.loc[signals['buy_signal'], 'action'] = 'buy'
        
        return signals[['action', 'stop_loss', 'tp1', 'tp2', 'tp3']]
    
    def backtest(self, df, params, initial_capital=10000):
        """Backtest yürütür - BASİTLEŞTİRİLMİŞ VERSİYON"""
        try:
            # Sinyalleri hesapla
            signals = self.swing_signal(df, params)
            
            if signals.empty:
                st.warning("❌ Sinyal hesaplanamadı - yeterli veri yok veya tüm değerler NaN")
                return pd.DataFrame(), pd.DataFrame()
            
            # Index kontrolü - aynı index'e sahip olmalılar
            if not df.index.equals(signals.index):
                st.warning("⚠️ Index uyumsuzluğu tespit edildi, ortak index kullanılıyor...")
                common_index = df.index.intersection(signals.index)
                if len(common_index) == 0:
                    st.error("❌ Ortak index bulunamadı")
                    return pd.DataFrame(), pd.DataFrame()
                df = df.loc[common_index]
                signals = signals.loc[common_index]
            
            st.info(f"🔧 Backtest başlıyor: {len(df)} veri noktası, {len(signals)} sinyal")
            
            trades = []
            position = None
            capital = initial_capital
            equity_curve = []
            
            # Tarih listesi
            dates = signals.index
            
            for i, current_date in enumerate(dates):
                current_signal = signals.loc[current_date]
                current_data = df.loc[current_date]
                
                current_price = current_data['Close']
                current_high = current_data['High']
                current_low = current_data['Low']
                
                # Equity curve güncelleme
                current_equity = capital
                if position is not None:
                    current_equity += position['shares'] * current_price
                
                equity_curve.append({
                    'date': current_date,
                    'equity': current_equity
                })
                
                # YENİ POZİSYON AÇMA
                if position is None and current_signal['action'] == 'buy':
                    risk_per_share = current_price - current_signal['stop_loss']
                    
                    if risk_per_share <= 0:
                        continue
                    
                    risk_amount = capital * params.get('risk_per_trade', 0.02)
                    shares = risk_amount / risk_per_share
                    
                    # Komisyon ve slippage
                    entry_price = current_price * (1 + self.slippage)
                    position_value = shares * entry_price
                    commission_paid = position_value * self.commission
                    total_cost = position_value + commission_paid
                    
                    # Sermaye kontrolü
                    if total_cost > capital * 0.8:
                        shares = (capital * 0.8) / entry_price
                        position_value = shares * entry_price
                        commission_paid = position_value * self.commission
                        total_cost = position_value + commission_paid
                    
                    if shares > 0 and total_cost <= capital:
                        position = {
                            'entry_date': current_date,
                            'entry_price': entry_price,
                            'shares': shares,
                            'stop_loss': current_signal['stop_loss'],
                            'tp1': current_signal['tp1'],
                            'tp2': current_signal['tp2'],
                            'tp3': current_signal['tp3'],
                            'tp1_hit': False,
                            'tp2_hit': False
                        }
                        capital -= total_cost
                        st.success(f"📈 Yeni pozisyon: {shares:.2f} hisse @ ${entry_price:.2f}")
                
                # POZİSYON YÖNETİMİ
                elif position is not None:
                    exit_reason = None
                    exit_price = None
                    exit_shares = 0
                    
                    # TP1 kontrolü (%50 kapat)
                    if not position['tp1_hit'] and current_high >= position['tp1']:
                        exit_price = position['tp1'] * (1 - self.slippage)
                        exit_shares = position['shares'] * 0.5
                        position['shares'] -= exit_shares
                        position['tp1_hit'] = True
                        exit_reason = 'TP1'
                    
                    # TP2 kontrolü (%30 kapat)
                    elif position['tp1_hit'] and not position['tp2_hit'] and current_high >= position['tp2']:
                        exit_price = position['tp2'] * (1 - self.slippage)
                        exit_shares = position['shares'] * 0.6
                        position['shares'] -= exit_shares
                        position['tp2_hit'] = True
                        exit_reason = 'TP2'
                    
                    # TP3 kontrolü (%20 kapat)
                    elif position['tp1_hit'] and position['tp2_hit'] and current_high >= position['tp3']:
                        exit_price = position['tp3'] * (1 - self.slippage)
                        exit_shares = position['shares']
                        position['shares'] = 0
                        exit_reason = 'TP3'
                    
                    # Stop-loss kontrolü
                    elif current_low <= position['stop_loss']:
                        exit_price = position['stop_loss'] * (1 - self.slippage)
                        exit_shares = position['shares']
                        position['shares'] = 0
                        exit_reason = 'SL'
                    
                    # Çıkış işlemi
                    if exit_reason and exit_shares > 0:
                        exit_value = exit_shares * exit_price
                        commission_paid = exit_value * self.commission
                        capital += exit_value - commission_paid
                        
                        entry_value = exit_shares * position['entry_price']
                        trade_pnl = exit_value - entry_value
                        trade_pnl -= (entry_value * self.commission) + commission_paid
                        
                        trades.append({
                            'entry_date': position['entry_date'],
                            'exit_date': current_date,
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'shares': exit_shares,
                            'pnl': trade_pnl,
                            'return_pct': (trade_pnl / entry_value) * 100 if entry_value > 0 else 0,
                            'exit_reason': exit_reason,
                            'hold_days': (current_date - position['entry_date']).days
                        })
                        
                        st.info(f"📊 Çıkış: {exit_reason} | PnL: ${trade_pnl:.2f}")
                        
                        if position['shares'] <= 0:
                            position = None
            
            # Açık pozisyonları kapat
            if position is not None:
                last_price = df['Close'].iloc[-1]
                exit_value = position['shares'] * last_price
                commission_paid = exit_value * self.commission
                capital += exit_value - commission_paid
                
                entry_value = position['shares'] * position['entry_price']
                trade_pnl = exit_value - entry_value
                trade_pnl -= (entry_value * self.commission) + commission_paid
                
                trades.append({
                    'entry_date': position['entry_date'],
                    'exit_date': dates[-1],
                    'entry_price': position['entry_price'],
                    'exit_price': last_price,
                    'shares': position['shares'],
                    'pnl': trade_pnl,
                    'return_pct': (trade_pnl / entry_value) * 100 if entry_value > 0 else 0,
                    'exit_reason': 'OPEN',
                    'hold_days': (dates[-1] - position['entry_date']).days
                })
                
                st.info(f"🔓 Açık pozisyon kapatıldı | PnL: ${trade_pnl:.2f}")
            
            trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
            equity_df = pd.DataFrame(equity_curve)
            
            st.success(f"✅ Backtest tamamlandı: {len(trades_df)} işlem")
            return trades_df, equity_df
            
        except Exception as e:
            st.error(f"❌ Backtest hatası: {str(e)}")
            import traceback
            st.code(f"Hata detayı: {traceback.format_exc()}")
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
                'profit_factor': 0.0,
                'avg_r_multiple': 0.0,
                'max_drawdown_%': 0.0,
                'sharpe_ratio': 0.0,
                'avg_hold_days': 0.0
            }
        
        try:
            # Basit ve güvenli metrik hesaplama
            final_equity = float(equity_df['equity'].iloc[-1])
            total_return = (final_equity - float(initial_capital)) / float(initial_capital) * 100.0
            
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0
            
            avg_win = float(trades_df[trades_df['pnl'] > 0]['pnl'].mean()) if winning_trades > 0 else 0.0
            avg_loss = float(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if (total_trades - winning_trades) > 0 else 0.0
            
            # Basit drawdown hesaplama
            equity_series = equity_df.set_index('date')['equity']
            rolling_max = equity_series.expanding().max()
            drawdown = (equity_series - rolling_max) / rolling_max * 100
            max_drawdown = float(drawdown.min())
            
            metrics = {
                'total_return_%': round(total_return, 2),
                'total_trades': total_trades,
                'win_rate_%': round(win_rate, 1),
                'avg_win': round(avg_win, 2),
                'avg_loss': round(avg_loss, 2),
                'profit_factor': 0.0,  # Basit tutuyoruz
                'avg_r_multiple': 0.0,  # Basit tutuyoruz
                'max_drawdown_%': round(max_drawdown, 2),
                'sharpe_ratio': 0.0,    # Basit tutuyoruz
                'avg_hold_days': round(trades_df['hold_days'].mean(), 1) if not trades_df.empty else 0.0
            }
            
            return metrics
            
        except Exception as e:
            st.error(f"Metrik hesaplama hatası: {e}")
            return {
                'total_return_%': 0.0,
                'total_trades': 0,
                'win_rate_%': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'avg_r_multiple': 0.0,
                'max_drawdown_%': 0.0,
                'sharpe_ratio': 0.0,
                'avg_hold_days': 0.0
            }

# =========================
# STREAMLIT UYGULAMASI
# =========================
st.set_page_config(page_title="Swing Backtest Pro", layout="wide")
st.title("🚀 Swing Trade Backtest Sistemi")
st.markdown("**Profesyonel Swing Strateji Test Platformu**")

# Sidebar kontrolleri
st.sidebar.header("⚙️ Backtest Ayarları")
ticker = st.sidebar.selectbox("Sembol", ["BTC-USD", "ETH-USD", "AAPL", "GOOGL", "MSFT", "TSLA"])
start_date = st.sidebar.date_input("Başlangıç Tarihi", datetime(2023, 1, 1))
end_date = st.sidebar.date_input("Bitiş Tarihi", datetime(2024, 1, 1))

st.sidebar.header("📊 Strateji Parametreleri")
rsi_oversold = st.sidebar.slider("RSI Aşırı Satım", 20, 40, 30)
atr_multiplier = st.sidebar.slider("ATR Çarpanı", 1.0, 3.0, 2.0)
risk_per_trade = st.sidebar.slider("İşlem Risk %", 1.0, 5.0, 2.0) / 100

# Ana içerik
tab1, tab2, tab3 = st.tabs(["📈 Backtest", "🔧 Optimizasyon", "📊 Sağlamlık Analizi"])

with tab1:
    st.header("Backtest Sonuçları")
    
    if st.button("Backtest Çalıştır", type="primary"):
        try:
            with st.spinner("Veriler yükleniyor..."):
                # Daha kısa tarih aralığı ile başla
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if data.empty:
                    st.error("❌ Veri çekilemedi - Sembolü ve tarihleri kontrol edin")
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
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Toplam Getiri", f"{metrics['total_return_%']}%")
                st.metric("Win Rate", f"{metrics['win_rate_%']}%")
                st.metric("Max Drawdown", f"{metrics['max_drawdown_%']}%")
            
            with col2:
                st.metric("Toplam İşlem", f"{metrics['total_trades']}")
                st.metric("Ort. Kazanç", f"${metrics['avg_win']:.2f}")
                st.metric("Ort. Kayıp", f"${metrics['avg_loss']:.2f}")
            
            with col3:
                st.metric("Ort. Hold Days", f"{metrics['avg_hold_days']}")
            
            # Grafikler
            if not trades.empty:
                st.subheader("📈 Performans Grafikleri")
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                
                # Equity curve
                ax1.plot(equity['date'], equity['equity'], color='green', linewidth=2)
                ax1.set_title('Portföy Değeri', fontweight='bold')
                ax1.set_ylabel('Equity ($)')
                ax1.grid(True, alpha=0.3)
                
                # Trades
                for _, trade in trades.iterrows():
                    color = 'green' if trade['pnl'] > 0 else 'red'
                    ax1.scatter(trade['exit_date'], trade['entry_price'] * trade['shares'], 
                              color=color, alpha=0.6)
                
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
                
                # Trade tablosu
                st.subheader("📋 İşlem Listesi")
                display_trades = trades.copy()
                display_trades['entry_date'] = display_trades['entry_date'].dt.strftime('%Y-%m-%d')
                display_trades['exit_date'] = display_trades['exit_date'].dt.strftime('%Y-%m-%d')
                st.dataframe(display_trades)
                
            else:
                st.info("ℹ️ Backtest süresinde işlem gerçekleşmedi. Parametreleri gevşetmeyi deneyin.")
                    
        except Exception as e:
            st.error(f"❌ Ana backtest hatası: {str(e)}")

with tab2:
    st.header("🔧 Parametre Optimizasyonu")
    st.info("Önce temel backtest'in çalıştığından emin olalım...")

with tab3:
    st.header("📊 Sağlamlık Analizi")
    st.info("Backtest stabil çalıştıktan sonra bu özellikleri ekleyeceğiz...")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>⚠️ Risk Uyarısı:</strong> Backtest sonuçları geçmiş performans göstergesidir.</p>
</div>
""", unsafe_allow_html=True)
