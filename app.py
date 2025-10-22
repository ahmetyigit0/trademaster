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
        """Swing trade sinyalleri üretir"""
        df = self.calculate_indicators(df)
        
        # NaN değerleri temizle
        df = df.dropna()
        
        if df.empty:
            return pd.DataFrame()
        
        # Swing sinyalleri
        trend_up = df['EMA_20'] > df['EMA_50']
        rsi_oversold = df['RSI'] < params.get('rsi_oversold', 35)
        macd_bullish = df['MACD_Hist'] > params.get('macd_threshold', 0)
        price_above_ema20 = df['Close'] > df['EMA_20']
        
        # AL sinyali
        buy_signal = trend_up & rsi_oversold & macd_bullish & price_above_ema20
        
        # Signals DataFrame'ini doğru şekilde oluştur - SADECE index kullan
        signals = pd.DataFrame({
            'action': 'hold'
        }, index=df.index)
        
        signals.loc[buy_signal, 'action'] = 'buy'
        
        # Stop ve TP seviyeleri
        atr_multiplier = params.get('atr_multiplier', 1.5)
        signals['stop_loss'] = df['Close'] - (df['ATR'] * atr_multiplier)
        risk_distance = df['ATR'] * atr_multiplier
        signals['tp1'] = df['Close'] + risk_distance * 1.0
        signals['tp2'] = df['Close'] + risk_distance * 2.0
        signals['tp3'] = df['Close'] + risk_distance * 3.0
        
        return signals
    
    def backtest(self, df, params, initial_capital=10000):
        """Backtest yürütür"""
        try:
            # Önce sinyalleri hesapla
            signals = self.swing_signal(df, params)
            
            if signals.empty:
                return pd.DataFrame(), pd.DataFrame()
            
            # Index'leri kontrol et ve hizala
            common_index = df.index.intersection(signals.index)
            if len(common_index) == 0:
                return pd.DataFrame(), pd.DataFrame()
                
            aligned_df = df.loc[common_index]
            aligned_signals = signals.loc[common_index]
            
            trades = []
            position = None
            capital = initial_capital
            equity_curve = []
            
            for i in range(len(aligned_signals)):
                idx = aligned_signals.index[i]
                row = aligned_signals.iloc[i]
                current_data = aligned_df.loc[idx]
                
                current_price = current_data['Close']
                current_high = current_data['High']
                current_low = current_data['Low']
                
                # Equity curve güncelleme
                current_equity = capital
                if position is not None:
                    current_equity += position['shares'] * current_price
                equity_curve.append({
                    'date': idx,
                    'equity': current_equity
                })
                
                if position is None and row['action'] == 'buy':
                    # Yeni pozisyon aç
                    risk_per_share = current_price - row['stop_loss']
                    if risk_per_share <= 0:
                        continue
                        
                    risk_amount = capital * params.get('risk_per_trade', 0.02)
                    shares = risk_amount / risk_per_share
                    
                    # Komisyon ve slippage
                    entry_price = current_price * (1 + self.slippage)
                    position_value = shares * entry_price
                    commission_paid = position_value * self.commission
                    
                    # Sermaye kontrolü
                    total_cost = position_value + commission_paid
                    if total_cost > capital * 0.8:  # Max %80 kullan
                        shares = (capital * 0.8) / entry_price
                        position_value = shares * entry_price
                        commission_paid = position_value * self.commission
                        total_cost = position_value + commission_paid
                    
                    if shares > 0 and total_cost <= capital:
                        position = {
                            'entry_date': idx,
                            'entry_price': entry_price,
                            'shares': shares,
                            'stop_loss': row['stop_loss'],
                            'tp1': row['tp1'],
                            'tp2': row['tp2'],
                            'tp3': row['tp3'],
                            'tp1_hit': False,
                            'tp2_hit': False,
                            'breakeven': False
                        }
                        
                        capital -= total_cost
                
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
                        position['breakeven'] = True
                        position['stop_loss'] = position['entry_price']
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
                    
                    if exit_reason and exit_shares > 0:
                        # Pozisyon kapat
                        exit_value = exit_shares * exit_price
                        commission_paid = exit_value * self.commission
                        capital += exit_value - commission_paid
                        
                        # Trade kaydı
                        entry_value = exit_shares * position['entry_price']
                        trade_pnl = exit_value - entry_value
                        trade_pnl -= (entry_value * self.commission) + commission_paid
                        
                        trades.append({
                            'entry_date': position['entry_date'],
                            'exit_date': idx,
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'shares': exit_shares,
                            'pnl': trade_pnl,
                            'return_pct': (trade_pnl / entry_value) * 100 if entry_value > 0 else 0,
                            'exit_reason': exit_reason,
                            'hold_days': (idx - position['entry_date']).days
                        })
                        
                        if position['shares'] <= 0:
                            position = None
            
            # Kapanmamış pozisyonları kapat
            if position is not None and len(aligned_df) > 0:
                last_price = aligned_df['Close'].iloc[-1]
                exit_value = position['shares'] * last_price
                commission_paid = exit_value * self.commission
                capital += exit_value - commission_paid
                
                entry_value = position['shares'] * position['entry_price']
                trade_pnl = exit_value - entry_value
                trade_pnl -= (entry_value * self.commission) + commission_paid
                
                trades.append({
                    'entry_date': position['entry_date'],
                    'exit_date': aligned_df.index[-1],
                    'entry_price': position['entry_price'],
                    'exit_price': last_price,
                    'shares': position['shares'],
                    'pnl': trade_pnl,
                    'return_pct': (trade_pnl / entry_value) * 100 if entry_value > 0 else 0,
                    'exit_reason': 'OPEN',
                    'hold_days': (aligned_df.index[-1] - position['entry_date']).days
                })
            
            trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
            equity_df = pd.DataFrame(equity_curve) if equity_curve else pd.DataFrame()
            
            return trades_df, equity_df
            
        except Exception as e:
            st.error(f"Backtest içi hata: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()
    
    def calculate_metrics(self, trades_df, equity_df, initial_capital):
        """Performans metriklerini hesaplar"""
        if trades_df.empty or equity_df.empty:
            return {
                'total_return_%': 0,
                'total_trades': 0,
                'win_rate_%': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'avg_r_multiple': 0,
                'max_drawdown_%': 0,
                'sharpe_ratio': 0,
                'avg_hold_days': 0
            }
        
        try:
            total_return = (equity_df['equity'].iloc[-1] - initial_capital) / initial_capital * 100
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] < 0])
            win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
            
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
            
            total_win = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
            total_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
            profit_factor = total_win / total_loss if total_loss > 0 else float('inf')
            
            # R Multiple
            if not trades_df.empty:
                trades_df['r_multiple'] = trades_df['pnl'] / (trades_df['entry_price'] * 0.01)  # Basitleştirilmiş
                avg_r = trades_df['r_multiple'].mean()
            else:
                avg_r = 0
            
            # Drawdown
            if not equity_df.empty:
                equity_df['peak'] = equity_df['equity'].cummax()
                equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
                max_drawdown = equity_df['drawdown'].min()
            else:
                max_drawdown = 0
            
            # Sharpe (basit)
            if not equity_df.empty:
                equity_df['daily_return'] = equity_df['equity'].pct_change().fillna(0)
                daily_returns = equity_df['daily_return']
                if len(daily_returns) > 1 and daily_returns.std() > 0:
                    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
                else:
                    sharpe = 0
            else:
                sharpe = 0
            
            metrics = {
                'total_return_%': round(total_return, 2),
                'total_trades': total_trades,
                'win_rate_%': round(win_rate, 1),
                'avg_win': round(avg_win, 2),
                'avg_loss': round(avg_loss, 2),
                'profit_factor': round(profit_factor, 2),
                'avg_r_multiple': round(avg_r, 2),
                'max_drawdown_%': round(max_drawdown, 2),
                'sharpe_ratio': round(sharpe, 2),
                'avg_hold_days': round(trades_df['hold_days'].mean(), 1) if not trades_df.empty else 0
            }
            
            return metrics
            
        except Exception as e:
            st.error(f"Metrik hesaplama hatası: {e}")
            return {
                'total_return_%': 0,
                'total_trades': 0,
                'win_rate_%': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'avg_r_multiple': 0,
                'max_drawdown_%': 0,
                'sharpe_ratio': 0,
                'avg_hold_days': 0
            }

# =========================
# STREAMLIT UYGULAMASI
# =========================
st.set_page_config(page_title="Swing Backtest Pro", layout="wide")
st.title("🚀 Swing Trade Backtest Sistemi")
st.markdown("**Profesyonel Swing Strateji Test Platformu**")

# Sidebar kontrolleri
st.sidebar.header("⚙️ Backtest Ayarları")
ticker = st.sidebar.selectbox("Sembol", ["BTC-USD", "ETH-USD", "ADA-USD", "BNB-USD", "XRP-USD", "SOL-USD", "AAPL", "GOOGL", "MSFT"])
start_date = st.sidebar.date_input("Başlangıç Tarihi", datetime(2020, 1, 1))
end_date = st.sidebar.date_input("Bitiş Tarihi", datetime(2024, 1, 1))

st.sidebar.header("📊 Strateji Parametreleri")
rsi_oversold = st.sidebar.slider("RSI Aşırı Satım", 20, 40, 35)
atr_multiplier = st.sidebar.slider("ATR Çarpanı", 1.0, 3.0, 1.5)
risk_per_trade = st.sidebar.slider("İşlem Risk %", 1.0, 5.0, 2.0) / 100

# Ana içerik
tab1, tab2, tab3 = st.tabs(["📈 Backtest", "🔧 Optimizasyon", "📊 Sağlamlık Analizi"])

with tab1:
    st.header("Backtest Sonuçları")
    
    if st.button("Backtest Çalıştır", type="primary"):
        try:
            with st.spinner("Veriler yükleniyor..."):
                # Tarih aralığını biraz genişlet (indikatörler için)
                extended_start = start_date - timedelta(days=100)
                data = yf.download(ticker, start=extended_start, end=end_date, progress=False)
                
                if data.empty:
                    st.error("❌ Veri çekilemedi - Sembolü ve tarihleri kontrol edin")
                    st.stop()
                
                # Sadece istenen tarih aralığını kullan
                data = data[data.index >= pd.to_datetime(start_date)]
                data = data[data.index <= pd.to_datetime(end_date)]
                
                if data.empty:
                    st.error("❌ Filtrelenmiş veri kalmadı - Tarihleri kontrol edin")
                    st.stop()
            
            st.success(f"✅ {len(data)} günlük veri yüklendi ({data.index[0].strftime('%Y-%m-%d')} - {data.index[-1].strftime('%Y-%m-%d')})")
            
            # Veri kalitesi kontrolü
            st.info(f"📊 Veri Özeti: Açılış: ${data['Open'].iloc[0]:.2f} - Kapanış: ${data['Close'].iloc[-1]:.2f}")
            
            # Backtest çalıştır
            backtester = SwingBacktest()
            params = {
                'rsi_oversold': rsi_oversold,
                'atr_multiplier': atr_multiplier,
                'risk_per_trade': risk_per_trade
            }
            
            with st.spinner("Backtest çalıştırılıyor..."):
                trades, equity = backtester.backtest(data, params)
                metrics = backtester.calculate_metrics(trades, equity, 10000)
            
            # Sonuçları göster
            st.subheader("📊 Performans Özeti")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Toplam Getiri", f"%{metrics.get('total_return_%', 0):.2f}")
                st.metric("Win Rate", f"%{metrics.get('win_rate_%', 0):.1f}")
                st.metric("Max Drawdown", f"%{metrics.get('max_drawdown_%', 0):.1f}")
            
            with col2:
                st.metric("Toplam İşlem", f"{metrics.get('total_trades', 0)}")
                st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
                st.metric("Ort. R Multiple", f"{metrics.get('avg_r_multiple', 0):.2f}")
            
            with col3:
                st.metric("Sharpe Oranı", f"{metrics.get('sharpe_ratio', 0):.2f}")
                st.metric("Ort. Kazanç", f"${metrics.get('avg_win', 0):.2f}")
                st.metric("Ort. Kayıp", f"${metrics.get('avg_loss', 0):.2f}")
            
            # Grafikler
            if not trades.empty and not equity.empty:
                st.subheader("📈 Performans Grafikleri")
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
                
                # Equity curve
                ax1.plot(equity['date'], equity['equity'], color='blue', linewidth=2)
                ax1.set_title('Equity Curve', fontweight='bold')
                ax1.set_ylabel('Portföy Değeri ($)')
                ax1.grid(True, alpha=0.3)
                
                # Drawdown
                if 'drawdown' in equity.columns:
                    ax2.fill_between(equity['date'], equity['drawdown'], 0, alpha=0.3, color='red')
                    ax2.set_title('Drawdown', fontweight='bold')
                    ax2.set_ylabel('Drawdown %')
                    ax2.grid(True, alpha=0.3)
                
                # R Multiple dağılımı
                if 'r_multiple' in trades.columns and not trades.empty:
                    ax3.hist(trades['r_multiple'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                    ax3.axvline(trades['r_multiple'].mean(), color='red', linestyle='--', 
                               label=f'Ort: {trades["r_multiple"].mean():.2f}')
                    ax3.set_title('R Multiple Dağılımı', fontweight='bold')
                    ax3.set_xlabel('R Multiple')
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)
                
                # Exit reason dağılımı
                if not trades.empty:
                    exit_counts = trades['exit_reason'].value_counts()
                    colors = sns.color_palette('pastel')[0:len(exit_counts)]
                    ax4.pie(exit_counts.values, labels=exit_counts.index, autopct='%1.1f%%', colors=colors)
                    ax4.set_title('Çıkış Nedenleri', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Trade tablosu
                st.subheader("📋 İşlem Listesi")
                display_trades = trades.copy()
                display_trades['entry_date'] = display_trades['entry_date'].dt.strftime('%Y-%m-%d')
                display_trades['exit_date'] = display_trades['exit_date'].dt.strftime('%Y-%m-%d')
                st.dataframe(display_trades)
                
                # CSV indirme
                csv = trades.to_csv(index=False)
                st.download_button(
                    "📥 İşlemleri CSV olarak indir",
                    csv,
                    f"swing_trades_{ticker}_{start_date}_{end_date}.csv",
                    "text/csv"
                )
            else:
                st.info("ℹ️ Backtest süresinde işlem gerçekleşmedi. Parametreleri değiştirmeyi deneyin.")
                    
        except Exception as e:
            st.error(f"❌ Backtest hatası: {str(e)}")
            st.info("🔍 Hata ayıklama için verileri kontrol edin:")

with tab2:
    st.header("🔧 Parametre Optimizasyonu")
    st.info("🚧 Optimizasyon özelliği geliştirme aşamasında...")

with tab3:
    st.header("📊 Sağlamlık Analizi")
    st.info("🚧 Monte Carlo ve Bootstrap analizleri geliştirme aşamasında...")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>⚠️ Risk Uyarısı:</strong> Bu backtest sonuçları geçmiş performans göstergesidir ve geleceği garanti etmez.</p>
    <p>Swing Backtest Pro v1.0 | Profesyonel Algoritmik Test Platformu</p>
</div>
""", unsafe_allow_html=True)
