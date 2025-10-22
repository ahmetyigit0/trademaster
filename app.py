import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =========================
# GELİŞTİRİLMİŞ ŞİFRE KORUMASI
# =========================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
        st.session_state["password_attempts"] = 0
        st.session_state["last_attempt"] = None
    
    def password_entered():
        current_time = datetime.now()
        
        # 3 başarısız denemeden sonra 10 dakika bekleme
        if st.session_state["password_attempts"] >= 3:
            last_attempt = st.session_state.get("last_attempt")
            if last_attempt and (current_time - last_attempt).seconds < 600:
                st.error("🚫 Lütfen 10 dakika sonra tekrar deneyin")
                return
        
        if st.session_state["password"] == "efe":
            st.session_state["password_correct"] = True
            st.session_state["password_attempts"] = 0
            st.session_state["last_attempt"] = None
            del st.session_state["password"]
        else:
            st.session_state["password_attempts"] += 1
            st.session_state["last_attempt"] = current_time
            st.session_state["password_correct"] = False
            
            remaining_attempts = 3 - st.session_state["password_attempts"]
            if remaining_attempts > 0:
                st.error(f"❌ Yanlış şifre! {remaining_attempts} deneme hakkınız kaldı.")
            else:
                st.error("🚫 3 başarısız giriş. Lütfen 10 dakika sonra tekrar deneyin.")
    
    if not st.session_state["password_correct"]:
        st.markdown("### 🔐 Yeni Kombine Stratejiye Giriş")
        
        # Şifre girişi
        col1, col2 = st.columns([1, 2])
        with col1:
            st.text_input(
                "Şifre", 
                type="password", 
                on_change=password_entered, 
                key="password",
                placeholder="Şifreyi giriniz...",
                help="3 başarısız denemeden sonra 10 dakika bekleyin"
            )
        
        # Bekleme süresi bilgisi
        if st.session_state["password_attempts"] >= 3:
            last_attempt = st.session_state.get("last_attempt")
            if last_attempt:
                wait_until = last_attempt + timedelta(minutes=10)
                st.warning(f"⏰ {wait_until.strftime('%H:%M:%S')} kadar bekleyin")
        
        return False
    return True

if not check_password():
    st.stop()

# =========================
# GELİŞTİRİLMİŞ BACKTEST MOTORU
# =========================
class EnhancedSwingBacktest:
    def __init__(self):
        self.commission = 0.001
        self.initial_capital = 10000
        self.max_drawdown = 0
    
    def calculate_drawdown(self, equity_series):
        """Maksimum drawdown hesaplama"""
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak * 100
        return drawdown.min() if not drawdown.empty else 0
    
    def calculate_indicators(self, df):
        df = df.copy()
        try:
            # Trend göstergeleri
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
            df['SMA_100'] = df['Close'].rolling(window=100).mean()
            
            # Momentum göstergeleri
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            period = 20
            df['BB_MA'] = df['Close'].rolling(window=period).mean()
            df['BB_STD'] = df['Close'].rolling(window=period).std()
            df['BB_Upper'] = df['BB_MA'] + (df['BB_STD'] * 2)
            df['BB_Lower'] = df['BB_MA'] - (df['BB_STD'] * 2)
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_MA']
            
            # MACD
            ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema_12 - ema_26
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
            
            # Fibonacci Support
            window_fib = 50
            high_50 = df['High'].rolling(window=window_fib).max()
            low_50 = df['Low'].rolling(window=window_fib).min()
            df['Fib_Support_382'] = low_50 + (high_50 - low_50) * 0.382
            df['Fib_Support_618'] = low_50 + (high_50 - low_50) * 0.618
            
            # Volatilite
            df['Volatility'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252) * 100
            
            df = df.fillna(method='bfill').fillna(method='ffill')
            return df
            
        except Exception as e:
            st.error(f"Gösterge hesaplama hatası: {e}")
            # Fallback değerleri
            for col in ['Close', 'Open', 'High', 'Low']:
                if col in df.columns:
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            
            df['EMA_20'] = df['Close']
            df['EMA_50'] = df['Close']
            df['RSI'] = 50
            df['BB_Lower'] = df['Close'] * 0.95
            df['MACD'] = 0
            df['Signal_Line'] = 0
            df['Fib_Support_382'] = df['Close'] * 0.9
            return df
    
    def generate_signals(self, df, params):
        df_copy = df.copy()
        
        # Gelişmiş sinyal koşulları
        df_copy['Trend_Up'] = (df_copy['EMA_20'] > df_copy['EMA_50']) & (df_copy['Close'] > df_copy['SMA_100'])
        df_copy['Momentum_Buy'] = df_copy['RSI'] < params['rsi_oversold']
        df_copy['Support_Touch'] = (df_copy['Close'] < df_copy['BB_Lower']) | (df_copy['Low'] < df_copy['BB_Lower'])
        df_copy['Fib_Support_Hit'] = (df_copy['Close'] <= df_copy['Fib_Support_382'] * 1.01)
        
        # MACD kesişimi
        df_copy['MACD_Cross_Up'] = (
            (df_copy['MACD'] > df_copy['Signal_Line']) & 
            (df_copy['MACD'].shift(1) <= df_copy['Signal_Line'].shift(1))
        )
        
        # Volatilite filtresi
        df_copy['Low_Volatility'] = df_copy['Volatility'] < params.get('max_volatility', 80)
        
        # Nihai alım sinyali
        df_copy['Buy_Signal'] = (
            df_copy['Trend_Up'] & 
            df_copy['Momentum_Buy'] & 
            (df_copy['Support_Touch'] | df_copy['Fib_Support_Hit']) & 
            df_copy['MACD_Cross_Up'] &
            df_copy['Low_Volatility']
        )
        
        # Sinyal kalitesi skoru
        df_copy['Signal_Score'] = (
            df_copy['Trend_Up'].astype(int) * 0.3 +
            (df_copy['RSI'] < 30).astype(int) * 0.2 +
            df_copy['Support_Touch'].astype(int) * 0.2 +
            df_copy['MACD_Cross_Up'].astype(int) * 0.2 +
            df_copy['Low_Volatility'].astype(int) * 0.1
        )
        
        signals = pd.DataFrame(index=df.index, data={'action': 'hold', 'signal_score': 0})
        signals[['stop_loss', 'take_profit']] = np.nan
        
        buy_indices = df_copy[df_copy['Buy_Signal']].index
        
        if not buy_indices.empty:
            risk_pct = params.get('risk_pct', 0.02)
            
            buy_data = df_copy.loc[buy_indices].copy()
            
            # Dinamik SL/TP - volatiliteye göre ayarlama
            volatility_factor = np.clip(buy_data['Volatility'] / 40, 0.5, 2.0)
            adjusted_risk_pct = risk_pct * volatility_factor
            
            stop_losses = buy_data['Close'] * (1 - adjusted_risk_pct)
            take_profits = buy_data['Close'] * (1 + (adjusted_risk_pct * params['reward_ratio']))
            
            signals.loc[buy_indices, 'action'] = 'buy'
            signals.loc[buy_indices, 'stop_loss'] = stop_losses
            signals.loc[buy_indices, 'take_profit'] = take_profits
            signals.loc[buy_indices, 'signal_score'] = df_copy.loc[buy_indices, 'Signal_Score']

        signals['action'] = signals['action'].fillna('hold')
        
        buy_count = signals['action'].value_counts().get('buy', 0)
        avg_score = signals[signals['action'] == 'buy']['signal_score'].mean()
        
        st.info(f"🎯 {buy_count} karmaşık alış sinyali bulundu (Ort. Skor: {avg_score:.2f})")
        return signals
    
    def run_backtest(self, data, params):
        df = self.calculate_indicators(data)
        signals = self.generate_signals(df, params)
        
        df_combined = df.merge(signals[['action', 'stop_loss', 'take_profit', 'signal_score']], 
                               left_index=True, right_index=True, how='left')
        
        df_combined['action'] = df_combined['action'].fillna('hold')
        df_combined[['stop_loss', 'take_profit']] = df_combined[['stop_loss', 'take_profit']].fillna(0.0)

        capital = float(self.initial_capital)
        position = None
        trades = []
        equity_curve = []
        max_capital = capital
        drawdowns = []
        
        for date in df_combined.index:
            row = df_combined.loc[date]
            current_price = float(row['Close'])
            signal_action = row['action']
            
            current_equity = float(capital)
            
            if position is not None:
                current_equity += float(position['shares']) * current_price
            
            # Drawdown hesaplama
            max_capital = max(max_capital, current_equity)
            current_drawdown = (current_equity - max_capital) / max_capital * 100
            drawdowns.append(current_drawdown)
            
            equity_curve.append({'date': date, 'equity': current_equity, 'drawdown': current_drawdown})
            
            # ALIM KOŞULU
            if position is None and signal_action == 'buy':
                stop_loss = float(row['stop_loss'])
                risk_per_share = current_price - stop_loss
                
                if risk_per_share > 0:
                    # Sinyal kalitesine göre pozisyon büyüklüğü ayarlama
                    signal_score = float(row.get('signal_score', 0.5))
                    adjusted_risk = params['risk_per_trade'] * min(signal_score * 2, 1.5)
                    
                    risk_amount = capital * adjusted_risk
                    shares = risk_amount / risk_per_share
                    
                    # Maksimum %95 sermaye kullanımı
                    max_shares = (capital * 0.95) / current_price
                    shares = min(shares, max_shares)
                    
                    if shares > 0:
                        position = {
                            'entry_date': date,
                            'entry_price': current_price,
                            'shares': shares,
                            'stop_loss': stop_loss,
                            'take_profit': float(row['take_profit']),
                            'signal_score': signal_score
                        }
                        capital -= shares * current_price * (1 + self.commission)
            
            # ÇIKIŞ KOŞULLARI
            elif position is not None:
                exited = False
                exit_price = None
                exit_reason = None

                # Stop Loss
                if current_price <= position['stop_loss']:
                    exit_price = position['stop_loss']
                    exit_reason = 'SL'
                    exited = True
                
                # Take Profit
                elif current_price >= position['take_profit']:
                    exit_price = position['take_profit']
                    exit_reason = 'TP'
                    exited = True
                
                # Trailing Stop (isteğe bağlı)
                elif params.get('trailing_stop', False):
                    new_stop = current_price * (1 - params.get('trailing_stop_pct', 0.03))
                    if new_stop > position['stop_loss']:
                        position['stop_loss'] = new_stop

                if exited:
                    exit_value = position['shares'] * exit_price
                    capital += exit_value * (1 - self.commission)
                    
                    entry_value = position['shares'] * position['entry_price']
                    pnl = exit_value - entry_value - (entry_value * self.commission)
                    
                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': date,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'shares': position['shares'],
                        'pnl': pnl,
                        'return_pct': (pnl / entry_value) * 100,
                        'exit_reason': exit_reason,
                        'signal_score': position['signal_score']
                    })
                    position = None
        
        # Açık pozisyonu kapat
        if position is not None:
            last_price = float(df_combined['Close'].iloc[-1])
            exit_value = position['shares'] * last_price
            capital += exit_value * (1 - self.commission)
            
            entry_value = position['shares'] * position['entry_price']
            pnl = exit_value - entry_value - (entry_value * self.commission)
            
            trades.append({
                'entry_date': position['entry_date'],
                'exit_date': df_combined.index[-1],
                'entry_price': position['entry_price'],
                'exit_price': last_price,
                'shares': position['shares'],
                'pnl': pnl,
                'return_pct': (pnl / entry_value) * 100,
                'exit_reason': 'OPEN',
                'signal_score': position['signal_score']
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
                'best_trade': "0.0%",
                'worst_trade': "0.0%",
                'profit_factor': "0.00",
                'max_drawdown': "0.0%",
                'sharpe_ratio': "0.00"
            }
        
        try:
            initial_equity = self.initial_capital
            final_equity = equity_df['equity'].iloc[-1]
            
            total_return = (final_equity - initial_equity) / initial_equity * 100 
            
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if (total_trades - winning_trades) > 0 else 0
            
            # Profit Factor
            gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
            gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            best_trade = trades_df['return_pct'].max() if not trades_df.empty else 0
            worst_trade = trades_df['return_pct'].min() if not trades_df.empty else 0
            
            # Maksimum Drawdown
            max_drawdown = equity_df['drawdown'].min() if 'drawdown' in equity_df.columns else 0
            
            # Sharpe Ratio (basit)
            if len(equity_df) > 1:
                returns = equity_df['equity'].pct_change().dropna()
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            else:
                sharpe_ratio = 0
            
            return {
                'total_return': f"{total_return:+.2f}%",
                'total_trades': str(total_trades),
                'win_rate': f"{win_rate:.1f}%",
                'avg_win': f"${avg_win:.2f}",
                'avg_loss': f"${abs(avg_loss):.2f}", 
                'best_trade': f"{best_trade:.2f}%",
                'worst_trade': f"{worst_trade:.2f}%",
                'profit_factor': f"{profit_factor:.2f}",
                'max_drawdown': f"{max_drawdown:.1f}%",
                'sharpe_ratio': f"{sharpe_ratio:.2f}"
            }
            
        except Exception as e:
            st.error(f"Metrik hesaplama hatası: {e}")
            return {
                'total_return': "HATA",
                'total_trades': "HATA",
                'win_rate': "HATA",
                'avg_win': "HATA",
                'avg_loss': "HATA",
                'best_trade': "HATA",
                'worst_trade': "HATA",
                'profit_factor': "HATA",
                'max_drawdown': "HATA",
                'sharpe_ratio': "HATA"
            }

# =========================
# GELİŞTİRİLMİŞ STREAMLIT UYGULAMASI
# =========================
st.set_page_config(page_title="Gelişmiş Kombine Swing Backtest", layout="wide")
st.title("🧠 Gelişmiş Kombine Swing Trading Backtest")
st.markdown("**6 Göstergeli Akıllı Kombinasyon Stratejisi: EMA, SMA, RSI, BB, MACD, Fibonacci + Volatilite**")

# Sidebar
st.sidebar.header("⚙️ Ayarlar")
ticker = st.sidebar.selectbox("Sembol", ["BTC-USD", "ETH-USD", "TSLA", "NVDA", "AAPL", "GOOGL", "MSFT", "AMZN", "META"])
start_date = st.sidebar.date_input("Başlangıç", datetime(2023, 1, 1))
end_date = st.sidebar.date_input("Bitiş", datetime(2024, 1, 1))

st.sidebar.header("📊 Strateji Parametreleri")
rsi_oversold = st.sidebar.slider("RSI Aşırı Satım", 25, 45, 30)
reward_ratio = st.sidebar.slider("Risk/Ödül Oranı", 1.5, 4.0, 2.5)
risk_per_trade = st.sidebar.slider("Risk %", 1.0, 5.0, 1.5) / 100
max_volatility = st.sidebar.slider("Maks. Volatilite %", 30, 100, 80)

st.sidebar.header("🎯 Gelişmiş Seçenekler")
use_trailing_stop = st.sidebar.checkbox("Trailing Stop Kullan", value=False)
if use_trailing_stop:
    trailing_stop_pct = st.sidebar.slider("Trailing Stop %", 1.0, 5.0, 3.0) / 100

params = {
    'rsi_oversold': rsi_oversold,
    'reward_ratio': reward_ratio,
    'risk_per_trade': risk_per_trade,
    'max_volatility': max_volatility,
    'trailing_stop': use_trailing_stop,
    'trailing_stop_pct': trailing_stop_pct if use_trailing_stop else 0.03
}

# Ana içerik
if st.button("🚀 Gelişmiş Backtest Çalıştır", type="primary"):
    try:
        with st.spinner("Veri yükleniyor ve göstergeler hesaplanıyor..."):
            extended_start = start_date - timedelta(days=200) 
            data = yf.download(ticker, start=extended_start, end=end_date, progress=False)
            
            if data.empty:
                st.error("❌ Veri bulunamadı")
                st.stop()
            
            data = data[data.index >= pd.to_datetime(start_date)]
            data = data[data.index <= pd.to_datetime(end_date)]
            
            st.success(f"✅ {len(data)} günlük veri yüklendi - {data.index[0].strftime('%d.%m.%Y')} - {data.index[-1].strftime('%d.%m.%Y')}")
        
        backtester = EnhancedSwingBacktest()
        
        with st.spinner("Gelişmiş backtest çalıştırılıyor..."):
            trades, equity = backtester.run_backtest(data, params)
            metrics = backtester.calculate_metrics(trades, equity)
        
        st.subheader("📊 Detaylı Performans Özeti")
        
        # Ana metrikler
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
            st.metric("En İyi İşlem", metrics['best_trade'])
            st.metric("En Kötü İşlem", metrics['worst_trade'])
            st.metric("Max Drawdown", metrics['max_drawdown'])
        
        with col4:
            st.metric("Sharpe Ratio", metrics['sharpe_ratio'])
            if not trades.empty:
                avg_holding = (trades['exit_date'] - trades['entry_date']).dt.days.mean()
                st.metric("Ort. Tutma Süresi", f"{avg_holding:.1f} gün")
            
                sl_count = len(trades[trades['exit_reason'] == 'SL'])
                tp_count = len(trades[trades['exit_reason'] == 'TP'])
                st.metric("SL/TP Oranı", f"{sl_count}/{tp_count}")
        
        if not trades.empty and 'equity' in equity.columns: 
            st.subheader("📈 Detaylı Performans Grafikleri")
            
            # Equity Curve ve Drawdown
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Equity Curve
            ax1.plot(equity['date'], equity['equity'], color='purple', linewidth=2, label='Portföy Değeri')
            ax1.set_title(f'{ticker} Portföy Değeri Gelişimi')
            ax1.set_ylabel('Equity ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Drawdown
            ax2.fill_between(equity['date'], equity['drawdown'], 0, color='red', alpha=0.3, label='Drawdown')
            ax2.set_title('Drawdown (%)')
            ax2.set_ylabel('Drawdown %')
            ax2.set_xlabel('Tarih')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # İşlem analizi
            st.subheader("📋 Detaylı İşlem Analizi")
            
            if not trades.empty:
                display_trades = trades.copy()
                
                # Tarih formatlama
                for col in ['entry_date', 'exit_date']:
                    if not display_trades[col].empty and isinstance(display_trades[col].iloc[0], (datetime, pd.Timestamp)):
                        display_trades[col] = display_trades[col].dt.strftime('%Y-%m-%d')
                
                # Sayısal sütunları yuvarla
                display_trades['pnl'] = display_trades['pnl'].round(2)
                display_trades['return_pct'] = display_trades['return_pct'].round(2)
                display_trades['signal_score'] = display_trades['signal_score'].round(3)
                
                # Renkli gösterim
                def color_pnl(val):
                    color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
                    return f'color: {color}'
                
                st.dataframe(
                    display_trades.style.applymap(color_pnl, subset=['pnl', 'return_pct']),
                    height=400
                )
                
                # İşlem dağılımı
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**İşlem Sonuçları Dağılımı**")
                    exit_reasons = display_trades['exit_reason'].value_counts()
                    fig2, ax2 = plt.subplots(figsize=(8, 6))
                    ax2.pie(exit_reasons.values, labels=exit_reasons.index, autopct='%1.1f%%', startangle=90)
                    ax2.set_title('İşlem Çıkış Nedenleri')
                    st.pyplot(fig2)
                
                with col2:
                    st.markdown("**Getiri Dağılımı**")
                    fig3, ax3 = plt.subplots(figsize=(8, 6))
                    ax3.hist(display_trades['return_pct'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                    ax3.axvline(0, color='red', linestyle='--', alpha=0.8)
                    ax3.set_xlabel('Getiri (%)')
                    ax3.set_ylabel('Frekans')
                    ax3.set_title('İşlem Getirileri Dağılımı')
                    ax3.grid(True, alpha=0.3)
                    st.pyplot(fig3)
            
        elif not trades.empty and 'equity' not in equity.columns:
            st.info("🤷 İşlemler gerçekleşti ancak grafik verisi (equity) bulunamadı.")
        else:
            st.info("🤷 Hiç işlem gerçekleşmedi. Parametreleri ayarlayıp tekrar deneyin.")
            
    except Exception as e:
        st.error(f"❌ Hata: {str(e)}")
        st.info("💡 İpucu: Tarih aralığını veya parametreleri değiştirmeyi deneyin")

st.markdown("---")
st.markdown("**Gelişmiş Backtest Sistemi v5.0 - 6'lı Kombinasyon Stratejisi**")
