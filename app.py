import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =========================
# ŞİFRE KORUMASI (Aynı kalıyor)
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
        st.markdown("### 🔐 Yeni Kombine Stratejiye Giriş")
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
# BACKTEST MOTORU - KOMBINASYON STRATEJİSİ
# =========================
class SwingBacktest:
    def __init__(self):
        self.commission = 0.001
        self.initial_capital = 10000
    
    def calculate_indicators(self, df):
        df = df.copy()
        
        try:
            # 1. EMA'lar (Hızlı/Yavaş Trend)
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
            
            # 2. RSI (Momentum)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # 3. Bollinger Bantları (Volatilite ve Kanal)
            period = 20
            df['BB_MA'] = df['Close'].rolling(window=period).mean()
            df['BB_STD'] = df['Close'].rolling(window=period).std()
            df['BB_Upper'] = df['BB_MA'] + (df['BB_STD'] * 2)
            df['BB_Lower'] = df['BB_MA'] - (df['BB_STD'] * 2)
            
            # 4. MACD (Daha derin momentum)
            ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema_12 - ema_26
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            # 5. Basitleştirilmiş Fibonacci (Son 50 günün %38.2 ve %61.8'i)
            window_fib = 50
            high_50 = df['High'].rolling(window=window_fib).max()
            low_50 = df['Low'].rolling(window=window_fib).min()
            
            fib_382 = low_50 + (high_50 - low_50) * 0.382
            df['Fib_Support_382'] = fib_382
            
            # NaN değerleri temizle
            df = df.fillna(method='bfill').fillna(method='ffill')
            
            return df
            
        except Exception as e:
            st.error(f"Gösterge hesaplama hatası: {e}")
            df['EMA_20'] = df['Close']
            df['EMA_50'] = df['Close']
            df['RSI'] = 50
            df['BB_Lower'] = df['Close'] * 0.95
            df['MACD'] = 0
            df['Signal_Line'] = 0
            df['Fib_Support_382'] = df['Close'] * 0.9
            return df
    
    def generate_signals(self, df, params):
        # Tüm sinyalleri Pandas vektör operasyonlarıyla hesapla (Döngüden daha güvenli)
        df_copy = df.copy()
        
        # 1. Trend Onayı (EMA)
        df_copy['Trend_Up'] = df_copy['EMA_20'] > df_copy['EMA_50']
        
        # 2. Momentum Onayı (RSI)
        df_copy['Momentum_Buy'] = df_copy['RSI'] < params['rsi_oversold']
        
        # 3. Volatilite/Kanal Desteği (BB)
        df_copy['Support_Touch'] = df_copy['Close'] < df_copy['BB_Lower'] 
        
        # 4. Makro Destek (Fibonacci)
        df_copy['Fib_Support_Hit'] = df_copy['Close'] <= df_copy['Fib_Support_382'] * 1.01
        
        # 5. MACD Kesişimi (Buradaki mantık düzeltildi)
        # MACD yukarı kesti AND önceki MACD altındaydı
        df_copy['MACD_Cross_Up'] = (
            (df_copy['MACD'] > df_copy['Signal_Line']) & 
            (df_copy['MACD'].shift(1) <= df_copy['Signal_Line'].shift(1))
        )
        
        # ALIM KOŞULU
        # Trend yukarı + RSI geri çekilmede + Fiyat destekte (BB veya Fib) + MACD Kesişimi
        df_copy['Buy_Signal'] = (
            df_copy['Trend_Up'] & 
            df_copy['Momentum_Buy'] & 
            (df_copy['Support_Touch'] | df_copy['Fib_Support_Hit']) & 
            df_copy['MACD_Cross_Up']
        )
        
        # Sonuçları listeye dönüştür
        signals = []
        for i in range(len(df_copy)):
            date = df_copy.index[i]
            if i < 50 or not df_copy.iloc[i]['Buy_Signal']:
                signals.append({'date': date, 'action': 'hold'})
            else:
                row = df_copy.iloc[i]
                close = row['Close']
                risk_pct = 0.02
                stop_loss = close * (1 - risk_pct)
                take_profit = close * (1 + (risk_pct * params['reward_ratio']))
                
                signals.append({
                    'date': date,
                    'action': 'buy',
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                })
        
        signals_df = pd.DataFrame(signals).set_index('date')
        
        buy_count = signals_df['action'].value_counts().get('buy', 0)
        st.info(f"🎯 {buy_count} karmaşık alış sinyali bulundu")
        return signals_df
    
    # run_backtest ve calculate_metrics metodları aynı kalacak
    # Sadece gereksiz yere tekrar yazmaktan kaçınmak için buraya koymuyorum.
    # Ancak yukarıdaki kod bloğuna dahil ettiğinizde çalışacaktır.

    def run_backtest(self, data, params):
        # Bu metotun içeriği, bir önceki tam kodunuzdaki gibi kalmalıdır.
        df = self.calculate_indicators(data)
        signals = self.generate_signals(df, params)
        
        capital = self.initial_capital
        position = None
        trades = []
        equity_curve = []
        
        for date in df.index:
            if date not in signals.index:
                # Sinyal yoksa equity'i güncelle
                current_price = df.loc[date, 'Close']
                current_equity = capital
                if position is not None:
                    current_equity += position['shares'] * current_price
                equity_curve.append({'date': date, 'equity': current_equity})
                continue
                
            current_price = df.loc[date, 'Close']
            signal = signals.loc[date]
            
            current_equity = capital
            if position is not None:
                current_equity += position['shares'] * current_price
            
            equity_curve.append({'date': date, 'equity': current_equity})
            
            if position is None and signal['action'] == 'buy':
                stop_loss = signal['stop_loss']
                risk_per_share = current_price - stop_loss
                
                if risk_per_share > 0:
                    risk_amount = capital * params['risk_per_trade']
                    shares = risk_amount / risk_per_share
                    
                    if shares > 0:
                        position = {
                            'entry_date': date,
                            'entry_price': current_price,
                            'shares': shares,
                            'stop_loss': stop_loss,
                            'take_profit': signal['take_profit']
                        }
                        capital -= shares * current_price
            
            elif position is not None:
                # Çıkış Kontrolü (Her gün SL/TP kontrol edilir)
                exited = False
                if current_price <= position['stop_loss']:
                    exit_price = position['stop_loss']
                    exit_reason = 'SL'
                    exited = True
                
                elif current_price >= position['take_profit']:
                    exit_price = position['take_profit']
                    exit_reason = 'TP'
                    exited = True

                # Eğer çıkış olduysa
                if exited:
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
                        'exit_reason': exit_reason
                    })
                    position = None
        
        if position is not None:
            last_price = df['Close'].iloc[-1]
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
                'exit_reason': 'OPEN'
            })
        
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        equity_df = pd.DataFrame(equity_curve)
        
        return trades_df, equity_df
    
    def calculate_metrics(self, trades_df, equity_df):
        # Bu metotun içeriği, bir önceki tam kodunuzdaki gibi kalmalıdır.
        if trades_df.empty:
            return {
                'total_return': "0.0%",
                'total_trades': "0",
                'win_rate': "0.0%",
                'avg_win': "$0.00",
                'avg_loss': "$0.00",
                'best_trade': "0.0%",
                'worst_trade': "0.0%"
            }
        
        try:
            initial_equity = self.initial_capital
            final_equity = equity_df['equity'].iloc[-1]
            
            # DÜZELTİLMİŞ TOPLAM GETİRİ HESABI
            total_return = (final_equity - initial_equity) / initial_equity * 100 
            
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if (total_trades - winning_trades) > 0 else 0
            
            best_trade = trades_df['return_pct'].max() if not trades_df.empty else 0
            worst_trade = trades_df['return_pct'].min() if not trades_df.empty else 0
            
            return {
                'total_return': f"{total_return:+.2f}%",
                'total_trades': str(total_trades),
                'win_rate': f"{win_rate:.1f}%",
                'avg_win': f"${avg_win:.2f}",
                'avg_loss': f"${abs(avg_loss):.2f}", 
                'best_trade': f"{best_trade:.2f}%",
                'worst_trade': f"{worst_trade:.2f}%"
            }
            
        except Exception as e:
            return {
                'total_return': "HATA",
                'total_trades': "HATA",
                'win_rate': "HATA",
                'avg_win': "HATA",
                'avg_loss': "HATA",
                'best_trade': "HATA",
                'worst_trade': "HATA"
            }

# =========================
# STREAMLIT UYGULAMASI (Aynı kalıyor)
# =========================
st.set_page_config(page_title="Kombine Swing Backtest", layout="wide")
st.title("🧠 Kombine Swing Trading Backtest")
st.markdown("**5 Göstergeli Agresif Kombinasyon Stratejisi: EMA, RSI, BB, MACD, Fibonacci**")

# Sidebar
st.sidebar.header("⚙️ Ayarlar")
ticker = st.sidebar.selectbox("Sembol", ["BTC-USD", "ETH-USD", "TSLA", "NVDA", "AAPL", "GOOGL", "MSFT"])
start_date = st.sidebar.date_input("Başlangıç", datetime(2023, 1, 1))
end_date = st.sidebar.date_input("Bitiş", datetime(2024, 1, 1))

st.sidebar.header("📊 Parametreler")
rsi_oversold = st.sidebar.slider("RSI Aşırı Satım (Buy Eşiği)", 25, 45, 30)
reward_ratio = st.sidebar.slider("Risk/Ödül Oranı (TP Multiplier)", 1.5, 4.0, 2.5)
risk_per_trade = st.sidebar.slider("Risk % (Pozisyon Büyüklüğü)", 1.0, 5.0, 1.5) / 100

params = {
    'rsi_oversold': rsi_oversold,
    'reward_ratio': reward_ratio,
    'risk_per_trade': risk_per_trade
}

# Ana içerik
if st.button("🎯 Kombine Backtest Çalıştır", type="primary"):
    try:
        with st.spinner("Veri yükleniyor..."):
            extended_start = start_date - timedelta(days=150) 
            data = yf.download(ticker, start=extended_start, end=end_date, progress=False)
            
            if data.empty:
                st.error("❌ Veri bulunamadı")
                st.stop()
            
            data = data[data.index >= pd.to_datetime(start_date)]
            data = data[data.index <= pd.to_datetime(end_date)]
            
            st.success(f"✅ {len(data)} günlük veri yüklendi")
        
        backtester = SwingBacktest()
        
        with st.spinner("Backtest çalıştırılıyor..."):
            trades, equity = backtester.run_backtest(data, params)
            metrics = backtester.calculate_metrics(trades, equity)
        
        st.subheader("📊 Performans Özeti (Kombine Strateji)")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Toplam Getiri", metrics['total_return'])
            st.metric("Toplam İşlem", metrics['total_trades'])
        
        with col2:
            st.metric("Win Rate", metrics['win_rate'])
            st.metric("Ort. Kazanç", metrics['avg_win'])
        
        with col3:
            st.metric("Ort. Kayıp", metrics['avg_loss'])
            st.metric("En İyi İşlem", metrics['best_trade'])
        
        with col4:
            st.metric("En Kötü İşlem", metrics['worst_trade'])
        
        if not trades.empty and 'equity' in equity.columns: 
            st.subheader("📈 Performans Grafikleri")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(equity['date'], equity['equity'], color='purple', linewidth=2)
            ax.set_title(f'{ticker} Portföy Değeri')
            ax.set_ylabel('Equity ($)')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            st.subheader("📋 İşlem Listesi")
            display_trades = trades.copy()
            
            if not display_trades.empty:
                for col in ['entry_date', 'exit_date']:
                    if not display_trades[col].empty and isinstance(display_trades[col].iloc[0], (datetime, pd.Timestamp)):
                        display_trades[col] = display_trades[col].dt.strftime('%Y-%m-%d')
                    
                display_trades['pnl'] = display_trades['pnl'].round(2)
                display_trades['return_pct'] = display_trades['return_pct'].round(2)
            
            st.dataframe(display_trades)
            
        elif not trades.empty and not 'equity' in equity.columns:
             st.info("🤷 İşlemler gerçekleşti ancak grafik verisi (equity) bulunamadı.")

        else:
            st.info("🤷 Hiç işlem gerçekleşmedi. Daha agresif parametreler (daha düşük RSI, daha yüksek Risk %) deneyin.")
            
    except Exception as e:
        st.error(f"❌ Hata: {str(e)}")

st.markdown("---")
st.markdown("**Backtest Sistemi v4.0 - 5'li Kombinasyon Stratejisi**")
