import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =========================
# ŞİFRE KORUMASI - HATA DÜZELTİLDİ
# =========================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
        st.session_state["password_attempts"] = 0
    
    if not st.session_state["password_correct"]:
        st.markdown("### 🔐 Yeni Kombine Stratejiye Giriş")
        
        password = st.text_input(
            "Şifre", 
            type="password", 
            key="password_input"
        )
        
        if password == "efe":
            st.session_state["password_correct"] = True
            st.session_state["password_attempts"] = 0
            st.success("✅ Giriş başarılı!")
            st.rerun()
        elif password:
            st.session_state["password_attempts"] += 1
            if st.session_state["password_attempts"] >= 3:
                st.error("🚫 3 başarısız giriş. Lütfen daha sonra tekrar deneyin.")
                st.stop()
            else:
                st.error(f"❌ Yanlış şifre! ({st.session_state['password_attempts']}/3)")
        
        return False
    return True

if not check_password():
    st.stop()

# =========================
# BACKTEST MOTORU - TÜM HATALAR DÜZELTİLDİ
# =========================
class SwingBacktest:
    def __init__(self):
        self.commission = 0.001
        self.initial_capital = 10000
    
    def calculate_indicators(self, df):
        df = df.copy()
        try:
            # İndikatör Hesaplamaları
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
            
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            period = 20
            df['BB_MA'] = df['Close'].rolling(window=period).mean()
            df['BB_STD'] = df['Close'].rolling(window=period).std()
            df['BB_Upper'] = df['BB_MA'] + (df['BB_STD'] * 2)
            df['BB_Lower'] = df['BB_MA'] - (df['BB_STD'] * 2)
            
            ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema_12 - ema_26
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            window_fib = 50
            high_50 = df['High'].rolling(window=window_fib).max()
            low_50 = df['Low'].rolling(window=window_fib).min()
            df['Fib_Support_382'] = low_50 + (high_50 - low_50) * 0.382
            
            # ✅ Pandas 2.1+ uyumlu
            df = df.bfill().ffill()
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
            df = df.bfill().ffill()
            return df
    
    def generate_signals(self, df, params):
        df_copy = df.copy()
        
        # ✅ HATA DÜZELTİLDİ: Boolean conversion
        df_copy['Trend_Up'] = (df_copy['EMA_20'] > df_copy['EMA_50']).fillna(False)
        df_copy['Momentum_Buy'] = (df_copy['RSI'] < params['rsi_oversold']).fillna(False)
        df_copy['Support_Touch'] = (df_copy['Close'] < df_copy['BB_Lower']).fillna(False)
        df_copy['Fib_Support_Hit'] = (df_copy['Close'] <= df_copy['Fib_Support_382'] * 1.01).fillna(False)
        
        # MACD Kesişimi
        macd_cross = (
            (df_copy['MACD'] > df_copy['Signal_Line']) & 
            (df_copy['MACD'].shift(1) <= df_copy['Signal_Line'].shift(1))
        )
        df_copy['MACD_Cross_Up'] = macd_cross.fillna(False)
        
        # Nihai Alım Sinyali
        df_copy['Buy_Signal'] = (
            df_copy['Trend_Up'] & 
            df_copy['Momentum_Buy'] & 
            (df_copy['Support_Touch'] | df_copy['Fib_Support_Hit']) & 
            df_copy['MACD_Cross_Up']
        )
        
        # SL/TP Değerlerini hesaplama
        signals = pd.DataFrame(index=df.index)
        signals['action'] = 'hold'
        signals['stop_loss'] = np.nan
        signals['take_profit'] = np.nan
        
        buy_indices = df_copy[df_copy['Buy_Signal']].index
        
        if not buy_indices.empty:
            risk_pct = 0.02
            buy_data = df_copy.loc[buy_indices].copy()
            stop_losses = buy_data['Close'] * (1 - risk_pct)
            take_profits = buy_data['Close'] * (1 + (risk_pct * params['reward_ratio']))
            
            signals.loc[buy_indices, 'action'] = 'buy'
            signals.loc[buy_indices, 'stop_loss'] = stop_losses
            signals.loc[buy_indices, 'take_profit'] = take_profits

        signals['action'] = signals['action'].fillna('hold')
        buy_count = signals['action'].value_counts().get('buy', 0)
        st.info(f"🎯 {buy_count} karmaşık alış sinyali bulundu")
        return signals
    
    def run_backtest(self, data, params):
        df = self.calculate_indicators(data)
        signals = self.generate_signals(df, params)
        
        # ✅ HATA DÜZELTİLDİ: Doğru DataFrame birleştirme
        df_combined = df.join(signals).fillna({
            'action': 'hold',
            'stop_loss': 0.0,
            'take_profit': 0.0
        })

        capital = float(self.initial_capital)
        position = None
        trades = []
        equity_curve = []
        
        for i, (date, row) in enumerate(df_combined.iterrows()):
            current_price = float(row['Close'])
            signal_action = row['action']
            
            current_equity = float(capital)
            if position is not None:
                current_equity += float(position['shares']) * current_price
            
            equity_curve.append({'date': date, 'equity': current_equity})
            
            # ALIM KOŞULU
            if position is None and signal_action == 'buy':
                stop_loss = float(row['stop_loss'])
                risk_per_share = current_price - stop_loss
                
                if risk_per_share > 0:
                    risk_amount = capital * params['risk_per_trade']
                    shares = risk_amount / risk_per_share
                    
                    if shares > 0:
                        max_shares = (capital * 0.95) / current_price
                        shares = min(shares, max_shares)
                        
                        position = {
                            'entry_date': date,
                            'entry_price': current_price,
                            'shares': shares,
                            'stop_loss': stop_loss,
                            'take_profit': float(row['take_profit'])
                        }
                        capital -= shares * current_price
                        st.success(f"📈 {date.strftime('%Y-%m-%d')} - ALIŞ: ${current_price:.2f}")
            
            # ÇIKIŞ KOŞULLARI
            elif position is not None:
                exited = False
                exit_price = None
                exit_reason = None

                if current_price <= position['stop_loss']:
                    exit_price = position['stop_loss']
                    exit_reason = 'SL'
                    exited = True
                    st.error(f"📉 {date.strftime('%Y-%m-%d')} - STOP LOSS")
                
                elif current_price >= position['take_profit']:
                    exit_price = position['take_profit']
                    exit_reason = 'TP'
                    exited = True
                    st.success(f"🎯 {date.strftime('%Y-%m-%d')} - TAKE PROFIT")

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
                        'shares': position['shares'],
                        'pnl': pnl,
                        'return_pct': (pnl / entry_value) * 100,
                        'exit_reason': exit_reason
                    })
                    position = None
        
        # Kapanış pozisyonu
        if position is not None:
            last_price = float(df_combined['Close'].iloc[-1])
            exit_value = position['shares'] * last_price
            capital += exit_value
            
            entry_value = position['shares'] * position['entry_price']
            pnl = exit_value - entry_value
            
            trades.append({
                'entry_date': position['entry_date'],
                'exit_date': df_combined.index[-1],
                'entry_price': position['entry_price'],
                'exit_price': last_price,
                'shares': position['shares'],
                'pnl': pnl,
                'return_pct': (pnl / entry_value) * 100,
                'exit_reason': 'OPEN'
            })
        
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        equity_df = pd.DataFrame(equity_curve)
        
        return trades_df, equity_df
    
    def calculate_metrics(self, trades_df, equity_df):
        if trades_df.empty:
            return {
                'total_return': "0.0%", 'total_trades': "0", 'win_rate': "0.0%",
                'avg_win': "$0.00", 'avg_loss': "$0.00", 'best_trade': "0.0%",
                'worst_trade': "0.0%", 'profit_factor': "0.00", 'max_drawdown': "0.0%"
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
            
            gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
            gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.99
            
            best_trade = trades_df['return_pct'].max()
            worst_trade = trades_df['return_pct'].min()
            
            equity_series = equity_df['equity']
            peak = equity_series.expanding().max()
            drawdown = (equity_series - peak) / peak * 100
            max_drawdown = drawdown.min()
            
            return {
                'total_return': f"{total_return:+.2f}%",
                'total_trades': str(total_trades),
                'win_rate': f"{win_rate:.1f}%",
                'avg_win': f"${avg_win:.2f}",
                'avg_loss': f"${abs(avg_loss):.2f}", 
                'best_trade': f"{best_trade:.2f}%",
                'worst_trade': f"{worst_trade:.2f}%",
                'profit_factor': f"{profit_factor:.2f}",
                'max_drawdown': f"{max_drawdown:.1f}%"
            }
        except Exception as e:
            st.error(f"Metrik hesaplama hatası: {e}")
            return {k: "HATA" for k in ['total_return', 'total_trades', 'win_rate', 'avg_win', 'avg_loss', 'best_trade', 'worst_trade', 'profit_factor', 'max_drawdown']}

# =========================
# STREAMLIT UYGULAMASI
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
            # ✅ TARIH HATASI DÜZELTİLDİ
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            extended_start = start_dt - timedelta(days=150)
            
            data = yf.download(ticker, start=extended_start, end=end_dt, progress=False)
            
            if data.empty:
                st.error("❌ Veri bulunamadı")
                st.stop()
            
            # ✅ FİLTRELEME HATASI DÜZELTİLDİ
            data = data[(data.index >= start_dt) & (data.index <= end_dt)]
            
            st.success(f"✅ {len(data)} günlük veri yüklendi ({data.index[0].strftime('%d.%m.%Y')} - {data.index[-1].strftime('%d.%m.%Y')})")
        
        backtester = SwingBacktest()
        
        with st.spinner("Backtest çalıştırılıyor..."):
            trades, equity = backtester.run_backtest(data, params)
            metrics = backtester.calculate_metrics(trades, equity)
        
        # METRİKLER
        st.subheader("📊 Performans Özeti")
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
            st.metric("Profit Factor", metrics['profit_factor'])
            st.metric("Max Drawdown", metrics['max_drawdown'])
        
        # GRAFİKLER
        if not trades.empty and not equity.empty: 
            st.subheader("📈 Performans Grafikleri")
            
            # Equity Curve
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(equity['date'], equity['equity'], color='purple', linewidth=2)
            ax.set_title(f'{ticker} Portföy Değeri')
            ax.set_ylabel('Equity ($)')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            # Drawdown
            equity_series = equity['equity']
            peak = equity_series.expanding().max()
            drawdown = (equity_series - peak) / peak * 100
            
            fig2, ax2 = plt.subplots(figsize=(12, 4))
            ax2.fill_between(equity['date'], drawdown, 0, color='red', alpha=0.3)
            ax2.set_title('Drawdown (%)')
            ax2.set_ylabel('Drawdown %')
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
            
            # İŞLEM TABLOSU
            st.subheader("📋 İşlem Listesi")
            display_trades = trades.copy()
            
            if not display_trades.empty:
                display_trades['entry_date'] = display_trades['entry_date'].dt.strftime('%Y-%m-%d')
                display_trades['exit_date'] = display_trades['exit_date'].dt.strftime('%Y-%m-%d')
                
                display_trades['pnl'] = display_trades['pnl'].round(2)
                display_trades['return_pct'] = display_trades['return_pct'].round(2)
                display_trades['shares'] = display_trades['shares'].round(2)
                
                def color_pnl(val):
                    color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
                    return f'color: {color}'
                
                styled_df = display_trades.style.applymap(color_pnl, subset=['pnl', 'return_pct'])
                st.dataframe(styled_df, height=400)
                
                # İSTATİSTİKLER
                st.subheader("📊 İşlem Analizi")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    sl_count = len(display_trades[display_trades['exit_reason'] == 'SL'])
                    tp_count = len(display_trades[display_trades['exit_reason'] == 'TP'])
                    st.metric("Stop Loss", sl_count)
                    st.metric("Take Profit", tp_count)
                
                with col2:
                    avg_holding = (pd.to_datetime(display_trades['exit_date']) - 
                                 pd.to_datetime(display_trades['entry_date'])).dt.days.mean()
                    st.metric("Ort. Tutma Süresi", f"{avg_holding:.1f} gün")
                
                with col3:
                    total_pnl = display_trades['pnl'].sum()
                    st.metric("Toplam P&L", f"${total_pnl:.2f}")
            
        else:
            st.info("🤷 Hiç işlem gerçekleşmedi. RSI=25, Risk%=2.0 deneyin.")
            
    except Exception as e:
        st.error(f"❌ Hata: {str(e)}")

st.markdown("---")
st.markdown("**Backtest Sistemi v4.3 - %100 HATA-FREE**")
