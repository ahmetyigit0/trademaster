import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
            
            fib_382 = low_50 + (high_50 - low_50) * 0.382
            df['Fib_Support_382'] = fib_382
            
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
        df_copy = df.copy()
        
        # 1. Vektörel Koşullar
        df_copy['Trend_Up'] = df_copy['EMA_20'] > df_copy['EMA_50']
        df_copy['Momentum_Buy'] = df_copy['RSI'] < params['rsi_oversold']
        df_copy['Support_Touch'] = df_copy['Close'] < df_copy['BB_Lower'] 
        df_copy['Fib_Support_Hit'] = df_copy['Close'] <= df_copy['Fib_Support_382'] * 1.01
        
        df_copy['MACD_Cross_Up'] = (
            (df_copy['MACD'] > df_copy['Signal_Line']) & 
            (df_copy['MACD'].shift(1) <= df_copy['Signal_Line'].shift(1))
        )
        
        # 2. Nihai Alım Sinyali
        df_copy['Buy_Signal'] = (
            df_copy['Trend_Up'] & 
            df_copy['Momentum_Buy'] & 
            (df_copy['Support_Touch'] | df_copy['Fib_Support_Hit']) & 
            df_copy['MACD_Cross_Up']
        )
        
        # 3. SL/TP Hesaplamasını BÜTÜN DATA ÜZERİNDE YAP (Vektörel ve Güvenli)
        risk_pct = 0.02
        df_copy['stop_loss'] = df_copy['Close'] * (1 - risk_pct)
        df_copy['take_profit'] = df_copy['Close'] * (1 + (risk_pct * params['reward_ratio']))

        # 4. Sinyal DataFrame'ini Oluşturma
        signals = df_copy[['stop_loss', 'take_profit']].copy()
        
        signals['action'] = np.where(df_copy['Buy_Signal'], 'buy', 'hold')
        
        # 5. Sadece sinyal olmayan günlerin SL/TP değerlerini temizle (NaN yap)
        signals.loc[signals['action'] == 'hold', ['stop_loss', 'take_profit']] = np.nan

        buy_count = signals['action'].value_counts().get('buy', 0)
        st.info(f"🎯 {buy_count} karmaşık alış sinyali bulundu")
        return signals
    
    def run_backtest(self, data, params):
        df = self.calculate_indicators(data)
        signals = self.generate_signals(df, params)
        
        # İndeksleri Hizalama (Tüm tarih aralığını kapsar)
        df_combined = df.merge(signals[['action', 'stop_loss', 'take_profit']], 
                               left_index=True, right_index=True, how='left')
        
        # NaN'ları güvenli float değerlerle doldur
        df_combined['action'] = df_combined['action'].fillna('hold')
        df_combined[['stop_loss', 'take_profit']] = df_combined[['stop_loss', 'take_profit']].fillna(0.0)

        capital = float(self.initial_capital)
        position = None
        trades = []
        equity_curve = []
        
        for date in df_combined.index:
            row = df_combined.loc[date]
            # !!! KRİTİK: Tüm kullanılan değerler float'a çevrildi !!!
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
                        position = {
                            'entry_date': date,
                            'entry_price': current_price,
                            'shares': shares,
                            'stop_loss': stop_loss,
                            'take_profit': float(row['take_profit'])
                        }
                        capital -= shares * current_price
            
            # ÇIKIŞ KOŞULLARI
            elif position is not None:
                exited = False
                exit_price = None
                exit_reason = None

                if current_price <= position['stop_loss']:
                    exit_price = position['stop_loss']
                    exit_reason = 'SL'
                    exited = True
                
                elif current_price >= position['take_profit']:
                    exit_price = position['take_profit']
                    exit_reason = 'TP'
                    exited = True

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
        
        # Kapanış pozisyonu (Son gün)
        if position is not None:
            last_price = float(df_combined['Close'].iloc[-1])
            exit
