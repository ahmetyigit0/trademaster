import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =========================
# ŞİFRE KORUMASI - DÜZELTİLDİ
# =========================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
        st.session_state["password"] = ""
    
    if not st.session_state["password_correct"]:
        st.markdown("### 🔐 Swing Stratejiye Giriş")
        password = st.text_input("Şifre:", type="password", key="password_input")
        
        if password == "efe":
            st.session_state["password_correct"] = True
            st.session_state["password"] = password
            st.success("✅ Giriş başarılı!")
            st.rerun()
        elif password:
            st.error("❌ Yanlış şifre!")
        
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
    
    def calculate_indicators(self, df):
        df = df.copy()
        
        # EMA'lar
        df['EMA_20'] = df['Close'].ewm(span=20, min_periods=1).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, min_periods=1).mean()
        
        # RSI - DÜZELTİLDİ
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-10)  # Sıfır bölme hatası önlendi
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # ATR - DÜZELTİLDİ
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift(1))
        low_close = np.abs(df['Low'] - df['Close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=14, min_periods=1).mean()
        
        # NaN doldurma - DÜZELTİLDİ
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    def generate_signals(self, df, rsi_oversold=40, atr_multiplier=2.0):
        signals = pd.DataFrame(index=df.index)
        signals['action'] = 'hold'
        signals['stop_loss'] = np.nan
        signals['take_profit'] = np.nan
        
        for date in df.index:
            row = df.loc[date]
            
            close_val = float(row['Close'])
            ema_20_val = float(row['EMA_20'])
            ema_50_val = float(row['EMA_50'])
            rsi_val = float(row['RSI'])
            atr_val = float(row['ATR'])
            
            trend_ok = ema_20_val > ema_50_val
            rsi_ok = rsi_val < rsi_oversold
            price_ok = close_val > ema_20_val
            
            if trend_ok and rsi_ok and price_ok:
                stop_loss = close_val - (atr_val * atr_multiplier)
                take_profit = close_val + (atr_val * atr_multiplier * 2)
                
                signals.loc[date, 'action'] = 'buy'
                signals.loc[date, 'stop_loss'] = stop_loss
                signals.loc[date, 'take_profit'] = take_profit
        
        buy_count = (signals['action'] == 'buy').sum()
        st.info(f"🎯 {buy_count} alım sinyali bulundu")
        return signals
    
    def run_backtest(self, data, rsi_oversold=40, atr_multiplier=2.0, risk_per_trade=0.02):
        df = self.calculate_indicators(data)
        signals = self.generate_signals(df, rsi_oversold, atr_multiplier)
        
        # DataFrame birleştirme - DÜZELTİLDİ
        df_combined = df.copy()
        df_combined['action'] = signals['action']
        df_combined['stop_loss'] = signals['stop_loss']
        df_combined['take
