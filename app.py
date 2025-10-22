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
        st.session_state["password_attempts"] = 0
    
    def password_entered():
        if st.session_state["password"] == "efe":
            st.session_state["password_correct"] = True
            st.session_state["password_attempts"] = 0
            del st.session_state["password"]
            st.rerun()
        else:
            st.session_state["password_attempts"] += 1
            if st.session_state["password_attempts"] >= 3:
                st.error("🚫 3 başarısız giriş. 5 dakika bekleyin.")
                st.stop()
    
    if not st.session_state["password_correct"]:
        st.markdown("""
        # 🔐 **Kombine Stratejiye Hoş Geldiniz**
        ### **5 Göstergeli Profesyonel Swing Sistemi**
        """)
        st.text_input(
            "Şifre:", 
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
    
    def calculate_indicators(self, df):
        df = df.copy()
        
        # EMA
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands - HATA DÜZELTİLDİ!
        period = 20
        df['BB_MA'] = df['Close'].rolling(window=period).mean()
        df['BB_STD'] = df['Close'].rolling(window=period).std()
        df['BB_Upper'] = df['BB_MA'] + (df['BB_STD'] * 2)
        df['BB_Lower'] = df['BB_MA'] - (df['BB_STD'] * 2)
        
        # MACD
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Fibonacci
        window_fib = 50
        high_50 = df['High'].rolling(window=window_fib).max()
        low_50 = df['Low'].rolling(window=window_fib).min()
        df['Fib_382'] = low_50 + (high_50 - low_50) * 0.382
        
        # Volume
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # NaN Doldurma
        df = df.fillna(method='bfill').fillna(method='ffill')
        return df
    
    def generate_signals(self, df, params):
        df_copy = df.copy()
        
        # Koşullar
        trend_up = df_copy['EMA_20'] > df_copy['EMA_50']
        rsi_oversold = df_copy['RSI'] < params['rsi_oversold']
        bb_support = df_copy['Close'] <= df_copy['BB_Lower'] * 1.02
