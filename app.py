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
    
    if not st.session_state["password_correct"]:
        st.markdown("""
        # 🔐 **Kombine Stratejiye Hoş Geldiniz**
        ### **5 Göstergeli Profesyonel Swing Sistemi**
        """)
        
        password = st.text_input(
            "Şifre:", 
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
                st.error("🚫 3 başarısız giriş. Sayfayı yenileyin.")
                st.stop()
            else:
                st.error(f"❌ Yanlış şifre! ({st.session_state['password_attempts']}/3)")
        
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
