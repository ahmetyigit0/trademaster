import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Streamlit sayfa ayarı
st.set_page_config(
    page_title="Yüksek Kazançlı Swing Stratejisi",
    page_icon="🚀",
    layout="wide"
)

# 🔐 Şifre Koruması
def check_password():
    """Şifre kontrolü"""
    def password_entered():
        """Şifre hash'ini kontrol et"""
        if st.session_state["password"]:
            password_hash = hashlib.sha256(st.session_state["password"].encode()).hexdigest()
            if password_hash == "8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918":  # admin
                st.session_state["password_correct"] = True
                del st.session_state["password"]
            else:
                st.session_state["password_correct"] = False
    
    if "password_correct" not in st.session_state:
        # Şifre girişi
        st.title("🔒 Yüksek Kazançlı Swing Stratejisi")
        st.text_input(
            "Şifre:", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.error("😔 Şifre yanlış" if "password_correct" in st.session_state and not st.session_state["password_correct"] else "")
        return False
    elif not st.session_state["password_correct"]:
        # Şifre yanlış
        st.text_input(
            "Şifre:", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.error("😔 Şifre yanlış")
        return False
    else:
        # Şifre doğru
        return True

# Şifre kontrolü
if not check_password():
    st.stop()

# 🔓 Şifre doğruysa uygulama devam eder
st.title("🚀 Triple Confirmation - Yüksek Kazanç Oranlı Strateji")
st.markdown("**3'lü Onay Sistemi ile %70+ Win Rate Hedefi**")

# Sidebar - parametreler
st.sidebar.header("🎯 Strateji Parametreleri")

# Hisse seçimi
ticker = st.sidebar.text_input("Hisse Kodu", "AAPL")
period = st.sidebar.selectbox("Zaman Periyodu", ["6mo", "1y", "2y", "3y"])

# Strateji parametreleri
st.sidebar.subheader("Strateji Ayarları")
ema_short = st.sidebar.slider("Kısa EMA", 5, 20, 9)
ema_long = st.sidebar.slider("Uzun EMA", 20, 100, 21)
rsi_period = st.sidebar.slider("RSI Period", 5, 21, 14)
volume_ma = st.sidebar.slider("Volume MA", 5, 30, 20)

# Risk yönetimi
st.sidebar.subheader("Risk Yönetimi")
stop_loss_pct = st.sidebar.slider("Stop Loss (%)", 1.0, 10.0, 3.0)
take_profit_pct = st.sidebar.slider("Take Profit (%)", 2.0, 20.0, 6.0)

def calculate_advanced_indicators(df, ema_short=9, ema_long=21, rsi_period=14, volume_ma=20):
    """Gelişmiş teknik göstergeleri hesapla"""
    
    try:
        # DataFrame'in kopyasını al
        df = df.copy()
        
        # 1. EMA Trend Filtresi
        df['EMA_short'] = df['Close'].ewm(span=ema_short).mean()
        df['EMA_long'] = df['Close'].ewm(span=ema_long).mean()
        df['EMA_trend'] = df['EMA_short'] > df['EMA_long']
        
        # 2. RSI + Momentum
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(span=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=rsi_period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # RSI Momentum
        df['RSI_5'] = df['RSI'].rolling(5).mean()
        df['RSI_trend'] = df['RSI'] > df['RSI_5']
        
        # 3. Volume Confirmation - HATA DÜZELTMESİ
        df['Volume_MA'] = df['Volume'].rolling(volume_ma).mean()
        
        # NaN değerleri False yap
        df['Volume_spike'] = False
        valid_volume_mask = ~df['Volume_MA'].isna()
        df.loc[valid_volume_mask, 'Volume_spike'] = (
            df.loc[valid_volume_mask, 'Volume'] > 
            (df.loc[valid_volume_mask, 'Volume_MA'] * 1.2)
        )
        
        # 4. Price Action
        df['High_5'] = df['High'].rolling(5).max()
        df['Low_5'] = df['Low'].rolling(5).min()
        df['Resistance'] = df['High_5'].shift(1)
        df['Support'] = df['Low_5'].shift(1)
        
        # 5. Volatility (ATR)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift(1))
        low_close = np.abs(df['Low'] - df['Close'].shift(1))
        
        df['TR'] = np.maximum(np.maximum(high_low, high_close), low_close)
        df['ATR'] = df['TR'].rolling(14).mean()
        
        # NaN değerleri temizle
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
        
    except Exception as e:
        st.error(f"Gösterge hesaplama hatası: {str(e)}")
        return df

def generate_signals(df, stop_loss_pct=3.0, take_profit_pct=6.0):
    """3'lü onay sistemi ile sinyal üret"""
    
    try:
        # NaN değerleri False yap
        df['Buy_Signal'] = False
        df['Sell_Signal'] = False
        
        # Geçerli veri olan satırlar için sinyal hesapla
        valid_mask = (
            ~df['EMA_trend'].isna() & 
            ~df['RSI'].isna() & 
            ~df['RSI_trend'].isna() &
            ~df['Volume_spike'].isna() &
            ~df['Support'].isna()
        )
        
        # ALIŞ sinyalleri (TÜM koşullar sağlanmalı)
        buy_conditions = (
            (df['EMA_trend'] == True) &                    # 1. Trend yukarı
            (df['RSI'] > 40) & (df['RSI'] < 70) &         # 2. RSI ideal bölge
            (df['RSI_trend'] == True) &                   # 3. RSI yükselişte
            (df['Volume_spike'] == True) &                # 4. Volume onayı
            (df['Close'] > df['Support']) &               # 5. Support üzerinde
            (df['Close'].shift(1) < df['Close'])          # 6. Son gün yükseliş
        )
        
        df.loc[valid_mask & buy_conditions, 'Buy_Signal'] = True
        
        # SATIŞ sinyalleri
        sell_conditions = (
            (df['EMA_trend'] == False) |                  # Trend aşağı VEYA
            (df['RSI'] > 80) |                           # RSI aşırı alım VEYA
            (df['Close'] < df['Support']) |               # Support kırılımı VEYA
            (df['Volume_spike'] == False)                 # Volume zayıf
        )
        
        df.loc[valid_mask & sell_conditions, 'Sell_Signal'] = True
        
        return df
        
    except Exception as e:
        st.error(f"Sinyal üretme hatası: {str(e)}")
        return df

def advanced_backtest(df, initial_capital=10000, stop_loss_pct=3.0, take_profit_pct=6.0):
    """Gelişmiş backtest sistemi"""
    
    try:
        capital = initial_capital
        position = 0
        trades = []
        entry_price = 0
        entry_date = None
        stop_loss = 0
        take_profit = 0
        trade_active = False
        
        for i in range(len(df)):
            if i < 20:  # İlk 20 gün teknik göstergeler henüz oluşmamış olabilir
                continue
                
            current_data = df.iloc[i]
            current_date = df.index[i]
            
            # ALIŞ Sinyali
            if current_data['Buy_Signal'] and not trade_active:
                # Pozisyon aç
                shares = capital / current_data['Close']
                position = shares
                entry_price = current_data['Close']
                entry_date = current_date
                stop_loss = entry_price * (1 - stop_loss_pct/100)
                take_profit = entry_price * (1 + take_profit_pct/100)
                trade_active = True
                
                trades.append({
                    'Entry_Date': entry_date,
                    'Action': 'BUY',
                    'Entry_Price': entry_price,
                    'Stop_Loss': stop_loss,
                    'Take_Profit': take_profit,
                    'RSI': current_data['RSI'],
                    'Volume_Ratio': current_data['Volume'] / current_data['Volume_MA'] if current_data['Volume_MA'] > 0 else 0
                })
            
            # SATIŞ Kontrolleri (aktif trade varsa)
            elif trade_active:
                current_price = current_data['Close']
                current_low = current_data['Low']
                
                # Stop Loss tetiklendi mi?
                stop_loss_hit = current_low <= stop_loss
                
                # Take Profit tetiklendi mi?
                take_profit_hit = current_price >= take_profit
                
                # Satış sinyali mi?
                sell_signal = current_data['Sell_Signal']
                
                # SATIŞ kararı (öncelik: stop loss > take profit > sinyal)
                if stop_loss_hit or take_profit_hit or sell_signal:
                    # Çıkış fiyatı belirle
                    if stop_loss_hit:
                        exit_price = stop_loss
                        exit_reason = 'STOP_LOSS'
                    elif take_profit_hit:
                        exit_price = take_profit
                        exit_reason = 'TAKE_PROFIT'
                    else:
                        exit_price = current_price
                        exit_reason = 'SELL_SIGNAL'
                    
                    # Pozisyon kapat
                    capital = position * exit_price
                    profit_pct = (exit_price - entry_price) / entry_price * 100
                    trade_duration = (current_date - entry_date).days
                    
                    trades.append({
                        'Exit_Date': current_date,
                        'Action': 'SELL',
                        'Exit_Price': exit_price,
                        'Exit_Reason': exit_reason,
                        'Profit_Pct': profit_pct,
                        'Trade_Duration': trade_duration,
                        'RSI_Exit': current_data['RSI']
                    })
                    
                    trade_active = False
                    position = 0
        
        return trades
        
    except Exception as e:
        st.error(f"Backtest hatası: {str(e)}")
        return []

# Veri çekme
@st.cache_data
def load_data(ticker, period):
    try:
        data = yf.download(ticker, period=period)
        if data.empty:
            st.error("❌ Veri çekilemedi. Hisse kodunu kontrol edin.")
            return None
        return data
    except Exception as e:
        st.error(f"❌ Veri yükleme hatası: {str(e)}")
        return None

# Uygulama
if st.sidebar.button("🔍 Stratejiyi Çalıştır"):
    with st.spinner('Veriler yükleniyor ve strateji analiz ediliyor...'):
        data = load_data(ticker, period)
        
        if data is not None and not data.empty:
            # Göstergeleri hesapla
            data = calculate_advanced_indicators(data, ema_short, ema_long, rsi_period, volume_ma)
            data = generate_signals(data, stop_loss_pct, take_profit_pct)
            
            # Backtest çalıştır
            trades = advanced_backtest(data, stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct)
            
            # Performans analizi
            st.subheader("📊 Strateji Performansı")
            
            if trades:
                # Trades'i işle
                buy_trades = [t for t in trades if t['Action'] == 'BUY']
                sell_trades = [t for t in trades if t['Action'] == 'SELL']
                
                if sell_trades:
                    profits = [t['Profit_Pct'] for t in sell_trades]
                    win_trades = [p for p in profits if p > 0]
                    loss_trades = [p for p in profits if p <= 0]
                    
                    win_rate = len(win_trades) / len(sell_trades) * 100
                    avg_profit = np.mean(profits)
                    avg_win = np.mean(win_trades) if win_trades else 0
                    avg_loss = np.mean(loss_trades) if loss_trades else 0
                    total_profit = sum(profits)
                    
                    # Profit factor hesapla
                    if loss_trades:
                        profit_factor = abs(sum(win_trades) / abs(sum(loss_trades)))
                    else:
                        profit_factor = float('inf')
                    
                    # Metrikler
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("🎯 Kazanç Oranı", f"{win_rate:.1f}%")
                    
                    with col2:
                        st.metric("💰 Ortalama Kar", f"{avg_profit:.2f}%")
                    
                    with col3:
                        st.metric("📈 Ort. Kazanan", f"{avg_win:.2f}%")
                    
                    with col4:
                        st.metric("📉 Ort. Kaybeden", f"{avg_loss:.2f}%")
                    
                    col5, col6, col7 = st.columns(3)
                    
                    with col5:
                        st.metric("🔄 Toplam İşlem", len(sell_trades))
                    
                    with col6:
                        st.metric("✅ Başarılı", len(win_trades))
                    
                    with col7:
                        st.metric("❌ Başarısız", len(loss_trades))
                    
                    # Exit nedenleri
                    exit_reasons = {}
                    for trade in sell_trades:
                        reason = trade['Exit_Reason']
                        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
                    
                    st.subheader("📈 Çıkış Nedenleri")
                    for reason, count in exit_reasons.items():
                        st.write(f"- **{reason}**: {count} işlem")
                    
                    # İşlem geçmişi
                    st.subheader("📋 Detaylı İşlem Geçmişi")
                    trades_df = pd.DataFrame(sell_trades)
                    
                    # Tarih formatını düzelt
                    if not trades_df.empty:
                        trades_df['Entry_Date'] = pd.to_datetime(trades_df['Entry_Date']).dt.strftime('%Y-%m-%d')
                        trades_df['Exit_Date'] = pd.to_datetime(trades_df['Exit_Date']).dt.strftime('%Y-%m-%d')
                    
                    # Renklendirme fonksiyonu
                    def color_profit(val):
                        if isinstance(val, (int, float)):
                            color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
                            return f'color: {color}; font-weight: bold'
                        return ''
                    
                    styled_df = trades_df.style.applymap(color_profit, subset=['Profit_Pct'])
                    st.dataframe(styled_df)
                    
                else:
                    st.warning("Henüz tamamlanmış işlem bulunmuyor.")
            
            else:
                st.info("Henüz işlem sinyali oluşmadı.")
            
            # Mevcut sinyal
            st.subheader("🎯 Mevcut Piyasa Durumu")
            current = data.iloc[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                trend_color = "🟢" if current['EMA_trend'] else "🔴"
                st.metric("Trend Yönü", f"{trend_color} {'YUKARI' if current['EMA_trend'] else 'AŞAĞI'}")
            
            with col2:
                rsi_color = "🔴" if current['RSI'] > 70 else "🟢" if current['RSI'] < 30 else "🟡"
                st.metric("RSI", f"{rsi_color} {current['RSI']:.1f}")
            
            with col3:
                volume_status = "YÜKSEK" if current['Volume_spike'] else "NORMAL"
                st.metric("Volume", volume_status)
            
            with col4:
                signal = "ALIŞ" if current['Buy_Signal'] else "SATIŞ" if current['Sell_Signal'] else "NÖTR"
                signal_color = "🟢" if current['Buy_Signal'] else "🔴" if current['Sell_Signal'] else "⚪"
                st.metric("Sinyal", f"{signal_color} {signal}")
            
            # Son 5 gün sinyalleri
            st.subheader("📅 Son 5 Gün Sinyalleri")
            recent_data = data.tail(5).copy()
            recent_data['Date'] = recent_data.index.strftime('%Y-%m-%d')
            display_columns = ['Date', 'Close', 'EMA_trend', 'RSI', 'Volume_spike', 'Buy_Signal']
            st.dataframe(recent_data[display_columns].style.format({
                'Close': '{:.2f}',
                'RSI': '{:.1f}'
            }))
            
        else:
            st.error("Veri yüklenemedi. Hisse kodunu kontrol edin.")

else:
    st.info("""
    ## 🚀 Yüksek Kazanç Oranlı Strateji
    
    **Özellikler:**
    - %70-85 kazanç oranı hedefi
    - 3'lü onay sistemi (Trend + Momentum + Volume)
    - Otomatik risk yönetimi
    - Swing trade (3-7 gün)
    
    **Önerilen Ayarlar:**
    - EMA: 9/21
    - RSI: 14 periyot  
    - Stop Loss: %3
    - Take Profit: %6
    
    👆 **Parametreleri ayarlayıp stratejiyi çalıştırın!**
    """)

# Çıkış butonu
if st.sidebar.button("🔒 Çıkış"):
    st.session_state["password_correct"] = False
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("🔐 **Varsayılan Şifre:** admin")
