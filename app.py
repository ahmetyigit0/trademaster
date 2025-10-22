import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Streamlit sayfa ayarı
st.set_page_config(
    page_title="Dalga Avcısı - Swing Trade Stratejisi",
    page_icon="🌊",
    layout="wide"
)

# Başlık
st.title("🌊 Dalga Avcısı - Swing Trade Stratejisi")
st.markdown("**Bollinger Bands + RSI ile dalgalanmalardan kar fırsatları yakalayın**")

# Sidebar - parametreler
st.sidebar.header("📊 Strateji Parametreleri")

# Hisse seçimi
ticker = st.sidebar.text_input("Hisse Kodu (Örnek: GARAN.IS, THYAO.IS)", "GARAN.IS")
period = st.sidebar.selectbox("Zaman Periyodu", ["3mo", "6mo", "1y", "2y"])

# Bollinger Bands parametreleri
st.sidebar.subheader("Bollinger Bands Ayarları")
bb_period = st.sidebar.slider("BB Period", 10, 30, 20)
bb_std = st.sidebar.slider("BB Standart Sapma", 1, 3, 2)

# RSI parametreleri
st.sidebar.subheader("RSI Ayarları")
rsi_period = st.sidebar.slider("RSI Period", 5, 21, 14)
rsi_oversold = st.sidebar.slider("Aşırı Satım Seviyesi", 10, 40, 30)
rsi_overbought = st.sidebar.slider("Aşırı Alım Seviyesi", 60, 90, 70)

# Ana uygulama
def calculate_technical_indicators(df, bb_period=20, bb_std=2, rsi_period=14):
    """Teknik göstergeleri hesapla"""
    
    # Bollinger Bands
    df['MA20'] = df['Close'].rolling(window=bb_period).mean()
    df['BB_upper'] = df['MA20'] + (df['Close'].rolling(window=bb_period).std() * bb_std)
    df['BB_lower'] = df['MA20'] - (df['Close'].rolling(window=bb_period).std() * bb_std)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['MA20']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Strateji sinyalleri
    df['Buy_Signal'] = (df['Close'] < df['BB_lower']) & (df['RSI'] < rsi_oversold)
    df['Sell_Signal'] = (df['Close'] > df['BB_upper']) & (df['RSI'] > rsi_overbought)
    
    return df

def backtest_strategy(df, initial_capital=10000):
    """Strateji backtest"""
    
    capital = initial_capital
    position = 0
    trades = []
    buy_price = 0
    
    for i, row in df.iterrows():
        if row['Buy_Signal'] and position == 0:
            # ALIŞ
            position = capital / row['Close']
            buy_price = row['Close']
            trades.append({'Date': i, 'Action': 'BUY', 'Price': row['Close']})
            
        elif row['Sell_Signal'] and position > 0:
            # SATIŞ
            capital = position * row['Close']
            profit_pct = (row['Close'] - buy_price) / buy_price * 100
            trades.append({
                'Date': i, 
                'Action': 'SELL', 
                'Price': row['Close'],
                'Profit_Pct': profit_pct
            })
            position = 0
            buy_price = 0
    
    return trades

# Veri çekme
@st.cache_data
def load_data(ticker, period):
    try:
        data = yf.download(ticker, period=period)
        if data.empty:
            st.error("Veri çekilemedi. Lütfen hisse kodunu kontrol edin.")
            return None
        return data
    except:
        st.error("Hata oluştu. Lütfen hisse kodunu kontrol edin.")
        return None

# Uygulama akışı
data = load_data(ticker, period)

if data is not None:
    # Teknik göstergeleri hesapla
    data = calculate_technical_indicators(data, bb_period, bb_std, rsi_period)
    
    # Backtest
    trades = backtest_strategy(data)
    
    # Sonuçları göster
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Toplam İşlem Sayısı", len([t for t in trades if t['Action'] == 'BUY']))
    
    with col2:
        if len(trades) > 1:
            total_profit = sum([t.get('Profit_Pct', 0) for t in trades if t['Action'] == 'SELL'])
            st.metric("Toplam Kar (%)", f"{total_profit:.2f}%")
    
    with col3:
        current_price = data['Close'].iloc[-1]
        st.metric("Mevcut Fiyat", f"{current_price:.2f}")
    
    # Grafikler
    st.subheader("📈 Fiyat ve Bollinger Bands")
    
    fig = go.Figure()
    
    # Fiyat ve BB
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Kapanış', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], name='20 Günlük MA', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_upper'], name='BB Üst', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_lower'], name='BB Alt', line=dict(color='green', dash='dash')))
    
    # Alım/Satım sinyalleri
    buy_signals = data[data['Buy_Signal']]
    sell_signals = data[data['Sell_Signal']]
    
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], 
                            mode='markers', name='ALIŞ Sinyali',
                            marker=dict(color='green', size=10, symbol='triangle-up')))
    
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], 
                            mode='markers', name='SATIŞ Sinyali',
                            marker=dict(color='red', size=10, symbol='triangle-down')))
    
    fig.update_layout(height=500, title=f"{ticker} - Bollinger Bands ve Alım/Satım Sinyalleri")
    st.plotly_chart(fig, use_container_width=True)
    
    # RSI Grafiği
    st.subheader("📊 RSI Göstergesi")
    
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')))
    fig_rsi.add_hline(y=rsi_overbought, line_dash="dash", line_color="red", annotation_text="Aşırı Alım")
    fig_rsi.add_hline(y=rsi_oversold, line_dash="dash", line_color="green", annotation_text="Aşırı Satım")
    fig_rsi.update_layout(height=300, yaxis_range=[0, 100])
    st.plotly_chart(fig_rsi, use_container_width=True)
    
    # İşlem Geçmişi
    st.subheader("📋 İşlem Geçmişi")
    if trades:
        trades_df = pd.DataFrame(trades)
        st.dataframe(trades_df.tail(10))
    else:
        st.info("Henüz işlem sinyali oluşmadı.")
    
    # Strateji Açıklaması
    with st.expander("🎯 Strateji Nasıl Çalışıyor?"):
        st.markdown("""
        **ALIŞ Sinyali (Yeşil Üçgen):**
        - Fiyat Bollinger Alt Band'ın ALTINDA
        - RSI Aşırı Satım bölgesinde (seçtiğiniz değerin altında)
        
        **SATIŞ Sinyali (Kırmızı Üçgen):**
        - Fiyat Bollinger Üst Band'ın ÜSTÜNDE  
        - RSI Aşırı Alım bölgesinde (seçtiğiniz değerin üstünde)
        
        **Swing Trade Mantığı:**
        - Dalgalanmaları fırsata çevir
        - Aşırı hareketlerde tersine dönüş bekle
        - 3-10 gün arası pozisyon tut
        """)

else:
    st.info("Lütfen geçerli bir hisse kodu girin (Örnek: GARAN.IS, THYAO.IS, ASELS.IS)")
