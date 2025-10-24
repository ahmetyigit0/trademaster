import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="4Saatlik Profesyonel TA", layout="wide")

# Şifre koruması
def check_password():
    def password_entered():
        if st.session_state["password"] == "efe":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Şifre", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Şifre", type="password", on_change=password_entered, key="password")
        st.error("❌ Şifre yanlış!")
        return False
    else:
        return True

if not check_password():
    st.stop()

st.title("🎯 4 Saatlik Profesyonel Teknik Analiz Stratejisi")

# Sidebar
with st.sidebar:
    st.header("⚙️ Strateji Ayarları")
    
    crypto_symbol = st.text_input("Kripto Sembolü", "BTC-USD")
    
    st.caption("Hızlı Seçim:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("BTC-USD"):
            st.session_state.crypto_symbol = "BTC-USD"
        if st.button("ETH-USD"):
            st.session_state.crypto_symbol = "ETH-USD"
    with col2:
        if st.button("ADA-USD"):
            st.session_state.crypto_symbol = "ADA-USD"
        if st.button("XRP-USD"):
            st.session_state.crypto_symbol = "XRP-USD"
    
    if 'crypto_symbol' in st.session_state:
        crypto_symbol = st.session_state.crypto_symbol
    
    lookback_period = st.slider("Analiz Periyodu (Gün)", 30, 200, 100)
    ema_period = st.slider("EMA Period", 20, 100, 50)
    min_touch_points = st.slider("Minimum Temas Noktası", 2, 5, 3)
    risk_reward_ratio = st.slider("Min Risk/Ödül Oranı", 1.0, 3.0, 1.5)

# Fiyat formatlama
def format_price(price):
    if price >= 1000:
        return f"${price:,.2f}"
    elif price >= 1:
        return f"${price:.3f}"
    else:
        return f"${price:.6f}"

# Veri çekme
@st.cache_data
def get_4h_data(symbol, days=3):
    try:
        symbol = symbol.upper().strip()
        if '-' not in symbol:
            symbol = symbol + '-USD'
        
        data = yf.download(symbol, period=f"{days}d", interval="4h", progress=False)
        return data if not data.empty else None
    except:
        return None

# Teknik göstergeler
def calculate_indicators(data, ema_period=50):
    df = data.copy()
    df['EMA'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
    return df

# Destek/direnç analizi
def find_congestion_zones(data, lookback=80, min_touch_points=3):
    try:
        df = data.tail(lookback).copy()
        
        price_levels = []
        for i in range(len(df)):
            price_levels.extend([float(df['Close'].iloc[i]), float(df['High'].iloc[i]), float(df['Low'].iloc[i])])
        
        price_levels = sorted(price_levels)
        if not price_levels:
            return [], []
        
        price_range = max(price_levels) - min(price_levels)
        bin_size = price_range * 0.01
        
        bins = {}
        current_bin = min(price_levels)
        
        while current_bin <= max(price_levels):
            bin_end = current_bin + bin_size
            count = sum(1 for price in price_levels if current_bin <= price <= bin_end)
            if count > 0:
                bins[(current_bin, bin_end)] = count
            current_bin = bin_end
        
        congestion_zones = []
        for (zone_start, zone_end), count in bins.items():
            if count >= min_touch_points:
                zone_center = (zone_start + zone_end) / 2
                congestion_zones.append({'price': zone_center, 'strength': count})
        
        current_price = float(df['Close'].iloc[-1])
        support_zones = [zone for zone in congestion_zones if zone['price'] < current_price]
        resistance_zones = [zone for zone in congestion_zones if zone['price'] > current_price]
        
        support_zones = sorted(support_zones, key=lambda x: x['price'], reverse=True)[:3]
        resistance_zones = sorted(resistance_zones, key=lambda x: x['price'])[:3]
        
        return support_zones, resistance_zones
    except:
        return [], []

# MUM GRAFİĞİ oluştur - KESİN ÇALIŞAN
def create_candlestick_chart_with_levels(data, support_zones, resistance_zones, crypto_symbol):
    """GERÇEK MUM GRAFİĞİ - KESİN ÇALIŞIYOR"""
    
    fig = go.Figure()
    
    # 1. ÖNCE MUM ÇUBUKLARI
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price',
        increasing_line_color='green',
        decreasing_line_color='red',
        increasing_fillcolor='green',
        decreasing_fillcolor='red',
        line=dict(width=1.5),
        whiskerwidth=0.8
    ))
    
    # 2. EMA çizgisi
    if 'EMA' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['EMA'],
            name=f'EMA {ema_period}',
            line=dict(color='orange', width=2)
        ))
    
    # 3. DESTEK çizgileri
    for i, zone in enumerate(support_zones):
        fig.add_hline(
            y=zone['price'],
            line_dash="solid",
            line_color="lime",
            line_width=3,
            annotation_text=f"S{i+1}",
            annotation_position="left"
        )
    
    # 4. DİRENÇ çizgileri
    for i, zone in enumerate(resistance_zones):
        fig.add_hline(
            y=zone['price'],
            line_dash="solid",
            line_color="red",
            line_width=3,
            annotation_text=f"R{i+1}",
            annotation_position="right"
        )
    
    # 5. Mevcut fiyat
    current_price = float(data['Close'].iloc[-1])
    fig.add_hline(
        y=current_price,
        line_dash="dot",
        line_color="yellow",
        line_width=2,
        annotation_text=f"Şimdi: {format_price(current_price)}"
    )
    
    # Grafik ayarları
    fig.update_layout(
        title=f"{crypto_symbol} - 4 Saatlik Mum Grafiği",
        xaxis_title="Tarih",
        yaxis_title="Fiyat (USD)",
        height=600,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    return fig

# Ana uygulama
def main():
    # Veri yükleme
    with st.spinner(f'⏳ {crypto_symbol} verileri yükleniyor...'):
        data_3days = get_4h_data(crypto_symbol, days=3)
        data_full = get_4h_data(crypto_symbol, days=lookback_period)
    
    if data_3days is None or data_3days.empty:
        st.error(f"❌ {crypto_symbol} için veri yüklenemedi!")
        st.info("Örnek: BTC-USD, ETH-USD, ADA-USD, XRP-USD")
        return
    
    st.success(f"✅ {crypto_symbol} için {len(data_3days)} mum verisi yüklendi")
    
    # Göstergeleri hesapla
    data_full = calculate_indicators(data_full, ema_period)
    
    # Destek/direnç bul
    support_zones, resistance_zones = find_congestion_zones(data_full, min_touch_points=min_touch_points)
    
    # Mevcut durum
    current_price = float(data_full['Close'].iloc[-1])
    ema_value = float(data_full['EMA'].iloc[-1]) if 'EMA' in data_full.columns else current_price
    
    # Layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader(f"📈 {crypto_symbol} - 4 Saatlik Mum Grafiği")
        
        # MUM GRAFİĞİNİ GÖSTER
        chart_fig = create_candlestick_chart_with_levels(data_3days, support_zones, resistance_zones, crypto_symbol)
        st.plotly_chart(chart_fig, use_container_width=True)
        
        st.info("""
        **📊 Grafik Açıklaması:**
        - 🟢 **Yeşil Mumlar:** Yükseliş (Kapanış > Açılış)
        - 🔴 **Kırmızı Mumlar:** Düşüş (Kapanış < Açılış)  
        - 🟢 **S1,S2,S3:** Destek Seviyeleri
        - 🔴 **R1,R2,R3:** Direnç Seviyeleri
        - 🟡 **Sarı Çizgi:** Mevcut Fiyat
        - 🟠 **Turuncu Çizgi:** EMA
        """)
    
    with col2:
        st.subheader("📊 Mevcut Durum")
        st.metric("Fiyat", format_price(current_price))
        st.metric(f"EMA {ema_period}", format_price(ema_value))
        
        trend = "YÜKSELİŞ" if current_price > ema_value else "DÜŞÜŞ"
        st.metric("Trend", trend)
        
        st.subheader("💎 Seviyeler")
        st.write("**🟢 Destek:**")
        for i, zone in enumerate(support_zones):
            st.write(f"S{i+1}: {format_price(zone['price'])}")
        
        st.write("**🔴 Direnç:**")
        for i, zone in enumerate(resistance_zones):
            st.write(f"R{i+1}: {format_price(zone['price'])}")
        
        # Basit sinyal
        if support_zones and current_price <= support_zones[0]['price'] * 1.01:
            st.success("🟢 Destek Yakını - ALIM Potansiyeli")
        elif resistance_zones and current_price >= resistance_zones[0]['price'] * 0.99:
            st.error("🔴 Direnç Yakını - SATIM Potansiyeli")
        else:
            st.info("⚪ Bekle - Piyasa Gözlemi")

if __name__ == "__main__":
    main()