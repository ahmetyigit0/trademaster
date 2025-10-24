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

# ÖRNEK MUM VERİSİ OLUŞTUR - KESİN GÖRÜNSÜN DİYE
def create_sample_candlestick_data():
    """Örnek mum verisi oluştur"""
    dates = pd.date_range(start='2024-01-20', end='2024-01-23', freq='4h')
    
    # Basit fiyat hareketi
    base_price = 40000
    prices = []
    
    for i in range(len(dates)):
        # Rastgele fiyat hareketi
        change = np.random.uniform(-0.02, 0.02)
        base_price = base_price * (1 + change)
        
        # Mum verisi oluştur
        open_price = base_price
        close_price = base_price * (1 + np.random.uniform(-0.01, 0.01))
        high = max(open_price, close_price) * (1 + np.random.uniform(0, 0.015))
        low = min(open_price, close_price) * (1 - np.random.uniform(0, 0.015))
        
        prices.append({
            'Date': dates[i],
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close_price
        })
    
    df = pd.DataFrame(prices)
    df.set_index('Date', inplace=True)
    return df

# Basit mum grafiği oluştur
def create_basic_candlestick_chart():
    """Kesin görünen basit mum grafiği"""
    
    # Örnek veri oluştur
    data = create_sample_candlestick_data()
    
    fig = go.Figure()
    
    # MUM ÇUBUKLARI - ÇOK NET
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Mumlar',
        increasing_line_color='green',
        decreasing_line_color='red',
        increasing_fillcolor='green',
        decreasing_fillcolor='red',
        line=dict(width=2),
        whiskerwidth=0.8
    ))
    
    # Bazı destek/direnç çizgileri ekle
    current_price = data['Close'].iloc[-1]
    fig.add_hline(y=current_price * 0.98, line_dash="solid", line_color="blue", line_width=2, annotation_text="Destek")
    fig.add_hline(y=current_price * 1.02, line_dash="solid", line_color="orange", line_width=2, annotation_text="Direnç")
    
    fig.update_layout(
        title="DENEME MUM GRAFİĞİ - MUTLAKA GÖRÜNMELİ!",
        xaxis_title="Zaman",
        yaxis_title="Fiyat",
        height=500,
        showlegend=False
    )
    
    return fig, data

# Ana uygulama
def main():
    st.header("🧪 DENEME MUM GRAFİĞİ")
    st.warning("BU GRAFİKTE MUTLAKA YEŞİL VE KIRMIZI MUM ÇUBUKLARI GÖRÜNMELİ!")
    
    # Grafik oluştur
    chart_fig, sample_data = create_basic_candlestick_chart()
    
    # Grafiği göster
    st.plotly_chart(chart_fig, use_container_width=True)
    
    # Veriyi göster
    st.subheader("Örnek Mum Verisi")
    st.dataframe(sample_data.tail(10))
    
    # Kontrol
    st.success("✅ Eğer yukarıda yeşil ve kırmızı dikdörtgen mumlar görüyorsanız, grafik çalışıyor!")
    
    if len(sample_data) > 0:
        st.info(f"📊 Toplam {len(sample_data)} mum çubuğu oluşturuldu")
        st.info(f"💰 Son fiyat: ${sample_data['Close'].iloc[-1]:.2f}")

if __name__ == "__main__":
    main()