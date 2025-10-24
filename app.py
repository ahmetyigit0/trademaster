import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots # Alt grafikler için eklendi

st.set_page_config(page_title="Kripto Teknik Analiz", layout="wide")
st.title("🎯 Kripto Teknik Analiz")

# --- GÖSTERGE AYARLARI ---
# Önceki yanıtta konuştuğumuz 4 saatlik zaman dilimi için yaygın/etkili ayarları kullanıyoruz.
# Bunları Streamlit sidebar'ına da taşıyabiliriz, ancak basitlik için şimdilik sabit tutalım.
EMA_SHORT = 20
EMA_LONG = 50
RSI_PERIOD = 14
BOLL_PERIOD = 20
BOLL_STDDEV = 2
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
# -------------------------

# Sidebar
crypto_symbol = st.sidebar.text_input("Kripto Sembolü:", "BTC-USD")
# Not: yfinance 4h için maksimum 730 gün, 1h için 60 gün destekler. Ayarları buna göre optimize edelim.
lookback_days = st.sidebar.slider("Gün Sayısı", 30, 365, 90) 
analysis_type = st.sidebar.selectbox("Analiz Türü", ["4 Saatlik", "1 Günlük", "1 Saatlik"])

interval_map = {"4 Saatlik": "4h", "1 Günlük": "1d", "1 Saatlik": "1h"}

def get_crypto_data(symbol, days, interval):
    """yfinance kullanarak kripto verilerini çeker."""
    # Yfinance'ın zaman aralığı kısıtlamalarını dikkate alarak period'u ayarla
    if interval == '1h' and days > 60:
        days = 60
        st.sidebar.caption("1 Saatlik veriler için maksimum 60 gün desteklenir.")
    elif interval == '4h' and days > 730:
        days = 730
        st.sidebar.caption("4 Saatlik veriler için maksimum 730 gün desteklenir.")

    try:
        data = yf.download(symbol, period=f"{days}d", interval=interval, progress=False)
        return data
    except Exception as e:
        st.error(f"Veri çekilemedi: {e}")
        return None

def calculate_indicators(data):
    """EMA, RSI, Bollinger Bantları ve MACD'yi hesaplar."""
    if data.empty:
        return data

    # 1. EMA Hesaplama (Exponential Moving Average)
    data['EMA_Short'] = data['Close'].ewm(span=EMA_SHORT, adjust=False).mean()
    data['EMA_Long'] = data['Close'].ewm(span=EMA_LONG, adjust=False).mean()

    # 2. RSI Hesaplama (Relative Strength Index)
    # Pandas kullanarak basit RSI hesaplaması
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # 3. Bollinger Bantları Hesaplama
    data['BB_Middle'] = data['Close'].rolling(window=BOLL_PERIOD).mean()
    std = data['Close'].rolling(window=BOLL_PERIOD).std()
    data['BB_Upper'] = data['BB_Middle'] + (std * BOLL_STDDEV)
    data['BB_Lower'] = data['BB_Middle'] - (std * BOLL_STDDEV)

    # 4. MACD Hesaplama (Moving Average Convergence Divergence)
    data['MACD_Fast'] = data['Close'].ewm(span=MACD_FAST, adjust=False).mean()
    data['MACD_Slow'] = data['Close'].ewm(span=MACD_SLOW, adjust=False).mean()
    data['MACD'] = data['MACD_Fast'] - data['MACD_Slow']
    data['MACD_Signal'] = data['MACD'].ewm(span=MACD_SIGNAL, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']

    return data.dropna() # Hesaplamadan sonra NaN değerleri düşür

def main():
    try:
        interval = interval_map[analysis_type]
        st.write(f"**{crypto_symbol}** için **{analysis_type}** verileri çekiliyor...")
        
        data = get_crypto_data(crypto_symbol, lookback_days, interval)
        
        if data is None or data.empty:
            st.error("Veri çekilemedi. Sembolü kontrol edin veya daha kısa bir gün aralığı seçin.")
            return
            
        data = calculate_indicators(data)
        
        if data.empty:
            st.warning("Göstergeleri hesaplamak için yeterli veri yok. Lütfen gün sayısını artırın.")
            return

        st.success(f"✅ {len(data)} adet mum verisi çekildi ve teknik göstergeler hesaplandı.")
        
        # --- GRAFİK OLUŞTURMA (Subplots ile) ---
        
        # 3 satır, 1 sütunlu alt grafikler oluştur (Fiyat, RSI, MACD)
        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25], # Fiyat grafiğine daha fazla alan
            subplot_titles=('Fiyat & Hareketli Ortalamalar & Bollinger Bantları', 'Göreceli Güç Endeksi (RSI)', 'MACD')
        )

        # 1. SATIR: Fiyat, EMA ve Bollinger
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Mum'
        ), row=1, col=1)

        # EMA'lar
        fig.add_trace(go.Scatter(x=data.index, y=data['EMA_Short'], line=dict(color='orange', width=1), name=f'EMA {EMA_SHORT}'), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['EMA_Long'], line=dict(color='purple', width=1.5), name=f'EMA {EMA_LONG}'), row=1, col=1)

        # Bollinger Bantları
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], line=dict(color='blue', width=0.5), name='BB Üst'), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], line=dict(color='blue', width=0.5), name='BB Alt', fill='tonexty', fillcolor='rgba(0,100,80,0.1)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Middle'], line=dict(color='gray', width=0.5, dash='dash'), name='BB Orta'), row=1, col=1)
        
        # 2. SATIR: RSI
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], line=dict(color='green', width=1.5), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, annotation_text="Aşırı Alım (70)", annotation_position="top left")
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, annotation_text="Aşırı Satım (30)", annotation_position="bottom left")
        fig.update_yaxes(range=[0, 100], row=2, col=1)

        # 3. SATIR: MACD
        # MACD Histogram
        colors = ['red' if val < 0 else 'green' for val in data['MACD_Hist']]
        fig.add_trace(go.Bar(x=data.index, y=data['MACD_Hist'], marker_color=colors, name='MACD Hist'), row=3, col=1)
        # MACD ve Sinyal Çizgileri
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], line=dict(color='blue', width=1.5), name='MACD'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], line=dict(color='red', width=1), name='Sinyal'), row=3, col=1)
        fig.add_hline(y=0, line_color="gray", line_width=1, row=3, col=1) # Sıfır çizgisi

        # Layout Ayarları
        fig.update_layout(
            title_text=f"{crypto_symbol} {analysis_type} Teknik Analiz", 
            xaxis_rangeslider_visible=False, # Alttaki kaydırıcıyı gizle
            height=900, 
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # --- Son Veriler Tablosu ---
        st.subheader("📜 Son Mum Verileri ve Göstergeler")
        
        # Gösterge sütunlarını da ekleyelim
        display_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'EMA_Short', 'EMA_Long', 'RSI', 'BB_Upper', 'BB_Lower', 'MACD', 'MACD_Signal']
        display_data = data.tail(10)[display_cols].copy()
        
        # Formatlama (Fiyat ve Hacim)
        # Sadece fiyat sütunlarını ($) ve hacim sütununu (virgül) formatla
        for col in ['Open', 'High', 'Low', 'Close', 'EMA_Short', 'EMA_Long', 'BB_Upper', 'BB_Lower']:
            if col in display_data.columns:
                display_data[col] = display_data[col].map('${:.2f}'.format)
        
        if 'RSI' in display_data.columns:
            display_data['RSI'] = display_data['RSI'].map('{:.2f}'.format) + '%'

        if 'MACD' in display_data.columns:
             for col in ['MACD', 'MACD_Signal']:
                display_data[col] = display_data[col].map('{:.4f}'.format)
        
        if 'Volume' in display_data.columns:
            display_data['Volume'] = display_data['Volume'].map('{:,.0f}'.format)
            
        st.dataframe(display_data)
        
    except Exception as e:
        st.error(f"❌ Hata oluştu: {str(e)}")

if __name__ == "__main__":
    main()
