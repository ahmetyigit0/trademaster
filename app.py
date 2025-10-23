import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Streamlit arayüzü
st.set_page_config(page_title="Kripto Destek & Direnç Analizi", layout="wide")
st.title("🎯 Kripto Destek ve Direnç Seviyeleri Analizi")

# Sidebar
st.sidebar.header("⚙️ Ayarlar")
crypto_symbol = st.sidebar.text_input("Kripto Sembolü (Örn: BTC-USD, ETH-USD):", "BTC-USD")
lookback_period = st.sidebar.slider("Analiz Periyodu (gün)", 30, 90, 60)

def get_crypto_data(symbol, period_days=60):
    """Kripto verilerini çek"""
    try:
        data = yf.download(symbol, period=f"{period_days}d", interval="4h", progress=False)
        return data
    except Exception as e:
        st.error(f"Veri çekilemedi: {e}")
        return None

def find_support_resistance_levels(data, window=5):
    """Destek ve direnç seviyelerini bul"""
    if data is None or len(data) == 0:
        return [], []
    
    # Series'i numpy array'e çevir - HATA BURADA OLUYORDU
    highs = data['High'].values
    lows = data['Low'].values
    
    support_levels = []
    resistance_levels = []
    
    # Yerel minimum ve maksimumları bul
    for i in range(window, len(data)-window):
        current_low = lows[i]
        current_high = highs[i]
        
        # Destek seviyesi kontrolü (yerel minimum)
        is_support = True
        for j in range(1, window+1):
            if current_low > lows[i-j] or current_low > lows[i+j]:
                is_support = False
                break
        
        if is_support:
            support_levels.append(current_low)
        
        # Direnç seviyesi kontrolü (yerel maksimum)
        is_resistance = True
        for j in range(1, window+1):
            if current_high < highs[i-j] or current_high < highs[i+j]:
                is_resistance = False
                break
        
        if is_resistance:
            resistance_levels.append(current_high)
    
    return support_levels, resistance_levels

def filter_and_group_levels(levels, current_price, tolerance_percent=2.0):
    """Seviyeleri filtrele ve grupla"""
    if not levels:
        return []
    
    # Mevcut fiyata yakın seviyeleri al (%20 band)
    relevant_levels = [level for level in levels 
                      if abs(level - current_price) / current_price * 100 <= 20]
    
    if not relevant_levels:
        return []
    
    # Seviyeleri sırala ve grupla
    relevant_levels.sort()
    grouped_levels = []
    current_group = [relevant_levels[0]]
    
    for level in relevant_levels[1:]:
        # HATA: Burada da Series karşılaştırması olabilir
        avg_current = float(np.mean(current_group))  # float'a çevir
        percent_diff = abs(level - avg_current) / avg_current * 100
        
        if percent_diff <= tolerance_percent:
            current_group.append(level)
        else:
            grouped_levels.append(float(np.mean(current_group)))  # float'a çevir
            current_group = [level]
    
    if current_group:
        grouped_levels.append(float(np.mean(current_group)))  # float'a çevir
    
    return grouped_levels

def calculate_pivot_points(data):
    """Klasik pivot point hesaplama"""
    if len(data) < 1:
        return {}
    
    last_candle = data.iloc[-1]
    # HATA: Burada Series olabilir, float'a çevir
    high_val = float(last_candle['High'])
    low_val = float(last_candle['Low'])
    close_val = float(last_candle['Close'])
    
    P = (high_val + low_val + close_val) / 3
    R1 = 2 * P - low_val
    S1 = 2 * P - high_val
    R2 = P + (high_val - low_val)
    S2 = P - (high_val - low_val)
    
    return {
        'Pivot': float(P),
        'Resistance 1': float(R1),
        'Resistance 2': float(R2),
        'Support 1': float(S1),
        'Support 2': float(S2)
    }

def main():
    try:
        # Veri çekme
        st.write(f"**{crypto_symbol}** için 4 saatlik veriler çekiliyor...")
        
        data = get_crypto_data(crypto_symbol, lookback_period)
        
        if data is None or data.empty:
            st.error("Veri çekilemedi. Lütfen sembolü kontrol edin.")
            return
        
        st.success(f"✅ {len(data)} adet 4 saatlik mum verisi çekildi")
        
        # Mevcut fiyat - float'a çevir
        current_price = float(data['Close'].iloc[-1])
        
        # Seviyeleri hesapla
        support_levels, resistance_levels = find_support_resistance_levels(data)
        pivot_points = calculate_pivot_points(data)
        
        # Seviyeleri filtrele
        key_support = filter_and_group_levels(support_levels, current_price)
        key_resistance = filter_and_group_levels(resistance_levels, current_price)
        
        # Sonuçları göster
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Destek Seviyeleri")
            if key_support:
                for i, level in enumerate(reversed(key_support)):
                    distance_pct = ((current_price - level) / current_price) * 100
                    if level < current_price:
                        st.success(f"**Destek {i+1}:** ${level:.2f} (%{distance_pct:.2f} altında)")
                    else:
                        st.warning(f"**Destek {i+1}:** ${level:.2f} (%{abs(distance_pct):.2f} üstünde)")
            else:
                st.info("Destek seviyesi bulunamadı")
            
            st.subheader("🔧 Klasik Pivot Destekleri")
            st.write(f"**S1:** ${pivot_points['Support 1']:.2f}")
            st.write(f"**S2:** ${pivot_points['Support 2']:.2f}")
        
        with col2:
            st.subheader("📉 Direnç Seviyeleri")
            if key_resistance:
                for i, level in enumerate(key_resistance):
                    distance_pct = ((level - current_price) / current_price) * 100
                    if level > current_price:
                        st.error(f"**Direnç {i+1}:** ${level:.2f} (%{distance_pct:.2f} üstünde)")
                    else:
                        st.info(f"**Direnç {i+1}:** ${level:.2f} (%{abs(distance_pct):.2f} altında)")
            else:
                st.info("Direnç seviyesi bulunamadı")
            
            st.subheader("🔧 Klasik Pivot Dirençleri")
            st.write(f"**R1:** ${pivot_points['Resistance 1']:.2f}")
            st.write(f"**R2:** ${pivot_points['Resistance 2']:.2f}")
        
        # Özet bilgiler
        st.subheader("📊 Özet Bilgiler")
        col3, col4, col5, col6 = st.columns(4)
        
        with col3:
            st.metric("Mevcut Fiyat", f"${current_price:.2f}")
        with col4:
            st.metric("Destek Seviyeleri", len(key_support))
        with col5:
            st.metric("Direnç Seviyeleri", len(key_resistance))
        with col6:
            price_change = ((current_price - float(data['Close'].iloc[-2])) / float(data['Close'].iloc[-2])) * 100
            st.metric("Son Değişim", f"%{price_change:.2f}")
        
        # Grafik
        st.subheader("📊 Fiyat Grafiği ve Seviyeler")
        
        fig = go.Figure()
        
        # Mum grafiği
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Fiyat'
        ))
        
        # Destek seviyeleri
        for level in key_support:
            fig.add_hline(y=level, line_dash="dash", line_color="green", 
                         line_width=2, opacity=0.7,
                         annotation_text=f"D: ${level:.2f}")
        
        # Direnç seviyeleri
        for level in key_resistance:
            fig.add_hline(y=level, line_dash="dash", line_color="red", 
                         line_width=2, opacity=0.7,
                         annotation_text=f"R: ${level:.2f}")
        
        # Pivot seviyeleri
        fig.add_hline(y=pivot_points['Support 1'], line_dash="dot", line_color="lightgreen",
                     line_width=1, opacity=0.5, annotation_text=f"S1: ${pivot_points['Support 1']:.2f}")
        fig.add_hline(y=pivot_points['Support 2'], line_dash="dot", line_color="lightgreen",
                     line_width=1, opacity=0.5, annotation_text=f"S2: ${pivot_points['Support 2']:.2f}")
        fig.add_hline(y=pivot_points['Resistance 1'], line_dash="dot", line_color="lightcoral",
                     line_width=1, opacity=0.5, annotation_text=f"R1: ${pivot_points['Resistance 1']:.2f}")
        fig.add_hline(y=pivot_points['Resistance 2'], line_dash="dot", line_color="lightcoral",
                     line_width=1, opacity=0.5, annotation_text=f"R2: ${pivot_points['Resistance 2']:.2f}")
        
        # Mevcut fiyat çizgisi
        fig.add_hline(y=current_price, line_color="blue", line_width=3,
                     annotation_text=f"Mevcut: ${current_price:.2f}")
        
        fig.update_layout(
            title=f"{crypto_symbol} 4 Saatlik Grafik - Destek & Direnç Seviyeleri",
            xaxis_title="Tarih",
            yaxis_title="Fiyat (USD)",
            height=600,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Hata oluştu: {str(e)}")
        st.info("Lütfen sembolü kontrol edin ve internet bağlantınızı doğrulayın.")

if __name__ == "__main__":
    main()