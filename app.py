import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Profesyonel Kripto Analiz", layout="wide")

# Şifre koruması
def check_password():
    """Şifre kontrolü yapar"""
    def password_entered():
        """Şifre girildiğinde kontrol eder"""
        if st.session_state["password"] == "efe":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Şifreyi temizle
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # İlk giriş, şifre giriş alanını göster
        st.text_input(
            "Şifre", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.write("🔒 Bu uygulama şifre ile korunmaktadır")
        return False
    elif not st.session_state["password_correct"]:
        # Şifre yanlış, tekrar dene
        st.text_input(
            "Şifre", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.error("❌ Şifre yanlış! Lütfen tekrar deneyin.")
        return False
    else:
        # Şifre doğru
        return True

# Şifre kontrolü
if not check_password():
    st.stop()  # Şifre doğru değilse uygulamayı durdur

# Şifre doğruysa ana uygulamayı göster
st.title("🎯 Profesyonel Kripto Trading Analizi - Destek/Direnç Analizi")

# Sidebar
with st.sidebar:
    st.success("🔓 Giriş Başarılı!")
    crypto_symbol = st.text_input("Kripto Sembolü:", "BTC-USD")
    lookback_days = st.slider("Gün Sayısı", 30, 365, 90)
    analysis_type = st.selectbox("Analiz Türü", ["4 Saatlik", "1 Günlük", "1 Saatlik"])
    
    # Analiz parametreleri
    st.subheader("📊 Analiz Ayarları")
    sensitivity = st.slider("Destek/Direnç Hassasiyeti", 1, 10, 3)
    min_touch_points = st.slider("Minimum Temas Noktası", 2, 5, 2)
    wick_analysis = st.checkbox("İğne (Wick) Analizi", value=True)
    
    # Çıkış butonu
    if st.button("🔒 Çıkış Yap"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

interval_map = {"4 Saatlik": "4h", "1 Günlük": "1d", "1 Saatlik": "1h"}

def get_crypto_data(symbol, days, interval):
    try:
        data = yf.download(symbol, period=f"{days}d", interval=interval, progress=False)
        return data
    except Exception as e:
        st.error(f"Veri çekilemedi: {e}")
        return None

def find_support_resistance_levels(data, sensitivity=3, min_touch_points=2, use_wick_analysis=True):
    """
    Mum kapanışları ve iğnelere dayalı destek/direnç seviyelerini bulur
    """
    try:
        df = data.copy()
        
        # Fiyat verilerini topla
        if use_wick_analysis:
            # İğne analizi: High, Low, Close kullan
            price_levels = []
            for i in range(len(df)):
                # Önemli seviyeler: High, Low, Close
                price_levels.extend([
                    float(df['High'].iloc[i]),
                    float(df['Low'].iloc[i]), 
                    float(df['Close'].iloc[i])
                ])
        else:
            # Sadece kapanış fiyatları
            price_levels = [float(x) for x in df['Close']]
        
        # Fiyat seviyelerini hassasiyete göre grupla
        price_levels = sorted(price_levels)
        
        # Benzersiz seviyeleri bul ve yakın seviyeleri birleştir
        unique_levels = []
        tolerance = (max(price_levels) - min(price_levels)) * (sensitivity / 1000.0)
        
        i = 0
        while i < len(price_levels):
            current_level = price_levels[i]
            group = [current_level]
            
            # Yakın seviyeleri grupla
            j = i + 1
            while j < len(price_levels) and price_levels[j] - current_level <= tolerance:
                group.append(price_levels[j])
                j += 1
            
            # Grup ortalamasını al
            if len(group) >= min_touch_points:
                unique_levels.append(np.mean(group))
            
            i = j
        
        # Seviyeleri destek ve direnç olarak ayır
        current_price = float(df['Close'].iloc[-1])
        
        support_levels = [level for level in unique_levels if level < current_price]
        resistance_levels = [level for level in unique_levels if level > current_price]
        
        # En güçlü seviyeleri seç (en çok temas edenler)
        support_levels = sorted(support_levels, reverse=True)[:5]  # En yakın 5 destek
        resistance_levels = sorted(resistance_levels)[:5]  # En yakın 5 direnç
        
        return support_levels, resistance_levels
        
    except Exception as e:
        st.error(f"Destek/direnç analiz hatası: {e}")
        return [], []

def calculate_pivot_points(data):
    """Klasik pivot point seviyelerini hesapla"""
    try:
        df = data.copy()
        recent = df.tail(1).iloc[0]
        
        high = float(recent['High'])
        low = float(recent['Low'])
        close = float(recent['Close'])
        
        # Pivot Point
        pivot = (high + low + close) / 3
        
        # Direnç seviyeleri
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        
        # Destek seviyeleri
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        return pivot, [s1, s2, s3], [r1, r2, r3]
        
    except Exception as e:
        return None, [], []

def generate_trading_signals_with_levels(data, support_levels, resistance_levels):
    """Destek/direnç seviyelerine göre trading sinyalleri üret"""
    signals = []
    
    if len(data) < 10:
        return signals
    
    try:
        current_price = float(data['Close'].iloc[-1])
        prev_price = float(data['Close'].iloc[-2])
        
        # Yakın destek/direnç seviyelerini bul
        nearest_support = max([level for level in support_levels if level < current_price], default=None)
        nearest_resistance = min([level for level in resistance_levels if level > current_price], default=None)
        
        if nearest_support:
            distance_to_support = ((current_price - nearest_support) / current_price) * 100
            if distance_to_support <= 2:  # %2'den yakınsa
                signals.append(f"DESTEK YAKIN: ${nearest_support:.2f} (%{distance_to_support:.1f} uzak)")
                if prev_price > current_price:  # Düşüş trendinde
                    signals.append("DESTEK TESTI - POTANSIYEL ALIM")
        
        if nearest_resistance:
            distance_to_resistance = ((nearest_resistance - current_price) / current_price) * 100
            if distance_to_resistance <= 2:  # %2'den yakınsa
                signals.append(f"DIRENÇ YAKIN: ${nearest_resistance:.2f} (%{distance_to_resistance:.1f} uzak)")
                if prev_price < current_price:  # Yükseliş trendinde
                    signals.append("DIRENÇ TESTI - POTANSIYEL SATIM")
        
        # Kırılma sinyalleri
        if nearest_support and current_price < nearest_support and prev_price >= nearest_support:
            signals.append("DESTEK KIRILDI - SATIM SİNYALİ")
        
        if nearest_resistance and current_price > nearest_resistance and prev_price <= nearest_resistance:
            signals.append("DIRENÇ KIRILDI - ALIM SİNYALİ")
        
        return signals
        
    except Exception as e:
        return [f"Sinyal hatası: {str(e)}"]

def main():
    try:
        interval = interval_map[analysis_type]
        st.write(f"**{crypto_symbol}** için {analysis_type} veriler çekiliyor...")
        
        data = get_crypto_data(crypto_symbol, lookback_days, interval)
        
        if data is None or data.empty:
            st.error("Veri çekilemedi.")
            return
        
        st.success(f"✅ {len(data)} adet mum verisi çekildi")
        
        # Destek/direnç seviyelerini bul
        support_levels, resistance_levels = find_support_resistance_levels(
            data, sensitivity, min_touch_points, wick_analysis
        )
        
        # Pivot point hesapla
        pivot, pivot_supports, pivot_resistances = calculate_pivot_points(data)
        
        # Mevcut fiyat
        current_price = float(data['Close'].iloc[-1])
        
        # Trading sinyalleri üret
        signals = generate_trading_signals_with_levels(data, support_levels, resistance_levels)
        
        # Ana panel
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📈 Destek/Direnç Grafik Analizi")
            
            fig = go.Figure()
            
            # Çizgi grafiği (kapanış fiyatları)
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                name='Kapanış Fiyatı',
                line=dict(color='blue', width=2),
                mode='lines'
            ))
            
            # Mevcut fiyat çizgisi
            fig.add_hline(y=current_price, line_dash="solid", line_color="black", line_width=2, 
                         annotation_text=f"Mevcut Fiyat: ${current_price:.2f}")
            
            # Destek seviyeleri
            for i, level in enumerate(support_levels):
                fig.add_hline(y=level, line_dash="dash", line_color="green", line_width=2,
                             annotation_text=f"Destek {i+1}: ${level:.2f}")
            
            # Direnç seviyeleri
            for i, level in enumerate(resistance_levels):
                fig.add_hline(y=level, line_dash="dash", line_color="red", line_width=2,
                             annotation_text=f"Direnç {i+1}: ${level:.2f}")
            
            # Pivot point
            if pivot:
                fig.add_hline(y=pivot, line_dash="dot", line_color="orange", line_width=2,
                             annotation_text=f"Pivot: ${pivot:.2f}")
            
            fig.update_layout(
                height=600,
                title=f"{crypto_symbol} - Destek/Direnç Analizi",
                xaxis_title="Tarih",
                yaxis_title="Fiyat (USD)",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 TRADING SİNYALLERİ")
            
            if signals:
                for signal in signals:
                    if "ALIM" in signal or "KIRILDI" in signal and "DIRENÇ" in signal:
                        st.success(f"✅ {signal}")
                    elif "SATIM" in signal or "KIRILDI" in signal and "DESTEK" in signal:
                        st.error(f"❌ {signal}")
                    elif "TEST" in signal or "YAKIN" in signal:
                        st.warning(f"⚠️ {signal}")
                    else:
                        st.info(f"📊 {signal}")
            else:
                st.info("📊 Net trading sinyali yok")
            
            st.subheader("📊 FİYAT ANALİZİ")
            st.metric("Mevcut Fiyat", f"${current_price:.2f}")
            
            if support_levels:
                nearest_support = max(support_levels)
                distance_support = ((current_price - nearest_support) / current_price) * 100
                st.metric("En Yakın Destek", f"${nearest_support:.2f}", f"%{distance_support:.1f}")
            else:
                st.metric("En Yakın Destek", "Bulunamadı")
            
            if resistance_levels:
                nearest_resistance = min(resistance_levels)
                distance_resistance = ((nearest_resistance - current_price) / current_price) * 100
                st.metric("En Yakın Direnç", f"${nearest_resistance:.2f}", f"%{distance_resistance:.1f}")
            else:
                st.metric("En Yakın Direnç", "Bulunamadı")
            
            st.subheader("💎 DESTEK SEVİYELERİ")
            if support_levels:
                for i, level in enumerate(sorted(support_levels, reverse=True)):
                    distance = ((current_price - level) / current_price) * 100
                    st.write(f"🟢 D{i+1}: ${level:.2f} (%{distance:.1f} aşağıda)")
            else:
                st.write("Destek seviyesi bulunamadı")
            
            st.subheader("🚀 DİRENÇ SEVİYELERİ")
            if resistance_levels:
                for i, level in enumerate(sorted(resistance_levels)):
                    distance = ((level - current_price) / current_price) * 100
                    st.write(f"🔴 R{i+1}: ${level:.2f} (%{distance:.1f} yukarıda)")
            else:
                st.write("Direnç seviyesi bulunamadı")
            
            # Pivot Point bilgisi
            if pivot:
                st.subheader("⚖️ PIVOT POINT")
                st.write(f"**Pivot:** ${pivot:.2f}")
                st.write(f"**S1:** ${pivot_supports[0]:.2f}")
                st.write(f"**R1:** ${pivot_resistances[0]:.2f}")
        
        # Detaylı analiz
        st.subheader("📋 DETAYLI ANALİZ RAPORU")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.write("**📈 Teknik Özet:**")
            st.write(f"- Analiz edilen mum sayısı: {len(data)}")
            st.write(f"- Tespit edilen destek seviyesi: {len(support_levels)}")
            st.write(f"- Tespit edilen direnç seviyesi: {len(resistance_levels)}")
            st.write(f"- İğne analizi: {'Açık' if wick_analysis else 'Kapalı'}")
            st.write(f"- Hassasiyet seviyesi: {sensitivity}/10")
            
        with col4:
            st.write("**🎯 Trading Önerileri:**")
            if not signals:
                st.write("- Net sinyal yok - Piyasa gözlemi önerilir")
            elif any("ALIM" in signal for signal in signals):
                st.write("- 🟢 ALIM yönünde sinyaller mevcut")
            elif any("SATIM" in signal for signal in signals):
                st.write("- 🔴 SATIM yönünde sinyaller mevcut")
            else:
                st.write("- 🟡 NÖTR - Bekle ve gör")
        
        # Son 10 mumun detayları
        with st.expander("📜 SON 10 MUM DETAYI"):
            display_data = data.tail(10)[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            
            # Formatlama
            for col in ['Open', 'High', 'Low', 'Close']:
                display_data[col] = display_data[col].map(lambda x: f"${x:.2f}" if not pd.isna(x) else "N/A")
            display_data['Volume'] = display_data['Volume'].map(lambda x: f"{x:,.0f}" if not pd.isna(x) else "N/A")
            
            st.dataframe(display_data)
            
    except Exception as e:
        st.error(f"❌ Hata oluştu: {str(e)}")

if __name__ == "__main__":
    main()