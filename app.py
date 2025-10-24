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
    
    # Kripto sembolü için text input
    crypto_symbol = st.text_input("Kripto Sembolü", "BTC-USD", 
                                 help="Örnek: BTC-USD, ETH-USD, ADA-USD, XRP-USD vb.")
    
    # Popüler kripto seçenekleri (hızlı erişim için)
    st.caption("Hızlı Seçim:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("BTC-USD", use_container_width=True):
            st.session_state.crypto_symbol = "BTC-USD"
        if st.button("ETH-USD", use_container_width=True):
            st.session_state.crypto_symbol = "ETH-USD"
    with col2:
        if st.button("ADA-USD", use_container_width=True):
            st.session_state.crypto_symbol = "ADA-USD"
        if st.button("XRP-USD", use_container_width=True):
            st.session_state.crypto_symbol = "XRP-USD"
    
    # Session state'ten sembolü al
    if 'crypto_symbol' in st.session_state:
        crypto_symbol = st.session_state.crypto_symbol
    
    lookback_period = st.slider("Analiz Periyodu (Gün)", 30, 200, 100)
    
    st.subheader("📊 Parametreler")
    ema_period = st.slider("EMA Period", 20, 100, 50)
    rsi_period = st.slider("RSI Period", 5, 21, 14)
    min_touch_points = st.slider("Minimum Temas Noktası", 2, 5, 3)
    risk_reward_ratio = st.slider("Min Risk/Ödül Oranı", 1.0, 3.0, 1.5)

# Fiyat formatlama fonksiyonu
def format_price(price):
    """Fiyatı uygun formatta göster"""
    if price >= 1000:
        return f"${price:,.2f}"
    elif price >= 1:
        return f"${price:.3f}"
    elif price >= 0.1:
        return f"${price:.4f}"
    elif price >= 0.01:
        return f"${price:.5f}"
    else:
        return f"${price:.6f}"

# Veri çekme - SON 3 GÜN için
@st.cache_data
def get_4h_data(symbol, days=3):
    try:
        # Sembolü temizle ve kontrol et
        symbol = symbol.upper().strip()
        if '-' not in symbol:
            symbol = symbol + '-USD'  # Varsayılan USD pair ekle
        
        data = yf.download(symbol, period=f"{days}d", interval="4h", progress=False)
        
        if data.empty:
            st.error(f"❌ {symbol} için veri bulunamadı!")
            return None
            
        return data
    except Exception as e:
        st.error(f"❌ {symbol} veri çekilemedi: {e}")
        return None

# Teknik göstergeler
def calculate_indicators(data, ema_period=50, rsi_period=14):
    df = data.copy()
    
    # EMA
    df['EMA'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

# Yoğunluk tabanlı destek/direnç analizi
def find_congestion_zones(data, lookback=80, min_touch_points=3):
    """Fiyatın en çok zaman geçirdiği yoğunluk alanlarını bul"""
    try:
        df = data.tail(lookback).copy()
        
        # Tüm önemli fiyat noktaları (kapanış, high, low)
        price_levels = []
        for i in range(len(df)):
            price_levels.extend([
                float(df['Close'].iloc[i]),
                float(df['High'].iloc[i]),
                float(df['Low'].iloc[i])
            ])
        
        price_levels = sorted(price_levels)
        if not price_levels:
            return [], []
        
        # Yoğunluk analizi
        price_range = max(price_levels) - min(price_levels)
        bin_size = price_range * 0.01  # %1'lik bölgeler
        
        bins = {}
        current_bin = min(price_levels)
        
        while current_bin <= max(price_levels):
            bin_end = current_bin + bin_size
            count = sum(1 for price in price_levels if current_bin <= price <= bin_end)
            if count > 0:
                bins[(current_bin, bin_end)] = count
            current_bin = bin_end
        
        # Yoğun bölgeleri bul
        congestion_zones = []
        for (zone_start, zone_end), count in bins.items():
            if count >= min_touch_points:
                zone_center = (zone_start + zone_end) / 2
                congestion_zones.append({
                    'price': zone_center,
                    'strength': count,
                    'start': zone_start,
                    'end': zone_end
                })
        
        # Destek ve direnç olarak ayır
        current_price = float(df['Close'].iloc[-1])
        support_zones = [zone for zone in congestion_zones if zone['price'] < current_price]
        resistance_zones = [zone for zone in congestion_zones if zone['price'] > current_price]
        
        # Güçlü olanları seç ve SIRALI olarak düzenle
        support_zones = sorted(support_zones, key=lambda x: x['price'], reverse=True)[:5]  # Yüksekten düşüğe
        resistance_zones = sorted(resistance_zones, key=lambda x: x['price'])[:5]  # Düşükten yükseğe
        
        return support_zones, resistance_zones
        
    except Exception as e:
        st.error(f"Yoğunluk analizi hatası: {e}")
        return [], []

# Fitil analizi
def analyze_wicks(data, zone_price, tolerance_percent=1.0):
    """Belirli bir fiyat bölgesindeki fitil tepkilerini analiz et"""
    try:
        df = data.tail(50).copy()  # Son 50 mum
        tolerance = zone_price * (tolerance_percent / 100)
        
        reactions = 0
        strong_rejections = 0
        
        for i in range(len(df)):
            high = float(df['High'].iloc[i])
            low = float(df['Low'].iloc[i])
            close = float(df['Close'].iloc[i])
            open_price = float(df['Open'].iloc[i])
            
            # Bölgeye yakın mı?
            if abs(high - zone_price) <= tolerance or abs(low - zone_price) <= tolerance:
                reactions += 1
                
                # Güçlü reddetme sinyali kontrolü
                # Uzun üst fitil (direnç reddi)
                if high > zone_price and (high - max(open_price, close)) > (abs(open_price - close)) * 1.5:
                    strong_rejections += 1
                # Uzun alt fitil (destek reddi)
                elif low < zone_price and (min(open_price, close) - low) > (abs(open_price - close)) * 1.5:
                    strong_rejections += 1
        
        return reactions, strong_rejections
        
    except Exception as e:
        return 0, 0

# Ana trading stratejisi
def generate_trading_signals(data, support_zones, resistance_zones, ema_period=50, min_rr_ratio=1.5):
    """Profesyonel trading sinyalleri üret"""
    signals = []
    analysis_details = []
    
    if len(data) < ema_period + 10:
        return signals, analysis_details
    
    try:
        current_price = float(data['Close'].iloc[-1])
        ema_value = float(data['EMA'].iloc[-1])
        rsi_value = float(data['RSI'].iloc[-1])
        
        # 1. TREND ANALİZİ
        trend_direction = "BULLISH" if current_price > ema_value else "BEARISH"
        distance_to_ema = abs(current_price - ema_value) / ema_value * 100
        
        analysis_details.append(f"📈 TREND: {'YÜKSELİŞ' if trend_direction == 'BULLISH' else 'DÜŞÜŞ'}")
        analysis_details.append(f"📊 EMA {ema_period}: {format_price(ema_value)}")
        analysis_details.append(f"📍 Fiyat-EMA Mesafesi: %{distance_to_ema:.2f}")
        analysis_details.append(f"📉 RSI: {rsi_value:.1f}")
        
        # 2. YOĞUNLUK BÖLGELERİ ANALİZİ
        analysis_details.append("---")
        analysis_details.append("🎯 YOĞUNLUK BÖLGELERİ:")
        
        # Destek bölgeleri analizi (S1 en yüksek, S3 en düşük)
        for i, zone in enumerate(support_zones[:3]):
            reactions, strong_rejections = analyze_wicks(data, zone['price'])
            level_name = f"S{i+1}"  # S1, S2, S3
            analysis_details.append(f"🟢 {level_name}: {format_price(zone['price'])} (Güç: {zone['strength']}, Tepki: {reactions}, Red: {strong_rejections})")
        
        # Direnç bölgeleri analizi (R1 en düşük, R3 en yüksek)
        for i, zone in enumerate(resistance_zones[:3]):
            reactions, strong_rejections = analyze_wicks(data, zone['price'])
            level_name = f"R{i+1}"  # R1, R2, R3
            analysis_details.append(f"🔴 {level_name}: {format_price(zone['price'])} (Güç: {zone['strength']}, Tepki: {reactions}, Red: {strong_rejections})")
        
        # 3. SİNYAL ÜRETİMİ
        analysis_details.append("---")
        analysis_details.append("🎪 SİNYAL DEĞERLENDİRMESİ:")
        
        # En güçlü destek/direnç bölgeleri
        strongest_support = support_zones[0] if support_zones else None  # S1 - en yüksek destek
        strongest_resistance = resistance_zones[0] if resistance_zones else None  # R1 - en düşük direnç
        
        # ALIM SİNYALİ KOŞULLARI
        if (trend_direction == "BULLISH" and strongest_support and 
            current_price <= strongest_support['price'] * 1.02):  # %2 tolerans
            
            reactions, strong_rejections = analyze_wicks(data, strongest_support['price'])
            
            # Çalışırlık değerlendirmesi
            conditions_met = 0
            total_conditions = 4
            
            # Koşul 1: Trend uyumu
            if trend_direction == "BULLISH":
                conditions_met += 1
                analysis_details.append("✅ Trend uyumlu (Yükseliş)")
            
            # Koşul 2: Bölge test edilmiş mi?
            if reactions >= 2:
                conditions_met += 1
                analysis_details.append("✅ Bölge test edilmiş")
            
            # Koşul 3: Güçlü reddetme var mı?
            if strong_rejections >= 1:
                conditions_met += 1
                analysis_details.append("✅ Güçlü reddetme mevcut")
            
            # Koşul 4: RSI aşırı satımda mı?
            if rsi_value < 35:
                conditions_met += 1
                analysis_details.append("✅ RSI aşırı satım bölgesinde")
            
            # Risk/Ödül kontrolü
            if strongest_resistance:
                potential_profit = strongest_resistance['price'] - current_price
                potential_loss = current_price - strongest_support['price'] * 0.98  # %2 stop loss
                rr_ratio = potential_profit / potential_loss if potential_loss > 0 else 0
                
                analysis_details.append(f"📊 Risk/Ödül: {rr_ratio:.2f}")
                
                if rr_ratio >= min_rr_ratio:
                    conditions_met += 1
                    analysis_details.append("✅ Risk/Ödül uygun")
            
            # Sinyal kararı
            success_rate = conditions_met / total_conditions
            if success_rate >= 0.6:  # %60 başarı oranı
                stop_loss = strongest_support['price'] * 0.98
                take_profit = current_price + (current_price - stop_loss) * min_rr_ratio
                
                signals.append({
                    'type': 'BUY',
                    'price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confidence': success_rate,
                    'reason': f"Destek bölgesinde yükseliş tepkisi - Güven: %{success_rate*100:.0f}"
                })
            else:
                analysis_details.append("❌ ALIM: Yetersiz koşul - BEKLE")
        
        # SATIM SİNYALİ KOŞULLARI
        elif (trend_direction == "BEARISH" and strongest_resistance and 
              current_price >= strongest_resistance['price'] * 0.98):  # %2 tolerans
            
            reactions, strong_rejections = analyze_wicks(data, strongest_resistance['price'])
            
            # Çalışırlık değerlendirmesi
            conditions_met = 0
            total_conditions = 4
            
            # Koşul 1: Trend uyumu
            if trend_direction == "BEARISH":
                conditions_met += 1
                analysis_details.append("✅ Trend uyumlu (Düşüş)")
            
            # Koşul 2: Bölge test edilmiş mi?
            if reactions >= 2:
                conditions_met += 1
                analysis_details.append("✅ Bölge test edilmiş")
            
            # Koşul 3: Güçlü reddetme var mı?
            if strong_rejections >= 1:
                conditions_met += 1
                analysis_details.append("✅ Güçlü reddetme mevcut")
            
            # Koşul 4: RSI aşırı alımda mı?
            if rsi_value > 65:
                conditions_met += 1
                analysis_details.append("✅ RSI aşırı alım bölgesinde")
            
            # Risk/Ödül kontrolü
            if strongest_support:
                potential_profit = current_price - strongest_support['price']
                potential_loss = strongest_resistance['price'] * 1.02 - current_price  # %2 stop loss
                rr_ratio = potential_profit / potential_loss if potential_loss > 0 else 0
                
                analysis_details.append(f"📊 Risk/Ödül: {rr_ratio:.2f}")
                
                if rr_ratio >= min_rr_ratio:
                    conditions_met += 1
                    analysis_details.append("✅ Risk/Ödül uygun")
            
            # Sinyal kararı
            success_rate = conditions_met / total_conditions
            if success_rate >= 0.6:
                stop_loss = strongest_resistance['price'] * 1.02
                take_profit = current_price - (stop_loss - current_price) * min_rr_ratio
                
                signals.append({
                    'type': 'SELL',
                    'price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confidence': success_rate,
                    'reason': f"Direnç bölgesinde düşüş tepkisi - Güven: %{success_rate*100:.0f}"
                })
            else:
                analysis_details.append("❌ SATIM: Yetersiz koşul - BEKLE")
        
        else:
            analysis_details.append("🎭 NET SİNYAL YOK - Piyasa gözlemi önerilir")
            
            # EMA'ya uzaklık kontrolü
            if distance_to_ema > 5:  # %5'ten fazla uzaksa
                analysis_details.append("⚠️ Fiyat EMA'dan çok uzak - Risk yüksek")
        
        return signals, analysis_details
        
    except Exception as e:
        st.error(f"Sinyal üretim hatası: {e}")
        return [], []

# Sabit boyutlu mum grafiği oluşturma
def create_fixed_size_candlestick_chart(data, crypto_symbol):
    """Sabit boyutlu mum grafiği oluştur"""
    
    # Grafik oluştur
    fig = go.Figure()
    
    # Mum çubukları - net görünen iğnelerle
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=crypto_symbol,
        increasing_line_color='#26a69a',  # Profesyonel yeşil
        decreasing_line_color='#ef5350',   # Profesyonel kırmızı
        increasing_fillcolor='#26a69a',
        decreasing_fillcolor='#ef5350',
        line=dict(width=1.2),
        whiskerwidth=0.8  # İğnelerin genişliği
    ))
    
    # Grafik ayarları - SABİT BOYUT
    fig.update_layout(
        width=1000,  # Sabit genişlik
        height=500,  # Sabit yükseklik
        title={
            'text': f"{crypto_symbol} - Son 3 Günlük 4 Saatlik Mum Grafiği",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': 'white'}
        },
        xaxis_title="Tarih",
        yaxis_title="Fiyat (USD)",
        showlegend=False,
        xaxis_rangeslider_visible=False,
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font=dict(color='white', size=12),
        xaxis=dict(
            gridcolor='#444',
            tickfont=dict(size=11),
            title_font=dict(size=14)
        ),
        yaxis=dict(
            gridcolor='#444',
            tickfont=dict(size=11),
            title_font=dict(size=14)
        ),
        margin=dict(l=60, r=60, t=80, b=60)  # Sabit margin
    )
    
    # X ekseni ayarları
    fig.update_xaxes(
        tickformat='%m/%d %H:%M',
        tickangle=45
    )
    
    return fig

# Ana uygulama
def main():
    # Veri yükleme - SON 3 GÜN
    with st.spinner(f'⏳ {crypto_symbol} için son 3 günlük 4 saatlik veriler yükleniyor...'):
        data_3days = get_4h_data(crypto_symbol, days=3)
        data_full = get_4h_data(crypto_symbol, days=lookback_period)
    
    if data_3days is None or data_3days.empty or data_full is None or data_full.empty:
        st.error(f"❌ {crypto_symbol} için veri yüklenemedi!")
        st.info("💡 Lütfen geçerli bir kripto sembolü girin (Örnek: BTC-USD, ETH-USD, XRP-USD)")
        return
    
    st.success(f"✅ {crypto_symbol} için {len(data_3days)} adet 4 saatlik mum verisi yüklendi (Son 3 gün)")
    
    # Göstergeleri hesapla (tüm veriyle)
    data_full = calculate_indicators(data_full, ema_period, rsi_period)
    
    # Yoğunluk bölgelerini bul (SIRALI olarak)
    support_zones, resistance_zones = find_congestion_zones(data_full, min_touch_points=min_touch_points)
    
    # Sinyal üret
    signals, analysis_details = generate_trading_signals(
        data_full, support_zones, resistance_zones, ema_period, risk_reward_ratio
    )
    
    # Mevcut durum
    current_price = float(data_full['Close'].iloc[-1])
    ema_value = float(data_full['EMA'].iloc[-1])
    rsi_value = float(data_full['RSI'].iloc[-1])
    
    # Layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader(f"📈 {crypto_symbol} - Son 3 Günlük 4 Saatlik Mum Grafiği")
        
        # Sabit boyutlu mum grafiği oluştur
        chart_fig = create_fixed_size_candlestick_chart(data_3days, crypto_symbol)
        
        # Grafiği sabit boyutlu ve etkileşimsiz göster
        st.plotly_chart(chart_fig, use_container_width=False, config={
            'displayModeBar': False,  # Araç çubuğunu gizle
            'staticPlot': False,      # Küçük etkileşimlere izin ver
            'responsive': False       # Responsive özelliği kapat
        })
        
        # Grafik bilgisi
        st.info("""
        **📊 Grafik Özellikleri:**
        - Son 3 günlük 4 saatlik mumlar
        - Sabit boyut (küçültme/büyütme yok)
        - Net yeşil/kırmızı iğneler
        - Profesyonel trading görünümü
        """)
    
    with col2:
        st.subheader("🎯 TRADING SİNYALLERİ")
        
        if signals:
            for signal in signals:
                if signal['type'] == 'BUY':
                    st.success(f"""
                    **✅ ALIM SİNYALİ**
                    - Giriş: {format_price(signal['price'])}
                    - Stop: {format_price(signal['stop_loss'])}
                    - TP: {format_price(signal['take_profit'])}
                    - Güven: %{signal['confidence']*100:.0f}
                    """)
                else:
                    st.error(f"""
                    **❌ SATIM SİNYALİ**
                    - Giriş: {format_price(signal['price'])}
                    - Stop: {format_price(signal['stop_loss'])}
                    - TP: {format_price(signal['take_profit'])}
                    - Güven: %{signal['confidence']*100:.0f}
                    """)
        else:
            st.info("""
            **🎭 NET SİNYAL YOK**
            - Piyasa gözlemi önerilir
            - Koşullar uygun değil
            - BEKLE stratejisi uygula
            """)
        
        st.subheader("📊 MEVCUT DURUM")
        st.metric("Fiyat", format_price(current_price))
        st.metric(f"EMA {ema_period}", format_price(ema_value))
        st.metric("RSI", f"{rsi_value:.1f}")
        
        trend = "YÜKSELİŞ" if current_price > ema_value else "DÜŞÜŞ"
        st.metric("TREND", trend)
        
        # Destek/Direnç Listesi - SIRALI olarak
        st.subheader("💎 SEVİYELER")
        
        st.write("**🟢 DESTEK (S1→S3):**")
        for i, zone in enumerate(support_zones[:3]):
            level_name = f"S{i+1}"
            st.write(f"{level_name}: {format_price(zone['price'])}")
        
        st.write("**🔴 DİRENÇ (R1→R3):**")
        for i, zone in enumerate(resistance_zones[:3]):
            level_name = f"R{i+1}"
            st.write(f"{level_name}: {format_price(zone['price'])}")
    
    # Detaylı analiz
    st.subheader("🔍 DETAYLI ANALİZ RAPORU")
    with st.expander("Analiz Detayları", expanded=True):
        for detail in analysis_details:
            if "✅" in detail:
                st.success(detail)
            elif "❌" in detail or "⚠️" in detail:
                st.error(detail)
            elif "🎯" in detail or "🎪" in detail:
                st.warning(detail)
            else:
                st.info(detail)

if __name__ == "__main__":
    main()