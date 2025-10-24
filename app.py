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

st.title("🎯 4 Saatlik Profesyonel Teknik Analiz Stratejisi - SON 3 GÜN")

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
            crypto_symbol = "BTC-USD"
        if st.button("ETH-USD", use_container_width=True):
            crypto_symbol = "ETH-USD"
    with col2:
        if st.button("ADA-USD", use_container_width=True):
            crypto_symbol = "ADA-USD"
        if st.button("XRP-USD", use_container_width=True):
            crypto_symbol = "XRP-USD"
    
    st.info("📅 Analiz: SON 3 GÜN")
    
    st.subheader("📊 Parametreler")
    ema_period = st.slider("EMA Period", 5, 50, 20)
    rsi_period = st.slider("RSI Period", 5, 21, 10)
    min_touch_points = st.slider("Minimum Temas Noktası", 2, 5, 2)
    risk_reward_ratio = st.slider("Min Risk/Ödül Oranı", 1.0, 3.0, 1.5)

# Fiyat formatlama fonksiyonu
def format_price(price):
    """Fiyatı uygun formatta göster"""
    if price is None or np.isnan(price):
        return "N/A"
    
    try:
        price = float(price)
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
    except (ValueError, TypeError):
        return "N/A"

# Veri çekme - SADECE SON 3 GÜN
@st.cache_data
def get_4h_data(symbol):
    try:
        # Sembolü temizle ve kontrol et
        symbol = symbol.upper().strip()
        if '-' not in symbol:
            symbol = symbol + '-USD'  # Varsayılan USD pair ekle
        
        # SADECE SON 3 GÜN verisi çek
        data = yf.download(symbol, period="3d", interval="4h", progress=False)
        
        if data.empty or len(data) == 0:
            st.error(f"❌ {symbol} için veri bulunamadı!")
            return None
            
        return data
    except Exception as e:
        st.error(f"❌ {symbol} veri çekilemedi: {e}")
        return None

# Teknik göstergeler
def calculate_indicators(data, ema_period=20, rsi_period=10):
    if data is None or len(data) == 0:
        return data
        
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

# Yoğunluk tabanlı destek/direnç analizi - SON 3 GÜN
def find_congestion_zones(data, min_touch_points=2):
    """Fiyatın en çok zaman geçirdiği yoğunluk alanlarını bul - SON 3 GÜN"""
    try:
        if data is None or len(data) == 0:
            return [], []
            
        df = data.copy()  # Tüm veri son 3 gün zaten
        
        if len(df) == 0:
            return [], []
        
        # Tüm önemli fiyat noktaları (kapanış, high, low)
        price_levels = []
        for i in range(len(df)):
            try:
                price_levels.extend([
                    float(df['Close'].iloc[i]),
                    float(df['High'].iloc[i]),
                    float(df['Low'].iloc[i])
                ])
            except (ValueError, IndexError):
                continue
        
        if not price_levels:
            return [], []
            
        price_levels = sorted(price_levels)
        
        # Yoğunluk analizi - daha hassas bölgeler (son 3 gün için)
        price_range = max(price_levels) - min(price_levels)
        if price_range == 0:
            return [], []
            
        bin_size = price_range * 0.005  # %0.5'lik daha küçük bölgeler (hassas)
        
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
        try:
            current_price = float(df['Close'].iloc[-1])
            support_zones = [zone for zone in congestion_zones if zone['price'] < current_price]
            resistance_zones = [zone for zone in congestion_zones if zone['price'] > current_price]
        except (ValueError, IndexError):
            return [], []
        
        # Güçlü olanları seç ve SIRALI olarak düzenle
        support_zones = sorted(support_zones, key=lambda x: x['price'], reverse=True)[:3]  # Sadece 3 destek
        resistance_zones = sorted(resistance_zones, key=lambda x: x['price'])[:3]  # Sadece 3 direnç
        
        return support_zones, resistance_zones
        
    except Exception as e:
        st.error(f"Yoğunluk analizi hatası: {e}")
        return [], []

# Fitil analizi - SON 3 GÜN
def analyze_wicks(data, zone_price, tolerance_percent=0.5):
    """Belirli bir fiyat bölgesindeki fitil tepkilerini analiz et - SON 3 GÜN"""
    try:
        if data is None or len(data) == 0:
            return 0, 0
            
        df = data.copy()  # Tüm veri son 3 gün
        tolerance = zone_price * (tolerance_percent / 100)  # Daha dar tolerans
        
        reactions = 0
        strong_rejections = 0
        
        for i in range(len(df)):
            try:
                high = float(df['High'].iloc[i])
                low = float(df['Low'].iloc[i])
                close = float(df['Close'].iloc[i])
                open_price = float(df['Open'].iloc[i])
                
                # Bölgeye yakın mı? (daha dar tolerans)
                if abs(high - zone_price) <= tolerance or abs(low - zone_price) <= tolerance:
                    reactions += 1
                    
                    # Güçlü reddetme sinyali kontrolü
                    body_size = abs(open_price - close)
                    upper_wick = high - max(open_price, close)
                    lower_wick = min(open_price, close) - low
                    
                    # Uzun üst fitil (direnç reddi)
                    if high > zone_price and upper_wick > body_size * 1.2:
                        strong_rejections += 1
                    # Uzun alt fitil (destek reddi)
                    elif low < zone_price and lower_wick > body_size * 1.2:
                        strong_rejections += 1
            except (ValueError, IndexError):
                continue
        
        return reactions, strong_rejections
        
    except Exception as e:
        return 0, 0

# Mum grafiği oluşturma - SON 3 GÜN
def create_candlestick_chart_manual(data, support_zones, resistance_zones, crypto_symbol):
    """MANUEL MUM ÇİZİMİ - SON 3 GÜN"""
    
    fig = go.Figure()
    
    if data is None or len(data) == 0:
        return fig
    
    # HER MUMU AYRI AYRI ÇİZ
    for i in range(len(data)):
        try:
            row = data.iloc[i]
            open_price = float(row['Open'])
            high = float(row['High'])
            low = float(row['Low'])
            close_price = float(row['Close'])
            
            # Renk belirle: Kapanış > Açılış ise yeşil, değilse kırmızı
            color = '#00C805' if close_price > open_price else '#FF0000'
            
            # MUM GÖVDESİ (kalın dikdörtgen)
            fig.add_trace(go.Scatter(
                x=[data.index[i], data.index[i]],
                y=[open_price, close_price],
                mode='lines',
                line=dict(color=color, width=8),
                showlegend=False
            ))
            
            # ÜST İĞNE (High)
            fig.add_trace(go.Scatter(
                x=[data.index[i], data.index[i]],
                y=[max(open_price, close_price), high],
                mode='lines',
                line=dict(color=color, width=1.5),
                showlegend=False
            ))
            
            # ALT İĞNE (Low)
            fig.add_trace(go.Scatter(
                x=[data.index[i], data.index[i]],
                y=[min(open_price, close_price), low],
                mode='lines',
                line=dict(color=color, width=1.5),
                showlegend=False
            ))
        except (ValueError, IndexError):
            continue
    
    # EMA çizgisi
    if 'EMA' in data.columns:
        try:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['EMA'],
                name=f'EMA {ema_period}',
                line=dict(color='orange', width=2)
            ))
        except Exception:
            pass
    
    # Mevcut fiyat çizgisi
    try:
        current_price = float(data['Close'].iloc[-1])
        fig.add_hline(
            y=current_price,
            line_dash="dot",
            line_color="yellow",
            line_width=2,
            opacity=0.7,
            annotation_text=f"Şimdi: {format_price(current_price)}",
            annotation_position="left",
            annotation_font_size=12,
            annotation_font_color="yellow"
        )
    except (ValueError, IndexError):
        pass
    
    # DESTEK çizgileri
    for i, zone in enumerate(support_zones[:3]):
        try:
            fig.add_hline(
                y=zone['price'],
                line_dash="solid",
                line_color="#00FF00",
                line_width=2,
                opacity=0.8,
                annotation_text=f"S{i+1}",
                annotation_position="left",
                annotation_font_size=12,
                annotation_font_color="#00FF00"
            )
        except Exception:
            continue
    
    # DİRENÇ çizgileri
    for i, zone in enumerate(resistance_zones[:3]):
        try:
            fig.add_hline(
                y=zone['price'],
                line_dash="solid",
                line_color="#FF0000",
                line_width=2,
                opacity=0.8,
                annotation_text=f"R{i+1}",
                annotation_position="right",
                annotation_font_size=12,
                annotation_font_color="#FF0000"
            )
        except Exception:
            continue
    
    # Grafik ayarları
    fig.update_layout(
        height=500,
        title=f"{crypto_symbol} - SON 3 GÜN 4 Saatlik Analiz",
        xaxis_title="Tarih",
        yaxis_title="Fiyat (USD)",
        showlegend=True,
        xaxis_rangeslider_visible=False,
        plot_bgcolor='#0E1117',
        paper_bgcolor='#0E1117',
        font=dict(color='white', size=10),
        xaxis=dict(gridcolor='#444'),
        yaxis=dict(gridcolor='#444')
    )
    
    return fig

# Ana trading stratejisi - SON 3 GÜN
def generate_trading_signals(data, support_zones, resistance_zones, ema_period=20, min_rr_ratio=1.5):
    """Profesyonel trading sinyalleri üret - SON 3 GÜN"""
    signals = []
    analysis_details = []
    
    if data is None or len(data) < 5:  # Minimum 5 mum verisi
        analysis_details.append("❌ Yetersiz veri - analiz yapılamıyor")
        return signals, analysis_details
    
    try:
        current_price = float(data['Close'].iloc[-1])
        ema_value = float(data['EMA'].iloc[-1])
        rsi_value = float(data['RSI'].iloc[-1])
        
        # 1. TREND ANALİZİ - SON 3 GÜN
        trend_direction = "BULLISH" if current_price > ema_value else "BEARISH"
        distance_to_ema = abs(current_price - ema_value) / ema_value * 100
        
        analysis_details.append(f"📈 SON 3 GÜN TREND: {'YÜKSELİŞ' if trend_direction == 'BULLISH' else 'DÜŞÜŞ'}")
        analysis_details.append(f"📊 EMA {ema_period}: {format_price(ema_value)}")
        analysis_details.append(f"📍 Fiyat-EMA Mesafesi: %{distance_to_ema:.2f}")
        analysis_details.append(f"📉 RSI: {rsi_value:.1f}")
        
        # 2. YOĞUNLUK BÖLGELERİ ANALİZİ - SON 3 GÜN
        analysis_details.append("---")
        analysis_details.append("🎯 SON 3 GÜN YOĞUNLUK BÖLGELERİ:")
        
        # Destek bölgeleri analizi
        for i, zone in enumerate(support_zones[:3]):
            reactions, strong_rejections = analyze_wicks(data, zone['price'])
            level_name = f"S{i+1}"
            analysis_details.append(f"🟢 {level_name}: {format_price(zone['price'])} (Güç: {zone['strength']}, Tepki: {reactions}, Red: {strong_rejections})")
        
        # Direnç bölgeleri analizi
        for i, zone in enumerate(resistance_zones[:3]):
            reactions, strong_rejections = analyze_wicks(data, zone['price'])
            level_name = f"R{i+1}"
            analysis_details.append(f"🔴 {level_name}: {format_price(zone['price'])} (Güç: {zone['strength']}, Tepki: {reactions}, Red: {strong_rejections})")
        
        # 3. SİNYAL ÜRETİMİ - SON 3 GÜN
        analysis_details.append("---")
        analysis_details.append("🎪 SON 3 GÜN SİNYAL DEĞERLENDİRMESİ:")
        
        # En güçlü destek/direnç bölgeleri
        strongest_support = support_zones[0] if support_zones else None
        strongest_resistance = resistance_zones[0] if resistance_zones else None
        
        # KISA VADELİ ALIM SİNYALİ KOŞULLARI
        buy_signal = False
        sell_signal = False
        
        # ALIM SİNYALİ - Daha agresif kurallar (kısa vade)
        if strongest_support and current_price <= strongest_support['price'] * 1.01:  # %1 tolerans
            reactions, strong_rejections = analyze_wicks(data, strongest_support['price'])
            
            conditions_met = 0
            total_conditions = 3
            
            # Koşul 1: Trend nötr veya yükseliş
            if trend_direction == "BULLISH" or distance_to_ema < 3:
                conditions_met += 1
                analysis_details.append("✅ Trend uyumlu")
            
            # Koşul 2: Bölge test edilmiş
            if reactions >= 1:  # 1 tepki yeterli (kısa vade)
                conditions_met += 1
                analysis_details.append("✅ Bölge test edilmiş")
            
            # Koşul 3: Güçlü reddetme veya RSI
            if strong_rejections >= 1 or rsi_value < 40:
                conditions_met += 1
                analysis_details.append("✅ Teknik destek mevcut")
            
            success_rate = conditions_met / total_conditions
            if success_rate >= 0.67:  # %67 başarı oranı
                buy_signal = True
                stop_loss = strongest_support['price'] * 0.99  # %1 stop
                take_profit = current_price + (current_price - stop_loss) * min_rr_ratio
                
                signals.append({
                    'type': 'BUY',
                    'price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confidence': success_rate,
                    'reason': f"Kısa vadeli destek tepkisi - Güven: %{success_rate*100:.0f}"
                })
        
        # SATIM SİNYALİ - Daha agresif kurallar (kısa vade)
        elif strongest_resistance and current_price >= strongest_resistance['price'] * 0.99:  # %1 tolerans
            reactions, strong_rejections = analyze_wicks(data, strongest_resistance['price'])
            
            conditions_met = 0
            total_conditions = 3
            
            # Koşul 1: Trend nötr veya düşüş
            if trend_direction == "BEARISH" or distance_to_ema < 3:
                conditions_met += 1
                analysis_details.append("✅ Trend uyumlu")
            
            # Koşul 2: Bölge test edilmiş
            if reactions >= 1:  # 1 tepki yeterli (kısa vade)
                conditions_met += 1
                analysis_details.append("✅ Bölge test edilmiş")
            
            # Koşul 3: Güçlü reddetme veya RSI
            if strong_rejections >= 1 or rsi_value > 60:
                conditions_met += 1
                analysis_details.append("✅ Teknik direnç mevcut")
            
            success_rate = conditions_met / total_conditions
            if success_rate >= 0.67:  # %67 başarı oranı
                sell_signal = True
                stop_loss = strongest_resistance['price'] * 1.01  # %1 stop
                take_profit = current_price - (stop_loss - current_price) * min_rr_ratio
                
                signals.append({
                    'type': 'SELL',
                    'price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confidence': success_rate,
                    'reason': f"Kısa vadeli direnç tepkisi - Güven: %{success_rate*100:.0f}"
                })
        
        if not buy_signal and not sell_signal:
            analysis_details.append("🎭 NET SİNYAL YOK - Kısa vadeli piyasa gözlemi")
            
            # Özel kısa vadeli uyarılar
            if distance_to_ema > 8:
                analysis_details.append("⚠️ Fiyat EMA'dan çok uzak - Kısa vadeli düzeltme riski")
            if rsi_value > 70:
                analysis_details.append("⚠️ RSI aşırı alım - Kısa vadeli satış baskısı beklenebilir")
            if rsi_value < 30:
                analysis_details.append("⚠️ RSI aşırı satım - Kısa vadeli toplanma beklenebilir")
        
        return signals, analysis_details
        
    except Exception as e:
        analysis_details.append(f"❌ Sinyal üretim hatası: {e}")
        return [], analysis_details

# Ana uygulama
def main():
    st.info("""
    **📊 SON 3 GÜN ANALİZİ - Kısa Vadeli Strateji**
    - 🕒 Zaman Periyodu: Son 3 Gün
    - ⏰ Mum Aralığı: 4 Saatlik
    - 🎯 Amaç: Kısa vadeli fırsatları tespit
    - ⚠️ Risk: Yüksek volatilite - Dikkatli kullanın
    """)
    
    # Veri yükleme - SADECE SON 3 GÜN
    with st.spinner(f'⏳ {crypto_symbol} için SON 3 GÜN 4 saatlik veriler yükleniyor...'):
        data_3days = get_4h_data(crypto_symbol)
    
    if data_3days is None or data_3days.empty:
        st.error(f"❌ {crypto_symbol} için veri yüklenemedi!")
        st.info("💡 Lütfen geçerli bir kripto sembolü girin (Örnek: BTC-USD, ETH-USD, XRP-USD)")
        return
    
    st.success(f"✅ {crypto_symbol} için SON 3 GÜN {len(data_3days)} adet 4 saatlik mum verisi yüklendi")
    
    # Göstergeleri hesapla
    data_3days = calculate_indicators(data_3days, ema_period, rsi_period)
    
    # Yoğunluk bölgelerini bul (SON 3 GÜN)
    support_zones, resistance_zones = find_congestion_zones(data_3days, min_touch_points=min_touch_points)
    
    # Sinyal üret (SON 3 GÜN)
    signals, analysis_details = generate_trading_signals(
        data_3days, support_zones, resistance_zones, ema_period, risk_reward_ratio
    )
    
    # Mevcut durum
    try:
        current_price = float(data_3days['Close'].iloc[-1])
        ema_value = float(data_3days['EMA'].iloc[-1])
        rsi_value = float(data_3days['RSI'].iloc[-1])
    except (ValueError, IndexError):
        current_price = 0
        ema_value = 0
        rsi_value = 0
    
    # Layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader(f"📈 {crypto_symbol} - SON 3 GÜN 4 Saatlik Analiz")
        
        # MUM GRAFİĞİNİ GÖSTER - SON 3 GÜN
        chart_fig = create_candlestick_chart_manual(data_3days, support_zones, resistance_zones, crypto_symbol)
        st.plotly_chart(chart_fig, use_container_width=True)
        
        st.warning("""
        **⚠️ KISA VADELİ STRATEJİ UYARISI:**
        - 📅 Analiz periyodu: SADECE SON 3 GÜN
        - ⏰ Kısa vadeli sinyaller (4-24 saat)
        - 🔄 Hızlı pozisyon giriş/çıkışı
        - 📉 Stop-loss KESİNLİKLE kullanın
        """)
    
    with col2:
        st.subheader("🎯 KISA VADELİ SİNYALLER")
        
        if signals:
            for signal in signals:
                if signal['type'] == 'BUY':
                    st.success(f"""
                    **✅ KISA ALIM**
                    - Giriş: {format_price(signal['price'])}
                    - Stop: {format_price(signal['stop_loss'])}
                    - TP: {format_price(signal['take_profit'])}
                    - Güven: %{signal['confidence']*100:.0f}
                    - Süre: 4-24 saat
                    """)
                else:
                    st.error(f"""
                    **❌ KISA SATIM**
                    - Giriş: {format_price(signal['price'])}
                    - Stop: {format_price(signal['stop_loss'])}
                    - TP: {format_price(signal['take_profit'])}
                    - Güven: %{signal['confidence']*100:.0f}
                    - Süre: 4-24 saat
                    """)
        else:
            st.info("""
            **🎭 NET SİNYAL YOK**
            - Kısa vadeli koşullar uygun değil
            - Piyasa gözlemi önerilir
            - BEKLE stratejisi uygula
            """)
        
        st.subheader("📊 SON DURUM")
        st.metric("Fiyat", format_price(current_price))
        st.metric(f"EMA {ema_period}", format_price(ema_value))
        st.metric("RSI", f"{rsi_value:.1f}")
        
        trend = "YÜKSELİŞ" if current_price > ema_value else "DÜŞÜŞ"
        st.metric("3 GÜN TREND", trend)
        
        # Destek/Direnç Listesi
        st.subheader("💎 KISA VADE SEVİYELER")
        
        st.write("**🟢 DESTEK:**")
        for i, zone in enumerate(support_zones[:3]):
            level_name = f"S{i+1}"
            distance_pct = ((current_price - zone['price']) / current_price * 100)
            st.write(f"{level_name}: {format_price(zone['price'])} (%{distance_pct:.1f})")
        
        st.write("**🔴 DİRENÇ:**")
        for i, zone in enumerate(resistance_zones[:3]):
            level_name = f"R{i+1}"
            distance_pct = ((zone['price'] - current_price) / current_price * 100)
            st.write(f"{level_name}: {format_price(zone['price'])} (%{distance_pct:.1f})")
    
    # Detaylı analiz
    st.subheader("🔍 SON 3 GÜN DETAYLI ANALİZ")
    with st.expander("Kısa Vadeli Analiz Detayları", expanded=True):
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