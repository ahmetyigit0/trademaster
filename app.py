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

# Veri çekme
@st.cache_data
def get_4h_data(symbol, days):
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
        
        # Güçlü olanları seç ve sırala
        support_zones = sorted(support_zones, key=lambda x: x['strength'], reverse=True)[:5]
        resistance_zones = sorted(resistance_zones, key=lambda x: x['strength'], reverse=True)[:5]
        
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
def generate_trading_signals(data, support_zones, resistance_zones, ema_period=50, min_rr_ratio=1.ratio=1.55):
    """Profesyon):
    """Profesyonel trading sinyel trading sinyalleri üalleri üret"""
    signalsret"""
    signals = []
    analysis = []
    analysis_details =_details = []
    
 []
    
    if len(data) < ema_period    if len(data) < ema_period + 10 + 10:
        return:
        return signals, analysis_details signals, analysis_details
    
   
    
    try:
        current try:
        current_price = float(data['Close_price = float(data['Close'].il'].iloc[-1])
oc[-1])
        em        ema_valuea_value = float = float(data['EMA'].(data['EMA'].ilociloc[-1])
       [-1])
        rsi rsi_value = float(data_value = float(data['RS['RSI'].ilocI'].iloc[-1[-1])
        
        #])
        
        # 1 1. TREND ANAL. TREND ANALİZİZİ
        trendİ
        trend_direction_direction = "BULL = "BULLISH"ISH" if current_price > ema if current_price > ema_value else "BE_value else "BEARARISH"
       ISH"
        distance_to_ema distance_to_ema = abs(current = abs(current_price - em_price - ema_value) / ema_value * 100a_value) / ema_value * 100
        
        analysis_details.append(f
        
        analysis_details.append(f"📈"📈 TREND: TREND: {'YÜKS {'YÜKSELİELİŞ' if trendŞ' if trend_direction_direction == 'B == 'BULLULLISH' else 'ISH' else 'DÜŞÜŞ'}")
DÜŞÜŞ'        analysis_details.append(f"}")
        analysis_details.append(f"📊 EMA {ema📊 EMA {ema_period}: {format_price_period}: {format_price((ema_value)}")
        analysis_details.append(f"ema_value)}")
        analysis_details.append(f"📍 Fiyat-EMA Mesafesi: %{distance📍 Fiyat-EMA Mesafesi: %{distance_to_ema:.2f}")
       _to_ema:.2f}")
        analysis analysis_details.append(f"_details.append(f"📉 RSI:📉 RSI: {rsi_value:.1 {rsi_value:.1ff}")
        
        # }")
        
        # 22. YOĞ. YOĞUNLUNLUK BÖLGUK BÖLGELERELERİ ANALİZİ ANALİZİ
        analysis_details.append("---")
       İ
        analysis_details.append("---")
        analysis analysis_details.append("🎯_details.append("🎯 YOĞUNLUK B YOĞUNLUK BÖLGELERÖLGELERİ:")
        
       İ:")
        
        # Dest # Destek bölgek bölgelerieleri analizi
        analizi
        for i for i, zone in enumerate, zone in enumerate(support(support_zones[:3_zones[:3]):
           ]):
            reactions, strong_re reactions, strong_rejectionsjections = analyze = analyze_wicks(data,_wicks(data, zone zone['price['price'])
            analysis'])
            analysis_details.append(f"_details.append(f"🟢 Dest🟢 Destek {i+ek {i+1}: {format1}: {format_price(_price(zone['price'])} (zone['price'])} (GüçGüç: {zone[': {zone['strestrength']}, Tngth']}, Tepki: {reepki: {reactions}, Redactions}, Red: {: {strong_rejectionsstrong_rejections})")
        
})")
        
        # D        # Direnç birenç bölölgeleri analizi
geleri analizi
        for i, zone in enumerate(resistance_zones[:3]):
            reactions, strong_re        for i, zone in enumerate(resistance_zones[:3]):
            reactionsjections = analyze_wicks(data, strong_rejections = analyze_w, zone['price'])
           icks(data, zone['price'])
            analysis_details.append(f analysis_details.append(f""🔴 Direnç🔴 Direnç {i {i+1}: {+1}: {format_priceformat_price(zone['price'])} (Güç(zone['price'])} (Güç: {: {zone['strength']zone['strength']}, Tepki: {re}, Tepki: {reactions}, Red: {strongactions}, Red: {strong_rejections})")
        
        #_rejections})")
        
        # 3 3. SİNY. SİNYAL ÜRETAL ÜRETİMİİMİ
        analysis_details
        analysis_details.append(".append("---")
        analysis---")
        analysis_details.append_details.append("🎪 S("🎪 SİNYİNYAL DEĞAL DEĞERLERLENDİRMESENDİRMESİ:")
İ:")
        
               
        # En güçlü # En güçlü destek/d destek/direnç bölirenç bölgelergeleri
        strongesti
        strongest_s_support = support_zones[0upport = support_zones[0] if support_z] if support_zones else None
        strongestones else None
        strongest_resistance_resistance = resistance_z = resistance_zones[ones[0] if resistance0] if resistance_zones else None
        
        # AL_zones else None
        
        # ALIM SİNYALIM SİNYALİİ KOŞULLARI
 KOŞULLARI
               if (trend_d if (trend_direction == "BULLISH" and strongest_support andirection == "BULLISH" and strongest_support and 
 
            current_price <= strongest            current_price <= strongest_s_support['price'] * upport['price'] * 1.02): 1.02):  # %2 tolerans
            
 # %2 tolerans
            
            reactions, strong            reactions, strong_rejections = analyze_wicks(data_rejections = analyze_wicks(data, strongest_support['price, strongest_support['price'])
            
            # Çal'])
            
            # Çalışırlıkışırlık değerlendirmesi
            değerlendirmesi
            conditions_met = 0 conditions_met = 0
            total_
            total_conditions = 4
            
           conditions = 4
            
            # Koşul 1 # Koşul 1: Trend uyumu
: Trend uyumu
            if trend_direction == "BULLISH":
                conditions_met += 1
                analysis_details            if trend_direction == "BULLISH":
                conditions_met += 1
                analysis_details.append("✅ Trend uy.append("✅ Trend uyumlu (Yüksumlu (Yükseliş)")
            
            #eliş)")
            
            # Koş Koşul ul 2:2: Bölge test edil Bölge test edilmiş mimiş mi?
           ?
            if reactions >= 2:
                if reactions >= 2:
                conditions_met += conditions_met += 1
                analysis_details 1
                analysis_details.append("✅ B.append("✅ Bölge test edilmiş")
ölge test edilmiş")
            
                       
            # Koşul # Koşul 3: 3: Güçl Güçlü reddetü reddetme var mme var mı?
ı?
            if strong            if strong_rejections_rejections >= 1:
 >= 1:
                conditions                conditions_met += _met += 11
                analysis_details.append
                analysis_details.append("✅("✅ Güçlü Güçlü reddet reddetme meme mevcut")
            
            # Koşulvcut")
            
            # Koşul 4: RSI 4: RSI aşırı aşırı satımda m satımda mı?
           ı?
            if if rsi_value < rsi_value < 35 35:
                conditions_met:
                conditions_met += += 1
                analysis 1
                analysis_details_details.append("✅ R.append("✅ RSI aşSI aşırı satım bırı satım bölölgesinde")
            
           gesinde")
            
            # # Risk/Ödül Risk/Ödül kontrolü
            if kontrolü
            if strongest_res strongest_resistance:
                potential_proistance:
                potential_profit = strongest_resistance['fit = strongest_resistance['priceprice'] - current_price
'] - current_price
                potential                potential_loss = current_price_loss = current_price - strongest_support['price - strongest_support['price'] *'] * 0.98 0.98  #  # %2 stop loss %2 stop loss
               
                rr_ratio = rr_ratio = potential_pro potential_profit / potential_lossfit / potential_loss if potential if potential_loss > 0_loss > 0 else  else 0
                
               0
                
                analysis_details.append(f"📊 Risk analysis_details.append(f"📊 Risk/Ödül/Ödül: {: {rr_ratio:.rr_ratio:.2f2f}")
                
                if}")
                
                if rr_ rr_ratio >= min_rratio >= min_rr_ratio:
r_ratio:
                    conditions_met += 1
                    analysis_details.append("✅ Risk                    conditions_met += 1
                    analysis_details.append("✅/Ödül u Risk/Ödül uygun")
            
ygun")
            
            #            # Sinyal kararı
            success_rate = conditions_met / total_conditions Sinyal kararı
            success_rate = conditions_met / total_conditions
            if
            if success_rate success_rate >=  >= 0.6:0.6:  #  # %60 baş %60 başarı oranı
               arı oranı
                stop_loss = strongest stop_loss = strongest_support['price'] *_support['price'] * 0.98
 0.98
                take_profit = current_price                take_profit = current_price + (current_price - + (current_price - stop_loss) stop_loss) * min_rr * min_rr_ratio_ratio
                
                signals
                
                signals.append({
.append({
                    'type':                    'type': 'BUY 'BUY',
                    '',
                    'price': currentprice': current_price,
                    '_price,
                    'stop_lossstop_loss': stop': stop_loss_loss,
,
                    'take_profit                    'take_profit': take_profit': take_profit,
                    'confidence': success_rate,
                    'confidence': success_rate,
                    'reason':,
                    'reason': f"Dest f"Destek bölek bölgesindegesinde yükseli yükseliş tepkisiş tepkisi - Gü - Güven: %{ven: %{success_ratesuccess_rate*100:.0*100:.0f}"
f}"
                })
            else                })
            else:
               :
                analysis_details.append analysis_details.append("❌ AL("❌ ALIM: YIM: Yetersiz koşetersiz koşul -ul - BEKLE BEKLE")
        
       ")
        
        # SATIM S # SATIM SİNYİNYALİ KOALİ KOŞŞULLARI
       ULLARI
        elif ( elif (trend_directiontrend_direction == "BE == "BEARISH"ARISH" and strongest_res and strongest_resistance and 
istance and 
              current_price >=              current_price >= strongest_res strongest_resistance['price']istance['price'] * 0 * 0.98.98):  #):  # %2 tolerans %2 tolerans
            
           
            
            reactions, strong_re reactions, strong_rejectionsjections = analyze_wicks = analyze_wicks(data, strongest(data, strongest_resistance['_resistance['price'])
price'])
            
            # Ç            
            # Çalışalışırlıkırlık değerlend değerlendirmesi
irmesi
            conditions_            conditions_met =met = 0
            0
            total_conditions total_conditions = 4
            
 = 4
            
            # Koşul 1: Trend uyumu
                       # Koşul 1: Trend uyumu
            if trend_direction == " if trend_direction == "BEARISH":
BEARISH":
                conditions_met += 1                conditions_met += 1

                analysis_details.append("✅ Trend uyumlu (Dü                analysis_details.append("✅ Trend uyumlu (Düşüş)")
            
            # Koşul 2şüş)")
            
            # Koşul 2: Bölge test edil: Bölge test edilmiş mi?
            if reactions >= 2:
                conditions_met +=miş mi?
            if reactions >= 2:
                conditions 1
                analysis_details_met += 1
                analysis_details.append("✅ B.append("✅ Bölge test edilmiş")
ölge test edilmiş")
            
            #            
            # Koşul 3: Güçl Koşul 3: Güçlü reddetü reddetmeme var mı var mı?
            if?
            if strong_rejections >= 1:
                strong_rejections >= 1:
                conditions_met += conditions_met += 1
                analysis_details 1
                analysis_details.append(".append("✅ Güçl✅ Güçlü redü reddetme mevdetme mevcut")
cut")
            
            # Koşul 4            
            # Koşul 4: RSI aşır: RSI aşırı alı alımda mıımda mı?
           ?
            if rsi_value if rsi_value >  > 65:
                conditions65:
                conditions_met_met += 1
 += 1
                analysis                analysis_details.append("✅_details.append("✅ RSI aşırı alım RSI aşırı alım böl bölgesindegesinde")
            
            # Risk/")
            
            # Risk/ÖdÖdül kontrolüül kontrolü
            if strongest_support:
                potential
            if strongest_support:
                potential_profit = current_price_profit = current_price - - strongest_support['price strongest_support['price']
']
                potential_loss = strongest                potential_loss = strongest_res_resistance['price'] *istance['price'] *  1.02 - current_price1.02 - current_price  # %2 stop loss  # %2 stop loss
                rr_ratio
                rr_ratio = potential_profit / = potential_profit / potential_loss if potential potential_loss if potential_loss >_loss > 0 else  0 else 0
0
                
                analysis_details                
                analysis_details.append(f"📊 Risk/Ö.append(f"📊 Risk/Ödül: {dül: {rr_rr_ratio:.2f}")
                
                if rr_ratio:.2f}")
                
                ifratio rr_ratio >= min_rr_ >= min_rr_ratio:
                    conditions_metratio:
                    conditions_met +=  += 1
                    analysis1
                    analysis_details.append_details.append("✅ Risk/("✅ Risk/ÖdÖdül uygunül uygun")
            
")
            
            # Siny            # Sinyal kararı
al kararı
            success            success_rate = conditions__rate = conditions_met /met / total_conditions
 total_conditions
            if            if success_rate success_rate >= 0 >= 0.6:
                stop.6:
                stop_loss =_loss = strongest_resistance[' strongest_resistance['price'] *price'] * 1 1.02
.02
                take_profit                take_profit = = current current_price -_price - (stop_loss - current (stop_loss - current_price) *_price) * min_rr_ratio
                
                signals.append({
                    'type': min_rr_ratio
                
                signals.append({
                    'type': 'SELL',
                    ' 'SELL',
                    'price': current_price,
                    'price': current_price,
                    'stop_loss': stop_loss,
stop_loss': stop_loss,
                    'take_profit': take_pro                    'take_profit': take_profit,
                    'confidencefit,
                    'confidence': success_rate,
                    'reason': success_rate,
                    'reason': f"Direnç bölges': f"Direnç bölgesinde düşüş tepkinde düşüş tepkisi - Güven: %isi - Güven: %{success_rate*100{success_rate*100:.:.0f}"
                })
           0f}"
                })
            else else:
                analysis_details.append:
                analysis_details.append("("❌ SATIM:❌ SATIM: Y Yetersiz koşul - BEKLE")
        
        else:
           etersiz koşul - BEKLE")
        
        else:
            analysis_details analysis_details.append("🎭 NET.append("🎭 NET S SİNYAL YİNYAL YOKOK - Piyasa göz - Piyasa gözlelemi önerilirmi önerilir")
            
            # EMA'")
            
            # EMA'ya uzaklıkya uzaklık kontrolü kontrolü
            if distance_to
            if distance_to__ema > 5ema > 5: :  # %5' # %5'ten fazten fazla uzaksa
                analysisla uzaksa
                analysis_details.append("⚠️ F_details.append("⚠️ Fiyat EMA'dan çiyat EMA'dan çok uzak - Risk yok uzak - Risk yüksek")
        
        returnüksek")
        
        return signals, analysis_details
        
    signals, analysis_details
        
    except Exception as e except Exception as e:
       :
        st.error(f"Sinyal st.error(f"Sinyal üretim hatası: üretim hatası: {e}")
        return [], []

# Gra {e}")
        return [], []

# Grafikte destek/dfikte destek/dirençirenç gösterimi için gösterimi için optimize optimize fonks fonksiyiyon
def add_zones_to_chon
def add_zones_to_chartart(fig, support(fig, support_zones, resistance_zones):
   _zones, resistance_zones):
    """Grafiğe """Grafiğe destek ve diren destek ve direnç bölgelerini ekç bölgelerini ekle (optimize edle (optimize edilmiş)"""
    
   ilmiş)"""
    
    # Destek bölg # Destek bölgeleri - Yeleri - Yatay çizgileratay çizgiler yer yerine alanlar
    for i, zone in enumerateine alanlar
    for i, zone in enumerate(support_zones[:3]):
        # Şeff(support_zones[:3]):
        # Şeffafaf yeşil alan
        fig.add_h yeşil alan
        fig.add_hrect(
            y0rect(
            y0==zone['start'], y1zone['start'], y1=zone['end'],
           =zone['end'],
            fillcolor="green", opacity= fillcolor="green", opacity=00.2,
            line.2,
            line_width_width=0,
            annotation=0,
            annotation_text_text=f"S{i=f"S{i++1}",
            annotation1}",
            annotation_position="_position="top left"
top left"
        )
        # İnce        )
        # İnce ye yeşil çizşil çizgi
gi
        fig.add_        fig.add_hline(
hline(
            y=zone            y=zone['price['price'],
            line'],
            line_dash_dash="solid",
           ="solid",
            line_color line_color="green",
           ="green",
            line_width line_width=1,
           =1,
            opacity= opacity=0.7
0.7
        )
        )
    
    # D    
    # Dirençirenç bölgeler bölgeleri -i - Yatay ç Yatay çizgizgiler yerine aliler yerine alanlar
    for i, zone in enumerate(resistance_zonesanlar
    for i, zone in enumerate(resistance_zones[:3]):
[:3]):
        # Şeffaf kırmız        # Şeffaf kırmızı alan
        figı alan
        fig.add_hrect(
            y.add_hrect(
            y0=zone['0=zone['start'], y1=zone['endstart'], y1=zone['end'],
            fillcolor="red", opacity=0.2'],
            fillcolor="red", opacity=0.2,
           ,
            line_width=0 line_width=0,
,
            annotation_text=f"            annotation_text=f"R{i+R{i+1}",
           1}",
            annotation_position annotation_position="top right"
="top right"
        )
        )
        # İnce        # İnce kı kırmızı çrmızı çizgiizgi
        fig
        fig.add_hline.add_hline(
            y(
            y=zone['price'],
=zone['price'],
            line_dash="            line_dash="solid",
            line_color="solid",
            line_color="red",
            linered",
            line_width=1,
           _width=1,
            opacity= opacity=0.70.7
        )
    
   
        )
    
    return fig

# return fig

# Ana uygul Ana uygulama
def main():
    # Verama
def main():
    # Veri yükleme
i yükleme
       with st.spinner(f with st.spinner(f'⏳'⏳ {c {crypto_symbol} içinrypto_symbol} için 4 4 saatlik ver saatlik veriler yükleniler yükleniyoriyor...'):
       ...'):
        data = get_ data = get_4h_data4h_data(crypto_symbol(crypto_symbol, look, lookback_perback_period)
iod)
    
    if data    
    if data is None or is None or data.empty data.empty:
        st.error:
        st.error(f"(f"❌ {❌ {crypto_symbol} için vercrypto_symbol} için veri yükleni yüklenemediemedi!")
        st.info!")
        st.info("("💡 Lütfen💡 Lütfen geç geçerli bir kerli bir kriptoripto sembolü g sembolü girin (irin (ÖrÖrnek: BTC-nek: BTC-USD,USD, ETH-USD, ETH-USD, XRP XRP-USD)")
       -USD)")
        return
    
 return
    
    st.success(f    st.success(f"✅"✅ {crypto_symbol {crypto_symbol} için {} için {len(data)}len(data)} adet adet 4 saatlik 4 saatlik mum ver mum verisi yüklisi yüklendi")
endi")
    
    # Gö    
    # Göstergstergeleri hesapla
eleri hesapla
    data = calculate    data = calculate_indicators_indicators(data, ema_period(data, ema_period, rsi, rsi_period)
    
_period)
    
    # Yo    # Yoğunluk bölgelerini bul
    supportğunluk bölgelerini_zones, resistance_z bul
    support_zones, resistanceones = find_cong_zones = find_congestion_zones(data, min_testion_zones(data, min_touch_points=min_touch_pointsouch_points=min_touch_points)
    
    # S)
    
    # Sinyinyal üret
   al üret
    signals signals, analysis_details =, analysis_details = generate generate_trading_signals(
       _trading_signals(
        data data, support_zones,, support_zones, resistance resistance_zones,_zones, em ema_period, riska_period, risk_re_reward_ratio
   ward_ratio
    )
    
 )
    
       # # Mevcut durum Mevcut durum

    current_price = float(data    current_price = float(data['['Close'].iloc[-1])
    ema_value = float(data['EMA'].iloc[-1])
    rsiClose'].iloc[-1])
    ema_value = float(data['EMA'].iloc[-1])
    rsi_value_value = float(data['RSI'].il = float(data['RSI'].iloc[-oc[-1])
    
    # Layout1])
    
    # Layout
   
    col1, col2 col1, col2 = = st.columns([3, st.columns([3,  1])
    
    with1])
    
    with col col1:
        st1:
        st.subheader.subheader(f"📈(f"📈 {c {crypto_symbol} -rypto_symbol} - 4 4 Saatlik Grafik Saatlik Grafik Anal Analizi")
        
       izi")
        
        # Gra # Grafik olufik oluşturştur
        fig
        fig = go = go..Figure()
        
        #Figure()
        
        # Mum gra Mum grafiği
        figfiği
        fig.add.add_trace(go_trace(go.Cand.Candlestick(
            xlestick(
            x=data=data.index,
            open=data['Open.index,
            open=data['Open'],
           '],
            high=data['High'],
            high=data['High'],
            low=data[' low=data['Low'],
            close=data['CloseLow'],
            close=data['Close'],
           '],
            name='Price'
        ))
 name='Price'
        ))
        
        # EMA        
        # EMA
        fig.add_trace
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['EMA'],
            name=f(go.Scatter(
            x=data.index,
            y=data['EMA'],
            name=f''EMA {ema_periodEMA {ema_period}',
            line=}',
            line=dictdict(color='orange', width(color='orange', width=2)
=2)
        ))
        
        # Optimize ed        ))
        
        # Optimize edilmiş destilmiş destek/direnek/direnç bölgç bölgelerieleri
        fig =
        fig = add_z add_zones_to_chart(fig, support_zones_to_chart(fig, support_zonesones, resistance_zones)
, resistance_zones)
        
        
        fig.update_layout(
        fig.update_layout(
                       height= height=600,
            title=f"{c600,
            title=f"{crypto_symbol}rypto_symbol} - 4 - 4 Saatlik Prof Saatlik Profesyonelesyonel Analiz",
 Analiz",
            xaxis_title            xaxis_title="Tari="Tarih",
            yaxis_titleh",
            yaxis_title="Fiyat="Fiyat (USD)",
            show (USD)",
            showlegend=Truelegend=True,
            xaxis_r,
            xaxis_rangeslangeslider_visible=Falseider_visible=False
       
        )
        
        st )
        
        st.plotly.plotly_chart(fig_chart(fig, use, use_container_width=True)
_container_width=True)
    
       
    with col2:
 with col2:
        st        st.subheader(".subheader("🎯 TR🎯 TRADING SADING SİİNYALLERİNYALLERİ")
        
")
        
        if signals:
            for        if signals:
            for signal in signals:
                if signal['type'] signal in signals:
                if signal['type'] == 'BUY':
 == 'BUY':
                    st.success(f"""
                    **✅ ALIM                    st.success(f"""
                    **✅ ALIM SİNYALİ**
 SİNYALİ**
                                       - Giriş: - Giriş: {format {format_price(signal['price_price(signal['price'])}
'])}
                    - Stop: {format_price                    - Stop: {format_price(s(signal['stopignal['stop_loss'])}
_loss'])}
                    - TP:                    - TP: {format {format_price(s_price(signal['take_proignal['take_profit'])}
                   fit'])}
                    - Gü - Güven: %{ven: %{signal['signal['confidence']*100confidence']*100:.0f}
                   :.0f}
                    """)
                """)
                else:
                    st else:
                    st.error.error(f"""
                    **(f"""
                    **❌ SATIM SİNY❌ SATIM SİNYALİ**
                    - GirişALİ**
                    - G: {format_price(signal['price'])}
                    -iriş: {format_price(signal['price'])}
                    - Stop: { Stop: {formatformat_price(signal['stop_loss_price(signal['stop_loss'])}
'])}
                    - TP:                    - TP: {format {format_price(signal['_price(signal['take_profittake_profit']'])}
                    -)}
                    - Güven: %{ Güven: %{signal['signal['confidence']confidence']*100:.0f*100:.0f}
                    """)
        else}
                    """)
        else:
            st:
            st.info("".info("""
            **"
            **🎭🎭 NET SİNY NET SİNYAL YAL YOK**
            -OK**
            - Piy Piyasa gözleasa gözlemimi ö önernerilir
ilir
            - Koşullar            - Koşullar uygun uygun değil
            - BE değil
            - BEKLE strKLE stratejisiatejisi u uygula
            """)
        
       ygula
            """)
 st.subheader("📊 MEVCUT D        
        st.subheader("📊 MEVCUT DURUM")
        st.metric("Fiyat", format_price(currentURUM")
        st.metric("Fiyat", format_price(current_price))
_price))
               st.metric st.metric(f"EMA {ema_period}",(f"EMA {ema_period}", format format_price(ema_value))
_price(ema_value))
        st.metric        st.metric("RS("RSI", f"{I", f"{rsi_value:.rsi_value:.1f1f}")
        
        trend}")
        
        trend = "Y = "YÜKSÜKSELİŞ" if current_priceELİŞ" if current_price > em > ema_value elsea_value else " "DÜŞÜŞ"
DÜŞÜŞ"
        st.metric("        st.metric("TRENDTREND", trend)
        
", trend)
        
        #        # Destek/D Destek/Dirençirenç Listesi
        Listesi
        st.sub st.subheader("💎header("💎 SEV SEVİYELERİYELER")
        
")
        
        st.write("        st.write("****🟢 DEST🟢 DESTEK:**EK:**")
        for")
        for i i, zone in enumerate(support_zones[:3]):
            st.write(f, zone in enumerate(support_zones[:3]):
            st.write(f"S{i+1}:"S{i+1}: {format_price(zone[' {format_price(zone['price'])}")
price'])}")
        
        st.write        
        st.write("**("**🔴 DİREN🔴 DİRENÇ:**")
        forÇ:**")
        for i i, zone in enumerate(resistance_zones[:3, zone in enumerate(resistance_zones[:3]):
            st]):
            st.write.write(f"(f"R{i+R{i+1}: {format_price(1}: {format_price(zone['price'])}zone['price'])}")
    
    #")
    
    # Detay Detaylı analizlı analiz

    st.subheader    st.subheader("🔍 DETAYLI ANALİZ R("🔍 DETAYLI ANALİZ RAPORAPORUU")
    with st.expander("Analiz")
    with st.expander("Analiz Detayları", expanded Detayları", expanded=True):
        for detail in analysis_details:
           =True):
        for detail in analysis_details if "✅" in detail:
:
            if "✅" in                st.success(detail)
 detail:
                st.success(detail)
            elif "❌            elif "❌"" in detail or "⚠️ in detail or "⚠️" in detail:
                st.error(d" in detail:
                st.error(detail)
etail)
            elif "🎯" in detail            elif "🎯" in detail or " or "🎪" in🎪" in detail detail:
                st.warning:
                st.warning(d(detail)
            else:
               etail)
            else:
                st.info(detail)

if st.info(detail)

if __name__ == "__main __name__ == "__main__":
    main()