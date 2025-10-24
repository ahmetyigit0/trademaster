import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional

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
            crypto_symbol = "BTC-USD"
        if st.button("ETH-USD", use_container_width=True):
            crypto_symbol = "ETH-USD"
    with col2:
        if st.button("ADA-USD", use_container_width=True):
            crypto_symbol = "ADA-USD"
        if st.button("XRP-USD", use_container_width=True):
            crypto_symbol = "XRP-USD"
    
    st.subheader("📊 Parametreler")
    ema_period = st.slider("EMA Period", 20, 100, 50)
    rsi_period = st.slider("RSI Period", 5, 21, 14)
    min_touch_points = st.slider("Minimum Temas Noktası", 2, 5, 3)
    risk_reward_ratio = st.slider("Min Risk/Ödül Oranı", 1.0, 3.0, 1.5)
    analysis_lookback_bars = st.slider("Analiz Lookback (Bars)", 80, 200, 120)

# =============================================================================
# YENİ YARDIMCI FONKSİYONLAR
# =============================================================================

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range (ATR) hesaplar
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr

class Zone:
    """Destek/Direnç Bölgesi Sınıfı"""
    def __init__(self, low: float, high: float, touches: int, last_touch_ts: Any, 
                 kind: str = "support"):
        self.low = low
        self.high = high
        self.touches = touches
        self.last_touch_ts = last_touch_ts
        self.kind = kind
        self.score = 0
        self.status = "valid"  # valid, fake, broken
        
    def __repr__(self):
        return f"Zone({self.kind}, low={self.low:.4f}, high={self.high:.4f}, touches={self.touches})"

def build_zones(df: pd.DataFrame, min_touch_points: int, lookback: int = 120) -> List[Zone]:
    """
    Yoğunluk tabanlı destek/direnç bölgeleri oluşturur
    """
    if len(df) < lookback:
        lookback = len(df)
    
    data = df.tail(lookback).copy()
    current_price = float(data['Close'].iloc[-1])
    
    # ATR hesapla ve bin genişliğini belirle
    atr = compute_atr(data).iloc[-1] if len(data) > 14 else current_price * 0.02
    bin_width = max(0.25 * atr, current_price * 0.0015)
    
    # Tüm fiyat noktalarını topla (kapanış + high + low)
    price_levels = []
    for i in range(len(data)):
        try:
            price_levels.extend([
                float(data['Close'].iloc[i]),
                float(data['High'].iloc[i]),
                float(data['Low'].iloc[i])
            ])
        except (ValueError, IndexError):
            continue
    
    if not price_levels:
        return []
    
    price_levels = sorted(price_levels)
    min_price, max_price = min(price_levels), max(price_levels)
    
    # Histogram oluştur
    bins = {}
    current_bin = min_price
    
    while current_bin <= max_price:
        bin_end = current_bin + bin_width
        count = sum(1 for price in price_levels if current_bin <= price <= bin_end)
        if count >= min_touch_points:
            bins[(current_bin, bin_end)] = count
        current_bin = bin_end
    
    # Zone'ları oluştur
    zones = []
    for (zone_low, zone_high), touches in bins.items():
        # Son temas zamanını bul
        last_touch_idx = 0
        for i in range(len(data)-1, -1, -1):
            close_price = float(data['Close'].iloc[i])
            high_price = float(data['High'].iloc[i])
            low_price = float(data['Low'].iloc[i])
            
            if (zone_low <= close_price <= zone_high or 
                zone_low <= high_price <= zone_high or 
                zone_low <= low_price <= zone_high):
                last_touch_ts = data.index[i]
                break
        else:
            last_touch_ts = data.index[-1]
        
        # Zone türünü belirle
        kind = "support" if zone_high < current_price else "resistance"
        
        zone = Zone(
            low=zone_low,
            high=zone_high,
            touches=touches,
            last_touch_ts=last_touch_ts,
            kind=kind
        )
        zones.append(zone)
    
    return zones

def eval_fake_breakout(df: pd.DataFrame, zone: Zone) -> Dict[str, Any]:
    """
    Fake kırılım değerlendirmesi yapar
    
    Fake kırılım kuralları:
    - < 2 kapanış bant dışında
    - Taşma mesafesi < 0.5×ATR veya < %0.35
    - ≤ 2 mum içinde banda geri kapanış
    """
    if len(df) < 10:
        return {"status": "valid", "details": "Yetersiz veri"}
    
    data = df.tail(50).copy()  # Son 50 mumu analiz et
    atr = compute_atr(data).iloc[-1] if len(data) > 14 else zone.high * 0.02
    
    # Kırılım istatistikleri
    breakouts = 0
    max_breakout_distance = 0
    reclaim_mums = 0
    
    for i in range(len(data)):
        close_price = float(data['Close'].iloc[i])
        
        if zone.kind == "support":
            # Destek kırılımı (aşağı yönlü)
            if close_price < zone.low:
                breakouts += 1
                distance = zone.low - close_price
                max_breakout_distance = max(max_breakout_distance, distance)
                
                # Geri dönüş kontrolü
                for j in range(i+1, min(i+3, len(data))):
                    if float(data['Close'].iloc[j]) >= zone.low:
                        reclaim_mums = j - i
                        break
        else:
            # Direnç kırılımı (yukarı yönlü)
            if close_price > zone.high:
                breakouts += 1
                distance = close_price - zone.high
                max_breakout_distance = max(max_breakout_distance, distance)
                
                # Geri dönüş kontrolü
                for j in range(i+1, min(i+3, len(data))):
                    if float(data['Close'].iloc[j]) <= zone.high:
                        reclaim_mums = j - i
                        break
    
    # Fake kırılım koşulları
    condition1 = breakouts < 2  # < 2 kapanış bant dışında
    condition2 = max_breakout_distance < 0.5 * atr or max_breakout_distance < zone.high * 0.0035
    condition3 = reclaim_mums <= 2 and reclaim_mums > 0  # ≤ 2 mumda geri dönüş
    
    fake_conditions = [condition1, condition2, condition3]
    fake_score = sum(fake_conditions)
    
    # Kalıcı kırılım koşulları
    permanent_conditions = [
        breakouts >= 2,
        max_breakout_distance >= 0.5 * atr,
        reclaim_mums == 0 or reclaim_mums > 2
    ]
    permanent_score = sum(permanent_conditions)
    
    # Durum belirleme
    if fake_score >= 2:
        status = "fake"
        details = f"Fake kırılım: {breakouts} kırılım, {max_breakout_distance:.4f} mesafe, {reclaim_mums} mumda geri dönüş"
    elif permanent_score >= 2:
        status = "broken"
        details = f"Kalıcı kırılım: {breakouts} kırılım, {max_breakout_distance:.4f} mesafe"
    else:
        status = "valid"
        details = f"Normal bölge: {breakouts} kırılım, {max_breakout_distance:.4f} mesafe"
    
    return {"status": status, "details": details, "breakouts": breakouts, 
            "max_distance": max_breakout_distance, "reclaim_mums": reclaim_mums}

def compute_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD göstergesi hesaplar
    """
    exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    
    return macd, signal_line, histogram

def score_zone(df: pd.DataFrame, zone: Zone, ema: float, rsi: float, atr: float) -> int:
    """
    Bölge çalışırlık skoru hesaplar (0-100)
    """
    score = 0
    current_price = float(df['Close'].iloc[-1])
    
    # 1. Temas/"Saygı" Puanı (0-30)
    touches_score = min(zone.touches * 3, 30)  # Her temas +3 puan
    score += touches_score
    
    # 2. Fake/Kalıcı Durum Puanı (0-25)
    fake_result = eval_fake_breakout(df, zone)
    if fake_result["status"] == "fake":
        score += 25  # Fake kırılım güçlü bölge
    elif fake_result["status"] == "valid":
        score += 15  # Normal bölge
    else:  # broken
        score += 0   # Kırılmış bölge
    
    # 3. EMA50 Hizası (0-20)
    if zone.kind == "support":
        ema_distance = abs(zone.high - ema) / atr
        if ema_distance <= 1.0:
            score += 20
        elif ema_distance <= 2.0:
            score += 10
    else:  # resistance
        ema_distance = abs(zone.low - ema) / atr
        if ema_distance <= 1.0:
            score += 20
        elif ema_distance <= 2.0:
            score += 10
    
    # 4. RSI Uyumu (0-15)
    if zone.kind == "support" and rsi < 40:
        score += 15
    elif 15
    elif zone.k zone.kind == "resistance" andind == "resistance" and r rsi > 60:
si > 60:
        score        score += 15
    elif += 15
    elif 40 <= rsi <= 40 <= rsi <= 60:
        60:
        score score += += 8  8  # Nö # Nötr RSI
    
    # 5. MACtr RSI
    
    # 5. MACD Momentum (0-D Momentum (0-10)
    try:
        macd, signal10)
    try:
        macd, signal, hist = compute_mac, hist = compute_macd(df)
        current_hist =d(df)
        current_hist = hist.il hist.iloc[-1]
oc[-1]
        prev_        prev_hist = hist.iloc[-hist = hist.iloc[-2]
        
       2]
        
        if zone.kind if zone.kind == " == "support"support" and current_ and current_hist > prevhist > prev_hist:
_hist:
            score +=             score += 10
10
        elif zone.k        elif zone.kind == "resind == "resistance" and currentistance" and current_hist < prev_hist < prev_hist:
           _hist:
            score += score += 10
        10
        elif abs(current_hist - prev_ elif abs(current_hist - prev_hist) < hist) < 0.00010.0001:
            score += 5:
            score += 5
   
    except:
 except:
        pass        pass  # MACD  # MACD hesa hesaplanamazsa puan verme
    
    returnplanamazsa puan verme
    
    return min(score, min(score, 100 100)  # Maksimum)  # Maksimum 100 puan

def risk_re 100 puan

def risk_reward(ward(entry: float, slentry: float, sl:: float, tp1: float float, tp1: float, tp, tp2: Optional2: Optional[float][float] = None) -> float = None) -> float:
    """
:
    """
    Risk    Risk/Öd/Ödülül oran oranı hesaplı hesaplar
   ar
    """
    risk = """
    risk = abs(entry abs(entry - sl)
    if - sl)
    if risk risk == 0:
        return == 0:
        return 0 0
    
    reward
    
    reward = abs(t = abs(tp1 - entry)
    if tp2:
        reward = max(rep1 - entry)
    ifward, abs(tp2 - entry tp2:
        reward = max(reward, abs(tp2 - entry))
    
    return reward / risk))
    
    return reward / risk

# =

# =========================================================================================================================================================
# G
# GÜNCELLENÜNCELLENMİMİŞ MEVCUT FŞ MEVCUT FONKSİONKSİYONYONLARLAR
# =============================================================================
# =============================================================================

def format

def format_price(_price(price):
    """Fprice):
    """Fiyatıiyatı uygun form uygun formatta gösteratta göster"""
    if price is"""
    if price is None or None or np.is np.isnannan((price):
        return "N/A"
    
    tryprice):
        return "N/A"
    
    try:
        price:
        price = float(price)
        if price >= 100 = float(price)
        if price >= 1000:
           0:
            return f"${price:,.2f}"
        elif price return f"${price:,.2f}"
        elif price >=  >= 1:
            return f1:
            return f""${price:.3f${price:.3f}"
       }"
        elif price elif price >= 0 >= 0.1:
            return f"${.1:
            return f"${priceprice:.4f:.4f}"
       }"
        elif price >= 0 elif price >= 0.01:
            return f"${.01:
            return f"${price:.5f}"
        else:
price:.5f}"
        else:
            return f"${price            return f"${price:.6f}"
    except (:.6f}"
    except (ValueError,ValueError, TypeError):
        return "N/A"

@st.cache TypeError):
        return "N/A"

@st.cache_data_data
def get_4h_data
def get_4h_data(symbol, days=30):
(symbol, days=30):
       """4 saatlik """4 saatlik ver veri çeker -i çeker - analiz analiz için için 30 gün"""
    30 gün"""
    try:
 try:
        symbol        symbol = symbol. = symbol.upper().strip()
        if '-' not inupper().strip()
        if '-' not in symbol:
            symbol = symbol + symbol:
            symbol = symbol + '-USD'
        
        data = '-USD'
        
        data = y yf.download(symbol, periodf.download(symbol, period=f"{days}d",=f"{days}d", interval interval="4h", progress=False="4h", progress=False)
        
        if data.empty or len)
        
        if data.empty or len(data) == 0:
(data) == 0:
            st.error(f"            st.error(f"❌ {❌ {symbol} için veri bulsymbol} için veri bulunamadı!")
           unamadı!")
            return None
            
        return data
    except Exception return None
            
        return data
    except Exception as e:
        st.error as e:
        st.error(f"❌ {symbol} veri çekilemedi: {e}")
        return None

def(f"❌ {symbol} veri çekilemedi: {e}")
        return None

 calculate_indicators(data, emadef calculate_indicators(data,_period=50, ema_period=50, r rsi_period=14):
si_period=14):
    """Teknik göstergeler    """Teknik göstergeleri hesaplar"""
i hesaplar"""
       if data is None or len(data if data is None or len(data) == 0:
        return) == 0:
        return data data
        
    df = data.copy
        
    df = data.copy()
()
    
    # EMA
    df    
    # EMA
    df['['EMA'] = df['Close'].EMA'] = df['Close'].ewm(span=ema_periodewm(span=ema_period, adjust=False, adjust=False).).meanmean()
    
    # RSI
    delta =()
    
    # RSI
    delta = df[' df['Close'].diff()
    gain = (delta.where(delta > 0,Close'].diff()
    gain = (delta.where(delta > 0, 0)). 0)).rolling(window=rsi_period).mean()
rolling(window=rsi_period).mean()
    loss =    loss = (-delta.where(delta (-delta.where(delta <  < 0, 0)).rolling(window0, 0)).rolling(window=rsi=rsi_period).mean()
   _period).mean()
    rs = rs = gain / gain / loss
 loss
    df['RSI']    df['RSI'] =  = 100 - (100 / (100 - (100 / (1 +1 + rs))
    
    # rs))
    
    # A ATR
    df['ATTR
    df['ATR']R'] = compute_atr(df = compute_atr(df)
)
    
    return df

def    
    return df

def find_c find_congestion_zones(data,ongestion_zones(data, min_t min_touch_points=3,ouch_points=3, lookback= lookback=120):
    """120):
    """buildbuild_zones'u_zones'u kullanarak destek/diren kullanarak destek/direnç bölgç bölgeleri bulureleri bulur"""
    if data"""
    if data is None or len(data) == 0 is None or len(data) == 0:
:
        return [], []
    
           return [], []
    
    zones = zones = build_zones(data build_zones(data, min, min_touch_points, lookback_touch_points, lookback)
   )
    current_price = float(data current_price = float(data['Close'].['Close'].iloc[-1iloc[-1])
    
])
    
    support_zones =    support_zones = [zone for zone [zone for zone in in zones if zone.kind == " zones if zone.kind == "support"]
    resistancesupport"]
    resistance_zones = [zone for zone in zones if_zones = [zone for zone in zones if zone.kind zone.kind == "resistance"]
 == "resistance"]
    
       
    # Bölg # Bölgeleri skoreleri skorla
    emla
    ema_value = floata_value = float(data['EMA'].iloc[-1])
    rsi_value = float(data['RSI'].iloc[-1(data['EMA'].iloc[-1])
    rsi_value = float(data['RSI'].iloc[-])
    atr_value = float1])
    atr_value = float(data['(data['ATR'].iloc[-1])
    
ATR'].iloc[-1])
    
    for zone    for zone in support_zones + resistance_zones in support_zones + resistance_zones:
       :
        zone.score = score_zone zone.score = score_zone(data,(data, zone, ema_value, zone, ema_value, rsi rsi_value, atr_value, atr_value)
        fake_result = eval_f_value)
        fake_result = eval_fake_breakout(data,ake_breakout(data, zone)
        zone zone)
        zone.status = fake_result.status = fake_result["status["status"]
    
    #"]
    
    # Skora gö Skora görere s sıırala ve en iyi 3rala ve en iyi 3'ü al
    support_zones'ü al
    support_zones = sorted(support_zones, = sorted(support_zones, key=lambda x: key=lambda x: x.score, reverse x.score, reverse=True)[:=True)[:3]
    resistance_z3]
    resistance_zones = sortedones = sorted(resistance_z(resistance_zones, key=lambda xones, key=lambda x: x.score: x.score, reverse=True, reverse=True)[:3)[:3]
    
   ]
    
    return support_zones, resistance_zones return support_zones, resistance_zones

def generate_trading

def generate_trading_signals(data, support_zones,_signals(data, support_zones, resistance_zones, em resistance_zones, ema_period=50a_period=50, min, min_rr_ratio=1_rr_ratio=1.5.5):
    """
    Geli):
    """
    Geliştirştirilmiş trading sinyilmiş trading sinyallalleri üretir
eri üretir
    """
    """
    signals = []
    analysis    signals = []
    analysis_details = []
    
    if_details = []
    
    if data data is None or len(data) is None or len(data) < ema < ema_period + _period + 10:
        analysis_details.append10:
        analysis_details.append("("❌ Yetersiz veri❌ Yetersiz veri - analiz y - analiz yapılamıapılamıyor")
        returnyor")
        return signals, analysis_details
    
    signals, analysis_details
    
    try:
        current try:
        current_price = float(data['_price = float(data['Close'].ilClose'].iloc[-1])
        ema_value = floatoc[-1])
        ema_value = float(data['EMA'].(data['EMA'].iloc[-1iloc[-1])
        rsi])
        rsi_value = float_value = float(data['RSI(data['RSI'].iloc'].iloc[-1])
       [-1])
        atr_value = atr_value = float(data['ATR float(data['ATR'].iloc'].iloc[-1])
[-1])
        
        #        
        # Trend analizi Trend analizi
        trend =
        trend = "bull" if current_price "bull" if current_price > em > ema_value else "beara_value else "bear"
        ema_distance = abs(current_price"
        ema_distance = abs - ema_value) / atr(current_price - ema_value) / atr_value
        
        analysis_value
        
        analysis_details.append(f_details.append(f"📈 TREND"📈 TREND: {': {'YYÜKSELİŞ' ifÜKSELİŞ' if trend == 'bull' trend == 'bull' else ' else 'DÜŞDÜŞÜÜŞ'}")
        analysis_details.append(f"Ş'}")
        analysis_details.append(f"📊📊 EMA{ema_period EMA{ema_period}: {format}: {format_price(ema_value_price(ema_value)}")
       )}")
        analysis_details.append(f" analysis_details.append(f"📍 F📍 Fiyat-EMAiyat-EMA Mesaf Mesafesi: {ema_distance:.esi: {ema_distance:.2f2f}} ATR")
        analysis_details.append(f ATR")
        analysis_details.append(f"📉 R"📉 RSI: {rsi_value:.1SI: {rsi_value:.1f}")
        analysisf}")
        analysis_details.append(f"📏 ATR: {_details.append(f"📏 ATR: {format_priceformat_price(atr_value)}(atr_value)}")
        
        # En iyi bölg")
        
        # En iyi bölgeleri seç
        besteleri seç
        best_support = support_zones_support = support_zones[0] if support_zones else[0] if support_zones else None None
        best_resistance = resistance
        best_resistance = resistance_zones[_zones[0] if0] if resistance_zones else resistance_zones else None
        
        # AL None
        
        # ALIM sinIM sinyali kontrolüyali kontrolü
       
        if best_support and best if best_support and best_support.score >= 65:
           _support.score >= 65:
            # G # Giriiriş, SL,ş, SL, TP he TP hesapla
            entry = min(current_price,sapla
            entry = min(current_price, best_support.high best_support.high)
            sl = best_support)
            sl = best_support.low -.low - 0.25 0.25 * atr_value * atr_value
            
            # TP1: Son yerel
            
            # TP1: Son yerel tepe veya bir tepe veya bir üst bölge
            tp üst bölge
            tp1 =1 = best_resistance.l best_resistance.low if best_resow if best_resistance else current_priceistance else current_price + 2 + 2 * (current_price - * (current_price - sl)
 sl)
            # TP2:            # TP2: Bir üst Bir üst bölge ( bölge (varsa)
varsa)
            tp2 =            tp2 = None
 None
            if len(resistance_z            if len(resistance_zones)ones) > 1:
                > 1:
                tp2 = tp2 = resistance_zones resistance_zones[1].[1].low if resistance_zoneslow if resistance_zones[1[1]] else None
            
            rr else None
            
            rr = risk_reward(entry = risk_reward(entry, sl, tp1, tp, sl, tp1, tp2)
            
            if rr >= min_r2)
            
            if rr >= min_rr_ratio:
r_ratio:
                explain                explain = [
                    f" = [
                    f"EMA50EMA50 trend: {tre trend: {trendnd.upper()}",
                    f".upper()}",
                    f"Zone validity: {best_support.status} -Zone validity: {best_support.status} - {eval_fake_breakout {eval_fake_breakout(data(data, best_support, best_support)['details']}",
                   )['details']}",
                    f"RSI/MAC f"RSI/MACD teyitD teyit: RSI {: RSI {rsi_valuersi_value:.1f}:.1f} - Support - Support b bölölgesinde",
gesinde",
                    f"RR                    f"RR kontrolü: {rr:.2 kontrolü: {rr:.2f} ≥ {f} ≥ {min_rr_ratiomin_rr_ratio}"
                ]
                
               }"
                ]
                
                signals.append signals.append({
                    "type":({
                    "type": "BUY "BUY",
                    "entry": entry,
",
                    "entry": entry,
                    "sl":                    "sl": sl,
                    "tp1": sl,
                    "tp1": tp1,
                    tp1,
                    "tp2": tp2,
 "tp2": tp2,
                    "rr":                    "rr": rr,
                    "confidence": best_s rr,
                    "confidence": best_support.score,
                    "zoneupport.score,
                    "zone": {"low": best_s": {"low": best_support.lowupport.low, "high": best, "high": best_support_support.high, "kind":.high, "kind": "support"},
                    "tre "support"},
                    "trend":nd": trend,
                    " trend,
                    "explainexplain": explain
                })
        
        #": explain
                })
        
        # SATIM SATIM sinyali kontrolü
 sinyali kontrolü
        elif        elif best_resistance and best best_resistance and best_resistance.score_resistance.score >= 65:
            # >= 65:
            # Giri Giriş, SL, TP heş, SL, TP hesaplasapla
           
            entry = max(current_price entry = max(current_price, best_resistance.l, best_resistance.low)
            sl = bestow)
            sl = best_resistance.high_resistance.high + 0.25 * + 0.25 * atr atr_value
            
_value
            
            # TP1            # TP1: Son yerel dip v: Son yerel dip veya bireya bir alt bölge
 alt bölge
            tp1 =            tp1 = best_support.high if best_support best_support.high if best_support else current_price - 2 * (sl else current_price - 2 * (sl - current_price)
            # - current_price)
            # TP2: Bir alt bölge TP2: Bir alt bölge (varsa (varsa)
            tp2 =)
            tp2 = None
            if len(support_zones) None
            if len(support_zones) > 1:
                tp2 = support_zones > 1:
                tp2 = support_zones[1].high if support_zones[1] else None
            
            rr = risk_reward(entry,[1].high if support_zones[1] else None sl, tp1, tp
            
            rr = risk_reward(entry, sl, tp1, tp22)
            
            if rr >=)
            
            if rr >= min_r min_rr_ratio:
               r_ratio:
                explain = explain = [
                    [
                    f" f"EMA50 trend: {EMA50 trend: {trend.uppertrend.upper()}",
                    f"Zone validity()}",
                    f"Zone validity: {best_res: {best_resistance.status}istance.status} - {eval - {eval_fake_breakout(data_fake_breakout(data, best, best_resistance)['details_resistance)['details']}",
']}",
                    f"RS                    f"RSII/MAC/MACD teyit: RD teyit: RSI {rsiSI {rsi_value:.1f}_value:.1f} - Resistance bölgesinde - Resistance bölgesinde", 
                    f"", 
                    f"RR kontrolü: {rrRR kontrolü: {rr:.2f} ≥ {:.2f} ≥ {min_rr_ratiomin_rr_ratio}"
                ]
                
                signals}"
                ]
                
                signals.append({
                    "type":.append({
                    "type": "SELL",
                    " "SELL",
                    "entry": entry,
                    "entry": entry,
                    "sl": slsl": sl,
                   ,
                    "tp1": "tp1": tp1,
 tp1,
                    "tp2":                    "tp2": tp2 tp2,
                    "rr":,
                    "rr": rr rr,
                   ,
                    "confidence": best_resistance.score,
                    " "confidence": best_resistance.score,
                    "zone": {"low": bestzone": {"low": best_resistance_resistance.low, ".low, "high": besthigh": best_resistance.high_resistance.high, ", "kind": "resistance"},
                    "kind": "resistance"},
                    "trend":trend": trend,
                    "expl trend,
                    "explain":ain": explain
                })
        
 explain
                })
        
        #        # BEKLE durumu
 BEKLE durumu
        if        if not signals:
            wait not signals:
            wait_reasons_reasons = []
 = []
            if not best_support            if not best_support and not best_res and not best_resistance:
                wait_reasons.append("istance:
                wait_reasons.append("YeterliYeterli bölge bölge bulunamadı")
            elif best bulunamadı")
            elif best_s_support and best_support.score < 65:
upport and best_support.score < 65:
                wait_reasons.append(f"Destek skoru                wait_reasons.append(f"Destek skoru yetersiz: {best_s yetersiz: {best_support.score}")
            elif best_resupport.score}")
            elif bestistance and best_resistance_resistance and best_resistance.score < 65:
                wait.score < 65:
                wait_reasons.append(f"Direnç_reasons.append(f"Direnç skoru yetersiz: {best_resistance.score}")
            elif skoru yetersiz: {best_resistance.score}")
            elif ema_distance > 1 ema_distance > 1.0:
                wait_reasons.append(f".0:
                wait_reasons.append(f"EMA'dan uzakEMA'dan uzak: {ema: {ema_distance:._distance:.2f2f} ATR")
            
            signals.append} ATR")
            
            signals.append({
                "type({
                "type": "WAIT": "WAIT",
                "",
                "entry": current_priceentry": current_price,
                "sl,
                "sl": None,
": None,
                "tp1":                "tp1": None,
                None,
                "tp2": "tp2": None,
                " None,
                "rr": 0rr": 0,
               ,
                "confidence": max(( "confidence": max((best_sbest_support.scoreupport.score if best_support if best_support else else 0), 
                                 0), 
                                 (best_resistance.score if best_resistance (best_resistance.score if best_resistance else  else 0)),
                "zone": None,
0)),
                "zone": None,
                "trend": trend,
                "explain":                "trend": trend,
                "explain": wait_reasons
            })
        
 wait_reasons
            })
        
               return signals, analysis_details
        
    except Exception return signals, analysis_details
        
    except Exception as e as e:
        analysis_details.append(f"❌ S:
        analysis_details.append(f"❌ Sinyalinyal üretim hatası: {e}")
 üretim hatası: {e}")
        return [], analysis_details

def create_candlest        return [], analysis_details

def create_candlestick_chart_manual(data,ick_chart_manual(data, support support_zones, resistance_zones, crypto_zones, resistance_zones, crypto_symbol):
    """Geli_symbol):
    """Geliştirilmiş mum grafiğiştirilmiş mum grafiği - bant çiz - bant çizimimleri ve fake kırıleri ve fake kırılım etiketlerilım etiketleri"""
"""
    fig = go    fig = go.Figure()
    
.Figure()
    
    if    if data is None or data is None or len(data) == 0 len(data) == 0:
        return fig:
        return fig
    
    #
    
    # Son 3 g Son 3 gün verisiün verisi (görsel (görselleştirme içinleştirme için)
    data_)
    data_3days3days = data.tail(18) = data.tail(18)  # 3 gün *  # 3 gün * 6 6 mum/gün ≈  mum/gün ≈ 18 mum
    
18 mum
    
    # Mum    # Mumları çları çiziz
   
    for i in range(len(data_ for i in range(len(data_3days3days)):
        try:
           )):
        try:
            row = data_3days.iloc[i]
            open_price = float(row['Open row = data_3days.iloc[i]
            open_price = float(row['Open'])
            high ='])
            high = float(row float(row['High'])
           ['High'])
            low = float(row low = float(row['Low'])
           ['Low'])
            close_price = float(row['Close'])
            
            color = '#00C805' if close_price = float(row['Close'])
            
            color = '#00C805' if close_price close_price > open_price else '#FF > open_price else '#FF00000000'
            
            #'
            
            # Mum gövdes Mum gövdesi
            fig.addi
            fig.add_trace_trace(go.Scatter(
(go.Scatter(
                x                x=[data_3days.index=[data_3days.index[i],[i], data_3days.index[i data_3days.index[i]],
               ]],
                y=[open_price y=[open_price, close_price, close_price],
                mode='lines],
                mode='lines',
',
                line=dict                line=dict(color=(color=color, width=8),
               color, width=8),
                showlegend=False
 showlegend=False
            ))
            
                       ))
            
            # Üst iğne # Üst iğne
            fig.add_trace
            fig.add_trace((gogo.Sc.Scatteratter(
               (
                x=[data_3days.index x=[data_3days.index[i],[i], data_3days.index[i]],
 data_3days.index[i]],
                y=[                y=[max(open_price, close_pricemax(open_price, close_price), high],
), high],
                mode='lines',
                line=                mode='lines',
                line=dict(color=dict(color=color, width=1.5),
color, width=1.5),
                showlegend=False                showlegend=False
            ))
            
            # Alt i
            ))
            
            # Alt iğne
           ğne
            fig.add_trace(go fig.add_trace(go.Scatter(
.Scatter(
                x=[data_3days.index[i                x=[data_3days.index[i], data_3days], data_3days.index[i]],
                y.index[i]],
                y=[min=[min(open_price, close_price(open_price, close_price), low), low],
                mode='lines],
                mode='lines',
                line',
                line=dict(color=color,=dict(color=color, width=1.5),
                show width=1.5),
                showlegend=False
            ))
       legend=False
            ))
        except except (ValueError, IndexError):
 (ValueError, IndexError):
                       continue
    
    # EMA çiz continue
    
    # EMA çizgisi
    ifgisi
    if ' 'EMA' in data_3EMA' in data_3days.columns:
        try:
days.columns:
        try:
                       fig.add_trace(go.Sc fig.add_trace(go.Scatteratter(
(
                               x=data_3days.index,
                y x=data_3days.index,
                y=data_=data_3days['EMA'],
                name=f3days['EMA'],
                name=f'EMA'EMA {ema_period}',
 {ema_period}',
                               line=dict(color='orange', width=2 line=dict(color='orange', width=2)
)
            ))
        except Exception:
            pass
    
    # Destek BANTLARI
    for            ))
        except Exception:
            pass
    
    # Destek BANTLARI
    for i, zone i, zone in enumerate(support_z in enumerate(support_zones[:ones[:3]):
        try:
3]):
        try:
            # B            # Bant çizimi
           ant çizimi
            fig.add fig.add_h_hrect(
                yrect(
                y0=zone.low,0=zone.low, y1=zone.high,
                y1=zone.high,
                fillcolor="green", opacity=0 fillcolor="green", opacity=0.2,
                line_width=0.2,
                line_width=0,
            )
            #,
            )
            # Or Orta çta çizizgi ve etiket
            fig.add_hlinegi ve etiket
            fig.add_hline(
               (
                y y=(zone.low + zone.high) / =(zone.low + zone.high) / 22,
                line_dash="solid,
                line_dash="solid",
               ",
                line_color="#00FF line_color="#00FF00",
00",
                line_width=2                line_width=2,
               ,
                annotation_text=f"S{i+ annotation_text=f"S{i+1}1} ({zone.status})",
                ({zone.status})",
                annotation_position annotation_position="left",
                annotation="left",
                annotation_font_font_size=10,
                annotation_size=10,
                annotation_font_font_color_color="#="#00FF00"
            )
00FF00"
            )
        except Exception        except Exception:
            continue
    
    # D:
            continue
    
    # Direnç Birenç BANTLARI
   ANTLARI
    for i, for i, zone in enumerate(resistance_zones[:3 zone in enumerate(resistance_zones[:3]):
        try:
]):
        try:
            # Bant çiz            # Bant çizimi
            fig.addimi
            fig.add_hrect(
                y0_hrect(
                y0=zone.l=zone.low, y1ow, y1=zone.high=zone.high,
                fillcolor=",
                fillcolor="red", opacityred", opacity=0.2,
=0.2,
                line                line_width=0,
_width=0,
            )
                       )
            # Orta çizgi # Orta çizgi ve etiket ve etiket
            fig
            fig.add_hline.add_(
                y=(zone.low + zone.hhline(
                y=(zone.low + zone.high) / 2,
igh) / 2,
                line_dash="solid",
                               line_dash="solid",
                line line_color="#FF0000",
_color="#FF0000",
                               line_width=2,
                annotation line_width=2,
                annotation_text_text=f"R{i+1=f"R{i+1} ({} ({zone.status})",
                annotationzone.status})",
                annotation_position_position="right", 
                annotation="right", 
                annotation_f_font_size=10,
               ont_size=10,
                annotation_f annotation_font_color="#FF000ont_color="#FF0000"
            )
        except Exception:
            continue
    
    # Mevcut fiyat çizgisi0"
            )
        except Exception:
            continue
    
    # Mevcut fiyat çizgisi

    try:
        current    try:
        current_price =_price = float(data_3days[' float(data_3days['Close'].Close'].iloc[-iloc[-1])
        fig1])
        fig.add_hline(
           .add_hline(
            y=current_price,
            line_dash="dot",
            line_color=" y=current_price,
            line_dash="dot",
            line_color="yellow",
yellow",
            line_width=2,
            line_width=2,
            opacity=0.7,
                       opacity=0.7,
            annotation_text=f annotation_text=f"Şimdi"Şimdi: {format: {format_price(current_price)}_price(current_price)}",
            annotation_position",
            annotation_position="left",
           ="left",
            annotation_f annotation_font_size=12ont_size=12,
            annotation_font,
            annotation_font_color="yellow_color="yellow"
        )
   "
        )
    except (ValueError except (ValueError, IndexError):
, IndexError):
        pass        pass
    
    # Gra
    
    # Grafik ayarlarıfik ayarları
    fig.update_layout
    fig.update_layout(
        height(
        height=600,
       =600,
        title=f"{ title=f"{crypto_symbol} -crypto_symbol} - 4 4 Saatlik Profesyon Saatlik Profesyonel Analel Analiz (Son 3iz (Son 3 Gün)",
 Gün)",
        xaxis_title="        xaxis_title="TTariarihh",
        yaxis_title="Fiyat",
        yaxis_title="Fiyat (USD (USD)",
        showlegend=True,
        xaxis_r)",
        showlegend=True,
        xaxis_rangesliderangeslider_visible=False,
        plot_bgcolor_visible=False,
        plot_bgcolor='#0E1117',
        paper='#0E1117',
        paper_bgcolor='#0E111_bgcolor='#0E7',
        font1117',
        font=dict(color='white', size=12),
=dict(color='white', size=12),
        xaxis=dict(gridcolor        xaxis=dict(gridcolor='='#444'),
        yaxis#444'),
        yaxis=dict(gridcolor='#444')
=dict(gridcolor='#444')
    )
    
    return fig

#    )
    
    return fig

 =============================================================================
# ANA U# =============================================================================
# ANAYGULAMA
 UYGULAMA
# =# =============================================================================

def main():
============================================================================

def main():
    #    # Veri yükleme Veri yükleme - anal - analiz için 30 giz için 30 gün, görün, görsel için 3sel için 3 gün
 gün
    with st.spinner    with st.spinner(f'(f'⏳ {⏳ {crypto_symbol} için 4 saatlikcrypto_symbol} için 4 saatlik veriler veriler yükleniy yükleniyor...'):
        dataor...'):
        data_30days = get_4h_data(crypto_symbol, days_30days = get_4h_data(crypto_symbol=30)  # Analiz için 30 gün
    
   , days=30)  # Analiz için 30 gün
    
 if data_30days is None or data_30    if data_30days is None or data_30days.empty:
        st.error(fdays.empty:
        st.error(f"❌ {crypto_symbol} için"❌ {crypto_symbol} için veri yüklenemedi veri yüklenemedi!")
!")
        st.info("💡        st.info("💡 Lüt Lütfen geçerfen geçerli bir kripto sembolü girin (Örnekli bir kripto sembolü girin (Örnek: BTC-USD, ETH-USD, X: BTC-USD, ETH-USD, XRP-USD)")
        return
    
RP-USD)")
        return
    
    st.success(f"✅ {    st.success(f"✅ {crypto_symbol} için {crypto_symbol} için {len(data_30days)} adet 4len(data_30days)} adet 4 saatlik mum verisi yü saatlik mum verisi yüklklendi")
    
    # Göendi")
    
    # Göstergeleri hesapla
stergeleri hesapla
    data    data__30days = calculate_indicators(data_30days30days = calculate_indicators(data_30days,, ema_per ema_period,iod, rsi_period)
    
 rsi_period)
    
    #    # Yoğ Yoğunluk bölgelerini bulunluk bölgelerini bul
    support
    support_zones_zones,, resistance_zones = find_c resistance_zones = find_congestion_zongestion_zones(
        data_30days,ones(
        data_30days, min_touch_points min_touch_points, analysis_lookback_bars
    )
    
    # Sinyal üret
   , analysis_lookback_bars
    )
    
    # Sinyal üret
    signals, analysis_details = signals, analysis_details = generate_trading_signals(
        data generate_trading_signals(
        data_30days, support_30days, support_zones, resistance_zones, ema_per_zones, resistance_zones, ema_period, risk_reward_ratioiod, risk_reward_ratio

    )
    
    # Me    )
    
    # Mevcutvcut durum
    try:
 durum
    try:
        current        current_price = float(data_30_price = float(data_30days['Close'].iloc[-1])
        ema_value = float(data_30days['EMA'].iloc[-1])
        rsi_value = float(data_30days['days['Close'].iloc[-1])
        ema_value = float(data_30days['EMA'].iloc[-1])
        rsi_value = float(data_30days['RSIRSI'].il'].iloc[-1])
       oc[-1])
        atr_value = float atr_value = float(data_30(data_30days['ATR'].days['ATR'].ilociloc[-1])
    except ([-1])
    except (ValueErrorValueError, Index, IndexError):
        current_price = 0
        ema_value = 0
        rsi_value = 0
        atrError):
        current_price = 0
        ema_value = 0
        rsi_value = 0
        atr_value = 0
    
   _value = 0
    
    # # Layout
    col1 Layout
    col1, col2 = st.columns([3,, col2 = st.columns([3, 1])
    
    with col 1])
    
    with col1:
        st.subheader1:
        st(f"📈 {crypto_symbol}.subheader(f"📈 {crypto_symbol} - 4 Saatlik Prof - 4 Saatlik Profesesyonel Analiz")
        
       yonel Analiz")
        
        # Mum grafiğini göster
        chart_fig = # Mum grafiğini göster
        chart_f create_candlestick_chart_ig = create_candlestick_chart_manual(data_30days, support_zmanual(data_30days, support_zones, resistance_zones, crypto_symbolones, resistance_zones, crypto_symbol)
        st.plotly_chart)
        st.plotly_chart(chart_fig, use_container(chart_fig, use_container_width=True)
        
_width=True)
        
        st.info("""
        **📊 GRAF        st.info("""
        **📊 GRAFİK AÇIKİK AÇLAMASI:**
        -IKLAMASI:**
        - 🟢 **Yeşil Bantlar 🟢 **Yeşil Bantlar:** Destek Bölgeleri:** Destek Bölgeleri (S1, S2 (S1, S2, S3)
        - 🔴, S3)
        - 🔴 **Kırmız **Kırmızıı Bantlar:** Diren Bantlar:** Dirençç Böl Bölggelerieleri (R1, R (R1, R2, R3)  
2, R3)  
        - 🟠 **Turunc        - 🟠 **Turuncu Çizgi:** EMA50 Trend Göstergesi
        - 🟡 **Sarı Çu Çizgi:** EMA50 Trend Göstergesi
        - 🟡 **Sarı Çizgi:**izgi:** Mevcut Fiyat Mevcut Fiyat
        -
        - 🏷️ ** 🏷️ **EtiketEtiketler:**ler:** Bölge dur Bölge durumu (valid/fake/bumu (valid/fake/broken)
roken)
        """)
    
           """)
    
    with col with col2:
        st2:
        st.subheader.subheader("🎯 TRAD("🎯 TRADING SİING SİNYALLERİNYALLERİ")
        
")
        
        if signals and signals        if signals and signals[0]["[0]["type"] != "type"] != "WAIT":
            signal = signalsWAIT":
            signal = signals[0]
            if signal['[0]
            if signal['typetype'] == ''] == 'BUY':
                st.success(f"""
BUY':
                st.success(f"""
                **                **✅ ALIM SİNYALİ✅ ALIM SİNYALİ**
                - Giriş**
                - Giriş: {format_price(signal['entry: {format_price(signal['entry'])}
                - Stop'])}
                - Stop: {: {format_price(signal['sl']format_price(signal['sl'])}
)}
                - TP1: {format                - TP1: {format_price(signal['tp1_price(signal['tp1'])}
                {f"-'])}
                {f"- TP2: {format_price(s TP2: {format_price(signalignal['tp2'])}"['tp2'])}" if signal['tp2'] else "" if signal['tp2'] else ""}
               }
                - RR: { - RR: {signal['rr']:.2signal['rr']:.2f}
f}
                - Güven: %                - Güven: %{signal['{signal['confidence']}
               confidence']}
                """)
            """)
            else:
                st else:
                st.error(f"""
               .error(f"""
                **❌ SATIM SİNYAL **❌ SATIM SİNYALİ**
                -İ**
                - Giriş Giriş: {: {format_price(signal['entryformat_price(signal['entry'])}
                -'])}
                - Stop: {format Stop: {format_price(signal['_price(signal['sl'])}
               sl'])}
                - TP - TP1: {format_price(signal['tp11: {format_price(signal['tp1'])}
                {f"- TP2: {format_price(signal['tp'])}
                {f"- TP2: {format_price2'])}" if signal['tp(signal['tp2'])}" if signal['tp2']2'] else "" else ""}
                - RR: {}
                - RR: {signal['signal['rr']:.2f}
rr']:.2f}
                -                - Güven: % Güven: %{signal['{signal['confidence']}
confidence']}
                               """)
        else:
            st.info """)
        else:
            st.info("""
            **🎭 BEKLE SİNYALİ("""
            **🎭 BEKLE SİNYAL**
            - Koşullar uyİ**
            - Koşullargun değil
            - uygun değil
            - Piyasa gözlemi Piyasa gözlemi önerilir
 önerilir
            - Yeni sin            - Yeni sinyalyal için bekleyin
            """)
        
        st.subheader("📊 MEVCUT için bekleyin
            """)
        
        st.sub DURUM")
        st.metric("Fiyheader("📊 MEVCUT DURUM")
        st.metric("Fat", format_price(current_price))
        st.metric(f"EMAiyat", format_price(current_price))
        st.metric(f"EMA{ema_period}", format_price({ema_period}", format_price(ema_value))
        st.metricema_value))
        st.metric("RSI", f"{("RSI", f"{rsi_value:.1f}")
        st.mrsi_value:.1f}")
        st.metric("ATR", formatetric("ATR", format_price(atr_value))
        
        trend = "_price(atr_value))
        
        trendYÜKSELİŞ = "YÜKSELİŞ" if current_price > em" if current_price > emaa_value else "DÜŞÜ_value else "DÜŞÜŞ"
        st.metricŞ"
        st.metric("TR("TREND", trend)
        
       END", trend)
