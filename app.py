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

st.title("🎯 4 Saatlik Profesyonel Teknik Analiz")

# Sidebar
with st.sidebar:
    st.header("⚙️ Ayarlar")
    
    crypto_symbol = st.text_input("Kripto Sembolü", "BTC-USD", 
                                 help="Örnek: BTC-USD, ETH-USD, ADA-USD, XRP-USD")
    
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
    
    st.subheader("Parametreler")
    ema_period = st.slider("EMA Period", 20, 100, 50)
    rsi_period = st.slider("RSI Period", 5, 21, 14)
    min_touch_points = st.slider("Min Temas", 2, 5, 3)
    risk_reward_ratio = st.slider("Min R/R", 1.0, 3.0, 1.5)
    analysis_lookback_bars = st.slider("Analiz Bars", 80, 200, 120)

# =============================================================================
# TEMEL FONKSİYONLAR
# =============================================================================

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (ATR) hesaplar"""
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
    """Yoğunluk tabanlı destek/direnç bölgeleri oluşturur"""
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
    """Fake kırılım değerlendirmesi yapar"""
    if len(df) < 10:
        return {"status": "valid", "details": "Yetersiz veri"}
    
    data = df.tail(50).copy()
    atr = compute_atr(data).iloc[-1] if len(data) > 14 else zone.high * 0.02
    
    # Kırılım istatistikleri
    breakouts = 0
    max_breakout_distance = 0
    reclaim_mums = 0
    
    for i in range(len(data)):
        close_price = float(data['Close'].iloc[i])
        
        if zone.kind == "support":
            if close_price < zone.low:
                breakouts += 1
                distance = zone.low - close_price
                max_breakout_distance = max(max_breakout_distance, distance)
                
                for j in range(i+1, min(i+3, len(data))):
                    if float(data['Close'].iloc[j]) >= zone.low:
                        reclaim_mums = j - i
                        break
        else:
            if close_price > zone.high:
                breakouts += 1
                distance = close_price - zone.high
                max_breakout_distance = max(max_breakout_distance, distance)
                
                for j in range(i+1, min(i+3, len(data))):
                    if float(data['Close'].iloc[j]) <= zone.high:
                        reclaim_mums = j - i
                        break
    
    # Fake kırılım koşulları
    condition1 = breakouts < 2
    condition2 = max_breakout_distance < 0.5 * atr or max_breakout_distance < zone.high * 0.0035
    condition3 = reclaim_mums <= 2 and reclaim_mums > 0
    
    fake_score = sum([condition1, condition2, condition3])
    permanent_score = sum([
        breakouts >= 2,
        max_breakout_distance >= 0.5 * atr,
        reclaim_mums == 0 or reclaim_mums > 2
    ])
    
    if fake_score >= 2:
        status = "fake"
        details = f"Fake kırılım: {breakouts} kırılım"
    elif permanent_score >= 2:
        status = "broken"
        details = f"Kalıcı kırılım: {breakouts} kırılım"
    else:
        status = "valid"
        details = f"Normal bölge: {breakouts} kırılım"
    
    return {"status": status, "details": details}

def compute_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD göstergesi hesaplar"""
    exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    
    return macd, signal_line, histogram

def score_zone(df: pd.DataFrame, zone: Zone, ema: float, rsi: float, atr: float) -> int:
    """Bölge çalışırlık skoru hesaplar (0-100)"""
    score = 0
    current_price = float(df['Close'].iloc[-1])
    
    # 1. Temas/"Saygı" Puanı (0-30)
    touches_score = min(zone.touches * 3, 30)
    score += touches_score
    
    # 2. Fake/Kalıcı Durum Puanı (0-25)
    fake_result = eval_fake_breakout(df, zone)
    if fake_result["status"] == "fake":
        score += 25
    elif fake_result["status"] == "valid":
        score += 15
    
    # 3. EMA50 Hizası (0-20)
    if zone.kind == "support":
        ema_distance = abs(zone.high - ema) / atr
        if ema_distance <= 1.0:
            score += 20
        elif ema_distance <= 2.0:
            score += 10
    else:
        ema_distance = abs(zone.low - ema) / atr
        if ema_distance <= 1.0:
            score += 20
        elif ema_distance <= 2.0:
            score += 10
    
    # 4. RSI Uyumu (0-15)
    if zone.kind == "support" and rsi < 40:
        score += 15
    elif zone.kind == "resistance" and rsi > 60:
        score += 15
    elif 40 <= rsi <= 60:
        score += 8
    
    # 5. MACD Momentum (0-10)
    try:
        macd, signal, hist = compute_macd(df)
        current_hist = hist.iloc[-1]
        prev_hist = hist.iloc[-2]
        
        if zone.kind == "support" and current_hist > prev_hist:
            score += 10
        elif zone.kind == "resistance" and current_hist < prev_hist:
            score += 10
        elif abs(current_hist - prev_hist) < 0.0001:
            score += 5
    except:
        pass
    
    return min(score, 100)

def risk_reward(entry: float, sl: float, tp1: float, tp2: Optional[float] = None) -> float:
    """Risk/Ödül oranı hesaplar"""
    risk = abs(entry - sl)
    if risk == 0:
        return 0
    
    reward = abs(tp1 - entry)
    if tp2:
        reward = max(reward, abs(tp2 - entry))
    
    return reward / risk

def format_price(price):
    """Fiyatı uygun formatta göster"""
    if price is None or np.isnan(price):
        return "N/A"
    
    try:
        price = float(price)
        if price >= 1000:
            return f"${price:,.0f}"
        elif price >= 1:
            return f"${price:.2f}"
        elif price >= 0.1:
            return f"${price:.3f}"
        else:
            return f"${price:.4f}"
    except (ValueError, TypeError):
        return "N/A"

@st.cache_data
def get_4h_data(symbol, days=30):
    """4 saatlik veri çeker - analiz için 30 gün"""
    try:
        symbol = symbol.upper().strip()
        if '-' not in symbol:
            symbol = symbol + '-USD'
        
        data = yf.download(symbol, period=f"{days}d", interval="4h", progress=False)
        
        if data.empty or len(data) == 0:
            st.error(f"❌ {symbol} için veri bulunamadı!")
            return None
            
        return data
    except Exception as e:
        st.error(f"❌ {symbol} veri çekilemedi: {e}")
        return None

def calculate_indicators(data, ema_period=50, rsi_period=14):
    """Teknik göstergeleri hesaplar"""
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
    
    # ATR
    df['ATR'] = compute_atr(df)
    
    return df

def find_congestion_zones(data, min_touch_points=3, lookback=120):
    """build_zones'u kullanarak destek/direnç bölgeleri bulur"""
    if data is None or len(data) == 0:
        return [], []
    
    zones = build_zones(data, min_touch_points, lookback)
    current_price = float(data['Close'].iloc[-1])
    
    support_zones = [zone for zone in zones if zone.kind == "support"]
    resistance_zones = [zone for zone in zones if zone.kind == "resistance"]
    
    # Bölgeleri skorla
    ema_value = float(data['EMA'].iloc[-1])
    rsi_value = float(data['RSI'].iloc[-1])
    atr_value = float(data['ATR'].iloc[-1])
    
    for zone in support_zones + resistance_zones:
        zone.score = score_zone(data, zone, ema_value, rsi_value, atr_value)
        fake_result = eval_fake_breakout(data, zone)
        zone.status = fake_result["status"]
    
    # Skora göre sırala ve en iyi 3'ü al
    support_zones = sorted(support_zones, key=lambda x: x.score, reverse=True)[:3]
    resistance_zones = sorted(resistance_zones, key=lambda x: x.score, reverse=True)[:3]
    
    return support_zones, resistance_zones

def generate_trading_signals(data, support_zones, resistance_zones, ema_period=50, min_rr_ratio=1.5):
    """Geliştirilmiş trading sinyalleri üretir"""
    signals = []
    analysis_details = []
    
    if data is None or len(data) < ema_period + 10:
        analysis_details.append("❌ Yetersiz veri - analiz yapılamıyor")
        return signals, analysis_details
    
    try:
        current_price = float(data['Close'].iloc[-1])
        ema_value = float(data['EMA'].iloc[-1])
        rsi_value = float(data['RSI'].iloc[-1])
        atr_value = float(data['ATR'].iloc[-1])
        
        # Trend analizi
        trend = "bull" if current_price > ema_value else "bear"
        ema_distance = abs(current_price - ema_value) / atr_value
        
        analysis_details.append(f"📈 TREND: {'YÜKSELİŞ' if trend == 'bull' else 'DÜŞÜŞ'}")
        analysis_details.append(f"📊 EMA{ema_period}: {format_price(ema_value)}")
        analysis_details.append(f"📍 Fiyat-EMA Mesafesi: {ema_distance:.2f} ATR")
        analysis_details.append(f"📉 RSI: {rsi_value:.1f}")
        analysis_details.append(f"📏 ATR: {format_price(atr_value)}")
        
        # En iyi bölgeleri seç
        best_support = support_zones[0] if support_zones else None
        best_resistance = resistance_zones[0] if resistance_zones else None
        
        # ALIM sinyali kontrolü
        if best_support and best_support.score >= 65:
            entry = min(current_price, best_support.high)
            sl = best_support.low - 0.25 * atr_value
            tp1 = best_resistance.low if best_resistance else current_price + 2 * (current_price - sl)
            tp2 = None
            if len(resistance_zones) > 1:
                tp2 = resistance_zones[1].low if resistance_zones[1] else None
            
            rr = risk_reward(entry, sl, tp1, tp2)
            
            if rr >= min_rr_ratio:
                explain = [
                    f"EMA50 trend: {trend.upper()}",
                    f"Zone validity: {best_support.status}",
                    f"RSI: {rsi_value:.1f} - Support bölgesinde",
                    f"RR: {rr:.2f} ≥ {min_rr_ratio}"
                ]
                
                signals.append({
                    "type": "BUY",
                    "entry": entry,
                    "sl": sl,
                    "tp1": tp1,
                    "tp2": tp2,
                    "rr": rr,
                    "confidence": best_support.score,
                    "zone": {"low": best_support.low, "high": best_support.high, "kind": "support"},
                    "trend": trend,
                    "explain": explain
                })
        
        # SATIM sinyali kontrolü
        elif best_resistance and best_resistance.score >= 65:
            entry = max(current_price, best_resistance.low)
            sl = best_resistance.high + 0.25 * atr_value
            tp1 = best_support.high if best_support else current_price - 2 * (sl - current_price)
            tp2 = None
            if len(support_zones) > 1:
                tp2 = support_zones[1].high if support_zones[1] else None
            
            rr = risk_reward(entry, sl, tp1, tp2)
            
            if rr >= min_rr_ratio:
                explain = [
                    f"EMA50 trend: {trend.upper()}",
                    f"Zone validity: {best_resistance.status}",
                    f"RSI: {rsi_value:.1f} - Resistance bölgesinde", 
                    f"RR: {rr:.2f} ≥ {min_rr_ratio}"
                ]
                
                signals.append({
                    "type": "SELL",
                    "entry": entry,
                    "sl": sl,
                    "tp1": tp1,
                    "tp2": tp2,
                    "rr": rr,
                    "confidence": best_resistance.score,
                    "zone": {"low": best_resistance.low, "high": best_resistance.high, "kind": "resistance"},
                    "trend": trend,
                    "explain": explain
                })
        
        # BEKLE durumu
        if not signals:
            wait_reasons = []
            if not best_support and not best_resistance:
                wait_reasons.append("Yeterli bölge bulunamadı")
            elif best_support and best_support.score < 65:
                wait_reasons.append(f"Destek skoru yetersiz: {best_support.score}")
            elif best_resistance and best_resistance.score < 65:
                wait_reasons.append(f"Direnç skoru yetersiz: {best_resistance.score}")
            elif ema_distance > 1.0:
                wait_reasons.append(f"EMA'dan uzak: {ema_distance:.2f} ATR")
            
            signals.append({
                "type": "WAIT",
                "entry": current_price,
                "sl": None,
                "tp1": None,
                "tp2": None,
                "rr": 0,
                "confidence": max((best_support.score if best_support else 0), 
                                 (best_resistance.score if best_resistance else 0)),
                "zone": None,
                "trend": trend,
                "explain": wait_reasons
            })
        
        return signals, analysis_details
        
    except Exception as e:
        analysis_details.append(f"❌ Sinyal üretim hatası: {e}")
        return [], analysis_details

# =============================================================================
# YENİ YARDIMCI FONKSİYONLAR - UI OPTİMİZASYONU
# =============================================================================

def merge_overlapping_zones(zones: List[Zone], atr: float) -> List[Zone]:
    """Üst üste binen/çok yakın bantları birleştirir"""
    if not zones:
        return []
    
    sorted_zones = sorted(zones, key=lambda x: (x.low + x.high) / 2)
    merged = []
    merge_threshold = 0.25 * atr
    
    for zone in sorted_zones:
        if not merged:
            merged.append(zone)
            continue
            
        last_zone = merged[-1]
        zone_center = (zone.low + zone.high) / 2
        last_center = (last_zone.low + last_zone.high) / 2
        
        if abs(zone_center - last_center) < merge_threshold:
            new_low = min(last_zone.low, zone.low)
            new_high = max(last_zone.high, zone.high)
            new_touches = last_zone.touches + zone.touches
            new_score = max(last_zone.score, zone.score)
            
            merged_zone = Zone(
                low=new_low,
                high=new_high,
                touches=new_touches,
                last_touch_ts=max(last_zone.last_touch_ts, zone.last_touch_ts),
                kind=last_zone.kind
            )
            merged_zone.score = new_score
            merged_zone.status = last_zone.status if last_zone.score >= zone.score else zone.status
            
            merged[-1] = merged_zone
        else:
            merged.append(zone)
    
    return merged

def select_nearest_zones(zones: List[Zone], current_price: float, k: int = 2) -> List[Zone]:
    """Mevcut fiyata en yakın k adet zone seçer"""
    if not zones:
        return []
    
    sorted_zones = sorted(zones, key=lambda x: abs((x.low + x.high) / 2 - current_price))
    return sorted_zones[:k]

def get_zone_border_style(status: str) -> Dict[str, Any]:
    """Zone statüsüne göre kenarlık stilini belirler"""
    if status == "fake":
        return {"color": "#FFA500", "dash": "dash", "width": 1}
    elif status == "broken":
        return {"color": "#7A7A7A", "dash": "dot", "width": 1}
    else:
        return {"color": "#8A8F98", "dash": None, "width": 1}

def format_zone_label(zone: Zone, index: int) -> str:
    """Zone etiketini formatlar"""
    prefix = "S" if zone.kind == "support" else "R"
    label = f"{prefix}{index + 1}"
    
    if zone.status == "fake":
        label += " (fake)"
    elif zone.status == "broken":
        label += " (broken)"
    
    return label

def create_clean_candlestick_chart(data, support_zones, resistance_zones, crypto_symbol, signals):
    """Sadeleştirilmiş mum grafiği - maksimum 2 destek + 2 direnç bandı"""
    fig = go.Figure()
    
    if data is None or len(data) == 0:
        return fig, [], [], [], []
    
    # Son 3 gün verisi
    data_3days = data.tail(18)
    current_price = float(data_3days['Close'].iloc[-1])
    atr_value = float(data['ATR'].iloc[-1]) if 'ATR' in data.columns else current_price * 0.02
    
    # Zone'ları birleştir ve filtrele
    all_support = merge_overlapping_zones(support_zones, atr_value)
    all_resistance = merge_overlapping_zones(resistance_zones, atr_value)
    
    # En yakın 2 destek ve 2 direnç seç
    nearest_support = select_nearest_zones(all_support, current_price, 2)
    nearest_resistance = select_nearest_zones(all_resistance, current_price, 2)
    
    # Mumları çiz
    for i in range(len(data_3days)):
        try:
            row = data_3days.iloc[i]
            open_price = float(row['Open'])
            high = float(row['High'])
            low = float(row['Low'])
            close_price = float(row['Close'])
            
            color = '#00C805' if close_price > open_price else '#FF0000'
            
            # Mum gövdesi
            fig.add_trace(go.Scatter(
                x=[data_3days.index[i], data_3days.index[i]],
                y=[open_price, close_price],
                mode='lines',
                line=dict(color=color, width=8),
                showlegend=False
            ))
            
            # Üst iğne
            fig.add_trace(go.Scatter(
                x=[data_3days.index[i], data_3days.index[i]],
                y=[max(open_price, close_price), high],
                mode='lines',
                line=dict(color=color, width=1.5),
                showlegend=False
            ))
            
            # Alt iğne
            fig.add_trace(go.Scatter(
                x=[data_3days.index[i], data_3days.index[i]],
                y=[min(open_price, close_price), low],
                mode='lines',
                line=dict(color=color, width=1.5),
                showlegend=False
            ))
        except (ValueError, IndexError):
            continue
    
    # EMA çizgisi
    if 'EMA' in data_3days.columns:
        try:
            fig.add_trace(go.Scatter(
                x=data_3days.index,
                y=data_3days['EMA'],
                name=f'EMA{ema_period}',
                line=dict(color='orange', width=2),
                showlegend=False
            ))
        except Exception:
            pass
    
    # DESTEK BANTLARI - maksimum 2
    for i, zone in enumerate(nearest_support):
        border_style = get_zone_border_style(zone.status)
        fig.add_hrect(
            y0=zone.low,
            y1=zone.high,
            fillcolor="green",
            opacity=0.12,
            line=border_style,
        )
        # Sağ kenar etiketi
        fig.add_annotation(
            x=data_3days.index[-1],
            y=(zone.low + zone.high) / 2,
            text=format_zone_label(zone, i),
            showarrow=False,
            xanchor='left',
            yanchor='middle',
            font=dict(size=10, color="#00FF00"),
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="#00FF00",
            borderwidth=1
        )
    
    # DİRENÇ BANTLARI - maksimum 2
    for i, zone in enumerate(nearest_resistance):
        border_style = get_zone_border_style(zone.status)
        fig.add_hrect(
            y0=zone.low,
            y1=zone.high,
            fillcolor="red",
            opacity=0.12,
            line=border_style,
        )
        # Sağ kenar etiketi
        fig.add_annotation(
            x=data_3days.index[-1],
            y=(zone.low + zone.high) / 2,
            text=format_zone_label(zone, i),
            showarrow=False,
            xanchor='left',
            yanchor='middle',
            font=dict(size=10, color="#FF0000"),
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="#FF0000",
            borderwidth=1
        )
    
    # Mevcut fiyat çizgisi
    try:
        fig.add_hline(
            y=current_price,
            line_dash="dot",
            line_color="yellow",
            line_width=1,
            opacity=0.7,
            annotation_text=f"{format_price(current_price)}",
            annotation_position="left top",
            annotation_font_size=10,
            annotation_font_color="yellow"
        )
    except (ValueError, IndexError):
        pass
    
    # Sinyal işareti (sadece sinyal varsa)
    if signals and signals[0]["type"] != "WAIT":
        signal = signals[0]
        marker_symbol = "triangle-up" if signal["type"] == "BUY" else "triangle-down"
        marker_color = "#00FF00" if signal["type"] == "BUY" else "#FF0000"
        
        fig.add_trace(go.Scatter(
            x=[data_3days.index[-1]],
            y=[current_price],
            mode='markers',
            marker=dict(
                symbol=marker_symbol,
                size=12,
                color=marker_color,
                line=dict(width=2, color="white")
            ),
            showlegend=False,
            name=f"{signal['type']} Sinyal"
        ))
    
    # Grafik ayarları
    fig.update_layout(
        height=500,
        title=f"{crypto_symbol} - 4H (Son 3 Gün)",
        xaxis_title="",
        yaxis_title="Fiyat (USD)",
        showlegend=False,
        xaxis_rangeslider_visible=False,
        plot_bgcolor='#0E1117',
        paper_bgcolor='#0E1117',
        font=dict(color='white', size=10),
        xaxis=dict(gridcolor='#444', showticklabels=True),
        yaxis=dict(gridcolor='#444'),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig, nearest_support, nearest_resistance, all_support, all_resistance

# =============================================================================
# ANA UYGULAMA - SADELEŞTİRİLMİŞ UI
# =============================================================================

def main():
    # Veri yükleme
    with st.spinner(f'⏳ {crypto_symbol} verileri yükleniyor...'):
        data_30days = get_4h_data(crypto_symbol, days=30)
    
    if data_30days is None or data_30days.empty:
        st.error(f"❌ {crypto_symbol} için veri yüklenemedi!")
        return
    
    # Göstergeleri hesapla
    data_30days = calculate_indicators(data_30days, ema_period, rsi_period)
    
    # Yoğunluk bölgelerini bul
    support_zones, resistance_zones = find_congestion_zones(
        data_30days, min_touch_points, analysis_lookback_bars
    )
    
    # Sinyal üret
    signals, analysis_details = generate_trading_signals(
        data_30days, support_zones, resistance_zones, ema_period, risk_reward_ratio
    )
    
    # Mevcut durum
    try:
        current_price = float(data_30days['Close'].iloc[-1])
        ema_value = float(data_30days['EMA'].iloc[-1])
        rsi_value = float(data_30days['RSI'].iloc[-1])
        atr_value = float(data_30days['ATR'].iloc[-1])
        trend = "bull" if current_price > ema_value else "bear"
    except (ValueError, IndexError):
        current_price = 0
        ema_value = 0
        rsi_value = 0
        atr_value = 0
        trend = "neutral"
    
    # Layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Sadeleştirilmiş mum grafiği
        chart_fig, nearest_support, nearest_resistance, all_support, all_resistance = create_clean_candlestick_chart(
            data_30days, support_zones, resistance_zones, crypto_symbol, signals
        )
        st.plotly_chart(chart_fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Sinyal")
        
        # Sinyal kartı
        if signals and signals[0]["type"] != "WAIT":
            signal = signals[0]
            signal_color = "🟢" if signal['type'] == 'BUY' else "🔴"
            
            st.markdown(f"### {signal_color} {signal['type']}")
            
            cols = st.columns(2)
            with cols[0]:
                st.metric("Giriş", format_price(signal['entry']))
                st.metric("TP1", format_price(signal['tp1']))
            with cols[1]:
                st.metric("SL", format_price(signal['sl']))
                if signal['tp2']:
                    st.metric("TP2", format_price(signal['tp2']))
            
            st.metric("R/R", f"{signal['rr']:.2f}")
            st.metric("Güven", f"%{signal['confidence']}")
            
        else:
            st.markdown("### ⚪ BEKLE")
            st.info("Koşullar uygun değil")
        
        st.divider()
        
        # Trend ve gösterge
        st.subheader("📈 Trend")
        trend_icon = "🟢" if trend == "bull" else "🔴"
        st.metric("EMA50", trend_icon + " " + ("YÜKSELİŞ" if trend == "bull" else "DÜŞÜŞ"))
        st.metric("RSI", f"{rsi_value:.1f}")
        
        st.divider()
        
        # Yakın bantlar
        st.subheader("🎯 Yakın Bantlar")
        
        for i, zone in enumerate(nearest_support):
            st.write(f"**S{i+1}:** {format_price(zone.low)}-{format_price(zone.high)}")
            st.caption(f"Skor: {zone.score}, Durum: {zone.status}")
        
        for i, zone in enumerate(nearest_resistance):
            st.write(f"**R{i+1}:** {format_price(zone.low)}-{format_price(zone.high)}")
            st.caption(f"Skor: {zone.score}, Durum: {zone.status}")
    
    # Detaylı bant listesi
    with st.expander("📋 Tüm Bant Detayları"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Destek Bantları**")
            for i, zone in enumerate(all_support):
                status_icon = "🟢" if zone.status == "valid" else "🟠" if zone.status == "fake" else "⚫"
                st.write(f"{status_icon} S{i+1}: {format_price(zone.low)}-{format_price(zone.high)}")
                st.caption(f"Skor: {zone.score}, Temas: {zone.touches}, Durum: {zone.status}")
        
        with col2:
            st.write("**Direnç Bantları**")
            for i, zone in enumerate(all_resistance):
                status_icon = "🔴" if zone.status == "valid" else "🟠" if zone.status == "fake" else "⚫"
                st.write(f"{status_icon} R{i+1}: {format_price(zone.low)}-{format_price(zone.high)}")
                st.caption(f"Skor: {zone.score}, Temas: {zone.touches}, Durum: {zone.status}")

if __name__ == "__main__":
    main()