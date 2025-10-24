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
# YENİ YARDIMCI FONKSİYONLAR - UI OPTİMİZASYONU
# =============================================================================

def merge_overlapping_zones(zones: List[Zone], atr: float) -> List[Zone]:
    """
    Üst üste binen/çok yakın bantları birleştirir
    """
    if not zones:
        return []
    
    # Zone'ları fiyatlarına göre sırala
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
        
        # Merkezler arası mesafe threshold'dan küçükse birleştir
        if abs(zone_center - last_center) < merge_threshold:
            # Yeni bant oluştur
            new_low = min(last_zone.low, zone.low)
            new_high = max(last_zone.high, zone.high)
            new_touches = last_zone.touches + zone.touches
            new_score = max(last_zone.score, zone.score)  # En yüksek skoru al
            
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
    """
    Mevcut fiyata en yakın k adet zone seçer
    """
    if not zones:
        return []
    
    # Zone'ları mevcut fiyata olan mesafelerine göre sırala
    sorted_zones = sorted(zones, key=lambda x: abs((x.low + x.high) / 2 - current_price))
    return sorted_zones[:k]

def get_zone_border_style(status: str) -> Dict[str, Any]:
    """
    Zone statüsüne göre kenarlık stilini belirler
    """
    if status == "fake":
        return {"color": "#FFA500", "dash": "dash", "width": 1}
    elif status == "broken":
        return {"color": "#7A7A7A", "dash": "dot", "width": 1}
    else:  # valid
        return {"color": "#8A8F98", "dash": None, "width": 1}

def get_zone_fill_color(kind: str) -> str:
    """
    Zone türüne göre dolgu rengini belirler
    """
    return "green" if kind == "support" else "red"

def format_zone_label(zone: Zone, index: int) -> str:
    """
    Zone etiketini formatlar
    """
    prefix = "S" if zone.kind == "support" else "R"
    label = f"{prefix}{index + 1}"
    
    if zone.status == "fake":
        label += " (fake)"
    elif zone.status == "broken":
        label += " (broken)"
    
    return label

# =============================================================================
# GÜNCELLENMİŞ GÖRSELLEŞTİRME FONKSİYONLARI
# =============================================================================

def create_clean_candlestick_chart(data, support_zones, resistance_zones, crypto_symbol, signals):
    """
    Sadeleştirilmiş mum grafiği - maksimum 2 destek + 2 direnç bandı
    """
    fig = go.Figure()
    
    if data is None or len(data) == 0:
        return fig
    
    # Son 3 gün verisi (görselleştirme için)
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

# Diğer fonksiyonlar aynı kalacak (compute_atr, Zone, build_zones, eval_fake_breakout, compute_macd, score_zone, risk_reward, get_4h_data, calculate_indicators, find_congestion_zones, generate_trading_signals)

if __name__ == "__main__":
    main()k