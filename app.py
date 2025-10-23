import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Streamlit arayüzü
st.set_page_config(page_title="Kripto Teknik Analiz", layout="wide")
st.title("🎯 Kripto Teknik Analiz - Mum, Trend, Sinyal ve Yorum")

# Sidebar
st.sidebar.header("⚙️ Analiz Ayarları")
crypto_symbol = st.sidebar.text_input("Kripto Sembolü (Örn: BTC-USD, ETH-USD):", "BTC-USD")
lookback_days = st.sidebar.slider("Gün Sayısı", 30, 365, 90)
analysis_type = st.sidebar.selectbox("Analiz Türü", ["4 Saatlik", "1 Günlük", "1 Saatlik"])

# Analiz periyodu mapping
interval_map = {"4 Saatlik": "4h", "1 Günlük": "1d", "1 Saatlik": "1h"}

def get_crypto_data(symbol, days, interval):
    """Kripto verilerini çek"""
    try:
        data = yf.download(symbol, period=f"{days}d", interval=interval, progress=False)
        return data
    except Exception as e:
        st.error(f"Veri çekilemedi: {e}")
        return None

def calculate_technical_indicators(data):
    """Teknik göstergeleri hesapla (ta kütüphanesi olmadan)"""
    df = data.copy()
    
    # Moving Average'lar
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    
    # RSI Hesaplama
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['RSI'] = calculate_rsi(df['Close'])
    
    # MACD Hesaplama
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = calculate_macd(df['Close'])
    
    # Bollinger Bands
    def calculate_bollinger_bands(prices, window=20, num_std=2):
        middle_band = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        return upper_band, middle_band, lower_band
    
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
    
    return df

def identify_candlestick_patterns(data):
    """Mum formasyonlarını tespit et"""
    df = data.copy()
    patterns = []
    
    if len(df) < 2:
        return patterns
    
    # Tüm değerleri float'a çevir - HATA DÜZELTME
    curr_open, curr_high, curr_low, curr_close = float(df['Open'].iloc[-1]), float(df['High'].iloc[-1]), float(df['Low'].iloc[-1]), float(df['Close'].iloc[-1])
    prev_open, prev_high, prev_low, prev_close = float(df['Open'].iloc[-2]), float(df['High'].iloc[-2]), float(df['Low'].iloc[-2]), float(df['Close'].iloc[-2])
    
    # Doji - Açılış ve kapanış çok yakın
    body_size = abs(curr_close - curr_open)
    total_range = curr_high - curr_low
    if total_range > 0 and (body_size / total_range) < 0.1:
        patterns.append("DOJI - Kararsızlık sinyali")
    
    # Bullish Engulfing
    if (prev_close < prev_open and  # Önceki mum bearish
        curr_close > curr_open and  # Şimdiki mum bullish
        curr_open < prev_close and  # Şimdiki açılış önceki kapanıştan düşük
        curr_close > prev_open):    # Şimdiki kapanış önceki açılıştan yüksek
        patterns.append("BULLISH ENGULFING - Güçlü yükseliş sinyali")
    
    # Bearish Engulfing
    if (prev_close > prev_open and  # Önceki mum bullish
        curr_close < curr_open and  # Şimdiki mum bearish
        curr_open > prev_close and  # Şimdiki açılış önceki kapanıştan yüksek
        curr_close < prev_open):    # Şimdiki kapanış önceki açılıştan düşük
        patterns.append("BEARISH ENGULFING - Güçlü düşüş sinyali")
    
    # Hammer
    lower_shadow = min(curr_open, curr_close) - curr_low
    upper_shadow = curr_high - max(curr_open, curr_close)
    body = abs(curr_close - curr_open)
    
    if lower_shadow > 2 * body and upper_shadow < body and curr_close > curr_open:
        patterns.append("HAMMER - Dip reversal sinyali")
    
    # Shooting Star
    if upper_shadow > 2 * body and lower_shadow < body and curr_close < curr_open:
        patterns.append("SHOOTING STAR - Tepe reversal sinyali")
    
    return patterns

def calculate_trend_lines(data):
    """Trend çizgilerini hesapla"""
    # Float'a çevir - HATA DÜZELTME
    closes = data['Close'].astype(float).values
    dates = np.arange(len(closes))
    
    if len(dates) > 1:
        # Lineer regresyon ile trend
        z = np.polyfit(dates, closes, 1)
        trend_line = np.poly1d(z)(dates)
        trend_slope = float(z[0])  # float'a çevir
        
        # Trend yönünü belirle
        if trend_slope > 0:
            trend_direction = "📈 YÜKSELİŞ TRENDİ"
            trend_strength = "GÜÇLÜ" if trend_slope > np.std(closes) * 0.1 else "ZAYIF"
        elif trend_slope < 0:
            trend_direction = "📉 DÜŞÜŞ TRENDİ"
            trend_strength = "GÜÇLÜ" if abs(trend_slope) > np.std(closes) * 0.1 else "ZAYIF"
        else:
            trend_direction = "➡️ YATAY TREND"
            trend_strength = "NÖTR"
        
        return trend_line, f"{trend_direction} ({trend_strength})", trend_slope
    
    return None, "BELİRSİZ", 0

def generate_trading_signals(data):
    """Alım-satım sinyalleri üret"""
    df = data.copy()
    signals = []
    
    if len(df) < 2:
        return signals
    
    # Tüm değerleri float'a çevir - HATA DÜZELTME
    rsi = float(df['RSI'].iloc[-1]) if not pd.isna(df['RSI'].iloc[-1]) else None
    macd = float(df['MACD'].iloc[-1]) if not pd.isna(df['MACD'].iloc[-1]) else None
    macd_signal = float(df['MACD_Signal'].iloc[-1]) if not pd.isna(df['MACD_Signal'].iloc[-1]) else None
    price = float(df['Close'].iloc[-1])
    ma_20 = float(df['MA_20'].iloc[-1]) if not pd.isna(df['MA_20'].iloc[-1]) else None
    ma_50 = float(df['MA_50'].iloc[-1]) if not pd.isna(df['MA_50'].iloc[-1]) else None
    
    # RSI Sinyalleri
    if rsi is not None:
        if rsi < 30:
            signals.append("🎯 RSI AŞIRI SATIM - Potansiyel ALIM fırsatı")
        elif rsi > 70:
            signals.append("⚠️ RSI AŞIRI ALIM - Potansiyel SATIM sinyali")
    
    # MACD Sinyalleri
    if macd is not None and macd_signal is not None:
        prev_macd = float(df['MACD'].iloc[-2]) if not pd.isna(df['MACD'].iloc[-2]) else None
        prev_macd_signal = float(df['MACD_Signal'].iloc[-2]) if not pd.isna(df['MACD_Signal'].iloc[-2]) else None
        
        if prev_macd is not None and prev_macd_signal is not None:
            if macd > macd_signal and prev_macd <= prev_macd_signal:
                signals.append("✅ MACD ALTI KESİŞİM - ALIM sinyali")
            elif macd < macd_signal and prev_macd >= prev_macd_signal:
                signals.append("❌ MACD ÜSTÜ KESİŞİM - SATIM sinyali")
    
    # Moving Average Sinyalleri
    if ma_20 is not None and ma_50 is not None:
        if price > ma_20 > ma_50:
            signals.append("🚀 GÜÇLÜ YÜKSELİŞ TRENDİ - MA'lar destekliyor")
        elif price < ma_20 < ma_50:
            signals.append("🔻 GÜÇLÜ DÜŞÜŞ TRENDİ - MA'lar direnç gösteriyor")
        elif price > ma_20 and ma_20 > ma_50:
            signals.append("↗️ YÜKSELİŞ EĞİLİMİ - MA düzeni uygun")
        elif price < ma_20 and ma_20 < ma_50:
            signals.append("↘️ DÜŞÜŞ EĞİLİMİ - MA düzeni uygun")
    
    return signals

def calculate_support_resistance(data, window=10):
    """Destek ve direnç seviyelerini hesapla"""
    # Series'i numpy array'e çevir ve float'a dönüştür - HATA DÜZELTME
    highs = data['High'].astype(float).values
    lows = data['Low'].astype(float).values
    
    support_levels = []
    resistance_levels = []
    
    for i in range(window, len(data)-window):
        current_high = float(highs[i])
        current_low = float(lows[i])
        
        # Direnç - yerel maksimum
        is_resistance = True
        for j in range(1, window+1):
            prev_high = float(highs[i-j])
            next_high = float(highs[i+j])
            if current_high <= prev_high or current_high <= next_high:
                is_resistance = False
                break
        
        if is_resistance:
            resistance_levels.append(current_high)
        
        # Destek - yerel minimum
        is_support = True
        for j in range(1, window+1):
            prev_low = float(lows[i-j])
            next_low = float(lows[i+j])
            if current_low >= prev_low or current_low >= next_low:
                is_support = False
                break
        
        if is_support:
            support_levels.append(current_low)
    
    return support_levels, resistance_levels

def generate_analysis_report(data, patterns, signals, trend_direction):
    """Detaylı analiz raporu oluştur"""
    # Float'a çevir - HATA DÜZELTME
    current_price = float(data['Close'].iloc[-1])
    prev_price = float(data['Close'].iloc[-2])
    change_pct = ((current_price - prev_price) / prev_price) * 100
    
    # Volatilite hesapla
    volatility = data['Close'].astype(float).pct_change().std() * 100
    
    report = f"""
    ## 📊 Teknik Analiz Raporu
    
    **🎯 Mevcut Durum:**
    - **Fiyat:** ${current_price:.2f}
    - **Değişim:** %{change_pct:+.2f}
    - **Volatilite:** %{volatility:.2f}
    - **Trend:** {trend_direction}
    
    **📈 Trend Analizi:**
    - Ana Trend: {trend_direction.split('(')[0].strip()}
    - Momentum: {'YÜKSELİŞ' if change_pct > 0 else 'DÜŞÜŞ'}
    - Volatilite: {'YÜKSEK' if volatility > 3 else 'DÜŞÜK'}
    """
    
    if patterns:
 patterns:
        report        report += "\n** += "\n**🕯️ Mum Form🕯️ Mum Formasyonlarıasyonları:**\n"
       :**\n"
        for pattern in patterns:
            report += f"- { for pattern in patterns:
            report += f"- {patternpattern}\n"
    else:
       }\n"
    else:
        report += "\n**🕯 report += "\n**🕯️ Mum Formasyon️ Mum Formasyonlarıları:** Belirgin formasyon yok\n"
:** Belirgin formasyon yok\n"
    
    if    
    if signals:
        report signals:
        report += "\ += "\n**🔔n**🔔 Trading Sinyall Trading Sinyalleri:**\neri:**\n"
       "
        for signal in signals:
            report += f"- for signal in signals:
            report += f"- {signal}\n {signal}\n"
    else"
    else:
        report += "\:
        report += "\n**n**🔔 Trading Siny🔔 Trading Sinyallerialleri:** Net sin:** Net sinyal yyal yok\n"
    
    # Rok\n"
    
    # RSI DetaySI Detaylı Yorum
lı Yorum
    r    rsi = float(datasi = float(data['RSI'].['RSI'].iloc[-1iloc[-1]) if not pd.isna]) if not pd.isna(data['RSI(data['RSI'].il'].iloc[-1]) elseoc[-1]) else None
    None
    if rsi is if rsi is not None:
 not None:
        if rsi        if rsi < 30 < 30:
           :
            report += f"\n report += f"\n**📉 RSI**📉 RSI Analizi Analizi:** AŞIR:** AŞIRI SATI SATIM bölIM bölgesindegesinde (RSI (RSI: {rsi: {rsi:.1f}):.1f}) - ⚠ - ⚠️ Potansiy️ Potansiyel ALIMel ALIM fırs fırsatı"
       atı"
        elif r elif rsi >si > 70:
            report += 70:
            report += f"\n** f"\n**📈📈 RSI Analizi:** RSI Analizi:** AŞIRI ALIM bölgesinde (RSI: {rsi:.1 AŞIRI ALIM bölgesinde (RSI: {rsi:.1f}) - ⚠️ Dikkatli olun, SATIM sinyali"
        elif 30 <= rsif}) - ⚠️ Dikkatli olun, SATIM sinyali"
        elif 30 <= rsi <= <= 70:
            report += 70:
            report += f"\ f"\n**⚖️n**⚖️ R RSISI Analizi:** Nö Analizi:** Nötr bölgede (RSI:tr bölgede (RSI: {rsi {rsi:.1:.1f}) -f}) - 🔄 Trend 🔄 Trend takibi takibi öner önerilir"
    
ilir"
    
    # Genel Öneri
       # Genel Öneri
    bullish_signals bullish_signals = len = len([s for s([s for s in signals if 'ALIM in signals if 'ALIM' in s or 'YÜKSELİŞ' in s])
   ' in s or 'YÜKSELİŞ' in s])
    bearish_signals bearish_signals = len = len([s for s in signals([s for s in signals if ' if 'SATIM' inSATIM' in s or s or 'DÜŞÜŞ 'DÜŞÜŞ' in' in s])
    
    s])
    
    if bullish if bullish_signals > bearish_signals > bearish_signals:
_signals:
        report        report += "\n\n**💎 GEN += "\n\n**💎 GENEL BAKIŞ:** YÜKSEL BAKIŞ:** YÜKSELİŞ eğELİŞ eğilimi ağır basıyorilimi ağır basıyor"
    elif bear"
    elif bearish_signish_signals > bullish_signalsals > bullish_signals:
:
        report += "\n\n        report += "\n\n**💎 GEN**💎 GENEL BAKIEL BAKIŞ:** DÜŞÜŞ:** DÜŞÜŞ eŞ eğilimi ağırğilimi ağır basıyor"
    basıyor"
    else:
        report += "\n\n** else:
        report += "\n\n**💎 GENEL B💎 GENEL BAKIŞ:** KARARSIZ piAKIŞ:** KARARSIZ piyasa, bekleyin"
yasa, bekleyin"
    
    return report

def main    
    return report

def main():
    try:
():
    try:
        # Veri çek        # Veri çekme
        interval = interval_map[analysis_type]
me
        interval = interval_map[analysis_type]
        st        st.write(f"**{crypto_symbol}**.write(f"**{crypto_symbol}** için için {analysis_type} veriler {analysis_type} veriler çekiliyor...")
 çekiliyor...")
        
        data =        
        data = get_crypto_data(crypto_symbol, get_crypto_data(crypto_symbol, lookback_days, interval)
        
        lookback_days, interval)
        
        if data is if data is None or data.empty None or data.empty:
           :
            st.error("Ver st.error("Veri çekilemedi. Lütfen sembolü kontrol edin.")
            return
        
        st.success(f"i çekilemedi. Lütfen sembolü kontrol edin.")
            return
        
        st.success(f"✅ {✅ {len(data)} adlen(data)} adet {et {analysis_type} mum veranalysis_type} mum verisi çekisi çekildi")
        
ildi")
        
        # Tek        # Teknik göstergelernik göstergelerii he hesaplasapla
        data =
        data = calculate_technical_indicators calculate_technical_indicators(data)
        
       (data)
        
        # Analizleri yap # Analizleri yap
        patterns = identify_c
        patterns = identify_candlestick_patternandlestick_patterns(datas(data)
        signals =)
        signals = generate_trading_signals(data)
        trend_line, trend_direction, trend_slope = calculate_trend_lines generate_trading_signals(data)
        trend_line, trend_direction, trend_slope = calculate_trend_lines(data)
        support_levels, resistance(data)
        support_levels, resistance_levels = calculate_levels = calculate_support_resistance(data)
        
        # Se_support_resistance(data)
        
        #viyeleri filtre Seviyeleri filtrele (mevcut fle (mevcut fiyiyata yakın olanlar)
ata yakın olanlar)
        current_price =        current_price = float(data['Close'].il float(data['Close'].iloc[-1])
        key_supportoc[-1])
        key_support = [level for level in = [level for level in support_level support_levels if abs(s if abs(levellevel - current_price) / current_price - current_price) / current_price *  * 100 <= 15]
100 <= 15]
               key_resistance = [level key_resistance = [level for level in resistance_levels if abs for level in resistance_levels if abs(level - current(level - current_price)_price) / current_price * 100 / current_price * 100 <= 15]
        
        <= 15]
        
        # Benz # Benzersiz seviyelerersiz seviyeler
        key_s
        key_support =upport = list(set(key_support list(set(key_support))
))
        key_resistance = list        key_resistance = list(set(set(key_resistance))
       (key_resistance))
        key_support.sort()
        key_res key_support.sort()
        key_resistance.sortistance.sort()
        
       ()
        
        # Analiz ra # Analiz raporu
        report = generate_analysisporu
        report = generate_analysis_report(data, patterns, signals, trend_direction)
_report(data, patterns        st.markdown(report)
        
        # Grafikler, signals, trend_direction)
        st.markdown(report)
        
        # Grafikler
        st.subheader("📊 Görsel Analiz")

        st.subheader("📊 Görsel Analiz")
        col1, col2        col1, col2 = st.columns(2)
        
        = st.columns(2)
        
        with col1:
            # with col1:
            # Ana grafik
            fig1 = Ana grafik
            fig1 = go.Figure()
            
            # Mum go.Figure()
            
            # Mum grafiği
            grafiği
            fig fig1.add_trace(go1.add_trace(go.Candlestick(
                x=data.Candlestick(
                x=data.index,
               .index open=data['Open'],
                high=data['High'],
                low,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['=data['Low'],
                close=data['Close'],
                name='Mum'
Close'],
                name='Mum'
            ))
            
            # Moving Average'            ))
            
            # Moving Average'lar
            fig1.addlar
            fig1.add_trace_trace(go.Scatter(x(go.Scatter(x=data=data.index, y=data['MA.index, y=data['MA_20_20'], name'], name='MA='MA 20', line= 20', line=dict(color='orange', width=2)))
            fig1.add_trace(godict(color='orange', width=2)))
            fig1.add_trace(go.Scatter.Scatter(x=data.index, y=data['MA_50'],(x=data.index, y=data['MA_50'], name name='MA 50', line='MA 50', line=dict=dict(color='red', width=2)))
            
           (color='red', width=2)))
            
            # Trend # Trend çizgisi
            if trend_line is not çizgisi
            if trend_line is not None None:
                fig1.add_t:
                fig1.add_trace(race(go.Scatter(x=data.indexgo.Scatter(x=data.index, y=tre, y=trend_line,nd_line, name='Trend Çizgisi name='Trend Çizgisi', 
                                        line=dict(color', 
                                        line=dict(color='='blue', dash='dashblue', dash='dash', width=3)))
            
            #', width=3)))
            
            # Destek seviyeler Destek seviyeleri
            for level in key_si
            for level in key_support[-3:upport[-3:]:
                fig1.add_]:
                fig1.add_hlinehline(y=level, line_dash(y=level, line_dash="dash", line_color="="dash", line_color="greengreen", 
                             line_width=", 
                             line_width=2, opacity=0.72, opacity=0.7,
,
                             annotation_text=f"D:                             annotation_text=f"D: ${level:.2f}")
            
            # Direnç seviy ${level:.2f}")
            
            # Direnç seviyeleri
eleri
            for level in            for level in key_res key_resistance[-3:istance[-3:]:
                fig]:
                fig1.add_hline(y1.add_hline(y=level=level, line_dash, line_dash="dash="dash", line_color="", line_color="red",red", 
                             line_width=2, opacity=0. 
                             line_width=2, opacity=0.7,
7,
                             annotation_text=f"R                             annotation_text=f"R: ${: ${level:.2flevel:.2f}")
            
}")
            
            fig1.update_layout(
            fig1.update_layout(
                height                height=500, 
=500, 
                title                title=f"{c=f"{cryptorypto_symbol_symbol}} {analysis_type} Grafik {analysis_type} Grafik - Mum Formasyonları ve Trend - Mum Formasyonları ve Trend",
                xaxis_title",
                xaxis_title="T="Tarih",
               arih",
                yaxis_title=" yaxis_title="FiyatFiyat (USD)"
            (USD)"
            )
            st )
            st.plotly_chart.plotly_chart(fig1,(fig1, use_container_width=True use_container_width=True)
        
)
        
        with col2        with col2:
           :
            # RSI Gra # RSI Grafifiği
           ği
            fig2 = go fig2 = go.Figure()
            fig2.Figure()
            fig2.add_t.add_trace(go.Scatterrace(go.Scatter(x=data(x=data.index, y=data['RSI.index, y=data['RSI'], name='RSI','], name='RSI', 
 
                                    line=dict(color='pur                                    line=dict(color='purple', width=2)))
            fig2.add_hline(y=70, line_dple', width=2)))
            fig2.add_hline(y=70, line_dash="ash="dash", line_color="dash", line_color="red", annotation_text="red", annotation_text="AşırAşırı Alım")
ı Alım")
                       fig2.add_hline(y=30 fig2.add_hline(y=30,, line_dash=" line_dash="ddash", line_color="green", annotationash", line_color="green", annotation_text="Aşır_text="Aşırı Satım")
            fig2.add_hline(y=50, line_dash="ı Satım")
            fig2.add_hline(y=50, line_dash="dot", linedot", line_color="gray_color="gray", annotation_text="", annotation_text="OrtaOrta")
            fig2")
            fig2.update_layout(height.update_layout(height=250,=250, title="RSI title="RSI (14) - (14) - Momentum Gö Momentum Göstergesi")
           stergesi")
            st.plotly st.plotly_chart(fig_chart(fig2, use2, use_container_width=True)
            
_container_width=True)
            
                       # MACD Gra # MACD Grafifiği
            figği
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter3 = go.Figure()
            fig3.add_trace(go.Scatter(x=data.index, y(x=data.index, y=data['MAC=data['MACD'], name='MACD', 
                                   D'], name='MACD', 
                                    line=dict(color='blue', width=2)))
 line=dict(color='blue', width=            fig3.add2)))
            fig3.add_trace_trace((go.Scatter(xgo.Scatter(x=data.index, y=data['MAC=data.index, y=data['MACD_Signal'],D_Signal'], name='S name='Sinyinyal', 
                                   al', 
                                    line=dict(color='red', width= line=dict(color='red', width=2)))
            #2)))
            # Histogram'ı sadece NaN Histogram'ı sadece NaN değilse değilse ekle
            if not data ekle
            if not data['MACD_Hist['MACD_Histogram'].isna().all():
               ogram'].isna().all():
                fig3 fig3.add_trace(go.add_trace(go.B.Bar(x=data.index,ar(x=data.index, y=data y=data['MACD_['MACD_Histogram'], nameHistogram'], name='Histogram',
='Histogram',
                                    marker                                    marker_color='gray',_color='gray', opacity= opacity=0.3))
0.3))
            fig3            fig3.update_layout(height=.update_layout(height=250, title250, title="MACD -="MACD - Trend Takip Trend Takip Gösterg Göstergesi")
esi")
            st.plotly_chart(fig            st.plotly_chart(fig3, use_container3, use_container_width=True)
        
_width=True)
        
        # Det        # Detaylı bilaylı bilgiler
       giler
        with st.expander with st.expander("📋("📋 Detaylı Tek Detaylı Teknik Veriler"):
            col3, col4nik Veriler"):
            col3, col4, col5 = st.columns(3)
            
            with col3:
                st.write("**, col5 = st.columns(3)
            
            with col3:
                st.write("**📈 Moving📈 Moving Average'lar:**")
 Average'lar:**")
                ma_20_val                ma_20_val = data = data['['MA_20'].iloc[-MA_20'].iloc[-1]
                ma1]
                ma_50_val_50_val = data[' = data['MA_50'].MA_50'].iloc[-1iloc[-1]
                ma_]
                ma_200_val =200_val = data['MA data['MA_200'].il_200'].iloc[-1]
oc[-1]
                
                st                
                st.m.metric("MA 20",etric("MA 20", f"${ma f"${ma_20_val_20_val:.2f}":.2f}" if not pd if not pd.isna(ma.isna(ma_20_val_20_val) else "Hesaplanıyor")
) else "Hesaplanı                st.metric("MA 50yor")
                st.metric("MA 50", f"${ma_", f"${ma_50_val50_val:.2f}" if:.2f}" if not pd not pd.isna(.isna(ma_ma_50_val) else50_val) else "Hesaplan "Hesaplanıyor")
               ıyor")
                st.metric st.metric("MA 200("MA 200",", f"${ma_200 f"${ma_200_val_val:.2f}" if:.2f}" if not pd.is not pd.isna(ma_200_valna(ma_200_val) else ") else "HesaplanıyorHesaplanıyor")
            
")
            
            with col4:
                st.write            with col4:
                st.write("**🔍 Oscillator'lar:**")
               ("**🔍 Oscillator'lar:**")
                rsi_val = data['RS rsi_val = data['RSI'].ilI'].iloc[-1]
                macdoc[-1]
                macd_val = data['_val = data['MACD'].iloc[-1MACD'].iloc[-1]
                mac]
                macd_signal_val = data['MACd_signal_val = data['MACD_Signal'].D_Signal'].iloc[-1]
                
               iloc[-1]
                
                st.metric st.metric("RS("RSI", f"{rI", f"{rsi_val:.1si_val:.1f}" iff}" if not pd.is not pd.isna(rsi_valna(rsi_val) else "H) else "Hesaplanıyor")
                st.metric("MACesaplanıyor")
                st.metric("MACD", fD", f"{macd_val"{macd_val:.4f}":.4f}" if not pd if not pd.isna(m.isna(macd_val)acd_val) else "Hesa else "Hesaplanıyor")
planıyor")
                st                st.metric("MACD.metric("MACD Siny Sinyal", f"{macal", f"{macd_sd_signal_valignal_val:.4f}":.4f}" if not pd.is if not pd.isna(mna(macd_signalacd_signal_val) else "_val) else "HesaplanıHesaplanıyor")
            
           yor")
            
            with col5 with col5:
                st:
                st.write("**.write("**💎 Piyasa💎 Piyasa Bilg Bilgileri:**")
ileri:**")
                st.metric                st.metric("Destek Se("Destek Seviyeleri", lenviyeleri", len(key_support))
(key_support))
                st                st.metric.metric("Direnç("Direnç Seviyeler Seviyeleri",i", len(key_resistance))
 len(key_resistance))
                st                st.metric("Trend E.metric("Trend Eğimğim", f"{tre", f"{trend_snd_slope:.6flope:.6f}")
        
       }")
        
        # Son 10 mum verisi
        with # Son 10 mum verisi
        with st.expander(" st.expander("📜📜 Son Mum Veriler Son Mum Verileri"):
i"):
            display_data =            display_data = data.tail(10)[ data.tail(10)[['Open',['Open', 'High', ' 'HighLow', 'Close', 'Volume']].round(2)
            st.dataframe(display_data.style', 'Low', 'Close', 'Volume']].round(2)
            st.dataframe(display_data.style.format({
                'Open':.format({
                'Open': '${ '${:.:.2f}', 'High':2f}', 'High': '${:.2f}', 'Low '${:.2f}', '': '${:.2f}', 'CloseLow': '${:.2f}', '': '${:.2f}'
            }))
            
    exceptClose': '${:.2f}'
            }))
            
    except Exception as e:
        st.error(f"❌ H Exception as e:
        st.error(f"❌ Hata oluştu: {str(e)}")
        st.info("ata oluştu: {str(e)}")
        st.infoLütfen sembolü kontrol("Lütfen sembolü kontrol edin ve internet bağ edin ve internet bağlantınızı doğrulaylantınızı doğrulın.")

if __ayın.")

if __name__ == "__main__":
    mainname__ == "__main__":
    main()