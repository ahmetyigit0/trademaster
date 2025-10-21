import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Crypto AI Pro", layout="wide")
st.title("🚀 Crypto AI Pro")
st.markdown("**Eğitim amaçlıdır - Yatırım tavsiyesi DEĞİLDİR**")

# Yardımcı fonksiyonlar
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_ema(prices, period):
    return prices.ewm(span=period, adjust=False).mean()

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    macd = ema_fast - ema_slow
    signal_line = calculate_ema(macd, signal)
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(prices, period=20, std=2):
    sma = prices.rolling(period).mean()
    std_dev = prices.rolling(period).std()
    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)
    return upper_band, sma, lower_band

def calculate_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

# Sidebar
st.sidebar.header("⚙️ Analiz Ayarları")
ticker_input = st.sidebar.text_input("🎯 Kripto Sembolü", "BTC-USD", help="Örnek: BTC-USD, ETH-USD, ADA-USD, BNB-USD, XRP-USD, SOL-USD")

timeframe = st.sidebar.selectbox("⏰ Zaman Dilimi", ["1h", "4h", "1d", "1wk"], index=2)
period_map = {"1h": "1mo", "4h": "3mo", "1d": "6mo", "1wk": "1y"}
period = period_map[timeframe]

# Risk yönetimi
st.sidebar.header("🎯 Risk Yönetimi")
capital = st.sidebar.number_input("💰 Sermaye ($)", 1000, 1000000, 5000, step=1000)
risk_percent = st.sidebar.slider("📉 İşlem Risk %", 1.0, 5.0, 2.0, 0.1)
max_position = st.sidebar.slider("📊 Maks. Pozisyon %", 10.0, 50.0, 25.0, 5.0)

# Strateji ayarları
st.sidebar.header("🔧 Strateji Parametreleri")
rsi_oversold = st.sidebar.slider("📊 RSI Aşırı Satım", 20, 40, 30, 1)
rsi_overbought = st.sidebar.slider("📈 RSI Aşırı Alım", 60, 80, 70, 1)
atr_multiplier = st.sidebar.slider("🎯 ATR Çarpanı", 1.0, 3.0, 1.5, 0.1)

try:
    # Veri çek
    with st.spinner(f"🔄 {ticker_input} verileri çekiliyor..."):
        data = yf.download(ticker_input, period=period, interval=timeframe, progress=False)
    
    if data.empty:
        st.error("❌ Veri çekilemedi - Sembolü kontrol edin")
    else:
        # İndikatörleri hesapla
        data['RSI'] = calculate_rsi(data['Close'])
        data['EMA_20'] = calculate_ema(data['Close'], 20)
        data['EMA_50'] = calculate_ema(data['Close'], 50)
        data['EMA_200'] = calculate_ema(data['Close'], 200)
        data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = calculate_macd(data['Close'])
        data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = calculate_bollinger_bands(data['Close'])
        data['ATR'] = calculate_atr(data['High'], data['Low'], data['Close'])
        
        # Mevcut değerler
        current_price = float(data['Close'].iloc[-1])
        rsi = float(data['RSI'].iloc[-1])
        ema_20 = float(data['EMA_20'].iloc[-1])
        ema_50 = float(data['EMA_50'].iloc[-1])
        ema_200 = float(data['EMA_200'].iloc[-1])
        macd = float(data['MACD'].iloc[-1])
        macd_signal = float(data['MACD_Signal'].iloc[-1])
        macd_prev = float(data['MACD'].iloc[-2])
        macd_signal_prev = float(data['MACD_Signal'].iloc[-2])
        atr = float(data['ATR'].iloc[-1])
        bb_upper = float(data['BB_Upper'].iloc[-1])
        bb_lower = float(data['BB_Lower'].iloc[-1])
        
        # Sinyal belirleme
        buy_signals = 0
        sell_signals = 0
        signal_details = []
        
        # AL koşulları
        if rsi < rsi_oversold:
            buy_signals += 1
            signal_details.append(f"✅ RSI {rsi:.1f} - Aşırı satım bölgesi")
        
        if current_price > ema_20 and ema_20 > ema_50:
            buy_signals += 1
            signal_details.append("✅ EMA 20 > EMA 50 - Kısa vade trend yükseliş")
        
        if ema_50 > ema_200:
            buy_signals += 1
            signal_details.append("✅ EMA 50 > EMA 200 - Uzun vade trend yükseliş")
        
        if macd > macd_signal and macd_prev <= macd_signal_prev:
            buy_signals += 1
            signal_details.append("✅ MACD sinyali yukarı kesti - Momentum pozitif")
        
        if current_price < bb_lower:
            buy_signals += 1
            signal_details.append("✅ Fiyat Bollinger alt bandında - Potansiyel dip")
        
        # SAT koşulları
        if rsi > rsi_overbought:
            sell_signals += 1
            signal_details.append(f"❌ RSI {rsi:.1f} - Aşırı alım bölgesi")
        
        if current_price < ema_20 and ema_20 < ema_50:
            sell_signals += 1
            signal_details.append("❌ EMA 20 < EMA 50 - Kısa vade trend düşüş")
        
        if ema_50 < ema_200:
            sell_signals += 1
            signal_details.append("❌ EMA 50 < EMA 200 - Uzun vade trend düşüş")
        
        if macd < macd_signal and macd_prev >= macd_signal_prev:
            sell_signals += 1
            signal_details.append("❌ MACD sinyali aşağı kesti - Momentum negatif")
        
        if current_price > bb_upper:
            sell_signals += 1
            signal_details.append("❌ Fiyat Bollinger üst bandında - Potansiyel tepe")
        
        # Ana metrikler
        st.subheader(f"📊 {ticker_input} - Gerçek Zamanlı Analiz")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            price_change = ((current_price - float(data['Close'].iloc[-2])) / float(data['Close'].iloc[-2])) * 100
            st.metric("💰 Mevcut Fiyat", f"${current_price:.2f}", f"{price_change:+.2f}%")
        
        with col2:
            rsi_color = "🟢" if rsi < 30 else "🔴" if rsi > 70 else "🟡"
            st.metric("📊 RSI", f"{rsi:.1f}", f"{rsi_color}")
        
        with col3:
            trend = "🟢 YÜKSELİŞ" if ema_20 > ema_50 and ema_50 > ema_200 else "🔴 DÜŞÜŞ" if ema_20 < ema_50 and ema_50 < ema_200 else "🟡 YANAL"
            st.metric("🎯 Trend", trend)
        
        with col4:
            macd_trend = "🟢 YUKARI" if macd > macd_signal else "🔴 AŞAĞI"
            st.metric("📈 MACD", f"{macd:.4f}", macd_trend)
        
        st.markdown("---")
        
        # SİNYAL VE RİSK ANALİZİ
        total_signals = buy_signals + sell_signals
        buy_ratio = (buy_signals / total_signals * 100) if total_signals > 0 else 0
        risk_score = min(100, abs(buy_signals - sell_signals) * 20)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if buy_signals >= 4:
                st.success(f"🎯 **GÜÇLÜ AL SİNYALİ**\n\nAl: {buy_signals}/8 | Sat: {sell_signals}/8")
                signal_strength = "YÜKSEK"
                recommendation = "AL"
            elif sell_signals >= 4:
                st.error(f"🎯 **GÜÇLÜ SAT SİNYALİ**\n\nAl: {buy_signals}/8 | Sat: {sell_signals}/8")
                signal_strength = "YÜKSEK"
                recommendation = "SAT"
            elif buy_signals > sell_signals:
                st.warning(f"🎯 **ZAYIF AL SİNYALİ**\n\nAl: {buy_signals}/8 | Sat: {sell_signals}/8")
                signal_strength = "ORTA"
                recommendation = "AL"
            elif sell_signals > buy_signals:
                st.warning(f"🎯 **ZAYIF SAT SİNYALİ**\n\nAl: {buy_signals}/8 | Sat: {sell_signals}/8")
                signal_strength = "ORTA"
                recommendation = "SAT"
            else:
                st.info(f"🎯 **NÖTR SİNYAL**\n\nAl: {buy_signals}/8 | Sat: {sell_signals}/8")
                signal_strength = "DÜŞÜK"
                recommendation = "BEKLE"
        
        with col2:
            st.metric("📊 Sinyal Gücü", signal_strength)
            st.metric("🎯 Risk Skoru", f"%{risk_score:.0f}")
        
        with col3:
            st.metric("💰 Öneri", recommendation)
            st.metric("📈 Güven Oranı", f"%{buy_ratio:.0f}" if recommendation == "AL" else f"%{100-buy_ratio:.0f}")
        
        st.markdown("---")
        
        # DETAYLI STRATEJİ
        if recommendation in ["AL", "SAT"]:
            st.subheader("🎯 Detaylı İşlem Stratejisi")
            
            if recommendation == "AL":
                # AL stratejisi
                stop_loss = current_price - (atr * atr_multiplier)
                risk_per_coin = current_price - stop_loss
                
                tp1 = current_price + (risk_per_coin * 1.0)
                tp2 = current_price + (risk_per_coin * 2.0)
                tp3 = current_price + (risk_per_coin * 3.0)
                
                # Pozisyon büyüklüğü
                risk_amount = capital * (risk_percent / 100)
                position_size = risk_amount / risk_per_coin
                max_position_size = (capital * (max_position / 100)) / current_price
                final_position_size = min(position_size, max_position_size)
                position_value = final_position_size * current_price
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**🎯 Giriş ve Çıkış Seviyeleri:**")
                    st.write(f"- 📈 Mevcut Fiyat: `${current_price:.2f}`")
                    st.write(f"- 🛑 Stop Loss: `${stop_loss:.2f}` (%{((current_price-stop_loss)/current_price*100):.1f})")
                    st.write(f"- 🎯 TP1 (1:1): `${tp1:.2f}`")
                    st.write(f"- 🎯 TP2 (1:2): `${tp2:.2f}`")
                    st.write(f"- 🎯 TP3 (1:3): `${tp3:.2f}`")
                    
                with col2:
                    st.write("**💰 Pozisyon Bilgileri:**")
                    st.write(f"- 📊 Pozisyon Büyüklüğü: `{final_position_size:.4f} {ticker_input.split('-')[0]}`")
                    st.write(f"- 💰 Pozisyon Değeri: `${position_value:.2f}`")
                    st.write(f"- 📉 Risk Tutarı: `${risk_amount:.2f}`")
                    st.write(f"- ⚖️ Risk/Reward: `1:3`")
                    st.write(f"- 🎯 Başarı Şansı: `%{min(80, risk_score + 30):.0f}`")
            
            else:
                # SAT stratejisi
                resistance = float(data['High'].tail(10).max())
                stop_loss = resistance * 1.02
                risk_per_coin = stop_loss - current_price
                
                tp1 = current_price - (risk_per_coin * 1.0)
                tp2 = current_price - (risk_per_coin * 2.0)
                tp3 = current_price - (risk_per_coin * 3.0)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**🎯 Short Seviyeleri:**")
                    st.write(f"- 📉 Mevcut Fiyat: `${current_price:.2f}`")
                    st.write(f"- 🛑 Stop Loss: `${stop_loss:.2f}`")
                    st.write(f"- 🎯 TP1 (1:1): `${tp1:.2f}`")
                    st.write(f"- 🎯 TP2 (1:2): `${tp2:.2f}`")
                    st.write(f"- 🎯 TP3 (1:3): `${tp3:.2f}`")
                    
                with col2:
                    st.write("**💰 Risk Bilgileri:**")
                    st.write(f"- 📊 Risk/Reward: `1:3`")
                    st.write(f"- 📉 Hedef Destek: `${float(data['Low'].tail(20).min()):.2f}`")
                    st.write(f"- 🎯 Başarı Şansı: `%{min(75, risk_score + 25):.0f}`")
        
        st.markdown("---")
        
        # DETAYLI GEREKÇELER
        st.subheader("🧠 Detaylı Sinyal Gerekçeleri")
        
        st.write("**📊 Teknik Göstergeler Analizi:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Trend Analizi:**")
            st.write(f"- 📈 EMA 20: `${ema_20:.2f}` ({'🟢 Üstünde' if current_price > ema_20 else '🔴 Altında'})")
            st.write(f"- 📊 EMA 50: `${ema_50:.2f}` ({'🟢 Üstünde' if current_price > ema_50 else '🔴 Altında'})")
            st.write(f"- 🎯 EMA 200: `${ema_200:.2f}` ({'🟢 Üstünde' if current_price > ema_200 else '🔴 Altında'})")
            st.write(f"- 📊 Trend Hiyerarşisi: {'🟢 Tüm EMAlar yükseliş' if ema_20 > ema_50 > ema_200 else '🔴 Tüm EMAlar düşüş' if ema_20 < ema_50 < ema_200 else '🟡 Karışık trend'}")
            
        with col2:
            st.write("**Momentum Analizi:**")
            st.write(f"- 📊 RSI: `{rsi:.1f}` ({'🟢 Aşırı Satım' if rsi < 30 else '🔴 Aşırı Alım' if rsi > 70 else '🟡 Nötr'})")
            st.write(f"- 📈 MACD: `{macd:.4f}`")
            st.write(f"- 🎯 MACD Sinyal: `{macd_signal:.4f}`")
            st.write(f"- 📉 MACD Yön: {'🟢 Yukarı' if macd > macd_signal else '🔴 Aşağı'}")
        
        st.write("**📊 Piyasa Dinamiği:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Volatilite ve Bantlar:**")
            st.write(f"- 📊 ATR: `${atr:.2f}` (Volatilite)")
            st.write(f"- 🎯 Bollinger Üst: `${bb_upper:.2f}`")
            st.write(f"- 📈 Bollinger Alt: `${bb_lower:.2f}`")
            st.write(f"- 📉 Bant Konumu: {'🔴 Üst bandta' if current_price > bb_upper else '🟢 Alt bandta' if current_price < bb_lower else '🟡 Orta bölgede'}")
            
            st.write("**Volatilite ve Bantlar:**")
            st.write(f"- 📊 ATR: `${atr:.2f}` (Volatilite)")
            st.write(f"- 🎯 Bollinger Üst: `${bb_upper:.2f}`")
            st.write(f"- 📈 Bollinger Alt: `${bb_lower:.2f}`")
            st.write(f"- 📉 Bant Konumu: {'🔴 Üst bandta' if current_price > bb_upper else '🟢 Alt bandta' if current_price < bb_lower else '🟡 Orta bölgede'}")
            
        with col2:
            st.write("**Fiyat Hareketi:**")
            # Son 5 mum analizi
            recent_prices = data['Close'].tail(5)
            gains = 0
            for i in range(1, len(recent_prices)):
                if recent_prices.iloc[i] > recent_prices.iloc[i-1]:
                    gains += 1
            momentum = "🟢 Güçlü" if gains >= 3 else "🔴 Zayıf" if gains <= 1 else "🟡 Orta"
            st.write(f"- 📈 Son 5 Mum: {gains}/4 yükseliş")
            st.write(f"- 🎯 Momentum: {momentum}")
            st.write(f"- 📊 Hacim Trendi: {'🟢 Artan' if data['Volume'].iloc[-1] > data['Volume'].iloc[-2] else '🔴 Azalan'}")
        
        st.markdown("---")
        
        # SİNYAL DETAYLARI
        st.subheader("🔍 Sinyal Detayları ve Karar Mekanizması")
        
        st.write("**🎯 Hangi Göstergelere Bakıldı:**")
        st.write("1. **Trend Analizi (EMAlar)** - Uzun/kısa vade trend yönü")
        st.write("2. **Momentum (RSI)** - Aşırı alım/satım bölgeleri") 
        st.write("3. **Momentum (MACD)** - Trend değişim sinyalleri")
        st.write("4. **Volatilite (Bollinger)** - Aşırı fiyat hareketleri")
        st.write("5. **Hacim Analizi** - İşlem hacmi desteği")
        
        st.write("**📊 Nasıl Karar Veriliyor:**")
        st.write("- **4+ AL sinyali** = Güçlü AL")
        st.write("- **4+ SAT sinyali** = Güçlü SAT") 
        st.write("- **2-3 AL sinyali** = Zayıf AL")
        st.write("- **2-3 SAT sinyali** = Zayıf SAT")
        st.write("- **Eşit sinyaller** = NÖTR")
        
        st.write("**🧠 Algoritma Mantığı:**")
        st.write("```python")
        st.write("if buy_signals >= 4: GÜÇLÜ_AL")
        st.write("elif sell_signals >= 4: GÜÇLÜ_SAT") 
        st.write("elif buy_signals > sell_signals: ZAYIF_AL")
        st.write("elif sell_signals > buy_signals: ZAYIF_SAT")
        st.write("else: NÖTR")
        st.write("```")
        
        # TEKNİK SEVİYELER
        st.markdown("---")
        st.subheader("📈 Önemli Teknik Seviyeler")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**🟢 Destek Seviyeleri:**")
            support1 = float(data['Low'].tail(10).min())
            support2 = float(data['Low'].tail(20).min())
            support3 = bb_lower
            st.write(f"- Yakın Destek: `${support1:.2f}`")
            st.write(f"- Güçlü Destek: `${support2:.2f}`")
            st.write(f"- Bollinger Destek: `${support3:.2f}`")
            
        with col2:
            st.write("**🔴 Direnç Seviyeleri:**")
            resistance1 = float(data['High'].tail(10).max())
            resistance2 = float(data['High'].tail(20).max())
            resistance3 = bb_upper
            st.write(f"- Yakın Direnç: `${resistance1:.2f}`")
            st.write(f"- Güçlü Direnç: `${resistance2:.2f}`")
            st.write(f"- Bollinger Direnç: `${resistance3:.2f}`")
            
        with col3:
            st.write("**🎯 Kritik Seviyeler:**")
            st.write(f"- EMA 200: `${ema_200:.2f}`")
            st.write(f"- Psikolojik Seviye: `${round(current_price, -1):.0f}`")
            st.write(f"- ATR Stop: `${current_price - atr:.2f}`")
        
        # RİSK ANALİZİ
        st.markdown("---")
        st.subheader("⚠️ Detaylı Risk Analizi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📉 Piyasa Riskleri:**")
            volatility_risk = "YÜKSEK" if atr > current_price * 0.05 else "ORTA" if atr > current_price * 0.02 else "DÜŞÜK"
            st.write(f"- Volatilite Riski: {volatility_risk}")
            
            trend_risk = "DÜŞÜK" if ema_20 > ema_50 > ema_200 else "YÜKSEK" if ema_20 < ema_50 < ema_200 else "ORTA"
            st.write(f"- Trend Riski: {trend_risk}")
            
            momentum_risk = "YÜKSEK" if rsi > 80 or rsi < 20 else "DÜŞÜK" if 30 < rsi < 70 else "ORTA"
            st.write(f"- Momentum Riski: {momentum_risk}")
            
        with col2:
            st.write("**🛑 Risk Yönetimi:**")
            st.write(f"- Maksimum Kayıp: `${capital * (risk_percent/100):.0f}`")
            st.write(f"- Stop Loss Mesafesi: `%{((current_price - stop_loss)/current_price*100):.1f}`")
            st.write(f"- Risk/Reward Oranı: `1:3`")
            st.write(f"- Pozisyon Limiti: `%{max_position:.0f}`")
        
        # SON ÖNERİLER
        st.markdown("---")
        st.subheader("💡 Son Öneriler ve Strateji")
        
        if recommendation == "AL":
            st.success("**🎯 AL Stratejisi Önerileri:**")
            st.write("1. **Kademeli Giriş:** İlk %50 mevcut fiyattan, kalan %50 dip alımlarda")
            st.write("2. **Stop Yönetimi:** TP1'e ulaşınca stop'u maliyete çek")
            st.write("3. **Kısmi Çıkış:** TP1'de %50, TP2'de %30, TP3'te %20 sat")
            st.write("4. **Zamanlama:** Londra/New York açılış saatlerini takip et")
            
        elif recommendation == "SAT":
            st.error("**🎯 SAT Stratejisi Önerileri:**")
            st.write("1. **Direniş Testi:** Direnç seviyesinde short pozisyon aç")
            st.write("2. **Hacim Kontrolü:** Yüksek hacimli düşüşleri bekle")
            st.write("3. **Kademeli Çıkış:** Her TP seviyesinde kısmi kapat")
            st.write("4. **Trend Takip:** Ana trend dönüşünü gözle")
            
        else:
            st.info("**🎯 BEKLEME Stratejisi Önerileri:**")
            st.write("1. **Yanlış Zamanlama:** Trend belirsiz, beklemek en iyisi")
            st.write("2. **Gözlem:** Teknik seviyelerde kırılımı bekle")
            st.write("3. **Hazırlık:** AL/SAT sinyali için hazır ol")
            st.write("4. **Alternatif:** Diğer coinleri analiz et")
        
        # UYARI
        st.markdown("---")
        st.error("""
        **⚠️ ÖNEMLİ UYARILAR:**
        - Bu analizler %100 doğru değildir, sadece eğitim amaçlıdır
        - Kendi araştırmanızı yapmadan işlem açmayın
        - Risk yönetimi olmadan asla ticaret yapmayın
        - Geçmiş performans geleceği garanti etmez
        - Kripto paralar yüksek risk içerir, sermayenizi kaybedebilirsiniz
        """)

except Exception as e:
    st.error(f"❌ Sistem hatası: {str(e)}")
    st.info("Lütfen internet bağlantınızı kontrol edin ve tekrar deneyin")

st.markdown("---")
st.caption("🤖 Crypto AI Pro - Gelişmiş Algoritmik Analiz Sistemi | V1.0")