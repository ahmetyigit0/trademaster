import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

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
                signals.append(f"🎯 DESTEK YAKIN: ${nearest_support:.2f} (%{distance_to_support:.1f} uzak)")
                if prev_price > current_price:  # Düşüş trendinde
                    signals.append("🔄 DESTEK TESTİ - POTANSİYEL ALIM")
        
        if nearest_resistance:
            distance_to_resistance = ((nearest_resistance - current_price) / current_price) * 100
            if distance_to_resistance <= 2:  # %2'den yakınsa
                signals.append(f"🎯 DİRENÇ YAKIN: ${nearest_resistance:.2f} (%{distance_to_resistance:.1f} uzak)")
                if prev_price < current_price:  # Yükseliş trendinde
                    signals.append("🔄 DİRENÇ TESTİ - POTANSİYEL SATIM")
        
        # Kırılma sinyalleri
        if nearest_support and current_price < nearest_support and prev_price >= nearest_s_price >= nearest_support:
upport:
            signals.append("            signals.append("❌ DEST❌ DESTEK KEK KIRILDI -IRILDI - SAT SATIM SİNYIM SİNYALİALİ")
        
        if")
        
        if nearest_res nearest_resistance and current_priceistance and current_price > nearest_res > nearest_resistance and previstance and prev_price <= nearest_res_price <= nearest_resistance:
           istance:
            signals.append(" signals.append("✅ DİR✅ DİRENÇENÇ KIRILDI KIRILDI - AL - ALIM SİNYIM SİNYALİALİ")
        
        return")
        
        return signals
        
 signals
        
    except Exception as e    except Exception as e:
        return [:
        return [f"Sinyf"Sinyal hatası: {al hatası: {str(e)}"]

defstr(e)}"]

def main():
    try:
        interval = main():
    try:
        interval = interval_map[analysis_type]
 interval_map[analysis_type]
        st.write(f"**        st.write(f"**{{crypto_symbol}** içincrypto_symbol}** için { {analysis_type} veriler çanalysis_type} veriler çekiliyor...")
        
ekiliyor...")
        
        data = get_crypto        data = get_crypto_data(crypto_symbol, look_data(crypto_symbol, lookback_days, interval)
back_days, interval)
        
        
        if data is None        if data is None or data or data.empty:
            st.empty:
            st.error("Veri çek.error("Veri çekilemediilemedi.")
            return
        
        st.")
            return
        
        st.success(f"✅ {len.success(f"✅ {len(data)} adet mum ver(data)} adet mum verisi çekildi")
isi çekildi")
        
        # Destek/d        
        # Destek/direnç seviyelerini bulirenç seviyelerini bul
        support_levels,
        support_levels, resistance_levels resistance_levels = = find_support_resistance_levels find_support_resistance_levels(
(
            data, sensitivity, min_t            data, sensitivity, min_touchouch_points, wick_analysis_points, wick_analysis
       
        )
        
        # Pivot point hesapla )
        
        # Pivot point hesapla
        pivot, pivot
        pivot, pivot_supports, pivot_resist_supports, pivot_resistances = calculate_pivot_points(dataances = calculate_pivot_points(data)
        
        # Mev)
        
        # Mevcutcut fiyat
        current_price = float(data['Close'].iloc[-1 fiyat
        current_price = float(data['Close'].iloc[-1])
        
        # Trading sinyall])
        
        # Trading sinyalleri üret
        signalseri üret
        signals = generate_trading_signals = generate_trading_signals_with_levels(data, support_with_levels(data, support_levels_levels, resistance_levels, resistance_levels)
        
)
        
        # Ana panel
        # Ana panel
               col1, col col1, col2 = st.columns([2, 2 = st.columns([2, 1])
        
        with col1])
        
        with col11:
            st.subheader:
            st.subheader("("📈 Destek/D📈 Destek/Direnç Grafik Analiziirenç Grafik Analizi")
            
            fig = go")
            
            fig = go.Figure()
            
            #.Figure()
            
            # Çizgi grafiği Çizgi grafiği (kapanış fiy (kapanış fiyatları)
            fig.add_traceatları)
            fig.add_trace(go.Scatter(
                x=data.index,
                y(go.Scatter(
                x=data.index,
                y=data['=data['Close'],
                name='Kapanış FiyatıClose'],
                name='Kapanış Fiyatı',
                line=dict(color',
                line=dict(color='blue', width='blue', width=2=2),
                mode='),
                mode='lines'
lines'
            ))
            
                       ))
            
            # Me # Mevcut fiyvcut fiyat çizgat çizgisi
isi
            fig.add_            fig.add_hline(yhline(y=current_price,=current_price, line_dash line_dash="solid",="solid", line_color="black line_color="black", line_width", line_width=2,=2, 
                         annotation_text 
                         annotation_text=f=f"Mevcut F"Mevcut Fiyatiyat: ${current_price: ${current_price:.2:.2f}")
            
           f}")
            
            # Destek # Destek seviy seviyeleri
            for i, level in enumerateeleri
            for i,(support level in enumerate(support_levels):
               _level fig.add_hline(y=level, line_dash="s):
                fig.add_hline(y=level, linedash", line_color="green_dash="dash", line_color="green", line_width", line_width=2,
=2,
                             annotation_text                             annotation_text=f"Destek=f"Destek {i+1 {i+1}: ${}: ${level:.2flevel:.2f}")
            
}")
            
            # Diren            # Direnç seç seviyviyelereleri
            for i, leveli
            for i, level in enumerate(res in enumerate(resistance_levels):
                fig.add_hline(y=istance_levels):
                fig.add_hlinelevel, line_dash="dash", line_color="(y=levelred", line_width=2, line_dash="dash", line_color="red", line_width=2,
                             annotation_text=f"D,
                             annotation_text=f"Direnirenç {i+1ç {i+1}: ${level:.2f}: ${level:.2f}")
            
}")
            
            # Pivot            # Pivot point
 point
            if pivot:
                           if pivot:
                fig fig.add_hline(y=p.add_hline(y=pivotivot, line_dash="dot, line_dash="dot", line_color="orange",", line_color="orange", line_width line_width=2,
                             annotation=2,
                             annotation_text_text=f"Pivot=f"Pivot:: ${pivot:.2f ${pivot:.2f}")
            
            fig.update_layout}")
            
            fig.update_layout(
                height=600,
(
                height=600,
                title                title=f"{=f"{crypto_symbol} - Destekcrypto_symbol} - Destek/Diren/Direnç Analizi",
               ç Analizi",
                xaxis xaxis_title="Tarih_title="Tarih",
                yaxis_title="",
                yaxis_title="Fiyat (USD)",
                showFiyat (USD)",
                showlegend=True
            )
            
legend=True
            )
            
            st            st.plotly_chart.plotly_chart(fig, use_container_width(fig, use_container_width=True=True)
        
        with col2:
            st.subheader)
        
        with col2:
            st.subheader("("🎯 TRADING🎯 TRADING Sİ SİNYALLERİ")
NYALLERİ")
            
            
            if signals:
                           if signals:
                for for signal in signals:
                    if signal in signals:
                    if " "ALIM" in signalALIM" in signal or or "KIRILDI "KIRILDI" in" in signal and "D signal and "DİRİRENENÇ" in signal:
                        st.success(f"✅ {Ç" in signal:
                        st.success(f"✅ {signal}")
                   signal}")
                    elif "SATIM" in signal or "KIRILDI elif "SATIM" in signal or "KIRILDI" in signal" in signal and "DEST and "DESTEK" in signal:
                        st.errorEK" in signal:
                        st(f"❌ {signal.error(f"❌ {signal}")
                    elif "TEST"}")
                    elif "TEST" in signal in signal or "YAKIN" in signal:
 or "YAKIN" in signal:
                                               st.warning(f"⚠ st.warning(f"⚠️️ {signal}")
                    else:
 {signal}")
                    else:
                        st.info(f"                        st.info(f"📊 {signal}")
            else📊 {signal}")
            else:
               :
                st.info(" st.info("📊 Net trading📊 Net trading sinyali sinyali yok")
 yok")
            
            st.sub            
            st.subheader("header("📊 FİYAT ANAL📊 FİYAT ANALİZİ")
            stİZİ")
            st.metric("Mevcut.metric("Mevcut F Fiyat", f"${iyat", f"${currentcurrent_price:.2f}")
            
_price:.2f}")
            
            if support_levels            if support_levels:
                nearest_support =:
                nearest_support = max(support_levels)
 max(support_levels)
                               distance_support = distance_support = ((current ((current_price - nearest_support)_price - nearest_support) / current_price) * / current_price) *  100
                st.metric100
                st.metric("En Yakın Destek("En Yakın Destek", f"${nearest_support:.2f}", f"%", f"${nearest_support:.2f}", f"%{distance{distance_support:.1_support:.1f}")
f}")
            else:
                           else:
                st.metric(" st.metric("En YakEn Yakın Destek",ın Destek", "Bul "Bulununamadı")
            
            if resistance_levels:
               amadı")
            
            if resistance_levels:
                nearest_resistance nearest_resistance = min(res = min(resistance_levels)
istance_levels)
                distance_res                distance_resistance = ((neistance = ((nearest_resarest_resistance - current_priceistance - current_price) /) / current_price) * current_price) * 100 100
                st.m
                st.metric("etric("En Yakın DEn Yakın Dirençirenç", f"${", f"${nearest_resnearest_resistance:.2istance:.2f}", f"%{distance_resistancef}", f"%{distance_resistance:.1:.1f}")
            elsef}")
            else:
               :
                st.metric(" st.metric("En YakEn Yakın Direnın Direnç",ç", "Bul "Bulununamadı")
            
            st.subamadı")
            
            st.subheader("💎header("💎 DEST DESTEK SEVİYELERİEK SEVİYELERİ")
           ")
            if support_levels if support_levels:
               :
                for i, level for i, level in enumerate in enumerate(sorted(support_levels(sorted(support_levels, reverse, reverse=True)):
                    distance=True)):
                    distance = (( = ((current_pricecurrent_price - level) / current_price) * - level) / current_price) * 100 100
                    st.write(f"
                    st.write(f"🟢 D{i+1🟢 D{i+1}: ${level:.2f}}: ${level:.2f} (%{ (%{distance:.1fdistance:.1f} aşağıda)")
            else:
               } aşağıda)")
            else:
                st.write st.write("Destek se("Destek seviyesviyesi bulunami bulunamadı")
adı")
            
            st            
            st.subheader(".subheader("🚀 Dİ🚀 DİRENRENÇ SEVİÇ SEVİYELYELERİ")
           ERİ")
            if resistance if resistance_levels:
               _levels:
                for i for i, level in enumerate(sorted(resistance_levels)):
                    distance, level in enumerate(sorted(resistance_levels)):
                    distance = ((level - = ((level - current_price current_price) / current_price) *) / current_price) * 100
                    100
                    st.write(f" st.write(f"🔴 R{i🔴 R{i+1+1}: ${level:.2f} (%{distance:.1f} yukarıda)")
            else:
                st}: ${level:.2f} (%{distance:.1f} yukarıda)")
            else:
                st.write("D.write("Direnç seirenç seviyesiviyesi bulunam bulunamadı")
            
adı")
            
            #            # Pivot Point bil Pivot Point bilgisigisi
            if pivot
            if pivot:
                st:
                st.subheader(".subheader("⚖️⚖️ PIVOT POINT")
                st.write(f"**Pivot:** PIVOT POINT")
                st.write(f"**Pivot:** ${p ${pivot:.2fivot:.2f}")
               }")
                st.write(f" st.write(f"**S1:** ${p**S1:** ${pivotivot_supports[0]:_supports[0]:.2.2f}")
                stf}")
                st.write(f.write(f"**R1"**R1:**:** ${pivot_resist ${pivot_resistancesances[0]:.2[0]:.2f}")
f}")
        
        #        
        # Det Detaylıaylı analiz analiz
       
        st.subheader("📋 DETAYLI ANALİZ st.subheader("📋 DETAYLI ANALİZ RAPOR RAPORU")
        
       U")
        
        col3 col3, col4 =, col4 = st.columns st.columns(2)
        
(2)
        
        with        with col3:
            st col3:
            st.write.write("**📈 Tek("**📈 Teknik Özet:**")
           nik Özet:**")
            st st.write(f"- Anal.write(f"- Analiz ediz edilen mum sayısilen mum sayısı:ı: {len(data)} {len(data)}")
           ")
            st.write(f"- Tespit edilen dest st.write(f"- Tespit edilen destek seviyesi: {ek seviyesi: {lenlen(support_levels(support_levels)}")
)}")
            st.write(f"-            st.write(f"- T Tespit edilen direnespit edilen direnç seç seviyesi:viyesi: {len {len(resistance_levels)}(resistance_levels)}")
")
            st.write(f"- İ            st.write(f"- İğne analizi: {'Ağne analizi: {'Açık'çık' if w if wick_analysis elseick_analysis else 'Kapalı'}")
            'Kapalı'}")
            st st.write(f"- Hass.write(f"- Hassasiyetasiyet seviyesi: { seviyesi: {sensitivity}/10")
            
sensitivity}/10")
            
               with col4:
            with col4:
            st.write st.write("**🎯("**🎯 Trading Önerileri:**")
            if not signals:
                st Trading Önerileri:**")
            if not signals:
                st.write("- Net.write("- Net sinyal yok - Piy sinyal yok - Piyasa gözlemi öasa gözlemi önerilir")
            elifnerilir")
            elif any any("ALIM"("ALIM" in signal for signal in signals):
                in signal for signal in signals):
                st.write("- 🟢 st.write("- 🟢 ALIM yönünde ALIM yönünde sinyaller mev sinyaller mecut")
            elif any("SATIM" in signal forvcut")
            elif any("SATIM" in signal for signal in signals):
                st.write signal in signals):
                st.write("- 🔴 SATIM y("- 🔴 SATIM yönünde sinyaller mevcut")
            elseönünde sinyaller mevcut")
            else:
                st.write("- 🟡 NÖTR - Bekle ve gör")
        
       :
                st.write("- 🟡 NÖTR - Bekle ve gör")
        
        # Son 10 mumun detayları
        with st.expander("📜 SON 10 M # Son 10 mumun detayları
        with st.expander("📜 SON 10 MUM DETAYI"):
            display_data = data.tail(10)[['Open', 'High', 'Low', 'Close', 'UM DETAYI"):
            display_data = data.tail(10)[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            
            # Formatlama
Volume']].copy()
            
            # Formatlama
            for col in ['Open            for col in ['Open', 'High', 'Low', 'Close']:
                display_data', 'High', 'Low', 'Close']:
                display_data[col] = display_data[col].map(lambda x: f[col] = display_data[col].map(lambda x"${x:.2f}" if not pd.isna(x) else ": f"${x:.2f}" if not pd.isna(x) else "N/A")
            display_data['Volume'] = display_data['Volume'].map(lambda x: f"{x:,.0f}" if notN/A")
            display_data['Volume'] = display_data['Volume'].map(lambda x: f"{x:,.0f pd.isna(x) else "N/A")
            
            st.dataframe(display_data)
            
    except Exception as e}" if not pd.isna(x) else "N/A")
            
            st.dataframe(display_data)
            
    except Exception as e:
        st.error:
        st.error(f"❌ Hata olu(f"❌ Hata oluştu: {str(e)}")

if __name__ == "__main__":
    main()