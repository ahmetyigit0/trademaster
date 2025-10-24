import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Streamlit arayüzü
st.set_page_config(page_title="Kripto Teknik Analiz", layout="wide")
st.title("🎯 Kripto Teknik Analiz")

# Sidebar
st.sidebar.header("⚙️ Analiz Ayarları")
crypto_symbol = st.sidebar.text_input("Kripto Sembolü:", "BTC-USD")
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

def safe_float_format(value):
    """Güvenli float formatlama"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def main():
    try:
        # Veri çekme
        interval = interval_map[analysis_type]
        st.write(f"**{crypto_symbol}** için {analysis_type} veriler çekiliyor...")
        
        data = get_crypto_data(crypto_symbol, lookback_days, interval)
        
        if data is None or data.empty:
            st.error("Veri çekilemedi. Lütfen sembolü kontrol edin.")
            return
        
        st.success(f"✅ {len(data)} adet {analysis_type} mum verisi çekildi")
        
        # Veri tiplerini kontrol et
        st.write("Veri Tipleri:")
        st.write(data.dtypes)
        
        # Basit veri gösterimi - ÇOK GÜVENLİ VERSİYON
        with st.expander("📜 Son Mum Verileri"):
            display_data = data.tail(10)[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            
            # Tüm değerleri float'a çevir ve formatla
            for col in ['Open', 'High', 'Low', 'Close']:
                display_data[col] = display_data[col].apply(lambda x: f"${safe_float_format(x):.2f}")
            
            # Volume için özel format
            display_data['Volume'] = display_data['Volume'].apply(lambda x: f"{safe_float_format(x):,.0f}")
            
            st.dataframe(display_data)
            
        # Mevcut fiyat bilgisi
        current_price = safe_float_format(data['Close'].iloc[-1])
        st.metric("Mevcut Fiyat", f"${current_price:.2f}")
        
    except Exception as e:
        st.error(f"❌ Hata oluştu: {str(e)}")
        import traceback
        st.error(f"Detay: {traceback.format_exc()}")

if __name__ == "__main__":
    main()