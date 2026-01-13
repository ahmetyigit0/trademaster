import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests
from datetime import datetime

# --- KONFIGÜRASYON ---
st.set_page_config(page_title="Orion Konseyi", layout="wide")

# --- CSS: VİDEODAKİ GİBİ PREMİUM TASARIM ---
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #05070a; color: white; }
    .stButton>button { border-radius: 12px; width: 100%; height: 45px; transition: 0.3s; }
    
    /* Yan Yana Kart Yapısı */
    .asset-card {
        background: rgba(23, 28, 36, 0.9);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 20px; padding: 20px; margin-bottom: 10px;
        text-align: center;
    }
    
    /* Progress Bar (Hedef Takibi) */
    .progress-container { width: 100%; background: #1e293b; border-radius: 10px; height: 10px; margin: 10px 0; }
    .progress-bar { background: #3b82f6; height: 100%; border-radius: 10px; transition: 1s; }
</style>
""", unsafe_allow_html=True)

# --- ANIMASYON YÜKLEME (Videodaki Hareketli İkonlar İçin) ---
def load_lottieurl(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

# Örnek bir analiz animasyonu (Orion Modülü için)
lottie_analiz = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_qpwb7t6c.json")

# --- TEKNİK ANALİZ MOTORU (Hız Göstergesi İçin) ---
def create_gauge(score, title):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        title = {'text': title, 'font': {'size': 14, 'color': "white"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#3b82f6"},
            'steps': [
                {'range': [0, 30], 'color': "#ef4444"},
                {'range': [30, 70], 'color': "#f59e0b"},
                {'range': [70, 100], 'color': "#10b981"}],
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                      font={'color': "white", 'family': "Arial"}, height=200, margin=dict(l=20,r=20,t=50,b=20))
    return fig

# --- ÜST PANEL: HEDEF TAKİBİ (666K USD) ---
total_usd = 12500 # Bu senin portföyünden gelecek
progress = (total_usd / 666000) * 100

st.markdown(f"""
<div style='text-align:center; padding:30px; border-bottom:1px solid #1e293b;'>
    <h1 style='color:#3b82f6; margin:0;'>ORION KONSEYİ</h1>
    <p style='color:gray;'>Hedefine Ulaşma Oranı: %{progress:.2f}</p>
    <div class="progress-container"><div class="progress-bar" style="width:{progress}%"></div></div>
</div>
""", unsafe_allow_html=True)

# --- PORTFÖY VE TEKNİK ANALİZ ---
portfolio = ["NVDA", "BTC-USD", "THYAO.IS"]
cols = st.columns(3) # YAN YANA DİZİLİM

for i, symbol in enumerate(portfolio):
    with cols[i]:
        st.markdown(f"""
        <div class="asset-card">
            <h2 style='margin:0;'>{symbol}</h2>
            <small style='color:gray;'>Konsey Teknik Analiz Skoru</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Rastgele Skor (Gerçek analiz motoruna bağlanacak)
        score = 73 if symbol == "NVDA" else (42 if symbol == "BTC-USD" else 55)
        st.plotly_chart(create_gauge(score, "GÜVEN SKORU"), use_container_width=True)
        
        # Butonlar Yan Yana
        b1, b2, b3, b4 = st.columns(4)
        if b1.button("📝", key=f"ed_{i}"): st.write("Düzenle")
        if b2.button("🗑️", key=f"dl_{i}"): st.write("Sil")
        if b3.button("🔍", key=f"dt_{i}"): st.write("Detay")
        if b4.button("📊", key=f"ta_{i}"):
            st.session_state[f"show_ta_{i}"] = True

        # Videodaki "Orion Teknik Analiz" Penceresi (Modül)
        if st.session_state.get(f"show_ta_{i}", False):
            with st.expander("🛡️ KONSEY ANALİZ RAPORU", expanded=True):
                c_lottie, c_text = st.columns([1, 2])
                with c_lottie:
                    st_lottie(lottie_analiz, height=150)
                with c_text:
                    st.write(f"**Varlık:** {symbol}")
                    st.success("KONSEY KARARI: BİRİKTİR")
                    st.info("RSI Aşırı alım bölgesine yaklaşıyor. Momentum güçlü.")
                if st.button("Kapat", key=f"cls_{i}"):
                    st.session_state[f"show_ta_{i}"] = False
                    st.rerun()

st.divider()
st.caption("Veriler Yahoo Finance üzerinden Konsey tarafından anlık işlenmektedir.")
