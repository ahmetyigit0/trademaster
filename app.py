import streamlit as st
import pandas as pd
from datetime import datetime

# Sayfa Yapılandırması
st.set_page_config(page_title="Pro-Trade Executive", layout="wide")

# Stil Dosyası (Kart Görünümü İçin Custom CSS)
st.markdown("""
    <style>
    .trade-card {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #00ffcc;
        margin-bottom: 15px;
    }
    .metric-label { color: #888; font-size: 14px; }
    .metric-value { font-size: 18px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# Veri Yönetimi
if 'trades' not in st.session_state:
    st.session_state.trades = []

# --- BAŞLIK VE ÖZET ---
st.title("⚡ Pro-Trade Executive Terminal")
c1, c2, c3 = st.columns(3)

# --- İŞLEM EKLEME ALANI ---
with st.expander("➕ Yeni Pozisyon Planla / Aç", expanded=True):
    with st.form("quick_trade"):
        col1, col2, col3, col4 = st.columns(4)
        asset = col1.text_input("Enstrüman", placeholder="Örn: BTCUSDT")
        side = col2.selectbox("Yön", ["LONG", "SHORT"])
        entry = col3.number_input("Giriş Fiyatı", format="%.4f")
        size = col4.number_input("Pozisyon Büyüklüğü ($)", min_value=0.0)
        
        col5, col6, col7 = st.columns(3)
        sl = col5.number_input("Stop Loss (SL)", format="%.4f")
        tp = col6.number_input("Take Profit (TP)", format="%.4f")
        status = col7.selectbox("Durum", ["Açık", "Kapalı"])
        
        submit = st.form_submit_button("Pozisyonu Kaydet")
        
        if submit and asset:
            # R:R ve Hesaplamalar
            risk = abs(entry - sl)
            reward = abs(tp - entry)
            rr_ratio = round(reward / risk, 2) if risk != 0 else 0
            
            new_trade = {
                "id": len(st.session_state.trades) + 1,
                "asset": asset.upper(),
                "side": side,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "rr": rr_ratio,
                "size": size,
                "status": status,
                "date": datetime.now().strftime("%H:%M:%S")
            }
            st.session_state.trades.append(new_trade)
            st.success(f"{asset} Kaydedildi! R:R Oranı: 1:{rr_ratio}")

# --- SEKMELER (Açık / Kapalı Pozisyonlar) ---
tab1, tab2 = st.tabs(["📂 Açık Pozisyonlar", "✅ Kapanmış İşlemler"])

def render_trade_card(t):
    # Renk belirleme (Yön ve Duruma göre)
    border_color = "#00ffcc" if t['side'] == "LONG" else "#ff4b4b"
    
    st.markdown(f"""
        <div style="background-color: #262730; border-radius: 10px; padding: 15px; border-left: 6px solid {border_color}; margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between;">
                <span style="font-size: 20px; font-weight: bold;">{t['asset']} ({t['side']})</span>
                <span style="color: #888;">{t['date']}</span>
            </div>
            <hr style="margin: 10px 0; border-color: #444;">
            <div style="display: flex; justify-content: space-between; text-align: center;">
                <div><div class="metric-label">Giriş</div><div class="metric-value">{t['entry']}</div></div>
                <div><div class="metric-label">Stop</div><div class="metric-value" style="color: #ff4b4b;">{t['sl']}</div></div>
                <div><div class="metric-label">Hedef</div><div class="metric-value" style="color: #00ffcc;">{t['tp']}</div></div>
                <div><div class="metric-label">Risk/Reward</div><div class="metric-value">1:{t['rr']}</div></div>
                <div><div class="metric-label">Büyüklük</div><div class="metric-value">${t['size']}</div></div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    if st.button(f"Durumu Değiştir / Kapat (#{t['id']})"):
        t['status'] = "Kapalı" if t['status'] == "Açık" else "Açık"
        st.rerun()

with tab1:
    open_trades = [t for t in st.session_state.trades if t['status'] == "Açık"]
    if not open_trades:
        st.write("Şu an açık pozisyonun yok. Disiplin iyidir!")
    for t in open_trades:
        render_trade_card(t)

with tab2:
    closed_trades = [t for t in st.session_state.trades if t['status'] == "Kapalı"]
    if not closed_trades:
        st.write("Henüz kapalı işlem yok.")
    for t in closed_trades:
        render_trade_card(t)

# --- İSTATİSTİK ÖZETİ (Dashboard Altı) ---
st.divider()
if st.session_state.trades:
    df = pd.DataFrame(st.session_state.trades)
    total_trades = len(df)
    avg_rr = df['rr'].mean()
    st.info(f"📊 Toplam İşlem Sayısı: {total_trades} | Ortalama R:R Oranı: 1:{avg_rr:.2f}")
