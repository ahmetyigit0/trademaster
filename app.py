import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# Sayfa Ayarları
st.set_page_config(page_title="Orion Trading Journal", layout="wide")

# Veri Saklama Simülasyonu (Gerçek projede SQLite veya CSV kullanılabilir)
if 'journal' not in st.session_state:
    st.session_state.journal = pd.DataFrame(columns=[
        "Tarih", "Enstrüman", "Yön", "Giriş", "Çıkış", "Miktar", "Kar/Zarar", "Strateji", "Duygu Skoru", "Notlar"
    ])

# --- Yan Panel (Veri Girişi) ---
with st.sidebar:
    st.header("➕ Yeni İşlem Ekle")
    with st.form("trade_form", clear_on_submit=True):
        asset = st.text_input("Enstrüman (Örn: BTC, THYAO, NVDA)")
        direction = st.selectbox("Yön", ["Long", "Short"])
        entry_price = st.number_input("Giriş Fiyatı", min_value=0.0, format="%.4f")
        exit_price = st.number_input("Çıkış Fiyatı", min_value=0.0, format="%.4f")
        amount = st.number_input("Miktar", min_value=0.0, format="%.4f")
        
        st.divider()
        strategy = st.selectbox("Strateji", ["Likidasyon Bölgesi", "RSI Uyumsuzluğu", "Trend Takibi", "Haber/FOMO"])
        emotion = st.slider("İşlem Anındaki Disiplin/Duygu (1-10)", 1, 10, 5)
        notes = st.text_area("İşlem Notu (Neden açtın?)")
        
        submitted = st.form_submit_button("İşlemi Kaydet")
        
        if submitted:
            pnl = (exit_price - entry_price) * amount if direction == "Long" else (entry_price - exit_price) * amount
            new_data = {
                "Tarih": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Enstrüman": asset.upper(),
                "Yön": direction,
                "Giriş": entry_price,
                "Çıkış": exit_price,
                "Miktar": amount,
                "Kar/Zarar": round(pnl, 2),
                "Strateji": strategy,
                "Duygu Skoru": emotion,
                "Notlar": notes
            }
            st.session_state.journal = pd.concat([st.session_state.journal, pd.DataFrame([new_data])], ignore_index=True)
            st.success("İşlem günlüğe eklendi!")

# --- Ana Ekran (Dashboard) ---
st.title("📈 Trading Disiplin Merkezi")

if not st.session_state.journal.empty:
    df = st.session_state.journal
    
    # Metrikler
    col1, col2, col3, col4 = st.columns(4)
    total_pnl = df["Kar/Zarar"].sum()
    win_rate = len(df[df["Kar/Zarar"] > 0]) / len(df) * 100
    
    col1.metric("Net Kar/Zarar", f"${total_pnl:,.2f}")
    col2.metric("Win Rate", f"%{win_rate:,.1f}")
    col3.metric("Toplam İşlem", len(df))
    col4.metric("Ort. Disiplin Skoru", f"{df['Duygu Skoru'].mean():,.1f}/10")

    st.divider()

    # Grafikler
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Kümülatif Kar/Zarar")
        df['Cum_PnL'] = df['Kar/Zarar'].cumsum()
        fig_pnl = px.line(df, x=df.index, y='Cum_PnL', markers=True, color_discrete_sequence=['#00FFCC'])
        st.plotly_chart(fig_pnl, use_container_width=True)

    with c2:
        st.subheader("Strateji Bazlı Performans")
        strat_pnl = df.groupby("Strateji")["Kar/Zarar"].sum().reset_index()
        fig_strat = px.bar(strat_pnl, x="Strateji", y="Kar/Zarar", color="Kar/Zarar", color_continuous_scale="RdYlGn")
        st.plotly_chart(fig_strat, use_container_width=True)

    # İşlem Geçmişi Tablosu
    st.subheader("📜 İşlem Geçmişi")
    st.dataframe(df.sort_index(ascending=False), use_container_width=True)

else:
    st.info("Henüz işlem girilmedi. Sol taraftaki panelden ilk işlemini ekleyerek başlayabilirsin.")
