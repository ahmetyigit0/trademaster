import streamlit as st
import pandas as pd
from datetime import datetime

# OKX Teması ve Kart Tasarımı için CSS
st.set_page_config(page_title="Terminal", layout="wide")

st.markdown("""
    <style>
    /* OKX Dark Theme Renkleri */
    .stApp { background-color: #000000; color: #FFFFFF; }
    div[data-testid="stExpander"] { background-color: #1a1a1a; border: 1px solid #333; border-radius: 4px; }
    .stButton>button { background-color: #ffffff; color: #000; border-radius: 4px; font-weight: bold; width: 100%; }
    .stButton>button:hover { background-color: #cccccc; }
    
    /* İşlem Kartları */
    .trade-card {
        background-color: #121212;
        border: 1px solid #2b2b2b;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 12px;
    }
    .metric-box { text-align: left; }
    .label { color: #5e5e5e; font-size: 12px; text-transform: uppercase; }
    .value { font-size: 16px; font-weight: 600; margin-top: 4px; }
    .long { color: #00b07c; } /* OKX Yeşil */
    .short { color: #ff3e2e; } /* OKX Kırmızı */
    </style>
    """, unsafe_allow_html=True)

# Session State Başlatma
if 'trades' not in st.session_state:
    st.session_state.trades = []

# --- FONKSİYONLAR ---
def delete_trade(index):
    st.session_state.trades.pop(index)
    st.rerun()

def close_trade(index, exit_price):
    trade = st.session_state.trades[index]
    trade['exit'] = exit_price
    trade['status'] = "Kapalı"
    # Kar/Zarar Hesapla
    if trade['side'] == "LONG":
        trade['pnl'] = (exit_price - trade['entry']) * (trade['size'] / trade['entry'])
    else:
        trade['pnl'] = (trade['entry'] - exit_price) * (trade['size'] / trade['entry'])
    st.rerun()

# --- ÜST PANEL: İSTATİSTİKLER ---
st.subheader("Finansal Özet")
if st.session_state.trades:
    df = pd.DataFrame(st.session_state.trades)
    closed_df = df[df['status'] == "Kapalı"]
    
    c1, c2, c3, c4, c5 = st.columns(5)
    
    total_pnl = closed_df['pnl'].sum() if not closed_df.empty else 0
    wins = len(closed_df[closed_df['pnl'] > 0])
    win_rate = (wins / len(closed_df) * 100) if not closed_df.empty else 0
    max_win = closed_df['pnl'].max() if not closed_df.empty else 0
    max_loss = closed_df['pnl'].min() if not closed_df.empty else 0

    c1.metric("Toplam PnL", f"${total_pnl:.2f}")
    c2.metric("Win Rate", f"%{win_rate:.1f}")
    c3.metric("Max Kazanç", f"${max_win:.2f}", delta_color="normal")
    c4.metric("Max Kayıp", f"${max_loss:.2f}", delta_color="inverse")
    c5.metric("Aktif Poz.", len(df[df['status'] == "Açık"]))

st.divider()

# --- İŞLEM GİRİŞ PANELİ ---
with st.expander("➕ Yeni Pozisyon Aç", expanded=False):
    with st.form("trade_entry"):
        col1, col2, col3, col4 = st.columns(4)
        asset = col1.text_input("Sembol", placeholder="BTC/USDT")
        side = col2.selectbox("Yön", ["LONG", "SHORT"])
        entry = col3.number_input("Giriş", format="%.4f")
        size = col4.number_input("Büyüklük (USDT)", min_value=0.0)
        
        col5, col6, col7 = st.columns(3)
        sl = col5.number_input("Stop Loss", format="%.4f")
        tp = col6.number_input("Take Profit", format="%.4f")
        
        if st.form_submit_button("EMRİ GÖNDER"):
            risk = abs(entry - sl) if sl != 0 else 1
            reward = abs(tp - entry)
            rr = round(reward / risk, 2)
            
            st.session_state.trades.append({
                "asset": asset.upper(), "side": side, "entry": entry,
                "sl": sl, "tp": tp, "size": size, "rr": rr,
                "status": "Açık", "pnl": 0, "exit": 0,
                "time": datetime.now().strftime("%H:%M")
            })
            st.rerun()

# --- LİSTELEME ---
tab1, tab2 = st.tabs(["Açık Pozisyonlar", "İşlem Geçmişi"])

def show_trades(status_filter):
    for idx, t in enumerate(st.session_state.trades):
        if t['status'] == status_filter:
            side_class = "long" if t['side'] == "LONG" else "short"
            with st.container():
                st.markdown(f"""
                <div class="trade-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 18px; font-weight: bold;">{t['asset']} <span class="{side_class}">{t['side']}</span></span>
                        <span style="color: #5e5e5e;">{t['time']}</span>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(5, 1fr); margin-top: 15px;">
                        <div class="metric-box"><div class="label">Giriş</div><div class="value">{t['entry']}</div></div>
                        <div class="metric-box"><div class="label">Hedef</div><div class="value">{t['tp']}</div></div>
                        <div class="metric-box"><div class="label">Stop</div><div class="value">{t['sl']}</div></div>
                        <div class="metric-box"><div class="label">R:R</div><div class="value">1:{t['rr']}</div></div>
                        <div class="metric-box"><div class="label">PnL</div><div class="value {side_class if t['status']=='Kapalı' else ''}">${t['pnl']:.2f}</div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                col_btn1, col_btn2, _ = st.columns([2, 2, 6])
                
                if status_filter == "Açık":
                    with col_btn1:
                        with st.popover("Pozisyonu Kapat"):
                            exit_p = st.number_input("Çıkış Fiyatı", key=f"exit_{idx}", format="%.4f")
                            if st.button("Onayla", key=f"conf_{idx}"):
                                close_trade(idx, exit_p)
                
                with col_btn2:
                    if st.button("Sil", key=f"del_{idx}"):
                        delete_trade(idx)
                st.write("")

with tab1:
    show_trades("Açık")

with tab2:
    show_trades("Kapalı")
