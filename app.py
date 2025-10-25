import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

st.set_page_config(page_title="4Saatlik Profesyonel TA", layout="wide")

# Basit şifre koruması
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    
    password = st.text_input("Şifre", type="password")
    if password:
        if password == "efe":
            st.session_state["password_correct"] = True
        else:
            st.error("❌ Şifre yanlış!")
    
    return st.session_state["password_correct"]

if not check_password():
    st.stop()

# Sembol listesi
def load_symbol_index():
    return [
        "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD",
        "AVAX-USD", "DOT-USD", "MATIC-USD", "LINK-USD", "ATOM-USD"
    ]

# Veri çekme
@st.cache_data
def get_4h_data(symbol, days=30):
    try:
        symbol = symbol.upper().strip()
        if '-' not in symbol:
            symbol = symbol + '-USD'
        data = yf.download(symbol, period=f"{days}d", interval="4h", progress=False)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"❌ {symbol} veri çekilemedi: {e}")
        return None

# Basit gösterge hesaplama
def calculate_indicators(data):
    if data is None or len(data) == 0:
        return data
    
    df = data.copy()
    
    # Temel göstergeler
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    
    # Donchian
    df['DONCH_HIGH'] = df['High'].rolling(20).max()
    df['DONCH_LOW'] = df['Low'].rolling(20).min()
    
    # Bollinger Bands
    df['BB_MID'] = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_UPPER'] = df['BB_MID'] + (bb_std * 2)
    df['BB_LOWER'] = df['BB_MID'] - (bb_std * 2)
    
    return df

# Basit rejim belirleme
def get_regime(df):
    try:
        current_price = df['Close'].iloc[-1]
        ema20 = df['EMA20'].iloc[-1]
        ema50 = df['EMA50'].iloc[-1]
        
        if current_price > ema20 and ema20 > ema50:
            return 'UP'
        elif current_price < ema20 and ema20 < ema50:
            return 'DOWN'
        else:
            return 'RANGE'
    except:
        return 'RANGE'

# Basit sinyal üretme
def generate_signals(df):
    try:
        current_data = df.iloc[-1]
        current_price = current_data['Close']
        rsi = current_data['RSI']
        donch_high = current_data['DONCH_HIGH']
        donch_low = current_data['DONCH_LOW']
        bb_upper = current_data['BB_UPPER']
        bb_lower = current_data['BB_LOWER']
        
        regime = get_regime(df)
        
        # Uptrend sinyalleri
        if regime == 'UP' and current_price >= donch_high and rsi < 70:
            sl = donch_low
            risk = current_price - sl
            if risk > 0:
                tp1 = current_price + risk * 1.0
                tp2 = current_price + risk * 2.0
                return {
                    "type": "BUY", "entry": current_price, "sl": sl,
                    "tp1": tp1, "tp2": tp2, "rr": 2.0,
                    "reason": "Uptrend Breakout", "strat_id": "A"
                }
        
        # Downtrend sinyalleri
        elif regime == 'DOWN' and current_price <= donch_low and rsi > 30:
            sl = donch_high
            risk = sl - current_price
            if risk > 0:
                tp1 = current_price - risk * 1.0
                tp2 = current_price - risk * 2.0
                return {
                    "type": "SELL", "entry": current_price, "sl": sl,
                    "tp1": tp1, "tp2": tp2, "rr": 2.0,
                    "reason": "Downtrend Breakdown", "strat_id": "B"
                }
        
        # Range sinyalleri
        elif regime == 'RANGE':
            if current_price >= bb_upper and rsi > 60:
                sl = bb_upper * 1.02
                risk = sl - current_price
                tp1 = current_data['BB_MID']
                tp2 = bb_lower
                return {
                    "type": "SELL", "entry": current_price, "sl": sl,
                    "tp1": tp1, "tp2": tp2, "rr": 1.5,
                    "reason": "Range Resistance", "strat_id": "C"
                }
            elif current_price <= bb_lower and rsi < 40:
                sl = bb_lower * 0.98
                risk = current_price - sl
                tp1 = current_data['BB_MID']
                tp2 = bb_upper
                return {
                    "type": "BUY", "entry": current_price, "sl": sl,
                    "tp1": tp1, "tp2": tp2, "rr": 1.5,
                    "reason": "Range Support", "strat_id": "C"
                }
        
        return {"type": "WAIT", "reason": "Koşullar uygun değil", "strat_id": "NONE"}
    
    except Exception as e:
        return {"type": "WAIT", "reason": f"Hata: {str(e)}", "strat_id": "NONE"}

# Grafik oluşturma
def create_chart(data, signals):
    if data is None or len(data) == 0:
        return go.Figure()
    
    df = data.tail(24)  # Son 24 bar (4 gün)
    fig = go.Figure()
    
    # Mum çubukları
    for i in range(len(df)):
        try:
            row = df.iloc[i]
            open_price = row['Open']
            high = row['High']
            low = row['Low']
            close = row['Close']
            
            color = 'green' if close > open_price else 'red'
            
            # Mum gövdesi
            fig.add_trace(go.Scatter(
                x=[df.index[i], df.index[i]],
                y=[open_price, close],
                mode='lines',
                line=dict(color=color, width=6),
                showlegend=False
            ))
            
            # Üst gölge
            fig.add_trace(go.Scatter(
                x=[df.index[i], df.index[i]],
                y=[max(open_price, close), high],
                mode='lines',
                line=dict(color=color, width=1),
                showlegend=False
            ))
            
            # Alt gölge
            fig.add_trace(go.Scatter(
                x=[df.index[i], df.index[i]],
                y=[min(open_price, close), low],
                mode='lines',
                line=dict(color=color, width=1),
                showlegend=False
            ))
            
        except:
            continue
    
    # EMA'lar
    if 'EMA20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['EMA20'],
            line=dict(color='orange', width=2),
            name='EMA20'
        ))
    
    if 'EMA50' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['EMA50'],
            line=dict(color='blue', width=2),
            name='EMA50'
        ))
    
    # Sinyal işareti
    if signals and signals["type"] in ["BUY", "SELL"]:
        current_price = df['Close'].iloc[-1]
        marker_symbol = "triangle-up" if signals["type"] == "BUY" else "triangle-down"
        marker_color = "green" if signals["type"] == "BUY" else "red"
        
        fig.add_trace(go.Scatter(
            x=[df.index[-1]],
            y=[current_price],
            mode='markers',
            marker=dict(symbol=marker_symbol, size=15, color=marker_color, line=dict(width=2, color="white")),
            name=f"{signals['type']} Sinyal"
        ))
    
    fig.update_layout(
        title=f"{st.session_state.get('selected_symbol', 'BTC-USD')} - 4H Chart",
        xaxis_title="",
        yaxis_title="Fiyat (USD)",
        template="plotly_dark",
        height=500
    )
    
    return fig

# Format helper
def format_price(price):
    try:
        price = float(price)
        if price >= 1000:
            return f"${price:,.0f}"
        elif price >= 1:
            return f"${price:.2f}"
        else:
            return f"${price:.4f}"
    except:
        return "N/A"

# Ana uygulama
def main():
    st.title("🎯 4 Saatlik Teknik Analiz")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Ayarlar")
        
        # Sembol seçimi
        if 'selected_symbol' not in st.session_state:
            st.session_state.selected_symbol = "BTC-USD"
        
        crypto_symbol = st.text_input("Kripto Sembolü", st.session_state.selected_symbol)
        
        # Hızlı seçim butonları
        st.caption("Hızlı Seçim:")
        cols = st.columns(3)
        symbols = ["BTC-USD", "ETH-USD", "ADA-USD", "XRP-USD", "SOL-USD", "DOT-USD"]
        
        for i, symbol in enumerate(symbols):
            if cols[i % 3].button(symbol, use_container_width=True):
                st.session_state.selected_symbol = symbol
                st.rerun()
        
        st.divider()
        st.subheader("Backtest Ayarları")
        run_bt = st.button("Backtest Çalıştır (30g)", type="primary")
    
    # Ana içerik
    symbol = st.session_state.selected_symbol
    
    with st.spinner(f"{symbol} verileri yükleniyor..."):
        data = get_4h_data(symbol, days=30)
    
    if data is None:
        st.error("Veri yüklenemedi!")
        return
    
    # Göstergeleri hesapla
    data = calculate_indicators(data)
    
    # Sinyal üret
    signals = generate_signals(data)
    
    # Mevcut durum
    current_price = data['Close'].iloc[-1] if len(data) > 0 else 0
    regime = get_regime(data)
    
    # Layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Grafik
        fig = create_chart(data, signals)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Sinyal")
        
        if signals["type"] in ["BUY", "SELL"]:
            color = "🟢" if signals["type"] == "BUY" else "🔴"
            st.markdown(f"### {color} {signals['type']}")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Giriş", format_price(signals['entry']))
                st.metric("TP1", format_price(signals['tp1']))
            with col_b:
                st.metric("SL", format_price(signals['sl']))
                st.metric("TP2", format_price(signals['tp2']))
            
            st.metric("R/R", f"{signals['rr']:.1f}")
            st.metric("Strateji", signals['strat_id'])
            st.metric("Sebep", signals['reason'])
            
        else:
            st.markdown("### ⚪ BEKLE")
            st.info(signals['reason'])
        
        st.divider()
        
        # Göstergeler
        st.subheader("📈 Göstergeler")
        try:
            rsi = data['RSI'].iloc[-1]
            atr = data['ATR'].iloc[-1]
            st.metric("RSI", f"{rsi:.1f}")
            st.metric("ATR", format_price(atr))
            st.metric("Rejim", regime)
            st.metric("Fiyat", format_price(current_price))
        except:
            pass
    
    # Basit backtest
    if run_bt:
        st.divider()
        st.header("🧪 Backtest Sonuçları (30 Gün)")
        
        with st.spinner("Backtest çalışıyor..."):
            # Basit backtest simülasyonu
            trades = []
            balance = 10000
            equity = [balance]
            
            for i in range(50, len(data) - 1):
                df_slice = data.iloc[:i+1]
                signal = generate_signals(df_slice)
                
                if signal["type"] in ["BUY", "SELL"]:
                    # Basit trade simülasyonu
                    entry = data['Open'].iloc[i+1]
                    exit_price = data['Close'].iloc[i+5] if i+5 < len(data) else data['Close'].iloc[-1]
                    
                    if signal["type"] == "BUY":
                        pnl = (exit_price - entry) * (balance * 0.01 / (entry - signal['sl']))
                    else:
                        pnl = (entry - exit_price) * (balance * 0.01 / (signal['sl'] - entry))
                    
                    balance += pnl
                    trades.append({
                        'type': signal["type"],
                        'entry': entry,
                        'exit': exit_price,
                        'pnl': pnl,
                        'strat': signal['strat_id']
                    })
                
                equity.append(balance)
            
            # Sonuçlar
            if trades:
                total_trades = len(trades)
                winning_trades = len([t for t in trades if t['pnl'] > 0])
                win_rate = (winning_trades / total_trades) * 100
                total_pnl = sum(t['pnl'] for t in trades)
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Toplam İşlem", total_trades)
                col2.metric("Win Rate", f"{win_rate:.1f}%")
                col3.metric("Toplam PnL", f"${total_pnl:,.0f}")
                col4.metric("Final Balance", f"${balance:,.0f}")
                
                # Equity curve
                fig_equity = go.Figure()
                fig_equity.add_trace(go.Scatter(
                    y=equity,
                    line=dict(color="green", width=2),
                    name="Equity"
                ))
                fig_equity.update_layout(
                    title="Equity Curve",
                    height=300,
                    template="plotly_dark"
                )
                st.plotly_chart(fig_equity, use_container_width=True)
                
                # Trade listesi
                st.subheader("İşlem Listesi")
                trades_df = pd.DataFrame(trades)
                st.dataframe(trades_df, use_container_width=True)
            else:
                st.warning("Backtest süresince hiç işlem yapılmadı!")

# Uygulamayı çalıştır
if __name__ == "__main__":
    main()