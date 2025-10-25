# app.py
# ─────────────────────────────────────────────────────────────────────────────
# 4H • YFinance • EMA50 Trend • RSI14 • S/R Bölgeleri
# Optimized for Higher Win Rate
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

st.set_page_config(page_title="4H Pro TA (Optimized Win Rate)", layout="wide")

# =============================================================================
# ŞİFRE
# =============================================================================
def check_password():
    def password_entered():
        if st.session_state.get("password") == "efe":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Şifre", type="password", on_change=password_entered, key="password")
        st.stop()
    elif not st.session_state["password_correct"]:
        st.text_input("Şifre", type="password", on_change=password_entered, key="password")
        st.error("❌ Şifre yanlış!")
        st.stop()

check_password()

# =============================================================================
# YARDIMCI FONKSİYONLAR - BASİTLEŞTİRİLMİŞ
# =============================================================================
def format_price(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "N/A"
    try:
        x = float(x)
        if x >= 1000: return f"${x:,.0f}"
        if x >= 1:    return f"${x:.2f}"
        if x >= 0.1:  return f"${x:.3f}"
        return f"${x:.4f}"
    except Exception:
        return "N/A"

@st.cache_data
def get_4h_data(symbol: str, days: int) -> pd.DataFrame:
    sym = symbol.upper().strip()
    if "-" not in sym:
        sym = sym + "-USD"
    df = yf.download(sym, period=f"{days}d", interval="4h", progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    return df.dropna()

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    d = df.copy()

    # EMA
    d["EMA"] = d["Close"].ewm(span=50, adjust=False).mean()

    # RSI - Basitleştirilmiş
    delta = d["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, 0.001)
    d["RSI"] = 100 - (100 / (1 + rs))
    d["RSI"] = d["RSI"].fillna(50)

    return d

# =============================================================================
# S/R BÖLGELERİ - BASİT VE ETKİLİ
# =============================================================================
class Zone:
    def __init__(self, low: float, high: float, touches: int, kind: str):
        self.low = float(low)
        self.high = float(high)
        self.touches = int(touches)
        self.kind = kind
        self.score = 0

def find_zones_simple(d: pd.DataFrame, min_touch_points: int = 2) -> Tuple[List[Zone], List[Zone]]:
    if d.empty or len(d) < 20:
        return [], []
    
    # Son 50 barı kullan
    data = d.tail(50).copy()
    current_price = float(data["Close"].iloc[-1])
    
    # Basit destek/direnç seviyeleri
    supports = []
    resistances = []
    
    # Yüksek ve düşük seviyelerde yoğunluk
    price_levels = []
    for i in range(len(data)):
        high = float(data["High"].iloc[i])
        low = float(data["Low"].iloc[i])
        close = float(data["Close"].iloc[i])
        price_levels.extend([high, low, close])
    
    if not price_levels:
        return [], []
    
    # Basit bölgeleme
    price_range = max(price_levels) - min(price_levels)
    bin_size = price_range * 0.02  # %2'lik bölgeler
    
    bins = {}
    current_bin = min(price_levels)
    while current_bin <= max(price_levels):
        bin_end = current_bin + bin_size
        count = sum(1 for p in price_levels if current_bin <= p <= bin_end)
        if count >= min_touch_points:
            bins[(current_bin, bin_end)] = count
        current_bin = bin_end
    
    # Zone'ları oluştur
    for (low, high), touches in bins.items():
        if high < current_price:
            supports.append(Zone(low, high, touches, "support"))
        elif low > current_price:
            resistances.append(Zone(low, high, touches, "resistance"))
    
    # Skorlama - basit
    for zone in supports + resistances:
        zone.score = min(zone.touches * 25, 80)  # Temas sayısı bazlı
    
    # En iyi 2'şer tane
    supports = sorted(supports, key=lambda z: z.score, reverse=True)[:2]
    resistances = sorted(resistances, key=lambda z: z.score, reverse=True)[:2]
    
    return supports, resistances

# =============================================================================
# SİNYAL MOTORU - YÜKSEK WIN RATE İÇİN OPTİMİZE
# =============================================================================
@dataclass
class Signal:
    typ: str              # BUY | SELL | WAIT
    entry: float
    sl: Optional[float]
    tp1: Optional[float]
    tp2: Optional[float]
    rr: float
    confidence: int
    trend: str
    reason: List[str]

def generate_high_winrate_signals(d: pd.DataFrame, 
                                 supports: List[Zone], 
                                 resistances: List[Zone],
                                 min_rr: float = 1.5) -> Tuple[List[Signal], List[str]]:
    notes = []
    signals = []
    
    if d.empty or len(d) < 20:
        notes.append("❌ Yetersiz veri")
        return [Signal("WAIT", 0, None, None, None, 0, 0, "neutral", ["Yetersiz veri"])], notes
    
    try:
        current_price = float(d["Close"].iloc[-1])
        ema = float(d["EMA"].iloc[-1])
        rsi = float(d["RSI"].iloc[-1])
        
        trend = "bull" if current_price > ema else "bear"
        notes.append(f"Trend: {'YÜKSELİŞ' if trend == 'bull' else 'DÜŞÜŞ'}")
        notes.append(f"RSI: {rsi:.1f}")
        
        # DAHA AZ STRICT KURALLAR - DAHA FAZLA SİNYAL
        best_support = supports[0] if supports else None
        best_resistance = resistances[0] if resistances else None
        
        # UPTREND - Daha gevşek kurallar
        if trend == "bull" and best_support and best_support.score >= 60:
            # Support bölgesine yakınsa ve RSI uygunsa
            if current_price <= best_support.high * 1.01 and rsi < 70:
                sl = best_support.low * 0.99  # %1 stop loss
                risk = current_price - sl
                if risk > 0:
                    tp1 = current_price + risk * (min_rr * 0.6)
                    tp2 = current_price + risk * min_rr
                    rr = (tp2 - current_price) / risk
                    if rr >= min_rr:
                        reason = [
                            "Uptrend destek bölgesi",
                            f"RSI {rsi:.1f} (aşırı değil)",
                            f"Risk/Reward: {rr:.2f}",
                            f"Destek skoru: {best_support.score}"
                        ]
                        signals.append(Signal("BUY", current_price, sl, tp1, tp2, rr, 
                                            best_support.score, trend, reason))
        
        # DOWNTREND - Daha gevşek kurallar  
        elif trend == "bear" and best_resistance and best_resistance.score >= 60:
            # Resistance bölgesine yakınsa ve RSI uygunsa
            if current_price >= best_resistance.low * 0.99 and rsi > 30:
                sl = best_resistance.high * 1.01  # %1 stop loss
                risk = sl - current_price
                if risk > 0:
                    tp1 = current_price - risk * (min_rr * 0.6)
                    tp2 = current_price - risk * min_rr
                    rr = (current_price - tp2) / risk
                    if rr >= min_rr:
                        reason = [
                            "Downtrend direnç bölgesi", 
                            f"RSI {rsi:.1f} (aşırı değil)",
                            f"Risk/Reward: {rr:.2f}",
                            f"Direnç skoru: {best_resistance.score}"
                        ]
                        signals.append(Signal("SELL", current_price, sl, tp1, tp2, rr,
                                            best_resistance.score, trend, reason))
        
        # RANGE MARKET - Mean reversion
        if not signals and abs(current_price - ema) / current_price < 0.02:  # EMA'ya yakınsa
            if best_support and current_price <= best_support.high * 1.005 and rsi < 40:
                sl = best_support.low * 0.99
                risk = current_price - sl
                tp1 = ema  # EMA'ya kadar
                tp2 = best_resistance.low if best_resistance else current_price * 1.03
                rr = (tp2 - current_price) / risk
                if rr >= min_rr:
                    reason = ["Range - Destekten bounce", f"RSI oversold: {rsi:.1f}"]
                    signals.append(Signal("BUY", current_price, sl, tp1, tp2, rr, 
                                        70, "range", reason))
            
            elif best_resistance and current_price >= best_resistance.low * 0.995 and rsi > 60:
                sl = best_resistance.high * 1.01
                risk = sl - current_price
                tp1 = ema  # EMA'ya kadar
                tp2 = best_support.high if best_support else current_price * 0.97
                rr = (current_price - tp2) / risk
                if rr >= min_rr:
                    reason = ["Range - Dirençten reject", f"RSI overbought: {rsi:.1f}"]
                    signals.append(Signal("SELL", current_price, sl, tp1, tp2, rr,
                                        70, "range", reason))
        
    except Exception as e:
        notes.append(f"Hata: {str(e)}")
    
    if not signals:
        wait_reason = ["Uygun sinyal bulunamadı"]
        if not supports and not resistances:
            wait_reason.append("S/R bölgesi yok")
        signals.append(Signal("WAIT", current_price, None, None, None, 0, 0, 
                            trend, wait_reason))
    
    return signals, notes

# =============================================================================
# BACKTEST - BASİT VE ETKİLİ
# =============================================================================
@dataclass
class Trade:
    entry: float
    exit: float
    side: str
    pnl: float
    pnl_percent: float

def simple_backtest(df: pd.DataFrame, 
                   min_rr: float = 1.5, 
                   risk_percent: float = 1.0) -> Dict[str, Any]:
    if df.empty or len(df) < 100:
        return {"trades": 0, "win_rate": 0, "total_return": 0, "final_balance": 10000}
    
    balance = 10000.0
    trades = []
    equity = [balance]
    
    for i in range(50, len(df) - 5):  # 5 bar forward testing
        try:
            data_slice = df.iloc[:i+1]
            supports, resistances = find_zones_simple(data_slice, min_touch_points=2)
            signals, _ = generate_high_winrate_signals(data_slice, supports, resistances, min_rr)
            
            if signals and signals[0].typ in ["BUY", "SELL"]:
                signal = signals[0]
                entry_price = float(df["Open"].iloc[i+1])  # Sonraki bar açılışı
                exit_price = float(df["Close"].iloc[i+5])  # 5 bar sonra kapat
                
                # Position sizing
                position_size = balance * (risk_percent / 100)
                
                if signal.typ == "BUY":
                    pnl = (exit_price - entry_price) * (position_size / entry_price)
                else:  # SELL
                    pnl = (entry_price - exit_price) * (position_size / entry_price)
                
                balance += pnl
                pnl_percent = (pnl / (position_size)) * 100
                
                trades.append(Trade(
                    entry=entry_price,
                    exit=exit_price,
                    side=signal.typ,
                    pnl=pnl,
                    pnl_percent=pnl_percent
                ))
            
            equity.append(balance)
            
        except Exception:
            continue
    
    # Metrics
    total_trades = len(trades)
    if total_trades == 0:
        return {"trades": 0, "win_rate": 0, "total_return": 0, "final_balance": 10000}
    
    winning_trades = len([t for t in trades if t.pnl > 0])
    win_rate = (winning_trades / total_trades) * 100
    total_return = ((balance - 10000) / 10000) * 100
    
    return {
        "trades": total_trades,
        "win_rate": win_rate,
        "total_return": total_return,
        "final_balance": balance,
        "equity_curve": equity
    }

# =============================================================================
# ARAYÜZ
# =============================================================================
st.title("🎯 4H Pro TA - Optimized for Win Rate")

with st.sidebar:
    st.header("⚙️ Ayarlar")
    symbol = st.text_input("Kripto Sembolü", "BTC-USD")
    min_rr = st.slider("Min R/R", 1.0, 3.0, 1.5, 0.1)
    risk_percent = st.slider("Risk %", 0.5, 5.0, 1.0, 0.1)
    
    st.divider()
    st.subheader("Backtest")
    run_backtest = st.button("🚀 Backtest Çalıştır (90g)", type="primary")

# Ana veri yükleme
with st.spinner("Veri yükleniyor..."):
    data_30d = get_4h_data(symbol, 30)
    if not data_30d.empty:
        data_30d = compute_indicators(data_30d)
        supports, resistances = find_zones_simple(data_30d)
        signals, notes = generate_high_winrate_signals(data_30d, supports, resistances, min_rr)
    else:
        st.error("❌ Veri alınamadı!")
        st.stop()

# Ana görünüm
col1, col2 = st.columns([2, 1])

with col1:
    if not data_30d.empty:
        # Basit grafik
        fig = go.Figure()
        
        # Son 24 bar
        view_data = data_30d.tail(24)
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=view_data.index,
            open=view_data['Open'],
            high=view_data['High'],
            low=view_data['Low'],
            close=view_data['Close'],
            name="Price"
        ))
        
        # EMA
        fig.add_trace(go.Scatter(
            x=view_data.index,
            y=view_data['EMA'],
            line=dict(color='orange', width=2),
            name="EMA50"
        ))
        
        # Support/Resistance lines
        for i, zone in enumerate(supports[:2]):
            fig.add_hline(y=zone.low, line_dash="dash", line_color="green", 
                         annotation_text=f"S{i+1}")
            fig.add_hline(y=zone.high, line_dash="dash", line_color="green")
            
        for i, zone in enumerate(resistances[:2]):
            fig.add_hline(y=zone.low, line_dash="dash", line_color="red",
                         annotation_text=f"R{i+1}")
            fig.add_hline(y=zone.high, line_dash="dash", line_color="red")
        
        fig.update_layout(
            title=f"{symbol} - 4H Chart",
            height=500,
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📊 Sinyal")
    
    if signals and signals[0].typ in ["BUY", "SELL"]:
        signal = signals[0]
        color = "🟢" if signal.typ == "BUY" else "🔴"
        st.markdown(f"### {color} {signal.typ}")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Entry", format_price(signal.entry))
            st.metric("TP1", format_price(signal.tp1))
        with col_b:
            st.metric("SL", format_price(signal.sl))
            st.metric("TP2", format_price(signal.tp2))
        
        st.metric("R/R", f"{signal.rr:.2f}")
        st.metric("Confidence", f"{signal.confidence}")
        
        st.write("**Reason:**")
        for reason in signal.reason:
            st.write(f"• {reason}")
            
    else:
        st.markdown("### ⚪ WAIT")
        if signals:
            for reason in signals[0].reason:
                st.write(f"• {reason}")

# Backtest sonuçları
if run_backtest:
    st.header("📈 Backtest Sonuçları (90 Gün)")
    
    with st.spinner("Backtest çalışıyor..."):
        data_90d = get_4h_data(symbol, 90)
        if not data_90d.empty:
            data_90d = compute_indicators(data_90d)
            results = simple_backtest(data_90d, min_rr, risk_percent)
            
            # Sonuçları göster
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Trades", results["trades"])
                st.metric("Win Rate", f"{results['win_rate']:.1f}%")
                
            with col2:
                st.metric("Total Return", f"{results['total_return']:.1f}%")
                st.metric("Final Balance", f"${results['final_balance']:,.0f}")
            
            # Equity curve
            if "equity_curve" in results and len(results["equity_curve"]) > 1:
                st.subheader("Equity Curve")
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(
                    y=results["equity_curve"],
                    line=dict(color="green", width=2),
                    name="Equity"
                ))
                fig_eq.update_layout(
                    height=300,
                    template="plotly_dark",
                    showlegend=False
                )
                st.plotly_chart(fig_eq, use_container_width=True)
                
            if results["trades"] == 0:
                st.warning("Backtest sırasında hiç işlem yapılmadı. Parametreleri gevşetmeyi deneyin.")
                
        else:
            st.error("Backtest için veri alınamadı!")

# Strateji açıklaması
with st.expander("ℹ️ Strateji Detayları"):
    st.write("""
    **Optimize Edilmiş Yüksek Win Rate Stratejisi:**
    
    - **EMA50 Trend Filter**: Fiyat EMA50 üstünde = Uptrend, altında = Downtrend
    - **RSI14 Momentum**: Aşırı alım/satım bölgelerinden kaçınma
    - **Basit S/R Bölgeleri**: Yoğunluk bazlı destek/direnç seviyeleri
    - **Gevşek Kurallar**: Daha fazla sinyal üretmek için
    - **Fixed Exit**: 5 bar sonra pozisyon kapatma
    
    **İyileştirmeler:**
    - Daha düşük S/R skor threshold (60 → daha fazla sinyal)
    - RSI filtreleri gevşetildi
    - Daha geniş entry toleransları
    - Basit position sizing
    """)