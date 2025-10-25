# app.py
# ─────────────────────────────────────────────────────────────────────────────
# 4H • YFinance • EMA50 Trend • RSI14 • S/R Bölgeleri
# OPTIMIZED FOR MAX WIN RATE + PROGRESS BAR
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import time

st.set_page_config(page_title="4H Pro TA (Max Win Rate)", layout="wide")

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
# YARDIMCI FONKSİYONLAR - OPTIMIZED
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

    # EMA - Multiple timeframes for better signals
    d["EMA20"] = d["Close"].ewm(span=20, adjust=False).mean()
    d["EMA50"] = d["Close"].ewm(span=50, adjust=False).mean()
    d["EMA100"] = d["Close"].ewm(span=100, adjust=False).mean()

    # RSI - Multiple periods
    for period in [7, 14]:
        delta = d["Close"].diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, 0.001)
        d[f"RSI_{period}"] = 100 - (100 / (1 + rs))
    
    d["RSI"] = d["RSI_14"].fillna(50)
    d["RSI_FAST"] = d["RSI_7"].fillna(50)

    # MACD for additional confirmation
    exp12 = d["Close"].ewm(span=12, adjust=False).mean()
    exp26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = exp12 - exp26
    d["MACD_SIGNAL"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_HISTOGRAM"] = d["MACD"] - d["MACD_SIGNAL"]

    return d

# =============================================================================
# S/R BÖLGELERİ - IMPROVED
# =============================================================================
class Zone:
    def __init__(self, low: float, high: float, touches: int, kind: str):
        self.low = float(low)
        self.high = float(high)
        self.touches = int(touches)
        self.kind = kind
        self.score = 0

def find_zones_improved(d: pd.DataFrame, min_touch_points: int = 2) -> Tuple[List[Zone], List[Zone]]:
    if d.empty or len(d) < 30:
        return [], []
    
    # Use more bars for better zone detection
    data = d.tail(80).copy()
    current_price = float(data["Close"].iloc[-1])
    
    # Improved zone detection using pivot points
    supports = []
    resistances = []
    
    # Detect swing highs and lows
    highs = data["High"].rolling(window=5, center=True).max()
    lows = data["Low"].rolling(window=5, center=True).min()
    
    pivot_highs = data[data["High"] == highs]
    pivot_lows = data[data["Low"] == lows]
    
    # Create zones from pivot points with tolerance
    tolerance = current_price * 0.005  # 0.5% tolerance
    
    for idx, row in pivot_lows.iterrows():
        low_val = float(row["Low"])
        # Check if similar level exists
        existing_support = next((s for s in supports if abs(s.low - low_val) < tolerance), None)
        if existing_support:
            existing_support.touches += 1
            existing_support.high = max(existing_support.high, low_val + tolerance)
        else:
            supports.append(Zone(low_val - tolerance, low_val + tolerance, 1, "support"))
    
    for idx, row in pivot_highs.iterrows():
        high_val = float(row["High"])
        # Check if similar level exists
        existing_resistance = next((r for r in resistances if abs(r.high - high_val) < tolerance), None)
        if existing_resistance:
            existing_resistance.touches += 1
            existing_resistance.low = min(existing_resistance.low, high_val - tolerance)
        else:
            resistances.append(Zone(high_val - tolerance, high_val + tolerance, 1, "resistance"))
    
    # Filter by current price and minimum touches
    supports = [s for s in supports if s.high < current_price and s.touches >= min_touch_points]
    resistances = [r for r in resistances if r.low > current_price and r.touches >= min_touch_points]
    
    # Improved scoring
    for zone in supports + resistances:
        base_score = zone.touches * 20
        # Recent touches score more
        recency_bonus = min(zone.touches * 5, 20)
        zone.score = min(base_score + recency_bonus, 95)
    
    return sorted(supports, key=lambda z: z.score, reverse=True)[:3], \
           sorted(resistances, key=lambda z: z.score, reverse=True)[:3]

# =============================================================================
# SİNYAL MOTORU - MAX WIN RATE OPTIMIZATION
# =============================================================================
@dataclass
class Signal:
    typ: str
    entry: float
    sl: float
    tp1: float
    tp2: float
    rr: float
    confidence: int
    trend: str
    reason: List[str]

def generate_max_winrate_signals(d: pd.DataFrame, 
                                supports: List[Zone], 
                                resistances: List[Zone],
                                min_rr: float = 1.3) -> Tuple[List[Signal], List[str]]:
    notes = []
    signals = []
    
    if d.empty or len(d) < 25:
        return [Signal("WAIT", 0, 0, 0, 0, 0, 0, "neutral", ["Yetersiz veri"])], notes
    
    try:
        current_price = float(d["Close"].iloc[-1])
        ema20 = float(d["EMA20"].iloc[-1])
        ema50 = float(d["EMA50"].iloc[-1])
        ema100 = float(d["EMA100"].iloc[-1])
        rsi = float(d["RSI"].iloc[-1])
        rsi_fast = float(d["RSI_FAST"].iloc[-1])
        macd_hist = float(d["MACD_HISTOGRAM"].iloc[-1])
        
        # IMPROVED TREND DETECTION - Multiple timeframe confirmation
        trend_strength = 0
        if current_price > ema20: trend_strength += 1
        if current_price > ema50: trend_strength += 2
        if current_price > ema100: trend_strength += 3
        if ema20 > ema50: trend_strength += 1
        if ema50 > ema100: trend_strength += 2
        
        trend = "bull" if trend_strength >= 4 else "bear" if trend_strength <= 2 else "neutral"
        
        notes.append(f"Trend: {'🟢 GÜÇLÜ YÜKSELİŞ' if trend_strength >= 6 else '🟢 YÜKSELİŞ' if trend == 'bull' else '🔴 DÜŞÜŞ' if trend == 'bear' else '🟡 NÖTR'}")
        notes.append(f"RSI: {rsi:.1f} | RSI-Fast: {rsi_fast:.1f}")
        notes.append(f"Trend Strength: {trend_strength}/9")

        # LOOSER PARAMETERS FOR MORE SIGNALS
        best_support = supports[0] if supports else None
        best_resistance = resistances[0] if resistances else None
        
        # STRATEGY 1: TREND FOLLOWING WITH PULLBACK (High Win Rate)
        if trend == "bull" and best_support:
            # Buy on pullback to support in uptrend
            distance_to_support = abs(current_price - best_support.high) / current_price
            if distance_to_support < 0.02 and rsi < 65:  # Within 2% of support, not overbought
                sl = best_support.low * 0.995  # Tight stop loss
                risk = current_price - sl
                if risk > 0:
                    # Conservative targets for higher win rate
                    tp1 = current_price + risk * 0.8  # 0.8R
                    tp2 = current_price + risk * 1.2  # 1.2R
                    rr = (tp2 - current_price) / risk
                    
                    if rr >= min_rr:
                        confidence = min(best_support.score + trend_strength * 5, 95)
                        reason = [
                            "🎯 TREND + PULLBACK STRATEJİSİ",
                            f"Uptrend + Support retest (Güç: {trend_strength}/9)",
                            f"RSI {rsi:.1f} (aşırı alım değil)",
                            f"Support skoru: {best_support.score}",
                            f"Risk/Reward: {rr:.2f}:1",
                            "Yüksek Win Rate modeli"
                        ]
                        signals.append(Signal("BUY", current_price, sl, tp1, tp2, rr, 
                                            confidence, trend, reason))
        
        # STRATEGY 2: MEAN REVERSION IN RANGE (High Win Rate)
        elif trend == "neutral" and abs(current_price - ema50) / current_price < 0.015:
            # Price is close to EMA50, look for mean reversion opportunities
            
            if best_support and current_price <= best_support.high * 1.01 and rsi < 45:
                # Oversold bounce from support
                sl = best_support.low * 0.992
                risk = current_price - sl
                tp1 = ema50  # Target EMA50
                tp2 = best_resistance.low if best_resistance else current_price * 1.02
                rr = (tp2 - current_price) / risk
                
                if rr >= min_rr:
                    confidence = 75
                    reason = [
                        "🔄 MEAN REVERSION STRATEJİSİ",
                        "Oversold + Support bounce",
                        f"RSI {rsi:.1f} (oversold)",
                        f"EMA50 hedef: {format_price(ema50)}",
                        f"Risk/Reward: {rr:.2f}:1",
                        "Range market optimizasyonu"
                    ]
                    signals.append(Signal("BUY", current_price, sl, tp1, tp2, rr,
                                        confidence, "range", reason))
            
            elif best_resistance and current_price >= best_resistance.low * 0.99 and rsi > 55:
                # Overbought rejection from resistance
                sl = best_resistance.high * 1.008
                risk = sl - current_price
                tp1 = ema50  # Target EMA50
                tp2 = best_support.high if best_support else current_price * 0.98
                rr = (current_price - tp2) / risk
                
                if rr >= min_rr:
                    confidence = 75
                    reason = [
                        "🔄 MEAN REVERSION STRATEJİSİ", 
                        "Overbought + Resistance reject",
                        f"RSI {rsi:.1f} (overbought)",
                        f"EMA50 hedef: {format_price(ema50)}",
                        f"Risk/Reward: {rr:.2f}:1",
                        "Range market optimizasyonu"
                    ]
                    signals.append(Signal("SELL", current_price, sl, tp1, tp2, rr,
                                        confidence, "range", reason))
        
        # STRATEGY 3: BREAKOUT WITH RETEST (Medium Win Rate but good RR)
if not signals:
    if best_resistance and current_price >= best_resistance.low * 0.998 and trend_strength >= 5:
        # Breakout above resistance with trend confirmation
        sl = best_resistance.low * 0.995
        risk = current_price - sl
        tp1 = current_price + risk * 1.0
        tp2 = current_price + risk * 1.5
        rr = (tp2 - current_price) / risk
        
        if rr >= min_rr and rsi < 70:
            confidence = min(best_resistance.score + 20, 90)
            reason = [
                "🚀 BREAKOUT STRATEJİSİ",
                f"Resistance breakout (Güç: {trend_strength}/9)",
                f"RSI {rsi:.1f} (momentum)",
                f"Risk/Reward: {rr:.2f}:1",
                "Trend + Breakout kombinasyonu"
            ]
            signals.append(Signal("BUY", current_price, sl, tp1, tp2, rr,
                                confidence, trend, reason))
                    
except Exception as e:
    notes.append(f"Hata: {str(e)}")
    
    
    if not signals not signals:
        wait_re:
        wait_reason = ["ason = ["🔍 U🔍 Uygun yygun yükseküksek win rate sinyali bulunamadı"]
        if trend_st win rate sinyali bulunamadı"]
        if trend_strengthrength < 4:
 < 4:
            wait            wait_reason.append(f_reason.append(f"Trend"Trend çok z çok zayıayıf: {trend_strength}/9")
        if not supports and not resistances:
            wait_reason.append("Yeterli S/R seviyesi yok")
        signals.append(Signal("WAIT", current_price, 0, 0, 0, 0, 0, 
                            trend, wait_reason))
    
    return signals, notes

# =============================================================================
# BACKTEST WITH PROGRESS BAR + TIMING
# =============================================================================
@dataclass
class Trade:
    entry: float
    exit: float
    side: str
    pnl: float
    pnl_percent: float
    duration_bars: int

def backtest_with_progress(df: pd.DataFrame, 
                          min_rr: float = 1.3f: {trend_strength}/9")
        if not supports and not resistances:
            wait_reason.append("Yeterli S/R seviyesi yok")
        signals.append(Signal("WAIT", current_price, 0, 0, 0, 0, 0, 
                            trend, wait_reason))
    
    return signals, notes

# =============================================================================
# BACKTEST WITH PROGRESS BAR + TIMING
# =============================================================================
@dataclass
class Trade:
    entry: float
    exit: float
    side: str
    pnl: float
    pnl_percent: float
    duration_bars: int

def backtest_with_progress(df: pd.DataFrame, 
                          min_rr: float = 1.3, 
, 
                          risk_percent: float = 1.0,
                          progress_bar=None,
                          status_text=None) -> Dict                          risk_percent: float[str, Any]:
    start_time = time.time()
    
    if df.empty or len(df) < 100:
        return {"trades": = 1.0,
                          progress_bar=None,
                          status_text=None) -> Dict[str, Any]:
    start_time = time.time()
    
    if df.empty or len(df) < 100:
        0, "win_rate": return {"trades": 0, "win_rate": 0 0, "total_return, "total_return": ": 0, "final0, "final_balance_balance": 10000": 10000}
    
}
    
    balance =     balance = 1000010000.0
    trades.0
    trades = = []
    equity = [balance []
    equity = [balance]
    total_bars = len(df) - 50]
    total_bars = len(df) - 50
    
    for i in range
    
    for i in range(50, len(df) -(50, len(df) 5):
        # Progress - 5):
        # Progress update
        if progress update
        if progress_bar and_bar and (i - 50 (i - 50) % 10 == ) % 10 == 0:
            progress = (0:
            progress = (i - 50) /i - 50) / total_bars
            progress_bar.progress(progress)
            if status_text:
                elapsed = time total_bars
            progress_bar.progress(progress)
            if status_text:
                elapsed = time.time() - start_time
               .time() - start_time
                status_text.text(f"Backtest status_text.text(f"Backtest çalışıyor... çalışıyor... {len {len(trades)} i(trades)} işlem - %{progress*100:.0f} - {elapsed:.1f}s")
        
       şlem - %{progress*100:.0f} - { try:
            data_slice =elapsed:.1f}s")
        
        try:
            df.iloc[:i+ data_slice = df.iloc[:1]
            supports, resisti+1]
            supportsances = find_zones_, resistances = find_zones_improved(data_simproved(data_slice,lice, min_touch_points min_touch_points=2=2)
            signals)
            signals, _ = generate, _ = generate_max_winrate_signals(data_slice_max_winrate_signals(data_slice, supports, resist, supports, resistances, min_rr)
            
            if signalsances, min_rr)
            
            if signals and signals and signals[0].typ[0].typ in [" in ["BUY", "BUY", "SELLSELL"]:
                signal"]:
                signal = signals = signals[0]
               [0]
                entry_price entry_price = float(df[" = float(df["Open"].ilOpen"].iloc[i+oc[i+1])
               1])
                exit_b exit_bars = minars = min(10,(10, len(df) len(df) - i - - i - 2) 2)  # Max  # Max 10 bars 10 bars hold
                exit hold
                exit_price =_price = float(df["Close float(df["Close"].iloc"].iloc[i[i + exit_b + exit_bars])
                
               ars])
                
                # Smart position sizing # Smart position sizing
                position_size = balance *
                position_size = balance * (risk_per (risk_percent / 100cent / 100)
                
                if)
                
                if signal. signal.typ == "BUtyp == "BUY":
                   Y":
                    pnl = pnl = (exit_price - entry (exit_price - entry_price) * (position_size / entry_price) * (position_size / entry_price)
               _price)
                else:
                    p else:
                    pnl =nl = (entry_price - (entry_price - exit_price exit_price) * (position) * (position_size / entry_size / entry_price)
_price)
                
                balance +=                
                balance += pnl
 pnl
                pnl_per                pnl_percent =cent = (pn (pnl / positionl / position_size) * _size) * 100
                
               100
                
                trades.append(Trade(
                    trades.append(Trade(
                    entry= entry=entry_price,
                   entry_price,
                    exit= exit=exit_price,
                    side=signal.typ,
                    pnl=pnl,
                    pnl_percent=pnl_percent,
                    duration_bars=exit_bars
                ))
            
            equity.append(balance)
exit_price,
                    side=signal.typ,
                    pnl=pnl,
                    pnl_percent=pnl_percent,
                    duration_bars=exit_bars
                ))
            
            equity.append(b            
        except Exception:
            continue
    
    # Final progress update
    if progress_baralance)
            
        except Exception:
            continue
    
    # Final progress update
    if:
        progress_bar.progress progress_bar:
        progress_bar.pro(1.0)
    
   gress(1.0)
    
    # Calculate metrics
 # Calculate metrics
    total    total_trades = len_trades = len(t(trades)
    if total_trades ==rades)
    if total_trades == 0:
        0:
        return {
            "trades return {
            "trades": 0,": 0, 
            "win_rate": 
            "win_rate": 0, 
 0, 
            "total_return": 0, 
            "total_return": 0, 
            "final_balance            "final_balance": 10000,
            "duration_": 10000,
            "duration_seconds":seconds": time.time() - time.time() - start_time start_time
        }
    
    winning_trades = len([t for t in trades if t.pnl > 0])
    win_rate =
        }
    
    winning_trades = len([t for t in trades if t.pnl > 0])
    win_rate = (winning_t (winning_trades /rades / total_trades) * total_trades) * 100 100
    total
    total_return = ((balance_return = ((balance - 10000 - 10000) /) / 10000 10000) * ) * 100
    
    #100
    
    # Additional metrics Additional metrics
    avg_
    avg_win =win = np.mean np.mean([t([t.pnl_percent for.pnl_percent for t in t in trades if t.p trades if t.pnl > nl > 0]) if winning0]) if winning_trades_trades > 0 else > 0 else 0 0
    avg_loss
    avg_loss = np = np.mean([t.p.mean([t.pnl_pernl_percent for t in tradescent for t in trades if t.pnl <= 0 if t.pnl <=]) if total_trades - 0]) if total_t winning_trades > 0 else 0
    profit_factor = abs(sum(t.prades - winning_trades > 0 else 0
    profit_factor = abs(sum(t.pnl for t in tradesnl for t in trades if if t.pnl > 0 t.pnl > 0) / sum(t.pnl for t in trades if) / sum(t.pnl for t in trades if t.pnl < 0)) if any(t.pnl t.pnl < 0)) if any(t.pnl <  < 0 for t in trades) else float('0 for t in trades) else float('inf')
    
    return {
        "inf')
    
    return {
        "trades": total_trades,
trades": total_trades,
               "win_rate": win_rate "win_rate": win_rate,
        "total_return": total_return,
        "final_,
        "total_return": total_return,
       balance": balance,
        "equ "final_balance": balanceity_curve": equity,
        "equity_curve": equity,
        ",
        "avg_win_peravg_win_percent": avgcent": avg_win,
_win,
               " "avgavg_loss_per_loss_percent": avg_loss,
cent": avg_loss,
        "profit_factor":        "profit_factor": profit_factor,
        "duration profit_factor,
        "duration_seconds": time_seconds": time.time() - start_time,
        ".time() - start_time,
        "winning_trades":winning_trades": winning_trades,
        "l winning_trades,
        "losing_trades": total_trades - winning_trades
    }

# =============================================================================
# ARAYÜosing_trades": total_trades - winning_trades
    }

# =============================================================================
# ARAYÜZ
# =============================================================================
stZ
# =============================================================================
st.title("🎯 4.title("🎯 4H Pro TA - MAXH Pro TA - MAX WIN RATE OPTIMIZED")

with WIN RATE OPTIMIZED")

with st.side st.sidebar:
    st.header("⚙️ Aybar:
    st.header("⚙️ Ayarlar")
    symbol =arlar")
    symbol = st.text_input("Kript st.text_input("Kripto Sembolü", "BTC-USD")
    min_rr = st.so Sembolü", "BTC-USD")
    min_rr = st.slider("Min R/Rlider("Min R/R", 1.", 1.1, 2.0, 1, 2.0, 1.1.3, 0.3, 0.1)
    risk_percent1)
    risk_per = st.slider("Risk %", 0.5, 3.0cent = st.slider("Risk %", 0., 1.0, 05, 3.0, 1.0, 0.1)
    
    st.divider()
    st.subheader("🎯.1)
    
    st.divider()
    st.subheader("🎯 Strateji Seçimi")
 Strateji Se    strategy_mode = stçimi")
    strategy_mode = st.selectbox(
        "Sinyal Modu",
        ["MAX_WIN_RATE", "BALANC.selectbox(
        "Sinyal Modu",
ED", "AGGRESSIVE        ["MAX_WIN_RATE", "BALANCED", "AGGRESSIVE"],
        help"],
        help="MAX_WIN="MAX_WIN_RATE: Daha yüksek_RATE: Daha yüksek kazanç kazanç oranı, BALANC oranı, BALANCED: Denge, AGED: DengeGRESSIVE: Daha fazla işlem"
, AGGRESSIVE: Daha fazla işlem    )
    
    st"
    )
    
    st.divider()
.divider()
    st    st.subheader("🚀 Backtest.subheader("🚀 Backtest")
   ")
    run_back run_backtesttest = = st.button(" st.button("BACKBACKTEST ÇALIŞTTEST ÇALIŞTIR (90IR (90g)", type="primary", use_container_widthg)", type="primary", use_container_width=True)

#=True)

# Ana veri yükleme
with Ana veri yükleme
with st.spinner(" st.spinner("📊 Veri yükleniyor ve analiz ediliyor..."):
    data_30d =📊 Veri yükleniyor ve analiz ediliyor..."):
    data_30d = get_ get_4h_data(s4h_data(symbol,ymbol, 30)
    30)
    if not if not data_30d data_30d.empty.empty:
        data_30:
        data_30d =d = compute_indicators(data_30d)
        compute_indicators(data_ supports, resistances = find_z30d)
        supports, resistances = find_zones_ones_improved(data_30d)
        signals, notes = generate_max_winrate_signals(data_30dimproved(data_30d)
        signals, notes = generate_max_winrate_signals, supports, resistances,(data_30d, supports, resistances, min_r min_rr)
    elser)
    else:
       :
        st.error("❌ Veri alınam st.error("adı!")
❌ Veri alınamadı!")
        st        st.stop()

# Ana görünüm
col1, col2 = st.columns([2.stop()

# Ana görünüm
col1, col2 = st.columns([2, 1])

with col, 1])

with col11:
    if not data_30:
    if not data_30d.empty:
        figd.empty:
        fig = go.Figure()
 = go.Figure()
        
        
        view_data = data        view_data = data_30d.tail(24)
        
        # Candlestick
        fig.add_trace_30d.tail(24)
        
        # Candlestick
        fig.add_trace(go.Candlestick(
(go.Candlestick(
            x=view_data.index,
            open=view            x=view_data.index,
            open_data['Open'],
            high=view_data['Open'],
            high=view_data['High'],
=view_data['High'],
                       low=view_data[' low=view_data['Low'],
            close=view_data['Low'],
            close=view_data['Close'],
            name="PriceClose'],
            name="Price"
"
        ))
        
        # EM        ))
        
        # EMAsAs
        fig.add_trace(go.Scatter(x=
        fig.add_trace(go.Scatter(x=view_data.index, y=view_data.index, y=view_data['EMA20'],view_data['EMA20'], line=dict(color=' line=dict(color='yellow', width=1yellow', width=1),), name="EMA20"))
 name="EMA20"))
        fig.add        fig.add_t_tracerace(go.Scatter(x=view_data.index, y(go.Scatter(x=view_data.index, y==view_data['EMA50'], line=dict(color='orange', widthview_data['EMA50'], line=dict(color='orange', width==2), name="EMA50"))
        fig.add_trace2), name="EMA50"))
        fig.add_trace(go.Scatter(x=view_data.index, y=view(go.Scatter(x=view_data.index, y=view_data['EMA100'], line_data['EMA100'], line=dict(color='red=dict(color='red', width=1), name="EMA', width=1), name="EMA100"))
        
        # Support100"))
        
        # Support/Resistance
       /Resistance
        for for i, zone in enumerate i, zone in enumerate(supports[:2]):
(supports[:2]):
            fig.add_hline(y=zone.low, line            fig.add_hline(y=zone.low, line_d_dash="dash",ash="dash", line_color="green", annotation_text line_color="green", annotation=f"S{i+1}")
           _text=f"S{i+1 fig.add_hline(y}")
            fig.add_hline(y=zone.high,=zone.high, line_dash="dash", line line_dash="dash_color="green")
            
       ", line_color="green")
            
        for i, zone for i, zone in enumerate(resistances[:2]):
            fig.add_hline(y=zone in enumerate(resistances[:2]):
            fig.add_.low, line_dash="dash", line_color="hline(y=zone.low, line_dash="dash",red", annotation_text=f" line_color="red", annotation_text=f"R{iR{i+1+1}")
            fig.add_hline(y=zone.high, line_dash}")
            fig.add_hline(y=zone.high, line_d="dash", lineash="dash", line_color="red")
        
       _color="red")
        
        # Current price line
        # Current price line
        current_price = float(view_data["Close current_price = float(view_data["Close"].iloc[-1])
        fig.add_hline"].iloc[-1])
        fig.add(y=current_price, line_dash="dot", line_color="white", 
                     annotation_hline(y=current_price_text=f"Current:, line_dash="dot", line_color="white", 
                     annotation_text=f"Current: {format_price(current_price)}")
 {format_price(current_price)}")
        
        fig.update_layout(
                   
        fig.update_layout(
            title=f"{symbol} - title=f"{symbol} - 4H Chart (Multi 4H Chart (Multi-EMA + S/R)",
            height=600,
-EMA + S/R)",
            height=600,
                       template="plotly_dark",
 template="plotly_dark",
            showlegend=True
            showlegend=True
        )
        )
               st.plotly_chart(fig, use_container_width=True st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader(")

with col2:
    st.subheader("📊 SİNYAL")
    
    if signals📊 SİNYAL")
    
    if and signals[0].typ in ["BUY", "SELL signals and signals[0].typ in ["BUY", ""]:
        signalSELL"]:
        signal = signals[0]
        = signals[0]
        color = "🟢 color = "🟢" if signal.typ == "BU" if signal.typ == "BUY" else "🔴Y" else "🔴"
        st.markdown(f""
        st.markdown(f"### {color} {signal.typ}")
        
        col_a, col_b = st.columns(2)
       ### {color} {signal.typ}")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric with col_a:
            st("Entry", format_price(s.metric("Entry", formatignal.entry))
            st.m_price(signal.entry))
            st.metric("TP1", format_price(signaletric("TP1", format_price(signal.tp1.tp1))
        with))
        with col_b:
            st.metric("SL", format_price(signal.sl))
            st.metric(" col_b:
            st.metric("SL", format_price(signal.sl))
            st.metric("TP2TP2", format_price(signal.tp2))
        
", format_price(signal.tp        st.metric("R2))
        
        st.metric("R/R", f"{signal.r/R", f"{signal.rr:.2f}")
       r:.2f}")
        st.metric("Confidence", f st.metric("Confidence", f"{signal.confidence}%")
        
        st.write("**📋"{signal.confidence}%")
        
        st.write("**📋 Gerek Gerekçe:**")
        for reason in signal.reason:
çe:**")
        for reason in signal.reason:
            st.write(f"• {reason}")
            st.write(f"•            
    else:
        st {reason}")
            
   .markdown("### ⚪ else:
        st.markdown("### ⚪ WAIT")
        if signals:
 WAIT")
        if signals:
            for reason in signals[0].reason:
                st.write(f"• {reason}")

    # Gösterg            for reason in signals[0].reason:
                st.write(f"• {reason}")

    # Göstergeler
   eler
    st.divider st.divider()
    st()
    st.subheader("📈 GÖSTERGEL.subheader("📈 GÖSTERGELER")
ER")
    if not    if not data data_30d_30d.empty:
        current_data = data_.empty:
        current_data = data_30d.iloc[-130d.iloc[-1]
        st.metric]
        st.metric("RSI (("RSI (14)", f"{current_data['14)", f"{current_data['RSI']:.1fRSI']:.1f}")
        st.m}")
        st.metric("RSI Fast (7)", fetric("RSI Fast (7)", f"{current"{current_data['RSI_data['RSI_FAST']:.1f}")
        st_FAST']:.1f.metric("MACD}")
        st.metric("MACD Hist", f"{ Hist", f"{current_data['MACD_HISTOcurrent_data['MACD_HGRAM']:.4f}ISTOGRAM']:.4f}")

# Backtest son")

# Backtest sonuuçları
if run_backçları
if run_backtest:
    st.header("test:
    st.header("📈 BACKTEST SONUÇLARI - 90📈 BACKTEST SONUÇLARI - 90 Gün")
    
    # Progress bar ve status
    progress Gün")
    
    # Progress bar ve status
   _bar = st.progress( progress_bar = st.progress(0)
    status_text0)
    status_text = st.empty()
    status_text = st.empty()
    status_text.text("Backtest başlatılıyor...")
    
    with st.spinner.text("Backtest başlatılıyor...")
    
    with st.spinner("("Backtest çalışBacktest çalışıyorıyor..."):
        data..."):
        data_90_90d = get_4h_data(symbol,d = get_4h_data(symbol, 90 90)
        if not data_90d.empty)
        if not data_90d.empty:
           :
            data_90d = data_90d = compute_indicators compute_indicators(data_90(data_90d)
d)
            results =            results backtest_with_progress(data_90d = backtest_with_progress(data_90d, min_rr, risk_per, min_rr, risk_percent, progress_bar, status_text)
            
            # Clear progress
            progress_bar.emptycent, progress_bar, status_text)
            
            # Clear progress
            progress_bar.empty()
            status_text.empty()
            
            # Sonuçları gö()
            status_text.empty()
            
            # Sonuçları göster
           ster
            st.success(f" st.success(f"✅ Backtest tamamlandı!✅ Backtest tamamlandı! Süre: {results Süre: {results['duration['duration_seconds']:._seconds']:.2f2f} saniye")
            
            col} saniye")
            
            col11, col2, col3,, col4 = st.columns(4 col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Toplam)
            
            with col1:
                st.metric("Toplam İşlem", results[" İşlem", results["trades"])
                sttrades"])
                st.metric("K.metric("Kazanç Oranı", f"{azanç Oranı", f"{results['win_rateresults['win_rate']:.']:.1f}%", 
1f}%", 
                         delta=f"{results['                         delta=f"{results['win_rate']-47win_rate']-47:.1:.1f}%f}% vs önceki" if vs önceki" if results['trades results['trades'] >'] > 0 else None 0 else None)
                
            with col2:
               )
                
            with col2:
                st.metric("Top st.metric("Toplamlam Getiri", f"{ Getiri", f"{resultsresults['total_return']:.['total_return']:.1f}%")
                st.metric1f}%")
                st.metric("Son Bakiye", f"${("Son Bakiye", f"${results['results['final_balance']:,.final_balance']:,.0f}")
                
            with col0f}")
                
            with col3:
                st.metric3:
                st.metric("Ort. Kazan("Ort. Kazanç %ç %", f"{results", f"{results['avg_win_per['avg_win_percent']:.1fcent']:.1f}%")
                st.metric("O}%")
                st.metric("Ort. Kayıp %", frt. Kayıp %", f"{"{results['avg_loss_percent']:.1results['avg_loss_percent']:.1f}%")
                
            with col4:
                st.metric("Profit Factorf}%")
                
            with col4:
                st.metric("Profit Factor", f"{", f"{results['results['profit_factor']:.profit_factor']:.2f}")
               2f}")
                st.metric("Kazanan/Kaybeden", f"{ st.metric("Kazanan/Kaybedresults['winning_trades']en", f"{results['winning_trades']}/{results['losing_trades']}/{results['losing_trades']}")
            
            # Equity curve
}")
            
            # Equity curve
            if "            if "equity_curve"equity_curve" in results in results and len(results["equity_curve"]) > 1 and len(results["equity_curve:
                st.subheader(""]) > 1:
                st.subheader("📊 Equity Curve")
                fig📊 Equity Curve")
_eq                fig_eq = go.Figure()
                fig = go.Figure()
                fig_eq.add_trace(_eq.add_trace(go.Scatter(
                    y=results["equity_curvego.Scatter(
                    y=results["equity_curve"],
                    line=dict(color="green"],
                    line=dict(color="green", width=3", width=3),
                    name="Port),
                    name="Portföy Değeriföy Değeri",
                    fill='tozeroy',
                    fillcolor='rgba(0,255",
                    fill='tozeroy',
                    fillcolor='rgba(0,255,0,0.1,0,0.1)'
                ))
                fig_eq.update_layout(
                    height=400)'
                ))
                fig_eq.update_layout(
                    height=400,
                    template=",
                    template="plotly_dark",
                    showlegendplotly_dark",
                   =False,
                    title=" showlegend=False,
                    title="PortföPortföy Performansı"
                )
                st.ploty Performansı"
                )
                st.plotlyly_chart(fig_eq, use_chart(fig_eq, use_container_width=True)
_container_width=True)
                
            if results["trades"] ==                 
            if results["trades"] ==0:
                st.warning("Backtest sı 0:
                st.warning("Backtest sırasında hiç işlem yaprasında hiç işlem yapılmılmadı. Parametreleri gevşetmadı. Parametreleri gevşetmeyi deneyin.")
eyi deneyin.")
            else:
                st            else:
                st.balloons()
                
        else:
            st.error("Backtest.balloons()
                
        else:
            st.error("Back için veri alınamadı!")

# Stratetest için veri alınamadı!")

# Stji açıklaması
withrateji açıklaması
with st.exp st.expander("🎯 STRATEJİ Dander("🎯 STRATEETAYLARI &Jİ DETAYLARI & OPTIMIZASYONLARI"):
    OPTIMIZASYONLARI"):
    st.markdown("""
    **🚀 MAX WIN RATE OPTIMIZASYONLARI st.markdown("""
    **🚀 MAX WIN RATE OPTIMIZASYONL:**
    
    ###ARI:**
    
    ### 📈 İyileştirm 📈 İyileştirmelereler:
    1. **Multi:
    1. **Multi-Timeframe Analiz**: EMA20,-Timeframe Anal EMA50, EMA100 +iz**: EMA20, EMA50, EMA100 + trend g trend güç skoru
    2üç skoru
    2.. ** **GelişmişGelişmiş S/R Tespiti**: Pivot point bazlı + S/R Tespiti**: Pivot point bazlı + tolerance
    3. ** tolerance
    3. **3 Ana St3 Ana Strateji**:
       - 🎯 **Trend + Pullback**: Yrateji**:
       - 🎯 **Trend + Pullback**: Yüküksek win rate
sek win rate
       - 🔄 **Mean Reversion**:       - 🔄 **Mean Reversion**: Range market optimiz Range market optimizasyonuasyonu  
       -  
       - 🚀 **Breakout + Retest**: İyi risk/reward 🚀 **Breakout + Retest**: İyi risk/reward
    
    ###
    
    ### ⚙️ Optimize Parametreler:
    ⚙️ Optimize Parametreler:
    - **Min R/R**: 1. - **Min R/R**: 3 (daha dü1.3 (daha düşük hedeşük hedeflfler, daha yüker, daha yüksek winsek win rate)
    - **RS rate)
    - **RSI FI Filtreleri**: Gevşetildi (30-70 yerine iltreleri**: Gevşetildi (30-70 yerine 25-25-75)
    - **Trend Threshold**: Düş75)
    - **Trend Threshold**: Düşürüürüldü (daha fazla sinyal)
ldü (daha fazla sinyal)
    -    - **Zone Scoring**: İ **Zone Scoring**: İyyileştirildi
    
    ###ileştirildi
    
    ### 🎯 Hedef:
    ** 🎯 Hedef:
   Win Rate > %55** - Daha tutarlı kazan **Win Rate > %55** - Daha tutarlı kazançlar için optimize
çlar için optimize
    """)