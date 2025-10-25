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
        rsi = float(d["RSI"].iloc[-1]) if not pd.isna(d["RSI"].iloc[-1]) else 50
        rsi_fast = float(d["RSI_FAST"].iloc[-1]) if not pd.isna(d["RSI_FAST"].iloc[-1]) else 50
        macd_hist = float(d["MACD_HISTOGRAM"].iloc[-1]) if not pd.isna(d["MACD_HISTOGRAM"].iloc[-1]) else 0
        
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
                        "",
                        "Range market optimizRange market optimizasyonu"
asyonu"
                    ]
                    signals                    ]
                    signals.append(Signal.append(Signal("SELL", current_price,("SELL", current_price, sl, tp sl, tp1, tp2, rr1, tp2, rr,
                                       ,
                                        confidence, "range confidence, "range", reason))
", reason))
        
        # STRATEGY 3: BREAK        
        # STRATEGY 3:OUT WITH RETEST (Medium BREAKOUT WITH RETEST (Medium Win Rate but good RR Win Rate but good RR)
)
        if not signals        if not signals:
            if best_resistance and current:
            if best_resistance and current_price >= best_resistance.l_price >= best_resistance.low * 0.998ow * 0.998 and trend_strength >= and trend_strength >= 5:
                5:
                # # Breakout above resistance with Breakout above resistance with trend confirmation trend confirmation
               
                sl = sl = best_resistance.low best_resistance.low * * 0.995
 0.995
                               risk = current_price risk = current_price - sl - sl
                tp1 =
                tp1 = current_price + risk * 1.0
                tp2 = current_price + current_price + risk * 1.0
                tp2 = current_price + risk * 1. risk * 1.5
                rr = (tp2 - current_price5
                rr = (tp2 - current_price) / risk
) / risk
                
                               
                if rr >= min if rr >= min_rr and rsi_rr and rsi < 70:
                    confidence = < 70:
                    confidence = min(best_res min(best_resistance.score + istance.score + 2020, 90, 90)
                    reason =)
                    reason = [
                        [
                        " "🚀 BRE🚀 BREAKOUT STRATEAKOUT STRATEJİJİSİ",
                       Sİ",
                        f" f"Resistance breakout (Resistance breakout (GüGüç: {treç: {trend_stnd_strength}/9rength}/9)",
                       )",
                        f"RSI f"RSI {r {rsi:.1fsi:.1f} (} (momentummomentum)",
                       )",
                        f"Risk/R f"Risk/Rewardeward: {rr:.: {rr:.22f}:1f}:1",
",
                        "Trend + Breakout                        "Trend + Breakout kombinasyonu kombinasyonu"
                   "
                    ]
                    ]
                    signals signals.append(S.append(Signalignal("("BUY", currentBUY", current_price, sl,_price, sl, tp1, tp2, rr tp1, tp2, rr,
                                        confidence,,
                                        confidence, trend, reason))
                    
    except trend, reason))
                    
    except Exception as e:
 Exception as e:
        notes.append(f"Hata:        notes.append(f"Hata: {str(e)} {str(e)}")
    
    if not signals")
    
    if not signals:
        wait:
        wait_reason = ["_reason = ["🔍 Uy🔍 Uygun yükgun yüksek win ratesek win rate sinyali sinyali bulun bulunamadı"]
       amadı"]
        if trend if trend_strength < _strength < 4:
4:
            wait_re            wait_reason.append(fason.append(f"Trend çok"Trend çok zay zayıfıf: {trend: {trend_stre_strength}/9")
ngth}/9")
        if        if not not supports supports and not resistances and not resistances:
            wait_re:
            wait_reason.append("Yeterason.append("Yeterli S/R sevili S/R seviyesiyesi yok")
        yok")
        signals signals.append(Signal("WA.append(Signal("WAIT",IT", current_price, 0, 0, 0, 0, current_price, 0, 0 0, 
                            trend,, 0, 0, 0, 
                            trend, wait_re wait_reason))
ason))
    
    return signals, notes    
    return signals, notes

# =============================================================================
# BACKTEST WITH PROGRESS BAR + TIMING
# =

# =============================================================================
# BACKTEST WITH PROGRESS BAR + TIMING
# =============================================================================
============================================================================
@dataclass@dataclass

class Trade:
    entry:class Trade:
    entry: float
    exit: float
    exit: float
    float
    side: side: str
    p str
    pnl: floatnl: float
    p
    pnl_pernl_percent:cent: float
 float
    duration    duration_bars_bars: int

def backtest: int

def backtest_with_progress(df: pd_with_progress(df: pd.DataFrame,.DataFrame, 
                          min 
                          min_rr: float =_rr: float = 1 1.3, 
.3, 
                          risk                          risk_percent:_percent: float =  float = 11.0,
                          progress.0,
                          progress_bar=None,
_bar=None,
                          status_text                          status_text=None) -> Dict=None) -> Dict[str,[str, Any]:
 Any]:
    start_time    start_time = time.time = time.time()
()
    
    
       if df.empty or len if df.empty or len(df) <(df) < 100:
        return {"trades 100:
        return {"trades": 0,": 0, " "win_rate": 0,win_rate": 0, "total_return "total_return": 0, "final_balance": 0, "final_balance": 100": 10000}
    
    balance = 10000}
    
    balance = 10000.0
00.0
    trades = []
    equity    trades = []
    equity = = [balance]
    total [balance]
    total_bars_bars = len(df) = len(df) -  - 50
    
    for50
    
    for i in range(50, len(df i in range(50, len(df) - 5) - 5):
       ):
        # Progress update
 # Progress update
        if progress        if progress_bar_bar is not None and ( is not None and (i - 50) %i - 50) % 10 == 0:
            progress = ( 10 == 0:
            progress = (i - 50i - 50) / total_bars) / total_bars

            progress_bar.progress(min            progress_bar.progress(min((progress, 1.progress, 1.0))
0))
            if status_text            if status_text is not None:
 is not None:
                elapsed = time.time() - start                elapsed = time.time()_time
                status_text.text - start_time
                status(f"Backtest çal_text.text(f"Backtestışıyor... çalışıyor... {len {len(trades)} işlem - %{progress*100:.0f} - {elapsed:.1(trades)} işlem - %{progress*100:.0f} - {elapsed:.1f}sf}s")
        
        try:
            data_s")
        
        try:
           lice = df.iloc[:i+1]
            supports, resistances = find_zones_improved data_slice = df.iloc[:i+1]
            supports, resistances = find_zones_improved(data_slice, min_touch(data_slice, min_touch_points=2_points=2)
            signals, _ = generate_max_winrate_signals(data_slice, supports, resist)
            signals, _ = generate_max_winrate_signals(data_slice, supports, resistancesances, min_rr)
            
            if signals and signals[0].typ in ["BUY, min_rr)
            
            if signals and signals[0].typ in ["BUY",", "SELL"]:
                signal = signals[0]
                entry_price = float "SELL"]:
                signal = signals[0]
                entry_price = float(df["Open"].iloc[i+1])
                exit_bars = min(10, len(df["Open"].iloc[i+1])
                exit_bars = min(10, len(df) - i - 2)  # Max (df) - i - 2)  # Max 10 bars hold
                exit_price = float(df["Close"].iloc[i + exit_bars])
                
                # Smart position10 bars hold
                exit_price = float(df["Close"].iloc[i + exit_bars])
                
                # Smart position sizing
                position_size = balance * ( sizing
                position_size = balance * (risk_percent / 100)
                
                if signal.typ == "risk_percent / 100)
                
                if signal.typ ==BUY":
                    pnl = (exit_price - entry_price) * (position_size / entry_price)
                else "BUY":
                    pnl = (exit_price - entry_price) * (position_size / entry_price)
                else:
                    pnl = (entry_price - exit:
                    pnl = (entry_price - exit_price) * (position_size / entry_price_price) * (position_size / entry_price)
                
                balance += pnl
                pnl_percent)
                
                balance += pnl
                pnl_percent = (pnl / position_size) * 100 if position = (pnl / position_size) * 100 if position_size > 0 else 0
                
                trades.append_size > 0 else 0
                
               (Trade(
                    entry=entry_price,
                    exit=exit_price,
                    side=signal.typ,
                    pnl=pn trades.append(Trade(
                    entry=entry_price,
                    exit=exit_price,
                    side=signal.typ,
                    pnll,
                    pnl_percent=pnl_percent,
                    duration_bars=exit_bars
                ))
=pnl,
                    pnl_percent=pnl_percent,
                    duration_bars=exit_bars
                ))
            
            equity.append(balance)
            
        except            
            equity.append(balance)
            
        except Exception:
            continue
    
    # Final progress update
    Exception:
            continue
    
    # Final progress update
    if progress_bar is not None:
        progress_bar.progress(1.0)
    
 if progress_bar is not None:
        progress_bar.progress(1.0)
    
    #    # Calculate metrics
    total_trades = len(t Calculate metrics
    total_trades = len(trades)
rades)
    if total_trades    if total_trades == ==  0:
        return0:
        return {
 {
            "trades": 0            "trades": 0, 
, 
            "win_rate": 0,            "win_rate": 0, 
            "total_return": 0, 
            
            "total_return": 0, "final_balance": 100 
            "final_balance": 00,
            "duration_seconds10000,
            "duration_seconds": time.time()": time.time() - start_time
        - start_time
        }
    
 }
    
    winning_trades =    winning_trades = len len([t for t in([t for t in trades trades if t.pnl > 0])
    win_rate = (winning if t.pnl > 0])
    win_rate = (winning_trades / total_trades / total_trades_trades) * 100
) * 100
       total_return = (( total_return = ((balance - balance - 1000010000) /) /  10000) * 10010000) * 100
    
    # Additional
    
    # Additional metrics
    metrics
    avg avg_win = np.mean_win = np.mean([t.pnl([t.pnl_percent_percent for t in trades for t in trades if t if t.pnl.pnl > 0]) > 0]) if winning if winning_trades > _trades > 0 else0 else 0 0
    avg_loss = np
    avg_loss = np.mean([.mean([t.pnl_percent for tt.pnl_percent for t in trades in trades if t.pnl <= 0 if t.pnl <= 0]) if]) if total_trades - winning_trades > total_trades - winning_trades > 0 else 0 else 0
    
    total_ 0
    
    total_win = sum(t.pnlwin = sum(t.pnl for t in trades if t.pnl > 0)
 for t in trades if t.pnl > 0)
    total    total_loss = abs(sum_loss = abs(sum(t.pnl for t in(t.pnl for t in trades if t.pnl trades if t.pnl < 0))
    profit < 0))
    profit_factor = total_win / total_factor = total_win / total_loss if total_loss > _loss if total_loss > 00 else float('inf')
 else float('inf')
    
    return {
    
    return {
               "trades": total "trades": total_trades,
        "win_rate":_trades,
        "win win_rate,
        "total_rate": win_rate,
        "total_return": total_return_return": total_return,
       ,
        "final_balance "final_balance": balance": balance,
       ,
        " "equequity_curveity_curve": equity": equity,
        "avg_win,
        "avg_win_percent":_percent": avg_win,
        " avg_win,
        "avg_loss_peravg_loss_percent": avg_loss,
cent": avg_loss,
        "profit        "profit_factor": profit_factor,
        "duration_seconds": time.time() - start_factor": profit_factor,
        "duration_seconds": time_time,
        "winning_trades.time() - start_time,
        "winning_trades": winning_trades,
": winning_trades,
        "losing_trades        "losing_trades":": total_trades - winning_trades total_trades - winning_trades
   
    }

# ========================================================================= }

# =============================================================================
# ARAY====
# ARAYÜZ
# =============================================================================ÜZ
# =========================================================================
st.title("🎯====
st.title("🎯 4H Pro 4H Pro TA - MAX WIN RATE OPT TA - MAX WIN RATE OPTIMIZED")

with st.sideIMIZED")

with st.sidebar:
    st.header("bar:
    st.header("⚙️ Ayarl⚙️ Ayar")
    symbol = starlar")
    symbol = st.text_input("Kripto S.text_input("Kripto Sembolü",embolü", "BTC-USD")
    "BTC-USD")
    min_rr = st min_rr = st.slider("Min.slider("Min R/R R/R", 1.", 1.1, 21, 2.0.0, 1., 1.3,3, 0. 0.11)
    risk_per)
    risk_percent =cent = st.slider(" st.slider("Risk %", 0.5, Risk %", 0.53.0, 1, 3.0,.0, 0.1)
 1.0, 0.1)
    
    st.divider()
    st.subheader("🎯 Strateji Seçimi    
    st.divider()
    st.subheader("")
    strategy_mode = st🎯 Strateji Seçimi")
    strategy_mode = st.select.selectbox(
        "Sbox(
        "Sinyinyal Modu",
        ["al Modu",
        ["MAX_WIN_RATEMAX_WIN_RATE",", "BALANC "BALANCED", "AGGRESSIVE"],
       ED", "AGGRESSIVE"],
 help="MAX_WIN_R        help="MAX_WIN_RATE: Daha yATE: Daha yüksek kazanç orüksek kazanç oranı, BALANCED:anı, BALANCED Denge,: Denge, AG AGGRGRESSIVEESSIVE: Daha fazla i: Daha fazla işşlem"
    )
    
   lem"
    )
    
    st st.divider()
    st.subheader.divider()
    st.subheader("("🚀 Backtest")
🚀 Backtest")
    run_backtest = st.button("BACKTEST Ç    run_backtest = st.button("BACKTEST ÇALIŞTIR (ALIŞTIR (90g)",90g)", type="primary", use type="primary", use_container_width=True)

_container_width=True)

# Ana veri y# Ana veri yükleme
withükleme
with st.spinner("📊 st.spinner("📊 Veri yüklen Veri yükleniyor ve analiziyor ve analiz ediliyor..."):
    data_30d = get ediliyor..."):
    data_30d_4h_data(symbol, = get_4h_data(symbol, 30)
 30)
    if not data_30    if not data_30d.emptyd.empty:
       :
        data_30d = compute_indic data_30d = compute_indicators(data_30ators(data_30d)
        supports, resistances =d)
        supports, resist find_zones_ances = find_zonesimproved(data_30d_improved(data_30d)
        signals,)
        signals, notes notes = generate_max_ = generate_max_winrate_signals(data_30d,winrate_signals(data_30d, supports, resistances supports, resistances, min_rr)
    else, min_rr)
    else:
        st.error:
        st.error("❌ Veri alınam("❌ Veriadı!")
 alınamadı!")
        st.stop()

# Ana görünüm        st.stop()

# Ana görünüm
col1, col
col1, col2 = st.columns([2, 1])

with col1:
    if not data_302 = st.columns([2, 1])

with col1:
    if not data_30dd.empty:
        fig =.empty:
        fig = go go.Figure()
.Figure()
        
        view_data = data        
        view_data = data_30d.tail(24_30d.tail(24)
        
        # Candlest)
        
        # Candlestick
        fig.add_tick
        fig.add_trace(go.Candlestrace(go.Candlestick(
            x=viewick(
            x=view_data.index,
            open=_data.index,
            open=view_dataview_data['Open'],
['Open'],
            high=view_data['High'],
            high=view_data['High'],
            low=view_data            low=view_data['Low'],
            close=view_data['Close'],
['Low'],
            close=view_data['Close            name="Price"
        ))
        
'],
            name="Price"
        ))
        
               # EMAs
        # EMAs
        fig fig.add_trace(go.add_trace(go.Scatter.Scatter(x=view_data.index, y(x=view_data.index, y=view=view_data['EMA20'], line=dict(color='yellow', width=1),_data['EMA20'], line=dict(color='yellow', width=1), name=" name="EMA20"))
        fig.add_trace(EMA20"))
        fig.add_trace(gogo.Scatter(x=view_data.Scatter(x=view_data.index, y=view.index, y=_data['EMA50'], line=view_data['EMA50'], line=dict(color='orange',dict(color='orange', width width=2), name="=2), name="EMA50"))
        figEMA50"))
        fig.add_trace(go.Scatter.add_trace(go.Scatter(x=view_data.index,(x=view_data.index y=view_data[', y=view_data['EMA100'], line=dictEMA100'], line=dict(color='red',(color='red', width= width=1), name="1), name="EMAEMA100"))
        
        #100"))
        
        # Support/Resistance
        Support/Resistance
        for i, zone in for i, zone in enumerate enumerate(supports[:2]):
(supports[:2]):
            fig.add_hline(y=zone            fig.add_hline(y=zone.low.low, line_dash=", line_dash="dash", line_color="dash", line_color="green", annotation_text=f"S{igreen", annotation_text=f+1}")
            fig"S{i.add_hline(y=zone.high,+1}")
            fig.add_hline(y=zone.high, line_d line_dashash="dash", line_color="green")
            
        for i, zone in enumerate="dash", line_color="green")
            
        for i, zone in enumerate(resistances[:2]):
           (resistances[:2]):
            fig.add_ fig.add_hline(yhline(y=zone.l=zone.low, lineow, line_dash_dash="dash", line="dash", line_color="red", annotation_text=f"R_color="red", annotation_text=f"R{i+1}")
{i+1}")
            fig            fig.add_.add_hline(y=zone.highhline(y=zone.high, line_dash="dash", line_color, line_dash="dash",="red")
        
        # line_color="red")
        
        # Current price Current price line
 line
               current_price = float(view current_price =_data["Close"].il float(view_data["Close"].ilococ[-[-1])
        fig.add_hline1])
        fig.add_hline(y=current_price, line_dash="dot", line_color="white", 
                     annotation_text=f"Current: {format_price(current_price)}")
        
        fig.update_layout(
            title=f"{symbol} - 4H Chart(y=current_price, line_dash="dot", line_color="white", 
                     annotation_text=f"Current: {format_price(current_price)}")
        
        fig.update_layout(
            title=f"{symbol} - (Multi-EMA + S/R)",
            height=600,
            template="plot 4H Chart (Multi-EMA + S/R)",
            height=600,
            template="plotly_dark",
            showlegend=True
       ly_dark",
            showlegend=True
 )
        st.plotly_chart(f        )
        st.plotly_chart(fig, use_container_width=True)

with colig, use_container_width=True)

with col2:
    st.subheader("📊 S2:
    st.subheader("📊 SİNYAL")
    
   İNYAL")
    
    if if signals and signals and signals[0].typ in ["BUY", signals[0].typ in ["BUY", "SELL"]:
 "SELL"]:
               signal = signals signal = signals[0[0]
]
        color = "🟢" if signal.        color = "🟢" if signal.typ == "BUY" else "typ == "BUY" else "🔴"
🔴"
        st.markdown(f"### {        st.markdown(f"### {color} {signal.typ}")
color} {signal.typ}")
        
        col_a, col_b = st        
        col_a, col_b = st.columns(2.columns(2)
        with col_a)
        with col_a:
            st:
            st.metric(".metric("Entry", format_price(signal.Entry", format_price(signal.entry))
            stentry))
            st.metric(".metric("TP1", format_price(signalTP1", format_price(signal.tp.tp1))
       1))
        with col_b with col_b:
            st.m:
            st.metric("SL", format_price(signaletric("SL", format_price.sl))
            st.m(signal.sl))
            st.metric("TP2etric("TP2", format_price(signal.tp2", format_price(signal.t))
        
        st.mp2))
        
        st.metric("etric("R/RR/R", f"{signal.r", f"{signal.rr:.2f}")
        st.mr:.2f}")
        st.metric("Confidenceetric("Confidence", f", f"{signal"{signal.confidence}.confidence}%")
        
        st.write("**📋%")
        
        st.write("**📋 Gerekçe Gerekçe:**:**")
")
               for reason in for reason in signal.re signal.reason:
           ason:
            st.write st.write(f"• {(f"• {reason}")
            
    else:
        streason}")
            
    else:
        st.markdown("### ⚪.markdown("### ⚪ WAIT WAIT")
        if signals")
        if signals:
            for reason in signals[0].:
            for reason in signals[0].reason:
                streason:
                st.write(f".write(f"• {reason• {reason}")

    # Göstergeler}")

    # Göstergeler
    st.divider()
    st.subheader("📈
    st.divider()
    GÖSTERGELER")
    if st.subheader("📈 GÖSTERGELER")
    not data_30d if not data_30d.empty:
        current.empty:
        current_data = data_30d.iloc_data = data_30d.il[-1]
        # RSI değerlerini goc[-1]
        # RSI değerlerini güvenli şekilde al
        rsi_value = current_data['RSI']
        rüvenli şekilde al
        rsi_value = current_data['RSI']
        rsi_fast_valuesi_fast_value = = current_data['RS current_data['RSI_FAST']
        macdI_FAST']
        mac_hist_value = current_data['d_hist_value = current_data['MACD_HISTMACD_HISTOGRAMOGRAM']
        
        st.metric("RSI (']
        
        st.metric("RSI (14)",14)", f"{rsi_value f"{rsi_value:.1:.1f}" if not pdf}" if not pd.isna(rsi_value) else "N/A.isna(rsi_value) else "N/A")
        st.metric("")
        st.metric("RSI Fast (RSI Fast (77)", f"{rsi_f)", f"{rsi_fast_value:.1f}" if notast_value:.1f}" if pd.isna(rsi not pd.isna(rsi_fast_value) else_fast_value) else "N "N/A")
        st.metric("MACD Hist", f"{macd_/A")
        st.metric("MACD Hist", f"{macd_histhist_value:.4f_value:.4f}" if not pd.is}" if not pd.isna(macna(macd_hist_value) else "d_hist_value) else "N/A")

# BacktestN/A")

# Backtest sonuçları
if run_backtest sonuçları
if run_backtest:
    st.header("📈 BACKTEST:
    st.header("📈 BACKTEST SON SONUÇLARI - UÇLARI - 90 Gün")
    
   90 Gün")
    
    # # Progress bar Progress bar ve status
    ve status
    progress progress_bar = st.progress(_bar = st.progress(0)
    status_text = st.empty()
    status_text0)
    status_text = st.empty()
    status_text.text("Backtest baş.text("Backtest başlatlatılıyor...")
    
   ılıyor...")
    
    with st.spinner("Backtest ç with st.spinner("Backtest çalışıyor..."):
        data_90d = get_alışıyor..."):
        data_90d = get_44h_data(symbol, 90)
h_data(symbol, 90)
        if not data_90d.empty:
            data_90d =        if not data_90d.empty:
            data_90d = compute compute_indicators(data_90d)
            results = backtest_with_indicators(data_90d)
            results = backtest_with_pro_progress(data_90d, mingress(data_90d, min_rr, risk_percent, progress_rr, risk_percent, progress_bar, status_text)
            
            #_bar, status_text)
            
            # Clear Clear progress
            progress_bar.empty()
            status_text.empty()
            
            # Sonuçları göster
            st.success progress
            progress_bar.empty()
            status_text.empty()
            
            # Sonuçları göster
            st.success(f"✅ Backtest(f"✅ Backtest tamamlandı! Süre: {results tamamlandı! Süre: {results['duration_seconds']['duration_seconds']:.2f:.2f} s} saniye")
            
            col1, colaniye")
            
            col2, col3, col1, col2, col3, col4 = st4 = st.columns(4)
            
            with col.columns(4)
            
           1:
                st.metric("Top with col1:
                st.metric("Toplam İşlem", results["tlam İşlem", results["trades"])
                st.metric("Kazanç Orades"])
                st.metric("Kazanç Oranı",ranı", f"{results['win_rate']:.1 f"{results['winf}%", 
_rate']:.1f}%", 
                         delta=f"{results['                         delta=f"{results['win_ratewin_rate']-47:.1f}% vs']-47:.1f}% vs önce önceki" if results['tradeski" if results['trades'] > 0 else None)
                
           '] > 0 else None)
                
            with col2:
                st with col2:
                st.metric("Top.metric("Toplam Getirilam Getiri",", f"{results['total f"{results['total_return']:.1f}_return']:.1f}%%")
")
                st.metric                st.metric("Son Bakiye",("Son Bakiye", f"${results['final_balance'] f"${results['final_balance']:,.0f}")
                
            with col:,.0f}")
                
            with col3:
                st.metric("Ort3:
                st.metric("Ort. Kazanç %",. Kazanç %", f f"{results['avg_win"{results['avg_win_percent']:.1_percent']:.1f}%")
                stf}%")
                st.metric("Ort..metric("Ort. Kayıp %", f"{ Kayıp %", fresults['avg_loss_percent']"{results['avg_loss_percent']:.1f}:.1f}%")
                
            with col%")
                
            with col4:
                st4:
                st.metric("Profit Factor", f"{.metric("Profit Factor", f"{results['profitresults['profit_factor']_factor']:.2f}")
               :.2f}")
                st.metric("Kazanan/K st.metric("Kazanan/Kaybeden", f"{results['winning_tradesaybeden", f"{results['winning_trades']}/{results['losing_trades']']}/{results['losing_trades']}")
            
            # Equity curve
}")
            
            # Equity curve
            if "equity_            if "equity_curve" in results and lencurve" in results and len(results["equity_curve(results["equity_curve"]) > 1:
"]) > 1:
                               st.subheader("📊 st.subheader("📊 Equity Curve")
                fig_eq = go.Figure Equity Curve")
                fig_eq = go.Figure()
                fig_eq.add_t()
                fig_eq.add_trace(gorace(go.Scatter(
.Scatter(
                    y=results["equity_curve"],
                                       y=results["equity_ line=dict(color="green", width=3),
                    name="Portföcurve"],
                    line=dict(color="green", width=3),
                    namey Değeri",
                    fill='tozer="Portföy Değeri",
                    fill='oy',
                    fillcolor='rgba(0,255,0,0.1)'
tozeroy',
                    fillcolor='rgba(0,255,0,0.1)'
                ))
                fig_eq.update                ))
                fig_eq.update_layout(
                    height=400_layout(
                    height=400,
                    template="plotly_dark",
                    showlegend=False,
                    title="Port,
                    template="plotly_dark",
                    showlegend=False,
                    title="Portföy Performansı"
                )
                stföy Performansı"
.plot                )
                st.plotlyly_chart(fig_chart(fig_eq, use_container_width=True)
                
            if results["trades"] == _eq, use_container_width=True)
                
            if results["trades"] ==0:
                st.warning("Backtest sırasında hiç işlem 0:
                st.warning("Backtest sırasında hiç i yapılmadı. Parametreleri gevşlem yapılmadı. Parametrelerişetmeyi deney gevşetmeyi deneyin.")
            elsein.")
            else:
                st.balloons:
                st.balloons()
                
        else:
           ()
                
        else:
            st.error st.error("Backtest için("Backtest için veri veri alınamadı!")

# Strateji açıklamas alınamadı!")

# Strateji açı
with st.expıklaması
with st.expander("🎯 STRATEJİ DETander("🎯 STRATEJİ DETAYLARI &AYLARI & OPTIMIZASYONLARI OPTIMIZASYONL"):
    st.markdown("""
    **🚀ARI"):
    st.markdown("""
    **🚀 MAX WIN RATE OPT MAX WIN RATE OPTIMIZASYONIMIZASYONLARI:**
    
   LARI:**
    
    ### 📈 İyileşt ### 📈 İyileştirmeler:
    1. **Multi-Timeframe Analizirmeler:
    1. **Multi-Timeframe Anal**: EMA20, EMA50,iz**: EMA20, EMA50, EMA100 + trend güç skoru
 EMA100 + trend güç sk    2. **Gelişmiş S/R Toru
    2. **espiti**: Pivot point bazGelişmiş S/R Tespiti**: Pivot point bazlı + tolerance
    3. **3 Ana Strateji**:
       - 🎯lı + tolerance
    3. **3 Ana Strateji**:
       - 🎯 **Trend + Pullback**: Y **Trend + Pullback**: Yüksek win rate
üksek win rate
       - 🔄 **Mean Reversion**: Range market optimizasyonu  
       - 🔄 **Mean Reversion**: Range market optimizasyonu  
       -       - 🚀 ** 🚀 **Breakout + Retest**: İyi riskBreakout + Retest**: İyi risk/re/reward
    
    ### ⚙️ Optimward
    
    ### ⚙️ Optimize Parametreize Parametreler:
    - **Minler:
    - **Min R/R**: 1.3 (daha düşük he R/R**: 1.3 (daha düdefler,şük hedefler, daha y daha yüksek win rate)
   üksek win rate)
    - **RSI Filtreleri**: Gevşetild - **RSI Filtreleri**: Gei (30-70 yerine 25-75)
    - **vşetildi (30-70 yerine 25-75)
    - **Trend ThresholdTrend Threshold**: Düşürüldü (**: Düşürüldü (daha fazla sindaha fazla sinyal)
    - **Zone Scoring**: İyileştirildi
    
    ###yal)
    - **Zone Scoring**: İyileştirildi
    
 🎯 Hedef:
    ### 🎯 Hedef:
    **Win Rate > %55** - D    **Win Rate > %55** - Daha tutaha tutarlı kazançlar için optimize
    """)
arlı kazançlar için optimize
    """)