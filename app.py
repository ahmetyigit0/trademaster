# app.py
# ─────────────────────────────────────────────────────────────────────────────
# 4H • YFinance • EMA50 Trend • RSI14 • ADX14 Rejim • ATR SL
# S/R + Onaylı Giriş • Kısmi TP + Break-even • Basit Trailing
# Optimized for Higher Win Rate
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

st.set_page_config(page_title="4H Pro TA (High Win Rate v2)", layout="wide")

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
# YARDIMCI - FORMAT
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

# =============================================================================
# VERİ
# =============================================================================
@st.cache_data
def get_4h_data(symbol: str, days: int) -> pd.DataFrame:
    sym = symbol.upper().strip()
    if "-" not in sym:
        sym = sym + "-USD"
    df = yf.download(sym, period=f"{days}d", interval="4h", progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.dropna()
    cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df = df[[c for c in cols if c in df.columns]]
    return df

def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(n).mean()
    avg_loss = loss.rolling(n).mean().replace(0, 1e-8)
    rs = avg_gain / avg_loss
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def adx(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """
    DÜZELTME: pd.Series içine 2D array gitmesini engellemek için
    np.where(...) çağrılarını .values ile 1D garanti ediyoruz ve index açıkça veriyoruz.
    """
    high, low, close = df["High"], df["Low"], df["Close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm_vals = np.where((up_move > down_move) & (up_move > 0), up_move.values, 0.0)
    minus_dm_vals = np.where((down_move > up_move) & (down_move > 0), down_move.values, 0.0)

    plus_dm = pd.Series(plus_dm_vals, index=df.index)
    minus_dm = pd.Series(minus_dm_vals, index=df.index)

    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_n = tr.rolling(n).mean().replace(0, 1e-8)

    plus_di = 100 * (plus_dm.rolling(n).mean() / atr_n).replace(0, 1e-8)
    minus_di = 100 * (minus_dm.rolling(n).mean() / atr_n).replace(0, 1e-8)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-8)
    adx_val = dx.rolling(n).mean()

    out = pd.DataFrame({"PLUS_DI": plus_di, "MINUS_DI": minus_di, "ADX": adx_val})
    return out.fillna(method="bfill").fillna(0)

def bollinger(series: pd.Series, n: int = 20, k: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ma = series.rolling(n).mean()
    sd = series.rolling(n).std()
    upper = ma + k * sd
    lower = ma - k * sd
    return lower, ma, upper

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    d = df.copy()
    d["EMA50"] = ema(d["Close"], 50)
    d["RSI14"] = rsi(d["Close"], 14)
    d["ATR14"] = atr(d, 14)
    adx_df = adx(d, 14)
    d = d.join(adx_df)
    bb_l, bb_m, bb_u = bollinger(d["Close"], 20, 2.0)
    d["BB_L"], d["BB_M"], d["BB_U"] = bb_l, bb_m, bb_u
    return d.dropna()

# =============================================================================
# S/R BÖLGELERİ
# =============================================================================
class Zone:
    def __init__(self, low: float, high: float, touches: int, kind: str):
        self.low = float(low)
        self.high = float(high)
        self.touches = int(touches)
        self.kind = kind
        self.score = 0

def find_zones_simple(d: pd.DataFrame, lookback: int = 80, min_touch_points: int = 3) -> Tuple[List[Zone], List[Zone]]:
    if d.empty or len(d) < lookback:
        return [], []
    data = d.tail(lookback).copy()
    current_price = float(data["Close"].iloc[-1])

    price_levels = []
    for i in range(len(data)):
        price_levels.extend([
            float(data["High"].iloc[i]),
            float(data["Low"].iloc[i]),
            float(data["Close"].iloc[i])
        ])
    if not price_levels:
        return [], []

    pr_min, pr_max = min(price_levels), max(price_levels)
    price_range = pr_max - pr_min
    if price_range <= 0:
        return [], []

    bin_size = price_range * 0.015  # %1.5'lik bölge
    bins = {}
    current_bin = pr_min
    while current_bin <= pr_max:
        bin_end = current_bin + bin_size
        count = sum(1 for p in price_levels if current_bin <= p <= bin_end)
        if count >= min_touch_points:
            bins[(current_bin, bin_end)] = count
        current_bin = bin_end

    supports, resistances = [], []
    for (low, high), touches in bins.items():
        z = Zone(low, high, touches, "support" if high < current_price else "resistance" if low > current_price else "mid")
        if z.kind == "support":
            supports.append(z)
        elif z.kind == "resistance":
            resistances.append(z)

    for z in supports + resistances:
        z.score = min(z.touches * 20, 100)

    supports = sorted(supports, key=lambda z: z.score, reverse=True)[:3]
    resistances = sorted(resistances, key=lambda z: z.score, reverse=True)[:3]
    return supports, resistances

# =============================================================================
# SİNYAL MOTORU
# =============================================================================
@dataclass
class Signal:
    typ: str              # BUY | SELL | WAIT
    entry: float
    sl: Optional[float]
    tp1: Optional[float]
    tp2: Optional[float]
    rr_tp2: float
    confidence: int
    trend: str
    reason: List[str]

def close_confirms_reversal(df: pd.DataFrame, idx: int, long: bool) -> bool:
    if idx < 1: return False
    if long:
        return df["Close"].iloc[idx] > df["Close"].iloc[idx-1]
    else:
        return df["Close"].iloc[idx] < df["Close"].iloc[idx-1]

def touched_band(df: pd.DataFrame, idx: int, long: bool) -> bool:
    if long:
        return df["Low"].iloc[idx] <= df["BB_L"].iloc[idx]
    else:
        return df["High"].iloc[idx] >= df["BB_U"].iloc[idx]

def generate_signals(
    d: pd.DataFrame,
    supports: List[Zone],
    resistances: List[Zone],
    min_rr: float = 1.2,
    use_bb_filter: bool = True,
    adx_trend_thr: float = 25.0,
    adx_range_thr: float = 18.0,
    atr_mult_sl: float = 1.0,
    tp1_r_mult: float = 0.6
) -> Tuple[List[Signal], List[str]]:
    notes, signals = [], []
    if d.empty or len(d) < 30:
        return [Signal("WAIT", 0, None, None, None, 0, 0, "neutral", ["Yetersiz veri"])], ["❌ Yetersiz veri"]

    i = len(d) - 1
    price = float(d["Close"].iloc[i])
    ema50 = float(d["EMA50"].iloc[i])
    rsi14 = float(d["RSI14"].iloc[i])
    atr14 = float(d["ATR14"].iloc[i])
    adx14 = float(d["ADX"].iloc[i])

    trend = "bull" if price > ema50 else "bear"
    regime = "trend" if adx14 >= adx_trend_thr else "range" if adx14 <= adx_range_thr else "mid"
    notes += [f"Trend: {trend.upper()} | ADX: {adx14:.1f} ({regime})", f"RSI: {rsi14:.1f}", f"ATR: {atr14:.3f}"]

    best_s = supports[0] if supports else None
    best_r = resistances[0] if resistances else None

    def build_long(zone: Zone, conf_base: int) -> Optional[Signal]:
        sl = min(zone.low - atr_mult_sl * atr14, price - 0.0001)
        risk = price - sl
        if risk <= 0: return None
        tp2 = price + risk * min_rr
        tp1 = price + risk * (min_rr * tp1_r_mult)
        rr2 = (tp2 - price) / risk
        conf = conf_base
        reason = [f"Support zone score {zone.score}", f"ATR SL x{atr_mult_sl}", f"RR(TP2) {rr2:.2f}"]
        return Signal("BUY", price, sl, tp1, tp2, rr2, conf, "bull", reason)

    def build_short(zone: Zone, conf_base: int) -> Optional[Signal]:
        sl = max(zone.high + atr_mult_sl * atr14, price + 0.0001)
        risk = sl - price
        if risk <= 0: return None
        tp2 = price - risk * min_rr
        tp1 = price - risk * (min_rr * tp1_r_mult)
        rr2 = (price - tp2) / risk
        conf = conf_base
        reason = [f"Resistance zone score {zone.score}", f"ATR SL x{atr_mult_sl}", f"RR(TP2) {rr2:.2f}"]
        return Signal("SELL", price, sl, tp1, tp2, rr2, conf, "bear", reason)

    # TREND REJİMİ
    if regime == "trend":
        if trend == "bull" and best_s and best_s.score >= 80:
            near_s = price <= best_s.high * 1.005
            rsi_ok = rsi14 <= 50
            confirm = close_confirms_reversal(d, i, long=True)
            bb_ok = (touched_band(d, i, True) if use_bb_filter else True)
            if near_s and rsi_ok and confirm and bb_ok:
                sig = build_long(best_s, conf_base=min(90, best_s.score))
                if sig: signals.append(sig)

        if trend == "bear" and best_r and best_r.score >= 80:
            near_r = price >= best_r.low * 0.995
            rsi_ok = rsi14 >= 50
            confirm = close_confirms_reversal(d, i, long=False)
            bb_ok = (touched_band(d, i, False) if use_bb_filter else True)
            if near_r and rsi_ok and confirm and bb_ok:
                sig = build_short(best_r, conf_base=min(90, best_r.score))
                if sig: signals.append(sig)

    # RANGE REJİMİ
    if regime == "range" and not signals:
        if best_s and price <= best_s.high * 1.003 and rsi14 <= 35:
            confirm = close_confirms_reversal(d, i, long=True)
            bb_ok = (touched_band(d, i, True) if use_bb_filter else True)
            if confirm and bb_ok:
                sig = build_long(best_s, conf_base=75)
                if sig: signals.append(sig)
        if best_r and price >= best_r.low * 0.997 and rsi14 >= 65 and not signals:
            confirm = close_confirms_reversal(d, i, long=False)
            bb_ok = (touched_band(d, i, False) if use_bb_filter else True)
            if confirm and bb_ok:
                sig = build_short(best_r, conf_base=75)
                if sig: signals.append(sig)

    if not signals:
        signals.append(Signal("WAIT", price, None, None, None, 0, 0, trend, ["Uygun sinyal yok"]))
    return signals, notes

# =============================================================================
# BACKTEST (Bar-bar, SL/TP, TP1→BE, Basit Trailing)
# =============================================================================
@dataclass
class Trade:
    entry_i: int
    entry: float
    exit_i: int
    exit: float
    side: str
    pnl: float
    pnl_pct: float
    hit_tp1: bool
    hit_tp2: bool
    stopped: bool

def simulate_trade_path(df: pd.DataFrame, i_entry: int, side: str, entry: float, sl: float, tp1: float, tp2: float,
                        max_hold_bars: int = 12) -> Tuple[float, int, bool, bool, bool]:
    hit_tp1 = False
    hit_tp2 = False
    stopped = False
    stop = sl

    def bar_hit(long: bool, hi: float, lo: float, level: float) -> bool:
        return (hi >= level) if long else (lo <= level)

    last_swing_low = df["Low"].iloc[i_entry] if side == "BUY" else None
    last_swing_high = df["High"].iloc[i_entry] if side == "SELL" else None

    for k in range(1, max_hold_bars + 1):
        idx = i_entry + k
        if idx >= len(df):
            break
        hi, lo, close = float(df["High"].iloc[idx]), float(df["Low"].iloc[idx]), float(df["Close"].iloc[idx])

        if side == "BUY":
            if bar_hit(True, hi, lo, tp2):
                hit_tp2 = True
                return tp2, idx, True, True, False
            if (not hit_tp1) and bar_hit(True, hi, lo, tp1):
                hit_tp1 = True
                stop = max(stop, entry)
            prev_low = float(df["Low"].iloc[idx-1])
            last_swing_low = max(last_swing_low, prev_low) if last_swing_low is not None else prev_low
            stop = max(stop, last_swing_low)
            if bar_hit(False, hi, lo, stop):
                stopped = True
                return stop, idx, hit_tp1, hit_tp2, True
        else:
            if bar_hit(False, hi, lo, tp2):
                hit_tp2 = True
                return tp2, idx, True, True, False
            if (not hit_tp1) and bar_hit(False, hi, lo, tp1):
                hit_tp1 = True
                stop = min(stop, entry)
            prev_high = float(df["High"].iloc[idx-1])
            last_swing_high = min(last_swing_high, prev_high) if last_swing_high is not None else prev_high
            stop = min(stop, last_swing_high if last_swing_high is not None else stop)
            if bar_hit(True, hi, lo, stop):
                stopped = True
                return stop, idx, hit_tp1, hit_tp2, True

    exit_price = float(df["Close"].iloc[min(i_entry + max_hold_bars, len(df)-1)])
    return exit_price, min(i_entry + max_hold_bars, len(df)-1), hit_tp1, hit_tp2, stopped

def backtest(
    df: pd.DataFrame,
    min_rr: float,
    risk_percent: float,
    use_bb_filter: bool,
    adx_trend_thr: float,
    adx_range_thr: float,
    atr_mult_sl: float,
    tp1_r_mult: float,
    max_hold_bars: int
) -> Dict[str, Any]:
    if df.empty or len(df) < 150:
        return {"trades": 0, "win_rate": 0, "total_return": 0, "final_balance": 10000, "equity_curve": []}

    balance = 10000.0
    equity = [balance]
    trades: List[Trade] = []

    for i in range(120, len(df)-max_hold_bars-1):
        data_slice = df.iloc[:i+1]
        supports, resistances = find_zones_simple(data_slice, lookback=80, min_touch_points=3)
        signals, _ = generate_signals(
            data_slice, supports, resistances,
            min_rr=min_rr, use_bb_filter=use_bb_filter,
            adx_trend_thr=adx_trend_thr, adx_range_thr=adx_range_thr,
            atr_mult_sl=atr_mult_sl, tp1_r_mult=tp1_r_mult
        )
        sig = signals[0]
        if sig.typ not in ["BUY", "SELL"]:
            equity.append(balance)
            continue

        entry = float(df["Open"].iloc[i+1])
        sl, tp1, tp2 = sig.sl, sig.tp1, sig.tp2
        if sl is None or tp1 is None or tp2 is None:
            equity.append(balance)
            continue

        position_size = balance * (risk_percent / 100.0)

        exit_price, exit_i, hit_tp1, hit_tp2, stopped = simulate_trade_path(
            df, i+1, sig.typ, entry, sl, tp1, tp2, max_hold_bars=max_hold_bars
        )

        if sig.typ == "BUY":
            pnl = (exit_price - entry) * (position_size / entry)
        else:
            pnl = (entry - exit_price) * (position_size / entry)

        balance += pnl
        pnl_pct = (pnl / position_size) * 100.0
        trades.append(Trade(
            entry_i=i+1, entry=entry, exit_i=exit_i, exit=exit_price,
            side=sig.typ, pnl=pnl, pnl_pct=pnl_pct,
            hit_tp1=hit_tp1, hit_tp2=hit_tp2, stopped=stopped
        ))
        equity.append(balance)

    total_trades = len(trades)
    if total_trades == 0:
        return {"trades": 0, "win_rate": 0, "total_return": 0, "final_balance": 10000, "equity_curve": equity}

    wins = sum(1 for t in trades if t.pnl > 0)
    win_rate = 100.0 * wins / total_trades
    total_return = 100.0 * (balance - 10000.0) / 10000.0

    return {
        "trades": total_trades,
        "win_rate": win_rate,
        "total_return": total_return,
        "final_balance": balance,
        "equity_curve": equity,
        "trades_list": trades
    }

# =============================================================================
# UI
# =============================================================================
st.title("🎯 4H Pro TA — High Win Rate v2")

with st.sidebar:
    st.header("⚙️ Ayarlar")
    symbol = st.text_input("Sembol (örn. BTC-USD)", "BTC-USD")
    days_view = st.slider("Grafik Gün (30–120)", 30, 120, 60, 5)

    st.subheader("Sinyal / Risk")
    min_rr = st.slider("Min R/R (TP2 için)", 1.0, 3.0, 1.2, 0.1)
    risk_percent = st.slider("Risk % (pozisyon)", 0.5, 5.0, 1.0, 0.1)
    atr_mult_sl = st.slider("SL ATR Çarpanı", 0.5, 2.5, 1.0, 0.1)
    tp1_r_mult = st.slider("TP1 oranı (TP2*..)", 0.3, 0.9, 0.6, 0.05)

    st.subheader("Rejim / Filtre")
    adx_trend_thr = st.slider("ADX Trend Eşiği", 20, 35, 25, 1)
    adx_range_thr = st.slider("ADX Range Eşiği", 10, 25, 18, 1)
    use_bb_filter = st.checkbox("Range’de Bollinger dokunuşu iste (Önerilir)", True)

    st.subheader("Backtest")
    run_backtest = st.button("🚀 Backtest (90g)")
    max_hold_bars = st.slider("Max Tutma (bar)", 6, 24, 12, 1)

# VERİ
with st.spinner("Veri yükleniyor..."):
    data = get_4h_data(symbol, max(90, days_view))
    if data.empty:
        st.error("❌ Veri alınamadı!")
        st.stop()
    data_ind = compute_indicators(data)

# S/R & Sinyal
supports, resistances = find_zones_simple(data_ind, lookback=80, min_touch_points=3)
signals, notes = generate_signals(
    data_ind, supports, resistances,
    min_rr=min_rr, use_bb_filter=use_bb_filter,
    adx_trend_thr=adx_trend_thr, adx_range_thr=adx_range_thr,
    atr_mult_sl=atr_mult_sl, tp1_r_mult=tp1_r_mult
)

# GRAFİK
col1, col2 = st.columns([2,1])
with col1:
    view = data_ind.tail(int(days_view*6))  # 4H ~ 6 bar/gün
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=view.index, open=view["Open"], high=view["High"],
        low=view["Low"], close=view["Close"], name="Price"
    ))
    fig.add_trace(go.Scatter(
        x=view.index, y=view["EMA50"], name="EMA50", line=dict(width=2)
    ))
    fig.add_trace(go.Scatter(
        x=view.index, y=view["BB_U"], name="BB Upper", line=dict(width=1, dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=view.index, y=view["BB_M"], name="BB Mid", line=dict(width=1, dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=view.index, y=view["BB_L"], name="BB Lower", line=dict(width=1, dash="dot")
    ))

    for i, z in enumerate(supports[:2]):
        fig.add_hline(y=z.low, line_dash="dash", line_color="green", annotation_text=f"S{i+1}L")
        fig.add_hline(y=z.high, line_dash="dash", line_color="green", annotation_text=f"S{i+1}H")
    for i, z in enumerate(resistances[:2]):
        fig.add_hline(y=z.low, line_dash="dash", line_color="red", annotation_text=f"R{i+1}L")
        fig.add_hline(y=z.high, line_dash="dash", line_color="red", annotation_text=f"R{i+1}H")

    fig.update_layout(title=f"{symbol} • 4H", height=520, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📊 Sinyal")
    s0 = signals[0]
    if s0.typ in ["BUY", "SELL"]:
        color = "🟢" if s0.typ == "BUY" else "🔴"
        st.markdown(f"### {color} {s0.typ}")
        ca, cb = st.columns(2)
        with ca:
            st.metric("Entry", format_price(s0.entry))
            st.metric("TP1", format_price(s0.tp1))
        with cb:
            st.metric("SL", format_price(s0.sl))
            st.metric("TP2", format_price(s0.tp2))
        st.metric("RR(TP2)", f"{s0.rr_tp2:.2f}")
        st.metric("Confidence", f"{s0.confidence}")
        st.write("**Reason:**")
        for r in s0.reason + notes:
            st.write(f"• {r}")
    else:
        st.markdown("### ⚪ WAIT")
        for r in s0.reason + notes:
            st.write(f"• {r}")

# BACKTEST
if run_backtest:
    st.header("📈 Backtest Sonuçları (90 Gün)")
    with st.spinner("Backtest çalışıyor..."):
        data_bt = get_4h_data(symbol, 90)
        if data_bt.empty:
            st.error("Backtest için veri alınamadı!")
        else:
            data_bt = compute_indicators(data_bt)
            res = backtest(
                data_bt, min_rr, risk_percent, use_bb_filter,
                adx_trend_thr, adx_range_thr, atr_mult_sl, tp1_r_mult, max_hold_bars
            )
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Total Trades", res["trades"])
            with c2:
                st.metric("Win Rate", f"{res['win_rate']:.1f}%")
            with c3:
                st.metric("Total Return", f"{res['total_return']:.1f}%")
            with c4:
                st.metric("Final Balance", f"${res['final_balance']:,.0f}")

            if "equity_curve" in res and len(res["equity_curve"]) > 1:
                st.subheader("Equity Curve")
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(y=res["equity_curve"], line=dict(width=2), name="Equity"))
                fig_eq.update_layout(height=300, template="plotly_dark", showlegend=False)
                st.plotly_chart(fig_eq, use_container_width=True)

            if res["trades"] == 0:
                st.warning("Hiç işlem oluşmadı. Eşikleri gevşetmeyi deneyin (örn. ADX, min R/R).")

# AÇIKLAMA
with st.expander("ℹ️ Strateji Özeti ve İpuçları"):
    st.write("""
**Yüksek Win Rate Odaklı İyileştirmeler**
- **ADX Rejimi**: ADX≥25 trend; ADX≤18 range. Yanlış rejimde işlem yok.
- **Onaylı Giriş**: S/R temasından sonra bar kapanışı trend yönünde teyit et.
- **ATR Tabanlı SL**: SL = zone sınırı ± ATR*x (varsayılan x=1.0).
- **Kısmi Kâr + Break-even**: TP1'de %50 kapat; kalan BE, basit trailing.
- **Bollinger (opsiyonel)**: Range’de band dokunuşu ek konfluans (false sinyalleri azaltır).
- **Backtest**: Bar-bar SL/TP/BE/Trailing mantığı; max tutma süresi varsayılan 12 bar (~2 gün).

**Not**
- Win rate’i yükseltirken R/R'yi aşırı kısmayın. En az ~1:1 hedefleyin.
- ADX eşiklerini (trend 25, range 18) ve RSI uçlarını (range: 35/65) enstrümana göre optimize edin.
""")