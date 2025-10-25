# app.py
# ─────────────────────────────────────────────────────────────────────────────
# 4H • YFinance • EMA50 Trend • RSI14 • S/R Bölgeleri • TP/SL Çizgileri
# 90 Gün Backtest • Ayarlanabilir R/R • Şifre: "efe"
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

st.set_page_config(page_title="4H Pro TA (EMA50 • RSI14 • S/R)", layout="wide")

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
# YARDIMCI
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
    if "-" not in sym:  # BTC → BTC-USD
        sym = sym + "-USD"
    df = yf.download(sym, period=f"{days}d", interval="4h", progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    # Bazı sembollerde volume None olabilir, doldur:
    df = df.dropna()
    return df

def compute_indicators(df: pd.DataFrame, ema_period: int = 50, rsi_period: int = 14) -> pd.DataFrame:
    if df.empty: return df
    d = df.copy()

    # EMA
    d["EMA"] = d["Close"].ewm(span=ema_period, adjust=False).mean()

    # RSI(14)
    delta = d["Close"].diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs = gain / (loss.replace(0, np.nan))
    d["RSI"] = 100 - (100 / (1 + rs))
    d["RSI"] = d["RSI"].fillna(method="bfill").fillna(50)

    # ATR(14)
    tr1 = d["High"] - d["Low"]
    tr2 = (d["High"] - d["Close"].shift()).abs()
    tr3 = (d["Low"] - d["Close"].shift()).abs()
    d["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    d["ATR"] = d["TR"].rolling(window=14).mean()
    d.drop(columns=["TR"], inplace=True)

    return d

# =============================================================================
# S/R BÖLGELERİ
# =============================================================================
class Zone:
    def __init__(self, low: float, high: float, touches: int, last_touch_ts, kind: str):
        self.low = float(low)
        self.high = float(high)
        self.touches = int(touches)
        self.last_touch_ts = last_touch_ts
        self.kind = kind  # "support" | "resistance"
        self.score = 0
        self.status = "valid"  # fake/valid/broken (etiket yazmayacağız, sadece dahili)

    def __repr__(self):
        return f"Zone({self.kind}, {self.low:.2f}-{self.high:.2f}, touches={self.touches}, score={self.score})"

def _compute_atr_series(d: pd.DataFrame) -> pd.Series:
    tr1 = d["High"] - d["Low"]
    tr2 = (d["High"] - d["Close"].shift()).abs()
    tr3 = (d["Low"] - d["Close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(14).mean()

def _fake_breakout_status(d: pd.DataFrame, zone: Zone) -> str:
    # Basit fake/broken mantığı: 50 bar içinde kısa taşma ve hızlı geri dönüş "fake"
    if len(d) < 20: return "valid"
    data = d.tail(50).copy()
    atr = _compute_atr_series(data).iloc[-1]
    if np.isnan(atr) or atr == 0: atr = (data["High"] - data["Low"]).mean()

    breakouts = 0
    max_dist = 0.0
    reclaim_bars = 0

    for i in range(len(data)):
        c = float(data["Close"].iloc[i])
        if zone.kind == "support":
            if c < zone.low:
                breakouts += 1
                max_dist = max(max_dist, zone.low - c)
                # 2 bar içinde geri kapandı mı?
                for j in range(i + 1, min(i + 3, len(data))):
                    if float(data["Close"].iloc[j]) >= zone.low:
                        reclaim_bars = j - i
                        break
        else:
            if c > zone.high:
                breakouts += 1
                max_dist = max(max_dist, c - zone.high)
                for j in range(i + 1, min(i + 3, len(data))):
                    if float(data["Close"].iloc[j]) <= zone.high:
                        reclaim_bars = j - i
                        break

    cond1 = breakouts < 2
    cond2 = max_dist < max(0.5 * atr, 0.0035 * float(data["Close"].iloc[-1]))
    cond3 = 0 < reclaim_bars <= 2

    if sum([cond1, cond2, cond3]) >= 2:
        return "fake"
    if (breakouts >= 2) or (max_dist >= 0.5 * atr) or (reclaim_bars == 0):
        return "broken"
    return "valid"

def build_zones(d: pd.DataFrame, min_touch_points: int = 3, lookback: int = 120) -> List[Zone]:
    if d.empty: return []
    data = d.tail(lookback).copy()
    cur = float(data["Close"].iloc[-1])
    atr = _compute_atr_series(data).iloc[-1]
    if np.isnan(atr) or atr == 0:
        atr = (data["High"] - data["Low"]).mean()

    # Histogram tabanlı yoğunluk → binWidth ATR bağlı
    bin_width = max(0.25 * atr, cur * 0.0015)
    prices = []
    for i in range(len(data)):
        prices.extend([float(data["High"].iloc[i]), float(data["Low"].iloc[i]), float(data["Close"].iloc[i])])
    if not prices: return []

    pmin, pmax = min(prices), max(prices)
    bins: Dict[Tuple[float, float], int] = {}
    b = pmin
    while b <= pmax:
        be = b + bin_width
        count = sum(1 for p in prices if b <= p <= be)
        if count >= min_touch_points:
            bins[(b, be)] = count
        b = be

    zones: List[Zone] = []
    for (zl, zh), touches in bins.items():
        # Son temas zamanını bul
        last_ts = data.index[-1]
        for i in range(len(data) - 1, -1, -1):
            high_i, low_i, close_i = float(data["High"].iloc[i]), float(data["Low"].iloc[i]), float(data["Close"].iloc[i])
            if (zl <= high_i <= zh) or (zl <= low_i <= zh) or (zl <= close_i <= zh):
                last_ts = data.index[i]
                break
        kind = "support" if zh < cur else "resistance"
        zones.append(Zone(zl, zh, touches, last_ts, kind))

    return zones

def score_zone(d: pd.DataFrame, z: Zone) -> int:
    # 0–100 skor: temaslar, trend (EMA mesafesi), RSI konumu, fake/broken durumu
    cur = float(d["Close"].iloc[-1])
    ema = float(d["EMA"].iloc[-1])
    rsi = float(d["RSI"].iloc[-1])
    atr = float(d["ATR"].iloc[-1])
    if atr <= 0 or np.isnan(atr): atr = (d["High"] - d["Low"]).tail(20).mean()

    score = 0
    score += min(z.touches * 4, 32)

    status = _fake_breakout_status(d, z)
    z.status = status
    if status == "fake":   score += 22
    if status == "valid":  score += 15
    if status == "broken": score += 0

    # EMA yakınlığı
    if z.kind == "support":
        ema_dist = abs(z.high - ema) / atr
    else:
        ema_dist = abs(z.low - ema) / atr
    if ema_dist <= 0.8: score += 18
    elif ema_dist <= 1.5: score += 8

    # RSI tarafı
    if z.kind == "support":
        if rsi < 45: score += 12
        elif 45 <= rsi <= 55: score += 6
    else:
        if rsi > 55: score += 12
        elif 45 <= rsi <= 55: score += 6

    return int(min(score, 100))

def find_zones(d: pd.DataFrame, min_touch_points: int = 3, lookback: int = 120) -> Tuple[List[Zone], List[Zone]]:
    zones = build_zones(d, min_touch_points, lookback)
    if not zones: return [], []
    for z in zones:
        z.score = score_zone(d, z)
    cur = float(d["Close"].iloc[-1])
    supports = [z for z in zones if z.kind == "support"]
    resistances = [z for z in zones if z.kind == "resistance"]
    # Skora göre sırala ve en güçlü 3'er taneyi al
    supports = sorted(supports, key=lambda z: z.score, reverse=True)[:3]
    resistances = sorted(resistances, key=lambda z: z.score, reverse=True)[:3]
    return supports, resistances

# =============================================================================
# SİNYAL MOTORU (EMA50 trend + RSI14 + S/R), TP/SL hesap
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
    trend: str            # bull/bear/neutral
    reason: List[str]
    zone: Optional[Dict[str, Any]]

def make_signal(d: pd.DataFrame, supports: List[Zone], resistances: List[Zone], min_rr: float) -> Tuple[List[Signal], List[str]]:
    notes: List[str] = []
    sigs: List[Signal] = []
    if d.empty or len(d) < 80:
        notes.append("❌ Yetersiz veri.")
        return sigs, notes

    c = float(d["Close"].iloc[-1])
    ema = float(d["EMA"].iloc[-1])
    rsi = float(d["RSI"].iloc[-1])
    atr = float(d["ATR"].iloc[-1])
    if atr <= 0 or np.isnan(atr):
        atr = (d["High"] - d["Low"]).tail(20).mean()

    trend = "bull" if c >= ema else "bear"
    notes.append(f"📈 Trend: {'YÜKSELİŞ' if trend=='bull' else 'DÜŞÜŞ'} (EMA50)")
    notes.append(f"RSI14: {rsi:.1f} | ATR: {atr:.2f}")

    best_s = supports[0] if supports else None
    best_r = resistances[0] if resistances else None

    # LONG sadece trend bull iken ve destek bölgesinde/üstünde
    if best_s and trend == "bull" and best_s.score >= 65:
        entry = min(c, best_s.high)  # entry destek üst sınırı veya mevcut fiyat
        sl = best_s.low - 0.25 * atr
        risk = entry - sl
        if risk > 0:
            tp1 = entry + risk * (min_rr * 0.5)
            tp2 = entry + risk * (min_rr)
            tp1, tp2 = sorted([tp1, tp2])
            rr = (tp2 - entry) / (entry - sl)
            if rr >= min_rr and rsi >= 45:  # çok zayıf momentumda alma
                reason = [
                    "EMA50 üstünde (trend bull)",
                    f"Destek bölgesinde/üstünde (Skor {best_s.score})",
                    f"RR {rr:.2f} ≥ {min_rr}",
                    "RSI≥45 momentum onayı"
                ]
                sigs.append(Signal("BUY", entry, sl, tp1, tp2, rr, best_s.score, trend,
                                   reason, {"kind": "support", "low": best_s.low, "high": best_s.high}))

    # SHORT sadece trend bear iken ve direnç bölgesinde/altında
    if best_r and trend == "bear" and best_r.score >= 65:
        entry = max(c, best_r.low)
        sl = best_r.high + 0.25 * atr
        risk = sl - entry
        if risk > 0:
            tp1 = entry - risk * (min_rr * 0.5)
            tp2 = entry - risk * (min_rr)
            tp1, tp2 = sorted([tp1, tp2], reverse=True)
            rr = (entry - tp2) / (sl - entry)
            if rr >= min_rr and rsi <= 55:
                reason = [
                    "EMA50 altında (trend bear)",
                    f"Direnç bölgesinde/altında (Skor {best_r.score})",
                    f"RR {rr:.2f} ≥ {min_rr}",
                    "RSI≤55 momentum onayı"
                ]
                sigs.append(Signal("SELL", entry, sl, tp1, tp2, rr, best_r.score, trend,
                                   reason, {"kind": "resistance", "low": best_r.low, "high": best_r.high}))

    if not sigs:
        wait_reason = ["Koşullar sinyal için yeterli değil."]
        if not best_s and not best_r:
            wait_reason.append("Yeterli S/R bölgesi yok.")
        elif best_s and best_s.score < 65:
            wait_reason.append(f"Destek skoru düşük: {best_s.score}")
        elif best_r and best_r.score < 65:
            wait_reason.append(f"Direnç skoru düşük: {best_r.score}")
        if abs(c - ema) > 1.2 * atr:
            wait_reason.append("Fiyat EMA50'den aşırı uzak (mean reversion riski).")
        sigs.append(Signal("WAIT", c, None, None, None, 0.0, 0, trend, wait_reason, None))

    return sigs, notes

# =============================================================================
# GRAFİK (TP/SL/ENTRY çizgileri + S/R bantları)
# =============================================================================
def plot_chart(d: pd.DataFrame, supports: List[Zone], resistances: List[Zone], sigs: List[Signal], symbol: str):
    if d.empty:
        return go.Figure()
    view = d.tail(18).copy()  # ~3 gün

    fig = go.Figure(data=[go.Candlestick(
        x=view.index, open=view["Open"], high=view["High"],
        low=view["Low"], close=view["Close"],
        increasing_line_color="#00C805", decreasing_line_color="#FF0000",
        showlegend=False
    )])

    # EMA
    if "EMA" in view.columns:
        fig.add_trace(go.Scatter(x=view.index, y=view["EMA"], mode="lines",
                                 line=dict(width=2, color="orange"),
                                 name="EMA50", showlegend=False))

    # En yakın 2+2 bandı çiz
    cur = float(view["Close"].iloc[-1])
    def mid(z): return (z.low + z.high) / 2.0
    nearest_s = sorted(supports, key=lambda z: abs(mid(z) - cur))[:2]
    nearest_r = sorted(resistances, key=lambda z: abs(mid(z) - cur))[:2]

    for i, z in enumerate(nearest_s):
        fig.add_hrect(y0=z.low, y1=z.high,
                      fillcolor="rgba(0,255,0,0.14)",
                      line=dict(width=1, color="#00FF00"), layer="below")
        fig.add_annotation(x=view.index[-1], y=mid(z), text=f"S{i+1}", showarrow=False,
                           xanchor="left", yanchor="middle",
                           font=dict(size=10, color="#00FF00"),
                           bgcolor="rgba(0,0,0,0.5)")

    for i, z in enumerate(nearest_r):
        fig.add_hrect(y0=z.low, y1=z.high,
                      fillcolor="rgba(255,0,0,0.14)",
                      line=dict(width=1, color="#FF0000"), layer="below")
        fig.add_annotation(x=view.index[-1], y=mid(z), text=f"R{i+1}", showarrow=False,
                           xanchor="left", yanchor="middle",
                           font=dict(size=10, color="#FF0000"),
                           bgcolor="rgba(0,0,0,0.5)")

    # Mevcut fiyat çizgisi
    cp = float(view["Close"].iloc[-1])
    fig.add_hline(y=cp, line_dash="dot", line_color="yellow", line_width=1,
                  annotation_text=f"{format_price(cp)}", annotation_position="left",
                  annotation_font_size=10, annotation_font_color="yellow")

    # TP/SL/Entry çizgileri (sinyal varsa)
    if sigs and sigs[0].typ in ("BUY", "SELL"):
        s = sigs[0]
        # entry
        fig.add_hline(y=s.entry, line_dash="solid", line_color="#B0E0E6", line_width=2,
                      annotation_text="Entry", annotation_position="right",
                      annotation_font_color="#B0E0E6")
        # sl
        fig.add_hline(y=s.sl, line_dash="dash", line_color="#FF6B6B", line_width=2,
                      annotation_text="SL", annotation_position="right",
                      annotation_font_color="#FF6B6B")
        # tp1
        fig.add_hline(y=s.tp1, line_dash="dash", line_color="#A1FF69", line_width=2,
                      annotation_text="TP1", annotation_position="right",
                      annotation_font_color="#A1FF69")
        # tp2
        fig.add_hline(y=s.tp2, line_dash="dash", line_color="#2ECC71", line_width=2,
                      annotation_text="TP2", annotation_position="right",
                      annotation_font_color="#2ECC71")
        # işaretleyici
        mcolor = "#00FF00" if s.typ == "BUY" else "#FF0000"
        msym = "triangle-up" if s.typ == "BUY" else "triangle-down"
        fig.add_trace(go.Scatter(
            x=[view.index[-1]], y=[cp], mode="markers",
            marker=dict(symbol=msym, size=12, color=mcolor, line=dict(width=2, color="white")),
            showlegend=False
        ))

    fig.update_layout(
        title=f"{symbol} • 4H (Son 3 Gün)",
        height=520,
        xaxis_rangeslider_visible=False,
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="white"),
        xaxis=dict(gridcolor="#444"),
        yaxis=dict(gridcolor="#444"),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

# =============================================================================
# BACKTEST 90 GÜN
# =============================================================================
@dataclass
class Trade:
    open_time: pd.Timestamp
    side: str              # LONG/SHORT
    entry: float
    sl: float
    tp1: float
    tp2: float
    size: float
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str
    r_multiple: float
    pnl: float

def backtest_90d(df: pd.DataFrame, min_rr: float, risk_percent: float, fee: float, slip: float) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty or len(df) < 180:
        return {"trades":0,"win_rate":0,"profit_factor":0,"expectancy_r":0,"max_drawdown_pct":0,"sharpe":0,"final_balance":10000,"total_return_pct":0}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    balance = 10000.0
    trades: List[Trade] = []
    equity = [balance]
    dd_series = [0.0]

    min_lookback = 120
    zone_recalc = 10
    cache = {}
    start_i = min_lookback
    for i in range(start_i, len(df)-1):
        # her 10 barda bir zone hesapla
        key = i // zone_recalc
        if key in cache:
            supports, resistances = cache[key]
        else:
            supports, resistances = find_zones(df.iloc[:i+1], min_touch_points=3, lookback=120)
            cache[key] = (supports, resistances)

        dslice = df.iloc[:i+1]
        sigs, _ = make_signal(dslice, supports, resistances, min_rr)
        if not sigs or sigs[0].typ == "WAIT":
            equity.append(balance)
            cur_eq = equity[-1]; peak = max(equity)
            dd_series.append(((cur_eq - peak) / peak) * 100 if peak>0 else 0)
            continue

        s = sigs[0]
        side = "LONG" if s.typ == "BUY" else "SHORT"
        next_open = float(df["Open"].iloc[i+1])

        # maliyet uygula
        entry = next_open * (1 + (fee + slip) if side=="LONG" else 1 - (fee + slip))
        sl = s.sl
        tp1 = s.tp1
        tp2 = s.tp2

        # pozisyon büyüklüğü (R$ = balance * risk%)
        risk_cap = balance * (risk_percent / 100.0)
        risk_per_unit = abs(entry - sl)
        if risk_per_unit <= 0:
            equity.append(balance); dd_series.append(((equity[-1]-max(equity))/max(equity))*100 if max(equity)>0 else 0); continue
        qty = risk_cap / risk_per_unit
        if qty <= 0:
            equity.append(balance); dd_series.append(((equity[-1]-max(equity))/max(equity))*100 if max(equity)>0 else 0); continue

        # ileri barlarda TP/SL kontrolü (max 60 bar ~ 10 gün)
        open_idx = i+1
        exit_idx = None
        exit_price = None
        exit_reason = None
        pnl = 0.0

        for j in range(open_idx, min(open_idx+60, len(df))):
            hi, lo, close_j = float(df["High"].iloc[j]), float(df["Low"].iloc[j]), float(df["Close"].iloc[j])
            if side == "LONG":
                if lo <= tp2 <= hi:
                    exit_idx, exit_price, exit_reason = j, tp2, "TP2"
                    pnl = (tp2 - entry) * qty
                    break
                if lo <= sl <= hi:
                    exit_idx, exit_price, exit_reason = j, sl, "SL"
                    pnl = (sl - entry) * qty
                    break
            else:
                if lo <= tp2 <= hi:
                    exit_idx, exit_price, exit_reason = j, tp2, "TP2"
                    pnl = (entry - tp2) * qty
                    break
                if lo <= sl <= hi:
                    exit_idx, exit_price, exit_reason = j, sl, "SL"
                    pnl = (entry - sl) * qty
                    break

        if exit_idx is None:
            exit_idx = min(open_idx+60, len(df)-1)
            lastc = float(df["Close"].iloc[exit_idx])
            exit_price = lastc
            exit_reason = "Time"
            pnl = (lastc - entry) * qty if side=="LONG" else (entry - lastc) * qty

        # çıkış maliyeti (fee)
        exit_price_costed = exit_price * (1 - fee if side=="LONG" else 1 + fee)
        # brüt PnL yerine fee sonrası approx PnL:
        pnl_after = pnl - (entry * qty * fee) - (exit_price_costed * qty * fee)

        # r-multiple
        R = risk_per_unit * qty
        r_mult = pnl_after / R if R > 0 else 0.0

        balance += pnl_after
        trades.append(Trade(
            open_time=df.index[open_idx], side=side, entry=entry, sl=sl, tp1=tp1, tp2=tp2, size=qty,
            exit_time=df.index[exit_idx], exit_price=exit_price_costed, exit_reason=exit_reason,
            r_multiple=r_mult, pnl=pnl_after
        ))

        equity.append(balance)
        cur_eq = equity[-1]; peak = max(equity)
        dd_series.append(((cur_eq - peak) / peak) * 100 if peak>0 else 0)

    # metrikler
    eq = pd.Series(equity)
    ret = eq.pct_change().replace([np.inf,-np.inf],0).fillna(0)
    wins = [t for t in trades if t.r_multiple > 0]
    losses = [t for t in trades if t.r_multiple <= 0]
    win_rate = (len(wins)/len(trades)*100) if trades else 0.0
    pf = (sum(t.pnl for t in wins) / max(1e-9, abs(sum(t.pnl for t in losses)))) if losses else float("inf")
    avg_win_r = np.mean([t.r_multiple for t in wins]) if wins else 0.0
    avg_loss_r = abs(np.mean([t.r_multiple for t in losses])) if losses else 0.0
    expectancy = (win_rate/100)*avg_win_r - (1-win_rate/100)*avg_loss_r
    max_dd = min(dd_series) if dd_series else 0.0
    sharpe = (ret.mean()/(ret.std()+1e-12))*np.sqrt(365*6) if ret.std()>0 else 0.0
    total_return = ((balance - 10000.0)/10000.0)*100

    report = {
        "trades": len(trades),
        "win_rate": win_rate,
        "profit_factor": pf,
        "expectancy_r": expectancy,
        "max_drawdown_pct": max_dd,
        "sharpe": sharpe,
        "final_balance": balance,
        "total_return_pct": total_return
    }

    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    eq_df = pd.DataFrame({"time": df.index[:len(equity)], "equity": equity})
    dd_df = pd.DataFrame({"time": df.index[:len(dd_series)], "drawdown": dd_series})
    return report, trades_df, eq_df, dd_df

# =============================================================================
# UI
# =============================================================================
st.title("🎯 4H Pro TA — EMA50 • RSI14 • S/R (Backtest’li)")

with st.sidebar:
    st.header("⚙️ Ayarlar")
    symbol = st.text_input("Kripto Sembolü", "BTC-USD", help="Örn: BTC-USD, ETH-USD, THETA-USD")
    ema_period = 50  # strateji gereği sabit
    rsi_period = 14  # talebe uygun
    min_touch_points = st.slider("Min Temas (S/R)", 2, 5, 3)
    rr_req = st.slider("Min R/R", 1.0, 3.0, 1.8, 0.1)
    lookback_bars = st.slider("Analiz Lookback (bar)", 80, 200, 120)

    st.divider()
    st.subheader("🧪 Backtest (90 Gün)")
    risk_percent = st.slider("Risk %", 0.1, 5.0, 1.0, 0.1)
    fee = st.number_input("Fee (taker, %)", 0.00, 1.00, 0.10, 0.01) / 100.0
    slip = st.number_input("Slippage (%)", 0.00, 0.50, 0.02, 0.01) / 100.0
    run_bt = st.button("Backtest Çalıştır (90d)", use_container_width=True, type="primary")

# VERİ
with st.spinner(f"⏳ Veri yükleniyor: {symbol} (30g / 4H)"):
    d_30 = get_4h_data(symbol, days=30)
if d_30.empty:
    st.error("❌ Veri alınamadı.")
    st.stop()

d_30 = compute_indicators(d_30, ema_period=ema_period, rsi_period=rsi_period)
supports, resistances = find_zones(d_30, min_touch_points=min_touch_points, lookback=lookback_bars)
signals, notes = make_signal(d_30, supports, resistances, rr_req)

# LAYOUT
c1, c2 = st.columns([3, 1])

with c1:
    fig = plot_chart(d_30, supports, resistances, signals, symbol)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("📊 Sinyal")
    s = signals[0]
    if s.typ in ("BUY", "SELL"):
        ic = "🟢" if s.typ == "BUY" else "🔴"
        st.markdown(f"### {ic} {s.typ}")
        a, b = st.columns(2)
        with a:
            st.metric("Giriş", format_price(s.entry))
            st.metric("TP1", format_price(s.tp1))
        with b:
            st.metric("SL", format_price(s.sl))
            st.metric("TP2", format_price(s.tp2))
        st.metric("R/R", f"{s.rr:.2f}")
        st.metric("Güven (Skor)", f"{s.confidence}")
        st.caption("Gerekçe:")
        for r in s.reason:
            st.write("• " + r)
    else:
        st.markdown("### ⚪ BEKLE")
        for r in s.reason:
            st.write("• " + r)

    st.divider()
    st.subheader("📈 Trend / Göstergeler")
    trend_icon = "🟢" if s.trend == "bull" else "🔴"
    st.metric("Trend (EMA50)", "YÜKSELİŞ" if s.trend == "bull" else "DÜŞÜŞ")
    last_rsi = float(d_30["RSI"].iloc[-1])
    st.metric("RSI(14)", f"{last_rsi:.1f}")

st.subheader("🎯 Yakın S/R Bantları")
colS, colR = st.columns(2)
with colS:
    st.write("**Destek**")
    for i, z in enumerate(supports):
        st.write(f"S{i+1}: {format_price(z.low)} – {format_price(z.high)}  |  Skor: {z.score}  |  Temas: {z.touches}")
with colR:
    st.write("**Direnç**")
    for i, z in enumerate(resistances):
        st.write(f"R{i+1}: {format_price(z.low)} – {format_price(z.high)}  |  Skor: {z.score}  |  Temas: {z.touches}")

# BACKTEST
if run_bt:
    st.header("📊 Backtest Sonuçları — 90 Gün (4H)")
    with st.spinner("Backtest çalışıyor..."):
        d_90 = get_4h_data(symbol, days=90)
        if d_90.empty:
            st.error("Backtest için veri alınamadı.")
        else:
            d_90 = compute_indicators(d_90, ema_period=ema_period, rsi_period=rsi_period)
            report, tdf, eqdf, dddf = backtest_90d(d_90, min_rr=rr_req, risk_percent=risk_percent, fee=fee, slip=slip)

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("İşlem", report["trades"])
                st.metric("Win %", f"{report['win_rate']:.1f}%")
            with k2:
                st.metric("Profit Factor", f"{report['profit_factor']:.2f}")
                st.metric("Expectancy (R)", f"{report['expectancy_r']:.2f}")
            with k3:
                st.metric("Max DD", f"{report['max_drawdown_pct']:.1f}%")
                st.metric("Sharpe", f"{report['sharpe']:.2f}")
            with k4:
                st.metric("Bakiye", f"${report['final_balance']:,.0f}")
                st.metric("Getiri", f"{report['total_return_pct']:.1f}%")

            c3, c4 = st.columns(2)
            with c3:
                if not eqdf.empty:
                    st.subheader("Equity Curve")
                    f1 = go.Figure()
                    f1.add_trace(go.Scatter(x=eqdf["time"], y=eqdf["equity"],
                                            line=dict(color="#00FF00", width=2),
                                            fill="tozeroy", fillcolor="rgba(0,255,0,0.1)"))
                    f1.update_layout(height=300, plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
                                     font=dict(color="white"), margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
                    st.plotly_chart(f1, use_container_width=True)
            with c4:
                if not dddf.empty:
                    st.subheader("Drawdown")
                    f2 = go.Figure()
                    f2.add_trace(go.Scatter(x=dddf["time"], y=dddf["drawdown"],
                                            line=dict(color="#FF4444", width=2),
                                            fill="tozeroy", fillcolor="rgba(255,0,0,0.3)"))
                    f2.update_layout(height=300, plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
                                     font=dict(color="white"), margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
                    st.plotly_chart(f2, use_container_width=True)

            if not tdf.empty:
                st.subheader("İşlem Listesi")
                disp = tdf.copy()
                for col in ["entry","sl","tp1","tp2","exit_price","pnl"]:
                    disp[col] = disp[col].astype(float)
                disp["entry"] = disp["entry"].apply(format_price)
                disp["sl"] = disp["sl"].apply(format_price)
                disp["tp1"] = disp["tp1"].apply(format_price)
                disp["tp2"] = disp["tp2"].apply(format_price)
                disp["exit_price"] = disp["exit_price"].apply(format_price)
                disp["r_multiple"] = disp["r_multiple"].round(2)
                disp["pnl"] = disp["pnl"].round(2)
                st.dataframe(
                    disp[["open_time","side","entry","sl","tp1","tp2","exit_time","exit_price","exit_reason","r_multiple","pnl"]],
                    use_container_width=True
                )