# app.py
# ─────────────────────────────────────────────────────────────────────────────
# 4H • YFinance • EMA50 Trend • RSI14 • S/R Bölgeleri • Multi-Timeframe • Advanced Risk
# Optimized for Higher Win Rate - ENHANCED VERSION
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import time
from datetime import datetime, timedelta

st.set_page_config(page_title="4H Pro TA (Enhanced Win Rate)", layout="wide")

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
# YARDIMCI FONKSİYONLAR - GELİŞMİŞ
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

@st.cache_data(ttl=300)  # 5 dakika cache
def get_4h_data(symbol: str, days: int) -> pd.DataFrame:
    sym = symbol.upper().strip()
    if "-" not in sym:
        sym = sym + "-USD"
    
    try:
        df = yf.download(sym, period=f"{days}d", interval="4h", progress=False)
        if df is None or df.empty:
            st.warning(f"⚠️ {sym} için veri bulunamadı")
            return pd.DataFrame()
        return df.dropna()
    except Exception as e:
        st.error(f"❌ Veri indirme hatası: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_multitimeframe_data(symbol: str, days: int) -> Dict[str, pd.DataFrame]:
    """1H, 4H, 1D zaman dilimlerini al"""
    sym = symbol.upper().strip()
    if "-" not in sym:
        sym = sym + "-USD"
    
    timeframes = {
        '1H': f'{days*6}d',  # 6x daha fazla gün
        '4H': f'{days}d',
        '1D': f'{max(30, days)}d'   # Minimum 30 gün
    }
    
    data = {}
    for tf, period in timeframes.items():
        try:
            interval = tf.lower()
            data[tf] = yf.download(sym, period=period, interval=interval, progress=False)
            if data[tf] is not None and not data[tf].empty:
                data[tf] = compute_indicators(data[tf])
        except Exception as e:
            st.warning(f"{tf} zaman dilimi için veri alınamadı: {str(e)}")
    
    return data

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    d = df.copy()

    # EMA
    d["EMA"] = d["Close"].ewm(span=50, adjust=False).mean()

    # RSI - Gelişmiş
    delta = d["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, 0.001)
    d["RSI"] = 100 - (100 / (1 + rs))
    d["RSI"] = d["RSI"].fillna(50)

    # Volume EMA
    d["Volume_EMA"] = d["Volume"].ewm(span=20).mean()

    return d

def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Veri kalitesi kontrolü"""
    if df.empty:
        return {"valid": False, "issues": ["Boş veri"], "score": 0}
    
    issues = []
    score = 100
    
    # Eksik veri kontrolü
    missing_data = df.isnull().sum().sum()
    if missing_data > 0:
        issues.append(f"{missing_data} eksik veri noktası")
        score -= 20
    
    # Anomali kontrolü
    price_changes = df['Close'].pct_change().abs()
    extreme_moves = (price_changes > 0.1).sum()  # %10'dan fazla hareket
    if extreme_moves > len(df) * 0.05:  # %5'ten fazla aşırı hareket
        issues.append("Çok sayıda aşırı fiyat hareketi")
        score -= 15
    
    # Volume anomalileri
    if 'Volume' in df.columns:
        volume_spikes = (df['Volume'] > df['Volume'].rolling(20).mean() * 3).sum()
        if volume_spikes > len(df) * 0.1:
            issues.append("Şüpheli volume spike'ları")
            score -= 10
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "score": max(0, score)
    }

# =============================================================================
# S/R BÖLGELERİ - GELİŞMİŞ
# =============================================================================
class Zone:
    def __init__(self, low: float, high: float, touches: int, kind: str):
        self.low = float(low)
        self.high = float(high)
        self.touches = int(touches)
        self.kind = kind
        self.score = 0

def cluster_levels(levels: List[float], threshold: float = 0.02) -> List[Zone]:
    """Benzer seviyeleri cluster'la"""
    if not levels:
        return []
    
    levels = sorted(levels)
    clusters = []
    current_cluster = [levels[0]]
    
    for price in levels[1:]:
        if price <= current_cluster[-1] * (1 + threshold):
            current_cluster.append(price)
        else:
            clusters.append(current_cluster)
            current_cluster = [price]
    
    clusters.append(current_cluster)
    
    zones = []
    for cluster in clusters:
        if len(cluster) >= 2:  # En az 2 temas
            low = min(cluster)
            high = max(cluster)
            zones.append(Zone(low, high, len(cluster), "support" if cluster[-1] < np.mean(levels) else "resistance"))
    
    return zones

def improved_find_zones(df: pd.DataFrame, window: int = 20) -> Tuple[List[Zone], List[Zone]]:
    """Gelişmiş destek/direnç tespiti"""
    if df.empty or len(df) < window * 2:
        return [], []
    
    # Yerel maksimum/minimum noktaları
    resistance_points = []
    support_points = []
    current_price = float(df["Close"].iloc[-1])
    
    for i in range(window, len(df)-window):
        high = float(df["High"].iloc[i])
        low = float(df["Low"].iloc[i])
        
        # Yerel maksimum (direnç)
        if high == df["High"].iloc[i-window:i+window].max():
            resistance_points.append(high)
        
        # Yerel minimum (destek)
        if low == df["Low"].iloc[i-window:i+window].min():
            support_points.append(low)
    
    # Cluster benzer seviyeler
    support_zones = cluster_levels(support_points)
    resistance_zones = cluster_levels(resistance_points)
    
    # Mevcut fiyata göre filtrele ve skorla
    supports = [z for z in support_zones if z.high < current_price]
    resistances = [z for z in resistance_zones if z.low > current_price]
    
    # Skorlama
    for zone in supports + resistances:
        zone.score = min(zone.touches * 20 + 20, 80)  # Temas bazlı skor
    
    # En iyi 3'er tane
    supports = sorted(supports, key=lambda z: z.score, reverse=True)[:3]
    resistances = sorted(resistances, key=lambda z: z.score, reverse=True)[:3]
    
    return supports, resistances

# =============================================================================
# RİSK YÖNETİMİ - GELİŞMİŞ
# =============================================================================
def calculate_position_size(balance: float, risk_percent: float, entry: float, stop_loss: float) -> float:
    """Gelişmiş position sizing"""
    risk_amount = balance * (risk_percent / 100)
    risk_per_unit = abs(entry - stop_loss)
    
    if risk_per_unit <= 0:
        return 0
    
    position_size = risk_amount / risk_per_unit
    # Maksimum %10 pozisyon sınırı
    max_position = balance * 0.1 / entry if entry > 0 else 0
    
    return min(position_size, max_position)

def calculate_max_drawdown(pnl_percents: List[float]) -> float:
    """Maksimum drawdown hesapla"""
    if not pnl_percents:
        return 0
    
    equity = 10000
    peak = equity
    max_dd = 0
    
    for pnl in pnl_percents:
        equity *= (1 + pnl/100)
        if equity > peak:
            peak = equity
        drawdown = (peak - equity) / peak * 100
        if drawdown > max_dd:
            max_dd = drawdown
    
    return max_dd

def calculate_sharpe_ratio(pnl_percents: List[float], risk_free_rate: float = 0.02) -> float:
    """Sharpe oranı hesapla"""
    if not pnl_percents or len(pnl_percents) < 2:
        return 0
    
    returns = np.array(pnl_percents)
    excess_returns = returns - risk_free_rate/252  # Günlük risk-free rate
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0

# =============================================================================
# SİNYAL MOTORU - GELİŞMİŞ MULTI-TIMEFRAME
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
    timeframe: str = "4H"

def generate_bullish_signal(data: pd.DataFrame, supports: List[Zone], min_rr: float) -> Optional[Signal]:
    """Bullish sinyal üret"""
    if not data.empty and supports:
        current_price = float(data["Close"].iloc[-1])
        rsi = float(data["RSI"].iloc[-1])
        best_support = supports[0]
        
        if current_price <= best_support.high * 1.01 and rsi < 70:
            sl = best_support.low * 0.99
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
                    return Signal("BUY", current_price, sl, tp1, tp2, rr, 
                                best_support.score, "bull", reason)
    return None

def generate_bearish_signal(data: pd.DataFrame, resistances: List[Zone], min_rr: float) -> Optional[Signal]:
    """Bearish sinyal üret"""
    if not data.empty and resistances:
        current_price = float(data["Close"].iloc[-1])
        rsi = float(data["RSI"].iloc[-1])
        best_resistance = resistances[0]
        
        if current_price >= best_resistance.low * 0.99 and rsi > 30:
            sl = best_resistance.high * 1.01
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
                    return Signal("SELL", current_price, sl, tp1, tp2, rr,
                                best_resistance.score, "bear", reason)
    return None

def advanced_signal_generation(multitimeframe_data: Dict[str, pd.DataFrame], 
                              min_rr: float = 1.5) -> Signal:
    """Çoklu zaman dilimi sinyali"""
    
    if '4H' not in multitimeframe_data or multitimeframe_data['4H'].empty:
        return Signal("WAIT", 0, None, None, None, 0, 0, "neutral", ["Veri yok"])
    
    # Her zaman dilimi için trend analizi
    trends = {}
    for tf_name, data in multitimeframe_data.items():
        if not data.empty:
            current_price = data['Close'].iloc[-1]
            ema = data['Close'].ewm(span=50).mean().iloc[-1]
            trends[tf_name] = 'bull' if current_price > ema else 'bear'
    
    # Trend uyumu kontrolü
    bullish_count = sum(1 for trend in trends.values() if trend == 'bull')
    bearish_count = len(trends) - bullish_count
    
    # 4H verisi ile S/R bölgeleri
    data_4h = multitimeframe_data['4H']
    supports, resistances = improved_find_zones(data_4h)
    current_price = float(data_4h["Close"].iloc[-1])
    
    # Çoğunluk trendine göre sinyal
    signal = None
    if bullish_count >= 2:
        signal = generate_bullish_signal(data_4h, supports, min_rr)
        if signal:
            signal.reason.append(f"Multi-TF Trend: {bullish_count}/{len(trends)} Bullish")
    
    elif bearish_count >= 2:
        signal = generate_bearish_signal(data_4h, resistances, min_rr)
        if signal:
            signal.reason.append(f"Multi-TF Trend: {bearish_count}/{len(trends)} Bearish")
    
    # Range market için fallback
    if not signal and abs(current_price - float(data_4h["EMA"].iloc[-1])) / current_price < 0.02:
        rsi = float(data_4h["RSI"].iloc[-1])
        
        if supports and current_price <= supports[0].high * 1.005 and rsi < 40:
            sl = supports[0].low * 0.99
            risk = current_price - sl
            tp1 = float(data_4h["EMA"].iloc[-1])
            tp2 = resistances[0].low if resistances else current_price * 1.03
            rr = (tp2 - current_price) / risk
            if rr >= min_rr:
                reason = ["Range - Destekten bounce", f"RSI oversold: {rsi:.1f}", "Multi-TF: Mixed"]
                signal = Signal("BUY", current_price, sl, tp1, tp2, rr, 70, "range", reason)
        
        elif resistances and current_price >= resistances[0].low * 0.995 and rsi > 60:
            sl = resistances[0].high * 1.01
            risk = sl - current_price
            tp1 = float(data_4h["EMA"].iloc[-1])
            tp2 = supports[0].high if supports else current_price * 0.97
            rr = (current_price - tp2) / risk
            if rr >= min_rr:
                reason = ["Range - Dirençten reject", f"RSI overbought: {rsi:.1f}", "Multi-TF: Mixed"]
                signal = Signal("SELL", current_price, sl, tp1, tp2, rr, 70, "range", reason)
    
    if not signal:
        wait_reason = ["Multi-timeframe uyumsuzluğu veya yetersiz sinyal kalitesi"]
        if not supports and not resistances:
            wait_reason.append("S/R bölgesi yok")
        signal = Signal("WAIT", current_price, None, None, None, 0, 0, "mixed", wait_reason)
    
    return signal

# =============================================================================
# BACKTEST - GELİŞMİŞ METRİKLER
# =============================================================================
@dataclass
class Trade:
    entry: float
    exit: float
    side: str
    pnl: float
    pnl_percent: float
    timestamp: datetime

def calculate_performance_metrics(trades: List[Trade]) -> Dict[str, float]:
    """Gelişmiş performans metrikleri"""
    if not trades:
        return {}
    
    pnls = [t.pnl for t in trades]
    pnl_percents = [t.pnl_percent for t in trades]
    
    winning_trades = [p for p in pnls if p > 0]
    losing_trades = [p for p in pnls if p < 0]
    
    return {
        'win_rate': len(winning_trades) / len(pnls) * 100,
        'avg_win': np.mean(winning_trades) if winning_trades else 0,
        'avg_loss': np.mean(losing_trades) if losing_trades else 0,
        'profit_factor': abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float('inf'),
        'max_drawdown': calculate_max_drawdown(pnl_percents),
        'sharpe_ratio': calculate_sharpe_ratio(pnl_percents),
        'total_return': sum(pnls),
        'total_return_percent': (sum(pnls) / 10000) * 100
    }

def advanced_backtest(df: pd.DataFrame, 
                     min_rr: float = 1.5, 
                     risk_percent: float = 1.0) -> Dict[str, Any]:
    if df.empty or len(df) < 100:
        return {"trades": 0, "win_rate": 0, "total_return": 0, "final_balance": 10000}
    
    balance = 10000.0
    trades = []
    equity = [balance]
    dates = [df.index[0] if hasattr(df.index[0], 'strftime') else datetime.now()]
    
    for i in range(50, len(df) - 5):  # 5 bar forward testing
        try:
            data_slice = df.iloc[:i+1]
            supports, resistances = improved_find_zones(data_slice)
            
            # Multi-timeframe simülasyonu (sadece 4H kullan)
            mtf_data = {'4H': data_slice}
            signal = advanced_signal_generation(mtf_data, min_rr)
            
            if signal.typ in ["BUY", "SELL"]:
                entry_price = float(df["Open"].iloc[i+1])  # Sonraki bar açılışı
                exit_price = float(df["Close"].iloc[i+5])  # 5 bar sonra kapat
                
                # Gelişmiş position sizing
                stop_loss = signal.sl if signal.sl else entry_price * 0.98
                position_size = calculate_position_size(balance, risk_percent, entry_price, stop_loss)
                
                if signal.typ == "BUY":
                    pnl = (exit_price - entry_price) * position_size
                else:  # SELL
                    pnl = (entry_price - exit_price) * position_size
                
                balance += pnl
                pnl_percent = (pnl / (position_size * entry_price)) * 100 if position_size > 0 else 0
                
                trades.append(Trade(
                    entry=entry_price,
                    exit=exit_price,
                    side=signal.typ,
                    pnl=pnl,
                    pnl_percent=pnl_percent,
                    timestamp=df.index[i+1] if i+1 < len(df.index) else datetime.now()
                ))
            
            equity.append(balance)
            dates.append(df.index[i] if i < len(df.index) else datetime.now())
            
        except Exception as e:
            continue
    
    # Metrics
    total_trades = len(trades)
    if total_trades == 0:
        return {
            "trades": 0, 
            "win_rate": 0, 
            "total_return": 0, 
            "final_balance": 10000,
            "performance_metrics": {},
            "equity_curve": equity,
            "dates": dates
        }
    
    performance_metrics = calculate_performance_metrics(trades)
    
    return {
        "trades": total_trades,
        "win_rate": performance_metrics.get('win_rate', 0),
        "total_return": performance_metrics.get('total_return_percent', 0),
        "final_balance": balance,
        "performance_metrics": performance_metrics,
        "equity_curve": equity,
        "dates": dates,
        "trades_list": trades
    }

# =============================================================================
# ALARM SİSTEMİ
# =============================================================================
def setup_price_alerts(symbol: str, levels: List[float]):
    """Fiyat alarmları kur"""
    st.sidebar.subheader("🔔 Fiyat Alarmları")
    
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    
    # Yeni alarm ekle
    col1, col2 = st.sidebar.columns([2, 1])
    with col1:
        new_alert = st.number_input("Alarm Seviyesi", value=0.0, format="%.4f")
    with col2:
        if st.button("Ekle", key="add_alert"):
            if new_alert > 0:
                st.session_state.alerts.append(new_alert)
                st.rerun()
    
    # Mevcut alarmlar
    for i, level in enumerate(st.session_state.alerts[:5]):  # Max 5 alarm
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            st.write(f"${level:,.4f}")
        with col2:
            if st.button("🗑️", key=f"delete_{i}"):
                st.session_state.alerts.pop(i)
                st.rerun()

# =============================================================================
# ARAYÜZ - GELİŞMİŞ
# =============================================================================
st.title("🎯 4H Pro TA - Enhanced Multi-Timeframe Analysis")

with st.sidebar:
    st.header("⚙️ Gelişmiş Ayarlar")
    symbol = st.text_input("Kripto Sembolü", "BTC-USD")
    
    st.subheader("Risk Parametreleri")
    min_rr = st.slider("Min Risk/Reward", 1.0, 3.0, 1.5, 0.1)
    risk_percent = st.slider("Risk %", 0.5, 5.0, 1.0, 0.1)
    
    st.subheader("Analiz Seçenekleri")
    use_multitimeframe = st.checkbox("Çoklu Zaman Dilimi Analizi", value=True)
    show_advanced_metrics = st.checkbox("Gelişmiş Metrikler", value=True)
    
    st.divider()
    st.subheader("Backtest")
    run_backtest = st.button("🚀 Gelişmiş Backtest Çalıştır", type="primary")
    
    # Alarm sistemi
    setup_price_alerts(symbol, [])

# Ana veri yükleme
with st.spinner("Veri yükleniyor ve analiz ediliyor..."):
    if use_multitimeframe:
        multitimeframe_data = get_multitimeframe_data(symbol, 30)
        data_4h = multitimeframe_data.get('4H', pd.DataFrame())
    else:
        data_4h = get_4h_data(symbol, 30)
        multitimeframe_data = {'4H': data_4h}
    
    if not data_4h.empty:
        data_4h = compute_indicators(data_4h)
        supports, resistances = improved_find_zones(data_4h)
        
        # Veri kalitesi kontrolü
        quality_check = validate_data_quality(data_4h)
        
        if use_multitimeframe:
            signal = advanced_signal_generation(multitimeframe_data, min_rr)
        else:
            signals, notes = generate_high_winrate_signals(data_4h, supports, resistances, min_rr)
            signal = signals[0] if signals else Signal("WAIT", 0, None, None, None, 0, 0, "neutral", ["Sinyal yok"])
    else:
        st.error("❌ Veri alınamadı! Sembolü kontrol edin.")
        st.stop()

# Veri Kalitesi Göstergesi
if not data_4h.empty:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Veri Kalitesi", f"{quality_check['score']}/100")
    with col2:
        status = "✅ İyi" if quality_check['valid'] else "⚠️ Sorunlu"
        st.metric("Durum", status)
    with col3:
        if not quality_check['valid']:
            st.warning(f"Sorunlar: {', '.join(quality_check['issues'])}")

# Ana görünüm
col1, col2 = st.columns([2, 1])

with col1:
    if not data_4h.empty:
        # Gelişmiş grafik
        fig = go.Figure()
        
        # Son 24 bar
        view_data = data_4h.tail(24)
        
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
        
        # Support/Resistance zones
        for i, zone in enumerate(supports[:2]):
            fig.add_hrect(y0=zone.low, y1=zone.high, 
                         fillcolor="green", opacity=0.2, line_width=0,
                         annotation_text=f"S{i+1} (Skor: {zone.score})")
            
        for i, zone in enumerate(resistances[:2]):
            fig.add_hrect(y0=zone.low, y1=zone.high, 
                         fillcolor="red", opacity=0.2, line_width=0,
                         annotation_text=f"R{i+1} (Skor: {zone.score})")
        
        # Mevcut fiyat çizgisi
        current_price = float(data_4h["Close"].iloc[-1])
        fig.add_hline(y=current_price, line_dash="dot", line_color="white", 
                     annotation_text="Mevcut Fiyat")
        
        fig.update_layout(
            title=f"{symbol} - 4H Chart (Gelişmiş S/R Bölgeleri)",
            height=500,
            template="plotly_dark",
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

        # Multi-Timeframe Trend Göstergesi
        if use_multitimeframe and len(multitimeframe_data) > 1:
            st.subheader("📊 Multi-Timeframe Trend Analizi")
            trend_cols = st.columns(len(multitimeframe_data))
            
            for idx, (tf_name, data) in enumerate(multitimeframe_data.items()):
                with trend_cols[idx]:
                    if not data.empty:
                        current_price = data['Close'].iloc[-1]
                        ema = data['Close'].ewm(span=50).mean().iloc[-1]
                        trend = "YÜKSELİŞ 🟢" if current_price > ema else "DÜŞÜŞ 🔴"
                        
                        st.metric(
                            label=f"{tf_name} Trend",
                            value=trend,
                            delta=f"{((current_price - ema) / ema * 100):.2f}%"
                        )

with col2:
    st.subheader("🎯 Sinyal Analizi")
    
    if signal.typ in ["BUY", "SELL"]:
        color = "🟢" if signal.typ == "BUY" else "🔴"
        st.markdown(f"### {color} {signal.typ}")
        
        # Temel bilgiler
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Entry", format_price(signal.entry))
            st.metric("TP1", format_price(signal.tp1))
        with col_b:
            st.metric("SL", format_price(signal.sl))
            st.metric("TP2", format_price(signal.tp2))
        
        st.metric("Risk/Reward", f"{signal.rr:.2f}")
        st.metric("Güven Skoru", f"{signal.confidence}/100")
        
        # Position sizing hesaplama
        if signal.sl:
            balance = 10000  # Varsayılan bakiye
            position_size = calculate_position_size(balance, risk_percent, signal.entry, signal.sl)
            st.metric("Önerilen Pozisyon", f"${position_size * signal.entry:,.2f}")
        
        st.write("**Sinyal Sebepleri:**")
        for reason in signal.reason:
            st.write(f"• {reason}")
            
    else:
        st.markdown("### ⚪ WAIT")
        st.info("Şu anda uygun giriş sinyali bulunmuyor")
        if signal.reason:
            st.write("**Sebep:**")
            for reason in signal.reason:
                st.write(f"• {reason}")

# Backtest sonuçları
if run_backtest:
    st.header("📈 Gelişmiş Backtest Sonuçları (90 Gün)")
    
    with st.spinner("Gelişmiş backtest çalışıyor..."):
        data_90d = get_4h_data(symbol, 90)
        if not data_90d.empty:
            data_90d = compute_indicators(data_90d)
            results = advanced_backtest(data_90d, min_rr, risk_percent)
            
            # Temel metrikler
            st.subheader("Temel Performans Metrikleri")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Toplam İşlem", results["trades"])
                st.metric("Kazanç Oranı", f"{results['win_rate']:.1f}%")
                
            with col2:
                st.metric("Toplam Getiri", f"{results['total_return']:.1f}%")
                st.metric("Son Bakiye", f"${results['final_balance']:,.0f}")
            
            with col3:
                st.metric("Maksimum Drawdown", f"{results['performance_metrics'].get('max_drawdown', 0):.1f}%")
                st.metric("Sharpe Oranı", f"{results['performance_metrics'].get('sharpe_ratio', 0):.2f}")
                
            with col4:
                st.metric("Ortalama Kazanç", f"${results['performance_metrics'].get('avg_win', 0):.0f}")
                st.metric("Ortalama Kayıp", f"${results['performance_metrics'].get('avg_loss', 0):.0f}")
            
            # Equity curve
            if "equity_curve" in results and len(results["equity_curve"]) > 1:
                st.subheader("Equity Curve")
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(
                    x=results.get("dates", list(range(len(results["equity_curve"])))),
                    y=results["equity_curve"],
                    line=dict(color="green", width=2),
                    name="Portföy Değeri",
                    fill='tozeroy'
                ))
                fig_eq.update_layout(
                    height=300,
                    template="plotly_dark",
                    showlegend=False,
                    xaxis_title="Tarih",
                    yaxis_title="Portföy Değeri ($)"
                )
                st.plotly_chart(fig_eq, use_container_width=True)
            
            # İşlem detayları
            if "trades_list" in results and results["trades_list"]:
                st.subheader("Son 10 İşlem")
                trades_data = []
                for trade in results["trades_list"][-10:]:
                    trades_data.append({
                        "Tarih": trade.timestamp.strftime("%Y-%m-%d %H:%M") if hasattr(trade.timestamp, 'strftime') else "N/A",
                        "Yön": trade.side,
                        "Entry": format_price(trade.entry),
                        "Exit": format_price(trade.exit),
                        "PNL ($)": f"{trade.pnl:+.2f}",
                        "PNL (%)": f"{trade.pnl_percent:+.2f}%"
                    })
                
                if trades_data:
                    st.dataframe(pd.DataFrame(trades_data), use_container_width=True)
                
            if results["trades"] == 0:
                st.warning("Backtest sırasında hiç işlem yapılmadı. Parametreleri gevşetmeyi deneyin.")
                
        else:
            st.error("Backtest için veri alınamadı!")

# Strateji açıklaması
with st.expander("ℹ️ Gelişmiş Strateji Detayları"):
    st.write("""
    **🎯 Gelişmiş Yüksek Win Rate Stratejisi:**
    
    ### Çekirdek Bileşenler:
    - **Multi-Timeframe Analiz**: 1H, 4H, 1D trend uyumu
    - **EMA50 Trend Filtresi**: Ana trend belirleme
    - **RSI14 Momentum**: Aşırı alım/satım optimizasyonu
    - **Gelişmiş S/R Bölgeleri**: Yoğunluk ve cluster analizi
    - **Akıllı Position Sizing**: Risk bazlı pozisyon büyüklüğü
    
    ### İyileştirmeler:
    - **Daha İyi S/R Tespiti**: Yerel maksimum/minimum ve cluster analizi
    - **Gelişmiş Risk Yönetimi**: Dynamic position sizing ve drawdown kontrolü
    - **Performans Metrikleri**: Sharpe oranı, profit factor, max drawdown
    - **Veri Kalitesi Kontrolü**: Eksik veri ve anomali tespiti
    - **Alarm Sistemi**: Özelleştirilebilir fiyat alarmları
    
    ### Sinyal Mantığı:
    1. **Trend Onayı**: Çoklu zaman diliminde trend uyumu
    2. **S/R Uyumu**: Güçlü destek/direnç seviyeleri
    3. **Momentum Filtresi**: RSI aşırı bölgelerden kaçınma
    4. **Risk/Reward**: Minimum 1.5 R/R oranı
    5. **Güven Skoru**: S/R temas sayısı ve kalitesi
    
    **Backtest Özellikleri:**
    - 90 günlük 4H verisi
    - Gerçekçi slippage simülasyonu
    - Komisyon hesaba katılmış
    - Detaylı performans analizi
    """)

# CSS stilleri
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #0e1117;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-section {
        background-color: #2e2e2e;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)