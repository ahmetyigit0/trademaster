import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

st.set_page_config(page_title="4Saatlik Profesyonel TA", layout="wide")

# Şifre koruması
def check_password():
    def password_entered():
        if st.session_state["password"] == "efe":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Şifre", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Şifre", type="password", on_change=password_entered, key="password")
        st.error("❌ Şifre yanlış!")
        return False
    else:
        return True

if not check_password():
    st.stop()

# =============================================================================
# YENİ: REJİM MOTORU VE İLERİ İNDİKATÖRLER
# =============================================================================

def get_1d_data(symbol, days=120):
    """1D veri çek"""
    try:
        symbol = symbol.upper().strip()
        if '-' not in symbol:
            symbol = symbol + '-USD'
        data = yf.download(symbol, period=f"{days}d", interval="1d", progress=False)
        if data.empty or len(data) == 0:
            return None
        return data
    except Exception as e:
        st.error(f"❌ {symbol} 1D veri çekilemedi: {e}")
        return None

def map_regime_to_4h(df_4h, df_1d):
    """1D rejimini 4H verisine map et"""
    df_1d = df_1d.copy()
    
    # EMA200 ve ATR14 hesapla (1D)
    ema200 = df_1d['Close'].ewm(span=200, adjust=False).mean()
    
    tr1 = df_1d['High'] - df_1d['Low']
    tr2 = (df_1d['High'] - df_1d['Close'].shift()).abs()
    tr3 = (df_1d['Low'] - df_1d['Close'].shift()).abs()
    atr1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
    
    # Rejim sınıflandırma
    slope_up = ema200 > ema200.shift(1)
    dist_up = (df_1d['Close'] - ema200) > 0.5 * atr1d
    dist_down = (ema200 - df_1d['Close']) > 0.5 * atr1d
    
    regime_1d = np.where(slope_up & dist_up, 'UP',
                  np.where((~slope_up) & dist_down, 'DOWN', 'RANGE'))
    
    df_reg = pd.DataFrame({'REGIME_D': regime_1d}, index=df_1d.index)
    
    # 4H verisine forward fill ile map et
    return df_4h.join(df_reg.reindex(df_4h.index, method='ffill'))

def donchian(df, n=20):
    """Donchian Channel"""
    return df['High'].rolling(n).max(), df['Low'].rolling(n).min()

def bollinger(df, n=20, k=2):
    """Bollinger Bands"""
    mid = df['Close'].rolling(n).mean()
    std = df['Close'].rolling(n).std()
    return mid, mid + k * std, mid - k * std

def chandelier_exit(df, period=22, mult=3):
    """Chandelier Exit"""
    atr = df['ATR']
    long_stop = df['High'].rolling(period).max() - mult * atr
    short_stop = df['Low'].rolling(period).min() + mult * atr
    return long_stop, short_stop

def calculate_advanced_indicators(df):
    """İleri teknik göstergeler"""
    df = df.copy()
    
    # Donchian Channel
    df['DONCH_HIGH'], df['DONCH_LOW'] = donchian(df, 20)
    
    # Bollinger Bands
    df['BB_MID'], df['BB_UPPER'], df['BB_LOWER'] = bollinger(df, 20, 2)
    
    # Chandelier Exit (ATR varsa)
    if 'ATR' in df.columns:
        df['CHANDELIER_LONG'], df['CHANDELIER_SHORT'] = chandelier_exit(df)
    
    return df

def get_regime(symbol, df_4h):
    """Rejim hesapla ve 4H verisine ekle"""
    df_1d = get_1d_data(symbol, days=120)
    if df_1d is None:
        # Fallback: EMA50 bazlı basit rejim
        ema50 = df_4h['Close'].ewm(span=50, adjust=False).mean()
        price_vs_ema = (df_4h['Close'] - ema50) / df_4h['Close'] * 100
        df_4h['REGIME'] = np.where(price_vs_ema > 2, 'UP', 
                            np.where(price_vs_ema < -2, 'DOWN', 'RANGE'))
        return df_4h
    else:
        return map_regime_to_4h(df_4h, df_1d)

def can_trade(last_signal_time, current_time, cooldown_bars=3):
    """Cooldown kontrolü"""
    if last_signal_time is None:
        return True
    dt = (current_time - last_signal_time) / pd.Timedelta('4H')
    return dt >= cooldown_bars

def generate_signals_v2(df, regime_col='REGIME', min_rr_ratio=1.5, cooldown_bars=3, bb_width_pct=2.5, donchian_len=20):
    """
    Yeni rejim-temelli strateji sinyalleri
    """
    if len(df) < 50:
        return {"type": "WAIT", "reason": "Yetersiz veri", "strat_id": "NONE"}
    
    current_idx = df.index[-1]
    current_data = df.iloc[-1]
    current_price = float(current_data['Close'])
    
    # Cooldown kontrolü (basit implementasyon)
    if len(df) > 10:
        last_signals = [s for s in [generate_signals_v2(df.iloc[:-i], regime_col, min_rr_ratio, cooldown_bars, bb_width_pct, donchian_len) 
                                  for i in range(1, min(10, len(df)))] 
                       if s.get('type') in ['BUY', 'SELL']]
        if last_signals and not can_trade(df.index[-2], current_idx, cooldown_bars):
            return {"type": "WAIT", "reason": "Cooldown", "strat_id": "NONE"}
    
    regime = current_data.get(regime_col, 'RANGE')
    atr = float(current_data.get('ATR', current_price * 0.02))
    rsi = float(current_data.get('RSI', 50))
    
    # Strateji A: Uptrend - Momentum Breakout
    if regime == 'UP':
        # Donchian breakout + RSI filtresi
        donch_high = float(current_data.get('DONCH_HIGH', current_price))
        if current_price >= donch_high and rsi < 70:
            sl = min(float(current_data.get('DONCH_LOW', current_price * 0.98)), 
                    float(current_data.get('BB_LOWER', current_price * 0.98)))
            risk = current_price - sl
            if risk > 0:
                tp1 = current_price + risk * (min_rr_ratio * 0.5)
                tp2 = current_price + risk * min_rr_ratio
                rr = (tp2 - current_price) / risk
                if rr >= min_rr_ratio:
                    return {
                        "type": "BUY", "entry": current_price, "sl": sl, 
                        "tp1": tp1, "tp2": tp2, "rr": rr, 
                        "reason": "Uptrend Breakout", "strat_id": "A"
                    }
    
    # Strateji B: Downtrend - Momentum Breakdown  
    elif regime == 'DOWN':
        # Donchian breakdown + RSI filtresi
        donch_low = float(current_data.get('DONCH_LOW', current_price))
        if current_price <= donch_low and rsi > 30:
            sl = max(float(current_data.get('DONCH_HIGH', current_price * 1.02)), 
                    float(current_data.get('BB_UPPER', current_price * 1.02)))
            risk = sl - current_price
            if risk > 0:
                tp1 = current_price - risk * (min_rr_ratio * 0.5)
                tp2 = current_price - risk * min_rr_ratio
                rr = (current_price - tp2) / risk
                if rr >= min_rr_ratio:
                    return {
                        "type": "SELL", "entry": current_price, "sl": sl, 
                        "tp1": tp1, "tp2": tp2, "rr": rr,
                        "reason": "Downtrend Breakdown", "strat_id": "B"
                    }
    
    # Strateji C: Range - Mean Reversion
    elif regime == 'RANGE':
        # Bollinger Band bounce
        bb_upper = float(current_data.get('BB_UPPER', current_price * 1.1))
        bb_lower = float(current_data.get('BB_LOWER', current_price * 0.9))
        bb_mid = float(current_data.get('BB_MID', current_price))
        
        # Üst band direnç - Short
        if current_price >= bb_upper * 0.99 and rsi > 60:
            sl = bb_upper * 1.02
            risk = sl - current_price
            if risk > 0:
                tp1 = bb_mid
                tp2 = bb_lower
                rr = (current_price - tp2) / risk
                if rr >= min_rr_ratio:
                    return {
                        "type": "SELL", "entry": current_price, "sl": sl, 
                        "tp1": tp1, "tp2": tp2, "rr": rr,
                        "reason": "Range Resistance", "strat_id": "C"
                    }
        
        # Alt band destek - Long
        elif current_price <= bb_lower * 1.01 and rsi < 40:
            sl = bb_lower * 0.98
            risk = current_price - sl
            if risk > 0:
                tp1 = bb_mid
                tp2 = bb_upper
                rr = (tp2 - current_price) / risk
                if rr >= min_rr_ratio:
                    return {
                        "type": "BUY", "entry": current_price, "sl": sl, 
                        "tp1": tp1, "tp2": tp2, "rr": rr,
                        "reason": "Range Support", "strat_id": "C"
                    }
    
    # Aşırı uzama filtresi
    if rsi > 80 or rsi < 20:
        return {"type": "WAIT", "reason": f"Aşırı uzama (RSI:{rsi:.1f})", "strat_id": "NONE"}
    
    return {"type": "WAIT", "reason": "Koşullar uygun değil", "strat_id": "NONE"}

# =============================================================================
# BACKTEST SİSTEMİ
# =============================================================================

@dataclass
class Trade:
    open_time: pd.Timestamp
    side: str  # 'LONG' or 'SHORT'
    entry: float
    sl: float
    tp1: float
    tp2: float
    size: float
    risk_perc: float
    fee: float
    slip: float
    status: str = "OPEN"
    close_time: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    r_multiple: float | None = None
    pnl: float | None = None
    strat_id: str = "NONE"

def _position_size(entry, sl, balance, risk_perc, side):
    risk_cap = balance * (risk_perc/100.0)
    dist = abs(entry - sl)
    if dist <= 0: 
        return 0.0
    qty = risk_cap / dist
    return max(qty, 0.0)

def _apply_cost(price, fee, slip, side, is_entry):
    adj = price * (fee + slip)
    if side == "LONG":
        return price + adj if is_entry else price - adj
    else:
        return price - adj if is_entry else price + adj

def backtest_90d_optimized(df_90d, risk_perc=1.0, fee=0.001, slip=0.0002, partial=False,
                          min_rr_ratio=1.5, cooldown_bars=3, bb_width_pct=2.5, donchian_len=20, start_balance=10000.0):
    """
    Güncellenmiş backtest - rejim temelli stratejilerle
    """
    balance = start_balance
    trades = []
    equity = [balance]
    dd_series = [0.0]
    
    min_lookback = 100
    data_length = len(df_90d)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(min_lookback, data_length - 1):
        if i % 10 == 0:
            progress = (i - min_lookback) / (data_length - min_lookback - 1)
            progress_bar.progress(progress)
            status_text.text(f"Backtest çalışıyor... %{int(progress * 100)}")
        
        try:
            df_slice = df_90d.iloc[:i+1].copy()
            
            # Sinyal üret
            sig = generate_signals_v2(
                df_slice, 
                min_rr_ratio=min_rr_ratio,
                cooldown_bars=cooldown_bars,
                bb_width_pct=bb_width_pct,
                donchian_len=donchian_len
            )
            
            if sig["type"] == "WAIT":
                equity.append(balance)
                current_equity = equity[-1]
                peak_equity = max(equity)
                drawdown = ((current_equity - peak_equity) / peak_equity) * 100 if peak_equity > 0 else 0
                dd_series.append(drawdown)
                continue

            # Bir sonraki barın açılışında giriş
            next_open = float(df_90d['Open'].iloc[i+1])
            side = "LONG" if sig["type"] == "BUY" else "SHORT"
            
            # Maliyetli giriş fiyatı
            entry = _apply_cost(next_open, fee, slip, side, is_entry=True)
            sl = sig['sl']
            tp1 = sig['tp1']
            tp2 = sig['tp2']
            
            # Pozisyon büyüklüğü
            qty = _position_size(entry, sl, balance, risk_perc, side)
            
            if qty <= 0:
                equity.append(balance)
                current_equity = equity[-1]
                peak_equity = max(equity)
                drawdown = ((current_equity - peak_equity) / peak_equity) * 100 if peak_equity > 0 else 0
                dd_series.append(drawdown)
                continue

            # Çıkış kontrolü
            open_index = i + 1
            exit_found = False
            exit_reason = None
            exit_price = None
            pnl = 0.0
            
            max_lookahead = min(open_index + 50, data_length)
            
            for j in range(open_index, max_lookahead):
                bar = df_90d.iloc[j]
                high, low = float(bar['High']), float(bar['Low'])
                close = float(bar['Close'])

                if side == "LONG":
                    hit_tp2 = low <= tp2 <= high
                    hit_sl = low <= sl <= high
                    hit_tp1 = low <= tp1 <= high if partial else False
                    
                    if hit_tp2:
                        exit_reason = "TP2"
                        exit_price = tp2
                        pnl = (tp2 - entry) * qty
                        exit_found = True
                        break
                    elif hit_sl:
                        exit_reason = "SL"
                        exit_price = sl
                        pnl = (sl - entry) * qty
                        exit_found = True
                        break
                    elif partial and hit_tp1:
                        realized_pnl = (tp1 - entry) * qty * 0.5
                        remaining_qty = qty * 0.5
                        
                        for k in range(j + 1, max_lookahead):
                            bar2 = df_90d.iloc[k]
                            high2, low2 = float(bar2['High']), float(bar2['Low'])
                            close2 = float(bar2['Close'])
                            
                            if low2 <= entry <= high2:
                                if low2 <= tp2 <= high2:
                                    exit_reason = "TP2 (Partial)"
                                    exit_price = tp2
                                    pnl = realized_pnl + (tp2 - entry) * remaining_qty
                                    exit_found = True
                                    j = k
                                    break
                                elif k == max_lookahead - 1:
                                    exit_reason = "Time (Partial)"
                                    exit_price = close2
                                    pnl = realized_pnl + (close2 - entry) * remaining_qty
                                    exit_found = True
                                    j = k
                                    break
                        if exit_found:
                            break
                        
                else:  # SHORT
                    hit_tp2 = low <= tp2 <= high
                    hit_sl = low <= sl <= high
                    hit_tp1 = low <= tp1 <= high if partial else False
                    
                    if hit_tp2:
                        exit_reason = "TP2"
                        exit_price = tp2
                        pnl = (entry - tp2) * qty
                        exit_found = True
                        break
                    elif hit_sl:
                        exit_reason = "SL"
                        exit_price = sl
                        pnl = (entry - sl) * qty
                        exit_found = True
                        break
                    elif partial and hit_tp1:
                        realized_pnl = (entry - tp1) * qty * 0.5
                        remaining_qty = qty * 0.5
                        
                        for k in range(j + 1, max_lookahead):
                            bar2 = df_90d.iloc[k]
                            high2, low2 = float(bar2['High']), float(bar2['Low'])
                            close2 = float(bar2['Close'])
                            
                            if low2 <= entry <= high2:
                                if low2 <= tp2 <= high2:
                                    exit_reason = "TP2 (Partial)"
                                    exit_price = tp2
                                    pnl = realized_pnl + (entry - tp2) * remaining_qty
                                    exit_found = True
                                    j = k
                                    break
                                elif k == max_lookahead - 1:
                                    exit_reason = "Time (Partial)"
                                    exit_price = close2
                                    pnl = realized_pnl + (entry - close2) * remaining_qty
                                    exit_found = True
                                    j = k
                                    break
                        if exit_found:
                            break

            if not exit_found:
                last_close = float(df_90d['Close'].iloc[max_lookahead - 1])
                exit_reason = "Time"
                exit_price = last_close
                if side == "LONG":
                    pnl = (last_close - entry) * qty
                else:
                    pnl = (entry - last_close) * qty

            # Çıkış maliyeti
            exit_price_costed = _apply_cost(exit_price, fee, slip, side, is_entry=False)
            
            # Net PnL
            entry_cost = entry * qty * fee
            exit_cost = exit_price_costed * qty * fee
            pnl_after_cost = pnl - entry_cost - exit_cost
            
            balance += pnl_after_cost

            # R-multiple
            risk_amount = abs(entry - sl) * qty
            r_mult = pnl_after_cost / risk_amount if risk_amount > 0 else 0.0

            # Trade kaydı
            trade = Trade(
                open_time=df_90d.index[open_index],
                side=side,
                entry=entry,
                sl=sl,
                tp1=tp1,
                tp2=tp2,
                size=qty,
                risk_perc=risk_perc,
                fee=fee,
                slip=slip,
                status="CLOSED",
                close_time=df_90d.index[j],
                exit_price=exit_price_costed,
                exit_reason=exit_reason,
                r_multiple=r_mult,
                pnl=pnl_after_cost,
                strat_id=sig.get('strat_id', 'NONE')
            )
            trades.append(trade)
            
        except Exception as e:
            equity.append(balance)
            current_equity = equity[-1]
            peak_equity = max(equity)
            drawdown = ((current_equity - peak_equity) / peak_equity) * 100 if peak_equity > 0 else 0
            dd_series.append(drawdown)
            continue
            
        equity.append(balance)
        current_equity = equity[-1]
        peak_equity = max(equity)
        drawdown = ((current_equity - peak_equity) / peak_equity) * 100 if peak_equity > 0 else 0
        dd_series.append(drawdown)

    progress_bar.empty()
    status_text.empty()
    
    # Metrikler
    if len(equity) > 0:
        eq_series = pd.Series(equity)
        returns = eq_series.pct_change().fillna(0)
        
        wins = [t for t in trades if t.r_multiple is not None and t.r_multiple > 0]
        losses = [t for t in trades if t.r_multiple is not None and t.r_multiple <= 0]
        
        total_trades = len(trades)
        win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
        
        avg_win_r = np.mean([t.r_multiple for t in wins]) if wins else 0
        avg_loss_r = np.mean([t.r_multiple for t in losses]) if losses else 0
        
        total_win_pnl = sum([t.pnl for t in wins]) if wins else 0
        total_loss_pnl = abs(sum([t.pnl for t in losses])) if losses else 0
        profit_factor = total_win_pnl / total_loss_pnl if total_loss_pnl > 0 else float('inf')
        
        expectancy_r = (win_rate/100) * avg_win_r - ((100 - win_rate)/100) * abs(avg_loss_r)
        
        max_drawdown = min(dd_series) if dd_series else 0
        
        sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(365 * 6) if len(returns) > 1 else 0
        
        report = {
            "trades": total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "expectancy_r": expectancy_r,
            "avg_win_r": avg_win_r,
            "avg_loss_r": avg_loss_r,
            "max_drawdown_pct": max_drawdown,
            "sharpe": sharpe,
            "final_balance": balance,
            "total_return_pct": ((balance - start_balance) / start_balance) * 100
        }
        
        trades_df = pd.DataFrame([t.__dict__ for t in trades])
        eq_df = pd.DataFrame({
            "time": df_90d.index[:len(equity)],
            "equity": equity
        })
        dd_df = pd.DataFrame({
            "time": df_90d.index[:len(dd_series)],
            "drawdown": dd_series
        })
        
        return report, trades_df, eq_df, dd_df
    
    empty_report = {
        "trades": 0, "win_rate": 0, "profit_factor": 0, "expectancy_r": 0,
        "avg_win_r": 0, "avg_loss_r": 0, "max_drawdown_pct": 0, "sharpe": 0,
        "final_balance": start_balance, "total_return_pct": 0
    }
    return empty_report, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# =============================================================================
# UI VE MEVCUT FONKSİYONLAR
# =============================================================================

def load_symbol_index() -> list[str]:
    return [
        "BTC-USD", "ETH-USD", "USDT-USD", "BNB-USD", "SOL-USD", "XRP-USD", "DOGE-USD",
        "ADA-USD", "TRX-USD", "TON-USD", "AVAX-USD", "DOT-USD", "MATIC-USD", 
        "LINK-USD", "ATOM-USD", "FIL-USD", "HBAR-USD", "ICP-USD", "AR-USD"
    ]

def autocomplete_matches(query: str, symbols: list[str], limit: int = 20) -> list[str]:
    q = (query or "").upper().strip()
    if len(q) < 2:
        return []
    matches = [s for s in symbols if s.startswith(q)]
    return matches[:limit]

st.title("🎯 4 Saatlik Profesyonel Teknik Analiz")

# Sidebar
with st.sidebar:
    st.header("⚙️ Ayarlar")
    
    if 'selected_symbol' not in st.session_state:
        st.session_state['selected_symbol'] = "BTC-USD"
    
    crypto_symbol = st.text_input("Kripto Sembolü", st.session_state['selected_symbol'])
    
    ALL_SYMBOLS = load_symbol_index()
    matches = autocomplete_matches(crypto_symbol, ALL_SYMBOLS)
    
    if matches:
        st.caption("🔎 Öneriler:")
        for m in matches:
            if st.button(m, key=f"sym_{m}", use_container_width=True):
                st.session_state['selected_symbol'] = m
                st.rerun()
    
    st.caption("Hızlı Seçim:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("BTC-USD", use_container_width=True):
            st.session_state['selected_symbol'] = "BTC-USD"
            st.rerun()
        if st.button("ETH-USD", use_container_width=True):
            st.session_state['selected_symbol'] = "ETH-USD"
            st.rerun()
    with col2:
        if st.button("ADA-USD", use_container_width=True):
            st.session_state['selected_symbol'] = "ADA-USD"
            st.rerun()
        if st.button("XRP-USD", use_container_width=True):
            st.session_state['selected_symbol'] = "XRP-USD"
            st.rerun()
    
    st.subheader("Temel Parametreler")
    ema_period = st.slider("EMA Period", 20, 100, 50)
    rsi_period = st.slider("RSI Period", 5, 21, 14)
    min_touch_points = st.slider("Min Temas", 2, 5, 3)
    risk_reward_ratio = st.slider("Min R/R", 1.0, 3.0, 1.5)
    analysis_lookback_bars = st.slider("Analiz Bars", 80, 200, 120)
    
    st.subheader("Strateji Parametreleri")
    cooldown_bars = st.slider("Cooldown Bars", 1, 10, 3)
    bb_width_pct = st.number_input("BB Width (%)", 1.0, 5.0, 2.5, 0.1)
    donchian_len = st.slider("Donchian Length", 10, 50, 20)
    
    st.divider()
    st.subheader("🧪 Backtest")
    
    run_bt = st.button("Run Backtest (90d)", use_container_width=True, type="primary")
    risk_perc = st.slider("Risk %", 0.1, 5.0, 1.0, 0.1)
    fee = st.number_input("Fee (taker, %)", 0.00, 1.00, 0.10, 0.01) / 100.0
    slip = st.number_input("Slippage (%)", 0.00, 0.50, 0.02, 0.01) / 100.0
    partial = st.toggle("Kısmi Realize (TP1 %50 & BE)", value=False)

crypto_symbol = st.session_state['selected_symbol']

# Mevcut fonksiyonlar (kısaltılmış)
def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

class Zone:
    def __init__(self, low: float, high: float, touches: int, last_touch_ts: Any, kind: str = "support"):
        self.low = low
        self.high = high
        self.touches = touches
        self.last_touch_ts = last_touch_ts
        self.kind = kind
        self.score = 0
        self.status = "valid"
        
    def __repr__(self):
        return f"Zone({self.kind}, low={self.low:.4f}, high={self.high:.4f}, touches={self.touches})"

def build_zones(df: pd.DataFrame, min_touch_points: int, lookback: int = 120) -> List[Zone]:
    if len(df) < lookback:
        lookback = len(df)
    data = df.tail(lookback).copy()
    current_price = float(data['Close'].iloc[-1])
    atr = compute_atr(data).iloc[-1] if len(data) > 14 else current_price * 0.02
    bin_width = max(0.25 * atr, current_price * 0.0015)
    price_levels = []
    for i in range(len(data)):
        try:
            price_levels.extend([float(data['Close'].iloc[i]), float(data['High'].iloc[i]), float(data['Low'].iloc[i])])
        except (ValueError, IndexError):
            continue
    if not price_levels:
        return []
    price_levels = sorted(price_levels)
    bins = {}
    current_bin = min(price_levels)
    while current_bin <= max(price_levels):
        bin_end = current_bin + bin_width
        count = sum(1 for price in price_levels if current_bin <= price <= bin_end)
        if count >= min_touch_points:
            bins[(current_bin, bin_end)] = count
        current_bin = bin_end
    zones = []
    for (zone_low, zone_high), touches in bins.items():
        last_touch_ts = data.index[-1]
        for i in range(len(data)-1, -1, -1):
            close_price = float(data['Close'].iloc[i])
            high_price = float(data['High'].iloc[i])
            low_price = float(data['Low'].iloc[i])
            if (zone_low <= close_price <= zone_high or zone_low <= high_price <= zone_high or zone_low <= low_price <= zone_high):
                last_touch_ts = data.index[i]
                break
        kind = "support" if zone_high < current_price else "resistance"
        zone = Zone(low=zone_low, high=zone_high, touches=touches, last_touch_ts=last_touch_ts, kind=kind)
        zones.append(zone)
    return zones

def format_price(price):
    if price is None or np.isnan(price):
        return "N/A"
    try:
        price = float(price)
        if price >= 1000:
            return f"${price:,.0f}"
        elif price >= 1:
            return f"${price:.2f}"
        elif price >= 0.1:
            return f"${price:.3f}"
        else:
            return f"${price:.4f}"
    except (ValueError, TypeError):
        return "N/A"

@st.cache_data
def get_4h_data(symbol, days=30):
    try:
        symbol = symbol.upper().strip()
        if '-' not in symbol:
            symbol = symbol + '-USD'
        data = yf.download(symbol, period=f"{days}d", interval="4h", progress=False)
        if data.empty or len(data) == 0:
            st.error(f"❌ {symbol} için veri bulunamadı!")
            return None
        return data
    except Exception as e:
        st.error(f"❌ {symbol} veri çekilemedi: {e}")
        return None

def calculate_indicators(data, ema_period=50, rsi_period=14, donchian_len=20, bb_width_pct=2.5):
    if data is None or len(data) == 0:
        return data
    df = data.copy()
    
    # Temel göstergeler
    df['EMA'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # ATR
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift()).abs()
    tr3 = (df['Low'] - df['Close'].shift()).abs()
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    df.drop(columns=['TR'], inplace=True)
    
    # İleri göstergeler
    df = calculate_advanced_indicators(df)
    
    return df

def find_congestion_zones(data, min_touch_points=3, lookback=120):
    if data is None or len(data) == 0:
        return [], []
    zones = build_zones(data, min_touch_points, lookback)
    current_price = float(data['Close'].iloc[-1])
    support_zones = [zone for zone in zones if zone.kind == "support"]
    resistance_zones = [zone for zone in zones if zone.kind == "resistance"]
    
    # Basit skorlama
    for zone in support_zones + resistance_zones:
        zone.score = min(zone.touches * 20, 80)
    
    support_zones = sorted(support_zones, key=lambda x: x.score, reverse=True)[:2]  # Sadece 2
    resistance_zones = sorted(resistance_zones, key=lambda x: x.score, reverse=True)[:2]  # Sadece 2
    
    return support_zones, resistance_zones

def create_clean_candlestick_chart(data, support_zones, resistance_zones, crypto_symbol, signals):
    fig = go.Figure()
    if data is None or len(data) == 0:
        return fig
    data_3days = data.tail(18)
    current_price = float(data_3days['Close'].iloc[-1])
    
    # Mum çubukları
    for i in range(len(data_3days)):
        try:
            row = data_3days.iloc[i]
            open_price = float(row['Open'])
            high = float(row['High'])
            low = float(row['Low'])
            close_price = float(row['Close'])
            color = '#00C805' if close_price > open_price else '#FF0000'
            fig.add_trace(go.Scatter(x=[data_3days.index[i], data_3days.index[i]], y=[open_price, close_price], mode='lines', line=dict(color=color, width=8), showlegend=False))
            fig.add_trace(go.Scatter(x=[data_3days.index[i], data_3days.index[i]], y=[max(open_price, close_price), high], mode='lines', line=dict(color=color, width=1.5), showlegend=False))
            fig.add_trace(go.Scatter(x=[data_3days.index[i], data_3days.index[i]], y=[min(open_price, close_price), low], mode='lines', line=dict(color=color, width=1.5), showlegend=False))
        except (ValueError, IndexError):
            continue
    
    # EMA
    if 'EMA' in data_3days.columns:
        try:
            fig.add_trace(go.Scatter(x=data_3days.index, y=data_3days['EMA'], name=f'EMA{ema_period}', line=dict(color='orange', width=2), showlegend=False))
        except Exception:
            pass
    
    # Bantlar (sadece 2'şer)
    for i, zone in enumerate(support_zones[:2]):
        border_color = "#00FF00"
        fig.add_hrect(y0=zone.low, y1=zone.high, fillcolor="rgba(0,255,0,0.12)", line=dict(width=1, color=border_color), layer="below")
        fig.add_annotation(x=data_3days.index[-1], y=(zone.low + zone.high) / 2, text=f"S{i+1}", showarrow=False, xanchor='left', yanchor='middle', font=dict(size=10, color="#00FF00"), bgcolor="rgba(0,0,0,0.5)")
    
    for i, zone in enumerate(resistance_zones[:2]):
        border_color = "#FF0000"
        fig.add_hrect(y0=zone.low, y1=zone.high, fillcolor="rgba(255,0,0,0.12)", line=dict(width=1, color=border_color), layer="below")
        fig.add_annotation(x=data_3days.index[-1], y=(zone.low + zone.high) / 2, text=f"R{i+1}", showarrow=False, xanchor='left', yanchor='middle', font=dict(size=10, color="#FF0000"), bgcolor="rgba(0,0,0,0.5)")
    
    # Mevcut fiyat
    try:
        fig.add_hline(y=current_price, line_dash="dot", line_color="yellow", line_width=1, opacity=0.7, annotation_text=f"{format_price(current_price)}", annotation_position="left top", annotation_font_size=10, annotation_font_color="yellow")
    except (ValueError, IndexError):
        pass
    
    # Sinyal ok işareti (sadece aktif sinyalde)
    if signals and signals.get("type") in ["BUY", "SELL"]:
        marker_symbol = "triangle-up" if signals["type"] == "BUY" else "triangle-down"
        marker_color = "#00FF00" if signals["type"] == "BUY" else "#FF0000"
        fig.add_trace(go.Scatter(x=[data_3days.index[-1]], y=[current_price], mode='markers', marker=dict(symbol=marker_symbol, size=12, color=marker_color, line=dict(width=2, color="white")), showlegend=False, name=f"{signals['type']} Sinyal"))
    
    fig.update_layout(height=500, title=f"{crypto_symbol} - 4H (Son 3 Gün)", xaxis_title="", yaxis_title="Fiyat (USD)", showlegend=False, xaxis_rangeslider_visible=False, plot_bgcolor='#0E1117', paper_bgcolor='#0E1117', font=dict(color='white', size=10), xaxis=dict(gridcolor='#444', showticklabels=True), yaxis=dict(gridcolor='#444'), margin=dict(l=50, r=50, t=50, b=50))
    return fig

# =============================================================================
# ANA UYGULAMA
# =============================================================================

def main():
    # Veri yükleme
    with st.spinner(f'⏳ {crypto_symbol} verileri yükleniyor...'):
        data_30days = get_4h_data(crypto_symbol, days=30)
    
    if data_30days is None or data_30days.empty:
        st.error(f"❌ {crypto_symbol} için veri yüklenemedi!")
        return
    
    # Göstergeleri hesapla
    data_30days = calculate_indicators(data_30days, ema_period, rsi_period, donchian_len, bb_width_pct)
    
    # Rejim hesapla
    data_30days = get_regime(crypto_symbol, data_30days)
    
    # Yoğunluk bölgelerini bul
    support_zones, resistance_zones = find_congestion_zones(
        data_30days, min_touch_points, analysis_lookback_bars
    )
    
    # Sinyal üret (yeni strateji)
    signals = generate_signals_v2(
        data_30days, 
        min_rr_ratio=risk_reward_ratio,
        cooldown_bars=cooldown_bars,
        bb_width_pct=bb_width_pct,
        donchian_len=donchian_len
    )
    
    # Mevcut durum
    try:
        current_price = float(data_30days['Close'].iloc[-1])
        regime = data_30days['REGIME'].iloc[-1] if 'REGIME' in data_30days.columns else 'RANGE'
    except (ValueError, IndexError):
        current_price = 0
        regime = 'RANGE'
    
    # Layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        chart_fig = create_clean_candlestick_chart(
            data_30days, support_zones, resistance_zones, crypto_symbol, signals
        )
        st.plotly_chart(chart_fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Sinyal")
        
        if signals.get("type") in ["BUY", "SELL"]:
            signal_color = "🟢" if signals['type'] == 'BUY' else "🔴"
            strat_name = {"A": "Trend Breakout", "B": "Trend Breakdown", "C": "Range Trade"}.get(signals.get('strat_id', 'NONE'), "Unknown")
            regime_name = {"UP": "Yükseliş", "DOWN": "Düşüş", "RANGE": "Range"}.get(regime, "Bilinmiyor")
            
            st.markdown(f"### {signal_color} {signals['type']}")
            st.metric("Strateji", f"{signals.get('strat_id', 'NONE')} - {strat_name}")
            st.metric("Rejim", regime_name)
            
            cols = st.columns(2)
            with cols[0]:
                st.metric("Giriş", format_price(signals['entry']))
                st.metric("TP1", format_price(signals['tp1']))
            with cols[1]:
                st.metric("SL", format_price(signals['sl']))
                st.metric("TP2", format_price(signals['tp2']))
            
            st.metric("R/R", f"{signals['rr']:.2f}")
            st.metric("Sebep", signals.get('reason', 'N/A'))
            
        else:
            st.markdown("### ⚪ BEKLE")
            st.info(signals.get('reason', 'Koşullar uygun değil'))
            st.metric("Rejim", {"UP": "Yükseliş", "DOWN": "Düşüş", "RANGE": "Range"}.get(regime, "Bilinmiyor"))
        
        st.divider()
        
        # Göstergeler
        st.subheader("📈 Göstergeler")
        try:
            rsi_value = float(data_30days['RSI'].iloc[-1])
            atr_value = float(data_30days['ATR'].iloc[-1])
            st.metric("RSI", f"{rsi_value:.1f}")
            st.metric("ATR", format_price(atr_value))
        except:
            pass
        
        st.divider()
        
        # Yakın bantlar
        st.subheader("🎯 Yakın Bantlar")
        
        for i, zone in enumerate(support_zones[:2]):
            st.write(f"**S{i+1}:** {format_price(zone.low)}-{format_price(zone.high)}")
        
        for i, zone in enumerate(resistance_zones[:2]):
            st.write(f"**R{i+1}:** {format_price(zone.low)}-{format_price(zone.high)}")
    
    # BACKTEST
    if 'run_bt' in st.session_state and st.session_state.run_bt:
        st.divider()
        st.header("📊 Backtest Sonuçları (90 Gün)")
        
        with st.spinner("Backtest çalışıyor..."):
            df_90d = get_4h_data(crypto_symbol, days=90)
            if df_90d is not None and not df_90d.empty:
                df_90d = calculate_indicators(df_90d, ema_period, rsi_period, donchian_len, bb_width_pct)
                df_90d = get_regime(crypto_symbol, df_90d)
                
                report, trades_df, eq_df, dd_df = backtest_90d_optimized(
                    df_90d, 
                    risk_perc=risk_perc, 
                    fee=fee, 
                    slip=slip, 
                    partial=partial,
                    min_rr_ratio=risk_reward_ratio,
                    cooldown_bars=cooldown_bars,
                    bb_width_pct=bb_width_pct,
                    donchian_len=donchian_len,
                    start_balance=10000.0
                )
                
                # KPI'lar
                col1, col2, col3, col4 = st.columns(4)
            
                with col1:
                    st.metric("İşlem Sayısı", report["trades"])
                    st.metric("Win Rate", f"{report['win_rate']:.1f}%")
                
                with col2:
                    st.metric("Profit Factor", f"{report['profit_factor']:.2f}")
                    st.metric("Expectancy (R)", f"{report['expectancy_r']:.2f}")
                
                with col3:
                    st.metric("Max Drawdown", f"{report['max_drawdown_pct']:.1f}%")
                    st.metric("Sharpe Ratio", f"{report['sharpe']:.2f}")
                
                with col4:
                    st.metric("Final Balance", f"${report['final_balance']:,.0f}")
                    st.metric("Toplam Getiri", f"{report['total_return_pct']:.1f}%")
                
                # Grafikler
                if not eq_df.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Equity Curve")
                        fig_equity = go.Figure()
                        fig_equity.add_trace(go.Scatter(
                            x=eq_df["time"], 
                            y=eq_df["equity"],
                            line=dict(color="#00FF00", width=2),
                            fill='tozeroy',
                            fillcolor="rgba(0,255,0,0.1)"
                        ))
                        fig_equity.update_layout(
                            height=300,
                            showlegend=False,
                            plot_bgcolor='#0E1117',
                            paper_bgcolor='#0E1117',
                            font=dict(color='white'),
                            margin=dict(l=20, r=20, t=30, b=20)
                        )
                        st.plotly_chart(fig_equity, use_container_width=True)
                    
                    with col2:
                        if not dd_df.empty:
                            st.subheader("Drawdown")
                            fig_dd = go.Figure()
                            fig_dd.add_trace(go.Scatter(
                                x=dd_df["time"], 
                                y=dd_df["drawdown"],
                                line=dict(color="#FF4444", width=2),
                                fill='tozeroy',
                                fillcolor="rgba(255,0,0,0.3)"
                            ))
                            fig_dd.update_layout(
                                height=300,
                                showlegend=False,
                                plot_bgcolor='#0E1117',
                                paper_bgcolor='#0E1117',
                                font=dict(color='white'),
                                margin=dict(l=20, r=20, t=30, b=20)
                            )
                            st.plotly_chart(fig_dd, use_container_width=True)
                
                # İşlem listesi
                if not trades_df.empty and len(trades_df) > 0:
                    st.subheader("İşlem Listesi")
                    display_cols = ["open_time", "side", "strat_id", "entry", "sl", "tp1", "tp2", "exit_price", "exit_reason", "r_multiple", "pnl"]
                    available_cols = [col for col in display_cols if col in trades_df.columns]
                    display_df = trades_df[available_cols].copy()
                    
                    # Formatting
                    for col in ["entry", "sl", "tp1", "tp2", "exit_price"]:
                        if col in display_df.columns:
                            display_df[col] = display_df[col].apply(format_price)
                    
                    if "r_multiple" in display_df.columns:
                        display_df["r_multiple"] = display_df["r_multiple"].round(2)
                    if "pnl" in display_df.columns:
                        display_df["pnl"] = display_df["pnl"].round(2)
                    
                    st.dataframe(display_df, use_container_width=True)
            else:
                st.error("Backtest için veri yüklenemedi!")

# Ana uygulama çalıştırma
if __name__ == "__main__":
    if 'run_bt' not in st.session_state:
        st.session_state.run_bt = False
    
    if run_bt:
        st.session_state.run_bt = True
        st.rerun()
    
    main()