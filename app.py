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
# YENİ: BACKTEST SİSTEMİ
# =============================================================================

@dataclass
class Trade:
    open_time: pd.Timestamp
    side: str  # 'LONG' or 'SHORT'
    entry: float
    sl: float
    tp1: float
    tp2: float
    size: float        # adet
    risk_perc: float   # % olarak (örn. 1.0)
    fee: float         # 0.001 = %0.1
    slip: float        # 0.0002 = %0.02
    status: str = "OPEN"  # OPEN/CLOSED
    close_time: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    r_multiple: float | None = None
    pnl: float | None = None

def _position_size(entry, sl, balance, risk_perc, side):
    """Pozisyon büyüklüğü hesaplar"""
    risk_cap = balance * (risk_perc/100.0)
    dist = abs(entry - sl)
    if dist <= 0: 
        return 0.0
    qty = risk_cap / dist
    return max(qty, 0.0)

def _apply_cost(price, fee, slip, side, is_entry):
    """
    Ücret ve slippage uygular
    side: 'LONG'/'SHORT'; entry ve exit için kaydırma
    """
    adj = price * (fee + slip)
    if side == "LONG":
        return price + adj if is_entry else price - adj
    else:
        return price - adj if is_entry else price + adj

def generate_signal_at_bar(df_slice, params):
    """
    df_slice: t dahil (t'ye kadar). Sadece df_slice kullan; geleceğe bakma.
    Return: None ya da {'type','entry','sl','tp1','tp2'}
    """
    try:
        signals, _ = generate_trading_signals(
            df_slice, 
            params['support_zones'], 
            params['resistance_zones'], 
            ema_period=params['ema_period'], 
            min_rr_ratio=params['min_rr_ratio']
        )
        
        if not signals or signals[0]["type"] == "WAIT":
            return None
            
        s = signals[0]
        
        if s['type'] == 'BUY':
            # LONG için: SL < Entry < TP1 < TP2
            tp1 = s.get('tp1', s['entry'] + (s['entry'] - s['sl']) * params['min_rr_ratio'] * 0.5)
            tp2 = s.get('tp2', s['entry'] + (s['entry'] - s['sl']) * params['min_rr_ratio'])
            tp1, tp2 = sorted([tp1, tp2])
            
            return {
                'type': 'LONG', 
                'sl': s['sl'], 
                'tp1': tp1, 
                'tp2': tp2,
                'entry_price': s['entry']
            }
            
        elif s['type'] == 'SELL':
            # SHORT için: TP2 < TP1 < Entry < SL
            tp1 = s.get('tp1', s['entry'] - (s['sl'] - s['entry']) * params['min_rr_ratio'] * 0.5)
            tp2 = s.get('tp2', s['entry'] - (s['sl'] - s['entry']) * params['min_rr_ratio'])
            tp1, tp2 = sorted([tp1, tp2], reverse=True)
            
            return {
                'type': 'SHORT', 
                'sl': s['sl'], 
                'tp1': tp1, 
                'tp2': tp2,
                'entry_price': s['entry']
            }
            
    except Exception as e:
        return None
    
    return None

# =============================================================================
# OPTİMİZE EDİLMİŞ BACKTEST SİSTEMİ
# =============================================================================

def backtest_90d_optimized(df_90d, risk_perc=1.0, fee=0.001, slip=0.0002, partial=False,
                          ema_period=50, min_rr_ratio=1.5, start_balance=10000.0):
    """
    Optimize edilmiş 90 günlük backtest - 10x daha hızlı
    """
    balance = start_balance
    trades = []
    equity = [balance]
    dd_series = [0.0]
    
    # Ön hesaplamalar
    min_lookback = max(120, ema_period + 20)
    data_length = len(df_90d)
    
    # Progress bar için
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Zone hesaplama sıklığını azalt (her 10 barda bir)
    zone_recalc_freq = 10
    cached_zones = {}
    
    for i in range(min_lookback, data_length - 1):
        # Progress güncelleme
        if i % 10 == 0:
            progress = (i - min_lookback) / (data_length - min_lookback - 1)
            progress_bar.progress(progress)
            status_text.text(f"Backtest çalışıyor... %{int(progress * 100)}")
        
        try:
            # Zone cache kontrolü
            cache_key = i // zone_recalc_freq
            if cache_key in cached_zones:
                support_zones, resistance_zones = cached_zones[cache_key]
            else:
                df_slice = df_90d.iloc[:i+1].copy()
                support_zones, resistance_zones = find_congestion_zones(
                    df_slice, min_touch_points=3, lookback=120
                )
                cached_zones[cache_key] = (support_zones, resistance_zones)
            
            # Hızlı sinyal kontrolü - sadece yüksek skorlu zone'ları kontrol et
            if not support_zones and not resistance_zones:
                equity.append(balance)
                current_equity = equity[-1]
                peak_equity = max(equity)
                drawdown = ((current_equity - peak_equity) / peak_equity) * 100 if peak_equity > 0 else 0
                dd_series.append(drawdown)
                continue
            
            # Sadece en iyi 2 zone'u kontrol et
            best_support = support_zones[0] if support_zones and support_zones[0].score >= 65 else None
            best_resistance = resistance_zones[0] if resistance_zones and resistance_zones[0].score >= 65 else None
            
            if not best_support and not best_resistance:
                equity.append(balance)
                current_equity = equity[-1]
                peak_equity = max(equity)
                drawdown = ((current_equity - peak_equity) / peak_equity) * 100 if peak_equity > 0 else 0
                dd_series.append(drawdown)
                continue
            
            # Hızlı sinyal üretimi
            current_price = float(df_90d['Close'].iloc[i])
            ema_value = float(df_90d['EMA'].iloc[i])
            atr_value = float(df_90d['ATR'].iloc[i])
            
            sig = None
            if best_support and current_price <= best_support.high * 1.02:  # %2 tolerans
                # ALIM sinyali
                entry = min(current_price, best_support.high)
                sl = best_support.low - 0.25 * atr_value
                risk_long = entry - sl
                
                if risk_long > 0:
                    tp1 = entry + risk_long * (min_rr_ratio * 0.5)
                    tp2 = entry + risk_long * min_rr_ratio
                    tp1, tp2 = sorted([tp1, tp2])
                    rr = (tp2 - entry) / risk_long
                    
                    if rr >= min_rr_ratio:
                        sig = {
                            'type': 'LONG', 
                            'sl': sl, 
                            'tp1': tp1, 
                            'tp2': tp2,
                            'entry_price': entry
                        }
            
            elif best_resistance and current_price >= best_resistance.low * 0.98:  # %2 tolerans
                # SATIM sinyali
                entry = max(current_price, best_resistance.low)
                sl = best_resistance.high + 0.25 * atr_value
                risk_short = sl - entry
                
                if risk_short > 0:
                    tp1 = entry - risk_short * (min_rr_ratio * 0.5)
                    tp2 = entry - risk_short * min_rr_ratio
                    tp1, tp2 = sorted([tp1, tp2], reverse=True)
                    rr = (entry - tp2) / risk_short
                    
                    if rr >= min_rr_ratio:
                        sig = {
                            'type': 'SHORT', 
                            'sl': sl, 
                            'tp1': tp1, 
                            'tp2': tp2,
                            'entry_price': entry
                        }
            
            if sig is None:
                equity.append(balance)
                current_equity = equity[-1]
                peak_equity = max(equity)
                drawdown = ((current_equity - peak_equity) / peak_equity) * 100 if peak_equity > 0 else 0
                dd_series.append(drawdown)
                continue

            # Bir sonraki barın açılışında giriş
            next_open = float(df_90d['Open'].iloc[i+1])
            side = sig['type']
            
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

            # HIZLI ÇIKIŞ KONTROLÜ - Vektörel yaklaşım
            open_index = i + 1
            exit_found = False
            exit_reason = None
            exit_price = None
            pnl = 0.0
            
            # Sadece sonraki 50 barı kontrol et (4H -> ~8 gün)
            max_lookahead = min(open_index + 50, data_length)
            
            for j in range(open_index, max_lookahead):
                bar = df_90d.iloc[j]
                high, low = float(bar['High']), float(bar['Low'])
                close = float(bar['Close'])

                if side == "LONG":
                    # LONG çıkış koşulları
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
                        # Kısmi realize
                        realized_pnl = (tp1 - entry) * qty * 0.5
                        remaining_qty = qty * 0.5
                        
                        # Kalan pozisyon için SL'yi BE'ye çek
                        for k in range(j + 1, max_lookahead):
                            bar2 = df_90d.iloc[k]
                            high2, low2 = float(bar2['High']), float(bar2['Low'])
                            close2 = float(bar2['Close'])
                            
                            if low2 <= entry <= high2:  # BE'ye çekildi
                                if low2 <= tp2 <= high2:
                                    exit_reason = "TP2 (Partial)"
                                    exit_price = tp2
                                    pnl = realized_pnl + (tp2 - entry) * remaining_qty
                                    exit_found = True
                                    j = k  # Çıkış zamanını güncelle
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
                    # SHORT çıkış koşulları
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
                        # Kısmi realize
                        realized_pnl = (entry - tp1) * qty * 0.5
                        remaining_qty = qty * 0.5
                        
                        # Kalan pozisyon için SL'yi BE'ye çek
                        for k in range(j + 1, max_lookahead):
                            bar2 = df_90d.iloc[k]
                            high2, low2 = float(bar2['High']), float(bar2['Low'])
                            close2 = float(bar2['Close'])
                            
                            if low2 <= entry <= high2:  # BE'ye çekildi
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

            # Çıkış yoksa vade sonu kapat
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
            
            # Net PnL (ücretler dahil)
            entry_cost = entry * qty * fee
            exit_cost = exit_price_costed * qty * fee
            pnl_after_cost = pnl - entry_cost - exit_cost
            
            # Bakiye güncelleme
            balance += pnl_after_cost

            # R-multiple hesapla
            risk_amount = abs(entry - sl) * qty
            r_mult = pnl_after_cost / risk_amount if risk_amount > 0 else 0.0

            # Trade kaydı oluştur
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
                pnl=pnl_after_cost
            )
            trades.append(trade)
            
        except Exception as e:
            # Hata durumunda equity'yi koru
            equity.append(balance)
            current_equity = equity[-1]
            peak_equity = max(equity)
            drawdown = ((current_equity - peak_equity) / peak_equity) * 100 if peak_equity > 0 else 0
            dd_series.append(drawdown)
            continue
            
        # Equity ve drawdown güncelleme
        equity.append(balance)
        current_equity = equity[-1]
        peak_equity = max(equity)
        drawdown = ((current_equity - peak_equity) / peak_equity) * 100 if peak_equity > 0 else 0
        dd_series.append(drawdown)

    # Progress bar'ı temizle
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
        
        # Sharpe oranı (yıllıklaştırılmış)
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
    
    # Varsayılan boş dönüş
    empty_report = {
        "trades": 0, "win_rate": 0, "profit_factor": 0, "expectancy_r": 0,
        "avg_win_r": 0, "avg_loss_r": 0, "max_drawdown_pct": 0, "sharpe": 0,
        "final_balance": start_balance, "total_return_pct": 0
    }
    return empty_report, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
# =============================================================================
# SEMBOL AUTOCOMPLETE SİSTEMİ (Önceki koddan)
# =============================================================================

def load_symbol_index() -> list[str]:
    """-USD ile biten popüler kripto sembollerinin yerleşik listesi."""
    return [
        "BTC-USD", "ETH-USD", "USDT-USD", "BNB-USD", "SOL-USD", "XRP-USD", "DOGE-USD",
        "ADA-USD", "TRX-USD", "TON-USD", "AVAX-USD", "DOT-USD", "MATIC-USD", 
        "LINK-USD", "ATOM-USD", "FIL-USD", "HBAR-USD", "ICP-USD", "AR-USD",
        "THETA-USD", "THE-USD", "THG-USD", "TIA-USD", "TUSD-USD", "LTC-USD",
        "BCH-USD", "XLM-USD", "UNI-USD", "ETC-USD", "XMR-USD", "EOS-USD",
        "AAVE-USD", "ALGO-USD", "BAT-USD", "COMP-USD", "DASH-USD", "ZEC-USD",
        "XTZ-USD", "NEAR-USD", "FTM-USD", "SAND-USD", "MANA-USD", "ENJ-USD",
        "GALA-USD", "APE-USD", "GRT-USD", "RUNE-USD", "KAVA-USD", "RNDR-USD",
        "OP-USD", "ARB-USD", "IMX-USD", "STX-USD", "APT-USD", "SUI-USD",
        "SEI-USD", "INJ-USD", "RPL-USD", "LDO-USD", "MKR-USD", "SNX-USD",
        "CRV-USD", "1INCH-USD", "BAL-USD", "YFI-USD", "SUSHI-USD", "CAKE-USD",
        "UMA-USD", "BADGER-USD", "KNC-USD", "REN-USD", "CVC-USD", "REP-USD",
        "ZRX-USD", "BAND-USD", "OXT-USD", "NMR-USD", "POLY-USD", "LRC-USD",
        "OMG-USD", "SKL-USD", "ANKR-USD", "STORJ-USD", "SXP-USD", "HNT-USD",
        "IOST-USD", "IOTA-USD", "VET-USD", "ONT-USD", "ZIL-USD", "SC-USD",
        "WAVES-USD", "RVN-USD", "DGB-USD", "ICX-USD", "STEEM-USD", "NANO-USD",
        "HOT-USD", "ONG-USD", "ONE-USD", "FUN-USD", "CELR-USD", "CHZ-USD",
        "COTI-USD", "DENT-USD", "DOCK-USD", "ELF-USD", "FET-USD", "KEY-USD",
        "LOOM-USD", "NKN-USD", "OCEAN-USD", "OGN-USD", "ORN-USD", "PERL-USD",
        "POND-USD", "POWR-USD", "QKC-USD", "QSP-USD", "REQ-USD", "RLC-USD",
        "ROSE-USD", "SLP-USD", "SNT-USD", "SRM-USD", "SYS-USD", "TCT-USD",
        "TFUEL-USD", "TOMO-USD", "TROY-USD", "TRB-USD", "TWT-USD", "VITE-USD",
        "WAN-USD", "WTC-USD", "YFII-USD", "ZEN-USD"
    ]

def autocomplete_matches(query: str, symbols: list[str], limit: int = 20) -> list[str]:
    """
    Sorguya göre sembol eşleşmelerini bulur
    """
    q = (query or "").upper().strip()
    if len(q) < 2:
        return []
    
    matches = [s for s in symbols if s.startswith(q)]
    return matches[:limit]

st.title("🎯 4 Saatlik Profesyonel Teknik Analiz")

# Sidebar
with st.sidebar:
    st.header("⚙️ Ayarlar")
    
    # Kripto sembolü + autocomplete
    if 'selected_symbol' not in st.session_state:
        st.session_state['selected_symbol'] = "BTC-USD"
    
    crypto_symbol = st.text_input("Kripto Sembolü", st.session_state['selected_symbol'],
                                 help="Örnek: BTC-USD, ETH-USD, ADA-USD, XRP-USD vb.")
    
    # Autocomplete önerileri
    ALL_SYMBOLS = load_symbol_index()
    matches = autocomplete_matches(crypto_symbol, ALL_SYMBOLS)
    
    if matches:
        st.caption("🔎 Öneriler:")
        for m in matches:
            if st.button(m, key=f"sym_{m}", use_container_width=True):
                st.session_state['selected_symbol'] = m
                st.rerun()
    
    # Popüler kripto seçenekleri
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
    
    st.subheader("Parametreler")
    ema_period = st.slider("EMA Period", 20, 100, 50)
    rsi_period = st.slider("RSI Period", 5, 21, 14)
    min_touch_points = st.slider("Min Temas", 2, 5, 3)
    risk_reward_ratio = st.slider("Min R/R", 1.0, 3.0, 1.5)
    analysis_lookback_bars = st.slider("Analiz Bars", 80, 200, 120)
    
    # YENİ: BACKTEST AYARLARI
    st.divider()
    st.subheader("🧪 Backtest")
    
    run_bt = st.button("Run Backtest (90d)", use_container_width=True, type="primary")
    risk_perc = st.slider("Risk %", 0.1, 5.0, 1.0, 0.1)
    fee = st.number_input("Fee (taker, %)", 0.00, 1.00, 0.10, 0.01) / 100.0
    slip = st.number_input("Slippage (%)", 0.00, 0.50, 0.02, 0.01) / 100.0
    partial = st.toggle("Kısmi Realize (TP1 %50 & BE)", value=False)

# Session state'ten sembolü al
crypto_symbol = st.session_state['selected_symbol']

# =============================================================================
# MEVCUT FONKSİYONLAR (Önceki koddan - kısaltılmış)
# =============================================================================

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

def eval_fake_breakout(df: pd.DataFrame, zone: Zone) -> Dict[str, Any]:
    if len(df) < 10:
        return {"status": "valid", "details": "Yetersiz veri"}
    data = df.tail(50).copy()
    atr = compute_atr(data).iloc[-1] if len(data) > 14 else zone.high * 0.02
    breakouts = 0
    max_breakout_distance = 0
    reclaim_mums = 0
    for i in range(len(data)):
        close_price = float(data['Close'].iloc[i])
        if zone.kind == "support":
            if close_price < zone.low:
                breakouts += 1
                distance = zone.low - close_price
                max_breakout_distance = max(max_breakout_distance, distance)
                for j in range(i+1, min(i+3, len(data))):
                    if float(data['Close'].iloc[j]) >= zone.low:
                        reclaim_mums = j - i
                        break
        else:
            if close_price > zone.high:
                breakouts += 1
                distance = close_price - zone.high
                max_breakout_distance = max(max_breakout_distance, distance)
                for j in range(i+1, min(i+3, len(data))):
                    if float(data['Close'].iloc[j]) <= zone.high:
                        reclaim_mums = j - i
                        break
    condition1 = breakouts < 2
    condition2 = max_breakout_distance < 0.5 * atr or max_breakout_distance < zone.high * 0.0035
    condition3 = reclaim_mums <= 2 and reclaim_mums > 0
    fake_score = sum([condition1, condition2, condition3])
    permanent_conditions = [breakouts >= 2, max_breakout_distance >= 0.5 * atr, reclaim_mums == 0 or reclaim_mums > 2]
    permanent_score = sum(permanent_conditions)
    if fake_score >= 2:
        status = "fake"
    elif permanent_score >= 2:
        status = "broken"
    else:
        status = "valid"
    return {"status": status}

def compute_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def score_zone(df: pd.DataFrame, zone: Zone, ema: float, rsi: float, atr: float) -> int:
    score = 0
    current_price = float(df['Close'].iloc[-1])
    touches_score = min(zone.touches * 3, 30)
    score += touches_score
    fake_result = eval_fake_breakout(df, zone)
    if fake_result["status"] == "fake":
        score += 25
    elif fake_result["status"] == "valid":
        score += 15
    if zone.kind == "support":
        ema_distance = abs(zone.high - ema) / atr
        if ema_distance <= 1.0:
            score += 20
        elif ema_distance <= 2.0:
            score += 10
    else:
        ema_distance = abs(zone.low - ema) / atr
        if ema_distance <= 1.0:
            score += 20
        elif ema_distance <= 2.0:
            score += 10
    if zone.kind == "support" and rsi < 40:
        score += 15
    elif zone.kind == "resistance" and rsi > 60:
        score += 15
    elif 40 <= rsi <= 60:
        score += 8
    try:
        macd, signal, hist = compute_macd(df)
        current_hist = hist.iloc[-1]
        prev_hist = hist.iloc[-2]
        if zone.kind == "support" and current_hist > prev_hist:
            score += 10
        elif zone.kind == "resistance" and current_hist < prev_hist:
            score += 10
        elif abs(current_hist - prev_hist) < 0.0001:
            score += 5
    except:
        pass
    return min(score, 100)

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

def calculate_indicators(data, ema_period=50, rsi_period=14):
    if data is None or len(data) == 0:
        return data
    df = data.copy()
    df['EMA'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift()).abs()
    tr3 = (df['Low'] - df['Close'].shift()).abs()
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    df.drop(columns=['TR'], inplace=True)
    return df

def find_congestion_zones(data, min_touch_points=3, lookback=120):
    if data is None or len(data) == 0:
        return [], []
    zones = build_zones(data, min_touch_points, lookback)
    current_price = float(data['Close'].iloc[-1])
    support_zones = [zone for zone in zones if zone.kind == "support"]
    resistance_zones = [zone for zone in zones if zone.kind == "resistance"]
    ema_value = float(data['EMA'].iloc[-1])
    rsi_value = float(data['RSI'].iloc[-1])
    atr_value = float(data['ATR'].iloc[-1])
    for zone in support_zones + resistance_zones:
        zone.score = score_zone(data, zone, ema_value, rsi_value, atr_value)
        fake_result = eval_fake_breakout(data, zone)
        zone.status = fake_result["status"]
    support_zones = sorted(support_zones, key=lambda x: x.score, reverse=True)[:3]
    resistance_zones = sorted(resistance_zones, key=lambda x: x.score, reverse=True)[:3]
    return support_zones, resistance_zones

def generate_trading_signals(data, support_zones, resistance_zones, ema_period=50, min_rr_ratio=1.5):
    signals = []
    analysis_details = []
    if data is None or len(data) < ema_period + 10:
        analysis_details.append("❌ Yetersiz veri")
        return signals, analysis_details
    try:
        current_price = float(data['Close'].iloc[-1])
        ema_value = float(data['EMA'].iloc[-1])
        rsi_value = float(data['RSI'].iloc[-1])
        atr_value = float(data['ATR'].iloc[-1])
        trend = "bull" if current_price > ema_value else "bear"
        ema_distance = abs(current_price - ema_value) / atr_value
        analysis_details.append(f"📈 TREND: {'YÜKSELİŞ' if trend == 'bull' else 'DÜŞÜŞ'}")
        analysis_details.append(f"📊 EMA{ema_period}: {format_price(ema_value)}")
        analysis_details.append(f"📍 Fiyat-EMA Mesafesi: {ema_distance:.2f} ATR")
        analysis_details.append(f"📉 RSI: {rsi_value:.1f}")
        analysis_details.append(f"📏 ATR: {format_price(atr_value)}")
        best_support = support_zones[0] if support_zones else None
        best_resistance = resistance_zones[0] if resistance_zones else None
        if best_support and best_support.score >= 65:
            entry = min(current_price, best_support.high)
            sl = best_support.low - 0.25 * atr_value
            risk_long = entry - sl
            tp1_long = entry + risk_long * (min_rr_ratio * 0.5)
            tp2_long = entry + risk_long * min_rr_ratio
            tp1, tp2 = sorted([tp1_long, tp2_long])
            rr = (tp2 - entry) / (entry - sl)
            if rr >= min_rr_ratio:
                explain = [f"EMA50 trend: {trend.upper()}", f"Zone validity: {best_support.status}", f"RSI: {rsi_value:.1f} - Support bölgesinde", f"RR kontrolü: {rr:.2f} ≥ {min_rr_ratio}"]
                signals.append({"type": "BUY", "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "rr": rr, "confidence": best_support.score, "zone": {"low": best_support.low, "high": best_support.high, "kind": "support"}, "trend": trend, "explain": explain})
        elif best_resistance and best_resistance.score >= 65:
            entry = max(current_price, best_resistance.low)
            sl = best_resistance.high + 0.25 * atr_value
            risk_short = sl - entry
            tp1_short = entry - risk_short * (min_rr_ratio * 0.5)
            tp2_short = entry - risk_short * min_rr_ratio
            tp1, tp2 = sorted([tp1_short, tp2_short], reverse=True)
            rr = (entry - tp2) / (sl - entry)
            if rr >= min_rr_ratio:
                explain = [f"EMA50 trend: {trend.upper()}", f"Zone validity: {best_resistance.status}", f"RSI: {rsi_value:.1f} - Resistance bölgesinde", f"RR kontrolü: {rr:.2f} ≥ {min_rr_ratio}"]
                signals.append({"type": "SELL", "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "rr": rr, "confidence": best_resistance.score, "zone": {"low": best_resistance.low, "high": best_resistance.high, "kind": "resistance"}, "trend": trend, "explain": explain})
        if not signals:
            wait_reasons = []
            if not best_support and not best_resistance:
                wait_reasons.append("Yeterli bölge bulunamadı")
            elif best_support and best_support.score < 65:
                wait_reasons.append(f"Destek skoru yetersiz: {best_support.score}")
            elif best_resistance and best_resistance.score < 65:
                wait_reasons.append(f"Direnç skoru yetersiz: {best_resistance.score}")
            elif ema_distance > 1.0:
                wait_reasons.append(f"EMA'dan uzak: {ema_distance:.2f} ATR")
            signals.append({"type": "WAIT", "entry": current_price, "sl": None, "tp1": None, "tp2": None, "rr": 0, "confidence": max((best_support.score if best_support else 0), (best_resistance.score if best_resistance else 0)), "zone": None, "trend": trend, "explain": wait_reasons})
        return signals, analysis_details
    except Exception as e:
        analysis_details.append(f"❌ Sinyal üretim hatası: {e}")
        return [], analysis_details

def create_clean_candlestick_chart(data, support_zones, resistance_zones, crypto_symbol, signals):
    fig = go.Figure()
    if data is None or len(data) == 0:
        return fig, [], [], [], []
    data_3days = data.tail(18)
    current_price = float(data_3days['Close'].iloc[-1])
    nearest_support = sorted(support_zones, key=lambda x: abs((x.low + x.high) / 2 - current_price))[:2]
    nearest_resistance = sorted(resistance_zones, key=lambda x: abs((x.low + x.high) / 2 - current_price))[:2]
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
    if 'EMA' in data_3days.columns:
        try:
            fig.add_trace(go.Scatter(x=data_3days.index, y=data_3days['EMA'], name=f'EMA{ema_period}', line=dict(color='orange', width=2), showlegend=False))
        except Exception:
            pass
    for i, zone in enumerate(nearest_support):
        border_color = "#FFA500" if zone.status == "fake" else "#7A7A7A" if zone.status == "broken" else "#00FF00"
        fig.add_hrect(y0=zone.low, y1=zone.high, fillcolor="rgba(0,255,0,0.12)", line=dict(width=1, color=border_color), layer="below")
        fig.add_annotation(x=data_3days.index[-1], y=(zone.low + zone.high) / 2, text=f"S{i+1}", showarrow=False, xanchor='left', yanchor='middle', font=dict(size=10, color="#00FF00"), bgcolor="rgba(0,0,0,0.5)")
    for i, zone in enumerate(nearest_resistance):
        border_color = "#FFA500" if zone.status == "fake" else "#7A7A7A" if zone.status == "broken" else "#FF0000"
        fig.add_hrect(y0=zone.low, y1=zone.high, fillcolor="rgba(255,0,0,0.12)", line=dict(width=1, color=border_color), layer="below")
        fig.add_annotation(x=data_3days.index[-1], y=(zone.low + zone.high) / 2, text=f"R{i+1}", showarrow=False, xanchor='left', yanchor='middle', font=dict(size=10, color="#FF0000"), bgcolor="rgba(0,0,0,0.5)")
    try:
        fig.add_hline(y=current_price, line_dash="dot", line_color="yellow", line_width=1, opacity=0.7, annotation_text=f"{format_price(current_price)}", annotation_position="left top", annotation_font_size=10, annotation_font_color="yellow")
    except (ValueError, IndexError):
        pass
    if signals and signals[0]["type"] != "WAIT":
        signal = signals[0]
        marker_symbol = "triangle-up" if signal["type"] == "BUY" else "triangle-down"
        marker_color = "#00FF00" if signal["type"] == "BUY" else "#FF0000"
        fig.add_trace(go.Scatter(x=[data_3days.index[-1]], y=[current_price], mode='markers', marker=dict(symbol=marker_symbol, size=12, color=marker_color, line=dict(width=2, color="white")), showlegend=False, name=f"{signal['type']} Sinyal"))
    fig.update_layout(height=500, title=f"{crypto_symbol} - 4H (Son 3 Gün)", xaxis_title="", yaxis_title="Fiyat (USD)", showlegend=False, xaxis_rangeslider_visible=False, plot_bgcolor='#0E1117', paper_bgcolor='#0E1117', font=dict(color='white', size=10), xaxis=dict(gridcolor='#444', showticklabels=True), yaxis=dict(gridcolor='#444'), margin=dict(l=50, r=50, t=50, b=50))
    return fig, nearest_support, nearest_resistance, support_zones, resistance_zones

# =============================================================================
# ANA UYGULAMA
# =============================================================================

def main():
    # Ana analiz için veri yükleme
    with st.spinner(f'⏳ {crypto_symbol} verileri yükleniyor...'):
        data_30days = get_4h_data(crypto_symbol, days=30)
    
    if data_30days is None or data_30days.empty:
        st.error(f"❌ {crypto_symbol} için veri yüklenemedi!")
        return
    
    # Göstergeleri hesapla
    data_30days = calculate_indicators(data_30days, ema_period, rsi_period)
    
    # Yoğunluk bölgelerini bul
    support_zones, resistance_zones = find_congestion_zones(
        data_30days, min_touch_points, analysis_lookback_bars
    )
    
    # Sinyal üret
    signals, analysis_details = generate_trading_signals(
        data_30days, support_zones, resistance_zones, ema_period, risk_reward_ratio
    )
    
    # Mevcut durum
    try:
        current_price = float(data_30days['Close'].iloc[-1])
        ema_value = float(data_30days['EMA'].iloc[-1])
        rsi_value = float(data_30days['RSI'].iloc[-1])
        atr_value = float(data_30days['ATR'].iloc[-1])
        trend = "bull" if current_price > ema_value else "bear"
    except (ValueError, IndexError):
        current_price = 0
        ema_value = 0
        rsi_value = 0
        atr_value = 0
        trend = "neutral"
    
    # Layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Sadeleştirilmiş mum grafiği
        chart_fig, nearest_support, nearest_resistance, all_support, all_resistance = create_clean_candlestick_chart(
            data_30days, support_zones, resistance_zones, crypto_symbol, signals
        )
        st.plotly_chart(chart_fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Sinyal")
        
        # Sinyal kartı
        if signals and signals[0]["type"] != "WAIT":
            signal = signals[0]
            signal_color = "🟢" if signal['type'] == 'BUY' else "🔴"
            
            st.markdown(f"### {signal_color} {signal['type']}")
            
            cols = st.columns(2)
            with cols[0]:
                st.metric("Giriş", format_price(signal['entry']))
                st.metric("TP1", format_price(signal['tp1']))
            with cols[1]:
                st.metric("SL", format_price(signal['sl']))
                st.metric("TP2", format_price(signal['tp2']))
            
            st.metric("R/R", f"{signal['rr']:.2f}")
            st.metric("Güven", f"%{signal['confidence']}")
            
        else:
            st.markdown("### ⚪ BEKLE")
            st.info("Koşullar uygun değil")
        
        st.divider()
        
        # Trend ve gösterge
        st.subheader("📈 Trend")
        trend_icon = "🟢" if trend == "bull" else "🔴"
        st.metric("EMA50", trend_icon + " " + ("YÜKSELİŞ" if trend == "bull" else "DÜŞÜŞ"))
        st.metric("RSI", f"{rsi_value:.1f}")
        
        st.divider()
        
        # Yakın bantlar
        st.subheader("🎯 Yakın Bantlar")
        
        for i, zone in enumerate(nearest_support):
            st.write(f"**S{i+1}:** {format_price(zone.low)}-{format_price(zone.high)}")
            st.caption(f"Skor: {zone.score}")
        
        for i, zone in enumerate(nearest_resistance):
            st.write(f"**R{i+1}:** {format_price(zone.low)}-{format_price(zone.high)}")
            st.caption(f"Skor: {zone.score}")
    
    # YENİ: BACKTEST SONUÇLARI
    # Backtest kısmını değiştirin:
if run_bt:
    st.divider()
    st.header("📊 Backtest Sonuçları (90 Gün)")
    
    with st.spinner("Backtest çalışıyor..."):
        df_90d = get_4h_data(crypto_symbol, days=90)
        if df_90d is not None and not df_90d.empty:
            df_90d = calculate_indicators(df_90d, ema_period, rsi_period)
            
            # Optimize edilmiş fonksiyonu kullan
            report, trades_df, eq_df, dd_df = backtest_90d_optimized(
                df_90d, 
                risk_perc=risk_perc, 
                fee=fee, 
                slip=slip, 
                partial=partial,
                ema_period=ema_period, 
                min_rr_ratio=risk_reward_ratio, 
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
                col1, col2 = st.columns(2)
                
                with col1:
                    if not eq_df.empty:
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
                if not trades_df.empty:
                    st.subheader("İşlem Listesi")
                    display_cols = ["open_time", "side", "entry", "sl", "tp1", "tp2", "exit_price", "exit_reason", "r_multiple", "pnl"]
                    display_df = trades_df[display_cols].copy()
                    display_df["entry"] = display_df["entry"].apply(format_price)
                    display_df["sl"] = display_df["sl"].apply(format_price)
                    display_df["tp1"] = display_df["tp1"].apply(format_price)
                    display_df["tp2"] = display_df["tp2"].apply(format_price)
                    display_df["exit_price"] = display_df["exit_price"].apply(format_price)
                    display_df["r_multiple"] = display_df["r_multiple"].round(2)
                    display_df["pnl"] = display_df["pnl"].round(2)
                    
                    st.dataframe(
                        display_df,
                        column_config={
                            "open_time": "Açılış",
                            "side": "Yön",
                            "entry": "Giriş",
                            "sl": "SL",
                            "tp1": "TP1", 
                            "tp2": "TP2",
                            "exit_price": "Çıkış",
                            "exit_reason": "Sebep",
                            "r_multiple": "R Multiple",
                            "pnl": "PnL ($)"
                        },
                        use_container_width=True
                    )
            else:
                st.error("Backtest için veri yüklenemedi!")
    
    # Detaylı bant listesi
    with st.expander("📋 Tüm Bant Detayları"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Destek Bantları**")
            for i, zone in enumerate(all_support):
                status_icon = "🟢" if zone.status == "valid" else "🟠" if zone.status == "fake" else "⚫"
                st.write(f"{status_icon} S{i+1}: {format_price(zone.low)}-{format_price(zone.high)}")
                st.caption(f"Skor: {zone.score}, Temas: {zone.touches}")
        
        with col2:
            st.write("**Direnç Bantları**")
            for i, zone in enumerate(all_resistance):
                status_icon = "🔴" if zone.status == "valid" else "🟠" if zone.status == "fake" else "⚫"
                st.write(f"{status_icon} R{i+1}: {format_price(zone.low)}-{format_price(zone.high)}")
                st.caption(f"Skor: {zone.score}, Temas: {zone.touches}")

if __name__ == "__main__":
    main()