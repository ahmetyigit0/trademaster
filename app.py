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
    """1D rejimini 4H verisine map et - BASİT VE ETKİLİ"""
    if df_1d is None or len(df_1d) < 50:
        # Fallback: EMA50 bazlı basit rejim
        try:
            ema50 = df_4h['Close'].ewm(span=50, adjust=False).mean()
            price_vs_ema = (df_4h['Close'] - ema50) / df_4h['Close'] * 100
            regime = np.where(price_vs_ema > 1, 'UP', 
                             np.where(price_vs_ema < -1, 'DOWN', 'RANGE'))
            return df_4h.assign(REGIME=regime)
        except:
            return df_4h.assign(REGIME='RANGE')
    
    try:
        # Basit rejim belirleme: Son 5 günün ortalaması
        recent_1d = df_1d.tail(5)
        avg_close = recent_1d['Close'].mean()
        avg_high = recent_1d['High'].mean()
        avg_low = recent_1d['Low'].mean()
        
        current_price = df_4h['Close'].iloc[-1]
        
        # Trend belirleme
        if current_price > avg_high * 0.99:
            regime = 'UP'
        elif current_price < avg_low * 1.01:
            regime = 'DOWN'
        else:
            regime = 'RANGE'
        
        # Tüm 4H verisine aynı rejimi uygula
        return df_4h.assign(REGIME=regime)
        
    except Exception as e:
        return df_4h.assign(REGIME='RANGE')

def donchian(df, n=20):
    """Donchian Channel"""
    return df['High'].rolling(n, min_periods=1).max(), df['Low'].rolling(n, min_periods=1).min()

def bollinger(df, n=20, k=2):
    """Bollinger Bands"""
    mid = df['Close'].rolling(n, min_periods=1).mean()
    std = df['Close'].rolling(n, min_periods=1).std().fillna(0.1)
    return mid, mid + k * std, mid - k * std

def calculate_advanced_indicators(df):
    """İleri teknik göstergeler - GÜVENLİ"""
    df = df.copy()
    
    try:
        # Donchian Channel
        donch_high, donch_low = donchian(df, 20)
        df['DONCH_HIGH'] = donch_high
        df['DONCH_LOW'] = donch_low
        
        # Bollinger Bands
        bb_mid, bb_upper, bb_lower = bollinger(df, 20, 2)
        df['BB_MID'] = bb_mid
        df['BB_UPPER'] = bb_upper
        df['BB_LOWER'] = bb_lower
        
        # NaN değerleri temizle
        for col in ['DONCH_HIGH', 'DONCH_LOW', 'BB_MID', 'BB_UPPER', 'BB_LOWER']:
            df[col] = df[col].fillna(method='bfill').fillna(method='ffill')
        
    except Exception as e:
        # Hata durumunda basit değerler
        current_price = df['Close'].iloc[-1] if len(df) > 0 else 100
        for col in ['DONCH_HIGH', 'BB_UPPER']:
            df[col] = current_price * 1.02
        for col in ['DONCH_LOW', 'BB_LOWER']:
            df[col] = current_price * 0.98
        df['BB_MID'] = current_price
    
    return df

def get_regime(symbol, df_4h):
    """Rejim hesapla ve 4H verisine ekle"""
    try:
        df_1d = get_1d_data(symbol, days=120)
        result_df = map_regime_to_4h(df_4h, df_1d)
        return result_df
    except Exception as e:
        # Hata durumunda basit rejim
        return df_4h.assign(REGIME='RANGE')

def generate_signals_v2(df, regime_col='REGIME', min_rr_ratio=1.5, cooldown_bars=3, bb_width_pct=2.5, donchian_len=20):
    """
    Yeni rejim-temelli strateji sinyalleri - GELİŞTİRİLMİŞ
    """
    if len(df) < 30:
        return {"type": "WAIT", "reason": "Yetersiz veri", "strat_id": "NONE"}
    
    try:
        current_idx = df.index[-1]
        current_data = df.iloc[-1]
        current_price = float(current_data['Close'])
        
        # Gerekli göstergeleri kontrol et
        required_cols = ['RSI', 'ATR', 'DONCH_HIGH', 'DONCH_LOW', 'BB_UPPER', 'BB_LOWER', 'BB_MID']
        for col in required_cols:
            if col not in current_data or pd.isna(current_data[col]):
                return {"type": "WAIT", "reason": f"{col} göstergesi hazır değil", "strat_id": "NONE"}
        
        regime = current_data.get(regime_col, 'RANGE')
        atr = float(current_data['ATR'])
        rsi = float(current_data['RSI'])
        donch_high = float(current_data['DONCH_HIGH'])
        donch_low = float(current_data['DONCH_LOW'])
        bb_upper = float(current_data['BB_UPPER'])
        bb_lower = float(current_data['BB_LOWER'])
        bb_mid = float(current_data['BB_MID'])
        
        # DEBUG: Değerleri kontrol et
        debug_info = f"Price: {current_price:.2f}, RSI: {rsi:.1f}, Regime: {regime}, Donch_H: {donch_high:.2f}, Donch_L: {donch_low:.2f}"
        
        # Strateji A: Uptrend - Momentum Breakout
        if regime == 'UP':
            # Donchian breakout + RSI filtresi
            if current_price >= donch_high and rsi < 75:
                sl = max(donch_low, bb_lower, current_price * 0.98)
                risk = current_price - sl
                if risk > 0 and risk / current_price < 0.03:  # Max %3 risk
                    tp1 = current_price + risk * (min_rr_ratio * 0.7)
                    tp2 = current_price + risk * min_rr_ratio
                    rr = (tp2 - current_price) / risk
                    if rr >= min_rr_ratio:
                        return {
                            "type": "BUY", "entry": current_price, "sl": sl, 
                            "tp1": tp1, "tp2": tp2, "rr": rr, 
                            "reason": f"Uptrend Breakout - {debug_info}", "strat_id": "A"
                        }
        
        # Strateji B: Downtrend - Momentum Breakdown  
        elif regime == 'DOWN':
            # Donchian breakdown + RSI filtresi
            if current_price <= donch_low and rsi > 25:
                sl = min(donch_high, bb_upper, current_price * 1.02)
                risk = sl - current_price
                if risk > 0 and risk / current_price < 0.03:  # Max %3 risk
                    tp1 = current_price - risk * (min_rr_ratio * 0.7)
                    tp2 = current_price - risk * min_rr_ratio
                    rr = (current_price - tp2) / risk
                    if rr >= min_rr_ratio:
                        return {
                            "type": "SELL", "entry": current_price, "sl": sl, 
                            "tp1": tp1, "tp2": tp2, "rr": rr,
                            "reason": f"Downtrend Breakdown - {debug_info}", "strat_id": "B"
                        }
        
        # Strateji C: Range - Mean Reversion
        elif regime == 'RANGE':
            # Bollinger Band bounce
            bb_width = (bb_upper - bb_lower) / bb_mid * 100
            
            # Sadece dar bantlarda işlem (volatilite düşük)
            if bb_width < 5:  # %5'ten dar bant
                # Üst band direnç - Short
                if current_price >= bb_upper * 0.995 and rsi > 65:
                    sl = bb_upper * 1.015
                    risk = sl - current_price
                    if risk > 0 and risk / current_price < 0.02:
                        tp1 = bb_mid
                        tp2 = bb_lower
                        rr = (current_price - tp2) / risk
                        if rr >= min_rr_ratio:
                            return {
                                "type": "SELL", "entry": current_price, "sl": sl, 
                                "tp1": tp1, "tp2": tp2, "rr": rr,
                                "reason": f"Range Resistance - {debug_info}", "strat_id": "C"
                            }
                
                # Alt band destek - Long
                elif current_price <= bb_lower * 1.005 and rsi < 35:
                    sl = bb_lower * 0.985
                    risk = current_price - sl
                    if risk > 0 and risk / current_price < 0.02:
                        tp1 = bb_mid
                        tp2 = bb_upper
                        rr = (tp2 - current_price) / risk
                        if rr >= min_rr_ratio:
                            return {
                                "type": "BUY", "entry": current_price, "sl": sl, 
                                "tp1": tp1, "tp2": tp2, "rr": rr,
                                "reason": f"Range Support - {debug_info}", "strat_id": "C"
                            }
        
        # Aşırı uzama filtresi
        if rsi > 85 or rsi < 15:
            return {"type": "WAIT", "reason": f"Aşırı uzama (RSI:{rsi:.1f})", "strat_id": "NONE"}
        
        return {"type": "WAIT", "reason": f"Koşullar uygun değil - {debug_info}", "strat_id": "NONE"}
    
    except Exception as e:
        return {"type": "WAIT", "reason": f"Hata: {str(e)}", "strat_id": "NONE"}

# =============================================================================
# BACKTEST SİSTEMİ - GELİŞTİRİLMİŞ
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
    Güncellenmiş backtest - DAHA FAZLA SİNYAL ÜRETECEK ŞEKİLDE
    """
    balance = start_balance
    trades = []
    equity = [balance]
    dd_series = [0.0]
    
    min_lookback = 50  # Daha az lookback ile daha fazla sinyal
    data_length = len(df_90d)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Sinyal sayacı
    signals_generated = 0
    last_signal_time = None
    
    for i in range(min_lookback, data_length - 1):
        if i % 10 == 0:
            progress = (i - min_lookback) / (data_length - min_lookback - 1)
            progress_bar.progress(progress)
            status_text.text(f"Backtest çalışıyor... {signals_generated} sinyal - %{int(progress * 100)}")
        
        try:
            df_slice = df_90d.iloc[:i+1].copy()
            
            # Cooldown kontrolü
            current_time = df_90d.index[i]
            if last_signal_time is not None:
                bars_passed = (current_time - last_signal_time) / pd.Timedelta(hours=4)
                if bars_passed < cooldown_bars:
                    equity.append(balance)
                    current_equity = equity[-1]
                    peak_equity = max(equity)
                    drawdown = ((current_equity - peak_equity) / peak_equity) * 100 if peak_equity > 0 else 0
                    dd_series.append(drawdown)
                    continue
            
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

            # Sinyal bulundu
            signals_generated += 1
            last_signal_time = current_time
            
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
            
            max_lookahead = min(open_index + 100, data_length)  # Daha uzun takip
            
            for j in range(open_index, max_lookahead):
                bar = df_90d.iloc[j]
                high, low = float(bar['High']), float(bar['Low'])
                close = float(bar['Close'])

                if side == "LONG":
                    # TP2 kontrolü (öncelikli)
                    if high >= tp2:
                        exit_reason = "TP2"
                        exit_price = tp2
                        pnl = (tp2 - entry) * qty
                        exit_found = True
                        break
                    # SL kontrolü
                    elif low <= sl:
                        exit_reason = "SL"
                        exit_price = sl
                        pnl = (sl - entry) * qty
                        exit_found = True
                        break
                    # Kısmi TP1
                    elif partial and high >= tp1:
                        # %50 kısmi çıkış
                        partial_qty = qty * 0.5
                        remaining_qty = qty * 0.5
                        partial_pnl = (tp1 - entry) * partial_qty
                        
                        # Kalan için BE'ye çek ve TP2'yi bekle
                        new_sl = entry  # Break-even
                        
                        for k in range(j + 1, max_lookahead):
                            bar2 = df_90d.iloc[k]
                            high2, low2 = float(bar2['High']), float(bar2['Low'])
                            
                            if low2 <= new_sl:
                                exit_reason = "SL (Partial)"
                                exit_price = new_sl
                                pnl = partial_pnl + (new_sl - entry) * remaining_qty
                                exit_found = True
                                j = k
                                break
                            elif high2 >= tp2:
                                exit_reason = "TP2 (Partial)"
                                exit_price = tp2
                                pnl = partial_pnl + (tp2 - entry) * remaining_qty
                                exit_found = True
                                j = k
                                break
                            elif k == max_lookahead - 1:
                                exit_reason = "Time (Partial)"
                                exit_price = float(bar2['Close'])
                                pnl = partial_pnl + (exit_price - entry) * remaining_qty
                                exit_found = True
                                j = k
                                break
                        
                        if exit_found:
                            break
                        
                else:  # SHORT
                    # TP2 kontrolü (öncelikli)
                    if low <= tp2:
                        exit_reason = "TP2"
                        exit_price = tp2
                        pnl = (entry - tp2) * qty
                        exit_found = True
                        break
                    # SL kontrolü
                    elif high >= sl:
                        exit_reason = "SL"
                        exit_price = sl
                        pnl = (entry - sl) * qty
                        exit_found = True
                        break
                    # Kısmi TP1
                    elif partial and low <= tp1:
                        # %50 kısmi çıkış
                        partial_qty = qty * 0.5
                        remaining_qty = qty * 0.5
                        partial_pnl = (entry - tp1) * partial_qty
                        
                        # Kalan için BE'ye çek ve TP2'yi bekle
                        new_sl = entry  # Break-even
                        
                        for k in range(j + 1, max_lookahead):
                            bar2 = df_90d.iloc[k]
                            high2, low2 = float(bar2['High']), float(bar2['Low'])
                            
                            if high2 >= new_sl:
                                exit_reason = "SL (Partial)"
                                exit_price = new_sl
                                pnl = partial_pnl + (entry - new_sl) * remaining_qty
                                exit_found = True
                                j = k
                                break
                            elif low2 <= tp2:
                                exit_reason = "TP2 (Partial)"
                                exit_price = tp2
                                pnl = partial_pnl + (entry - tp2) * remaining_qty
                                exit_found = True
                                j = k
                                break
                            elif k == max_lookahead - 1:
                                exit_reason = "Time (Partial)"
                                exit_price = float(bar2['Close'])
                                pnl = partial_pnl + (entry - exit_price) * remaining_qty
                                exit_found = True
                                j = k
                                break
                        
                        if exit_found:
                            break

            # Çıkış yoksa son fiyattan kapat
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
    
    # DEBUG: Sinyal bilgisi
    st.info(f"Toplam {signals_generated} sinyal üretildi, {len(trades)} işlem yapıldı")
    
    # Metrikler
    if len(equity) > 0:
        eq_series = pd.Series(equity)
        returns = eq_series.pct_change().fillna(0)
        
        wins = [t for t in trades if t.pnl is not None and t.pnl > 0]
        losses = [t for t in trades if t.pnl is not None and t.pnl <= 0]
        
        total_trades = len(trades)
        win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
        
        # R-multiple bazlı metrikler
        win_r = [t.r_multiple for t in trades if t.r_multiple is not None and t.r_multiple > 0]
        loss_r = [t.r_multiple for t in trades if t.r_multiple is not None and t.r_multiple <= 0]
        
        avg_win_r = np.mean(win_r) if win_r else 0
        avg_loss_r = np.mean(loss_r) if loss_r else 0
        
        total_win_pnl = sum([t.pnl for t in wins]) if wins else 0
        total_loss_pnl = abs(sum([t.pnl for t in losses])) if losses else 0
        profit_factor = total_win_pnl / total_loss_pnl if total_loss_pnl > 0 else float('inf')
        
        expectancy_r = (win_rate/100) * avg_win_r - ((100 - win_rate)/100) * abs(avg_loss_r)
        
        max_drawdown = min(dd_series) if dd_series else 0
        
        sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(365 * 6) if len(returns) > 1 else 0
        
        # Strateji bazlı analiz
        strat_stats = {}
        for trade in trades:
            strat = trade.strat_id
            if strat not in strat_stats:
                strat_stats[strat] = {'count': 0, 'wins': 0, 'pnl': 0}
            strat_stats[strat]['count'] += 1
            strat_stats[strat]['pnl'] += trade.pnl if trade.pnl else 0
            if trade.pnl and trade.pnl > 0:
                strat_stats[strat]['wins'] += 1
        
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
            "total_return_pct": ((balance - start_balance) / start_balance) * 100,
            "strat_stats": strat_stats,
            "signals_generated": signals_generated
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
        "final_balance": start_balance, "total_return_pct": 0,
        "strat_stats": {}, "signals_generated": signals_generated
    }
    return empty_report, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Kalan kod aynı şekilde devam ediyor...
# [Önceki koddaki UI ve diğer fonksiyonlar buraya gelecek]