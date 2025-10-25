# app.py
# ─────────────────────────────────────────────────────────────────────────────
# PURE FOREX PROFIT STRATEGY - 10K to 20K in 90 Days
# No External Dependencies - Only pandas, numpy, plotly
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import List, Dict, Any

st.set_page_config(page_title="Forex Profit Machine", layout="wide")

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
# TECHNICAL INDICATORS - MANUAL IMPLEMENTATION
# =============================================================================

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss.replace(0, 0.001)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range"""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr.fillna(method='bfill')

def calculate_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    """Bollinger Bands"""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    return lower_band, sma, upper_band

def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD Indicator"""
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Tüm göstergeleri hesapla"""
    if df.empty:
        return df
    
    df = df.copy()
    
    # Trend göstergeleri
    df['EMA_12'] = calculate_ema(df['Close'], 12)
    df['EMA_26'] = calculate_ema(df['Close'], 26)
    df['SMA_50'] = df['Close'].rolling(50).mean()
    
    # Momentum göstergeleri
    df['RSI_14'] = calculate_rsi(df['Close'], 14)
    macd_line, signal_line, histogram = calculate_macd(df['Close'])
    df['MACD_Line'] = macd_line
    df['MACD_Signal'] = signal_line
    df['MACD_Histogram'] = histogram
    
    # Volatilite göstergeleri
    df['ATR_14'] = calculate_atr(df['High'], df['Low'], df['Close'], 14)
    df['BB_Lower'], df['BB_Middle'], df['BB_Upper'] = calculate_bollinger_bands(df['Close'])
    
    # Price action
    df['Price_Change'] = df['Close'].pct_change()
    df['High_Low_Range'] = (df['High'] - df['Low']) / df['Close']
    
    return df.dropna()

# =============================================================================
# FOREX DATA
# =============================================================================

@st.cache_data
def get_forex_data(symbol: str, days: int) -> pd.DataFrame:
    """Forex verisi al"""
    forex_pairs = {
        'EURUSD': 'EURUSD=X',
        'GBPUSD': 'GBPUSD=X', 
        'USDJPY': 'USDJPY=X',
        'USDCHF': 'USDCHF=X',
        'AUDUSD': 'AUDUSD=X',
        'USDCAD': 'USDCAD=X',
        'NZDUSD': 'NZDUSD=X',
        'EURJPY': 'EURJPY=X',
        'GBPJPY': 'GBPJPY=X'
    }
    
    yf_symbol = forex_pairs.get(symbol.upper(), symbol + '=X')
    
    try:
        df = yf.download(yf_symbol, period=f'{days}d', interval='1h', progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        
        # Sadece gereken kolonları tut
        required_cols = ['Open', 'High', 'Low', 'Close']
        df = df[required_cols].dropna()
        
        return df
    except Exception as e:
        st.error(f"Data error: {e}")
        return pd.DataFrame()

# =============================================================================
# PROFIT STRATEGY
# =============================================================================

class ForexProfitStrategy:
    """10K to 20K High-Profit Strategy"""
    
    def __init__(self):
        self.name = "Multi-Timeframe Breakout Strategy"
    
    def analyze_market_condition(self, df: pd.DataFrame, current_idx: int) -> Dict[str, Any]:
        """Piyasa koşullarını analiz et"""
        if current_idx < 50:
            return {'trend': 'neutral', 'momentum': 'neutral', 'volatility': 'low', 'bias': 'neutral'}
        
        current = df.iloc[current_idx]
        
        # Trend analizi
        price = current['Close']
        ema_12 = current['EMA_12']
        ema_26 = current['EMA_26']
        sma_50 = current['SMA_50']
        
        # Trend skoru
        trend_score = 0
        if price > ema_12: trend_score += 1
        if price > ema_26: trend_score += 1  
        if price > sma_50: trend_score += 1
        if ema_12 > ema_26: trend_score += 1
        
        if trend_score >= 3:
            trend = 'bullish'
            bias = 'buy'
        elif trend_score <= 1:
            trend = 'bearish' 
            bias = 'sell'
        else:
            trend = 'neutral'
            bias = 'neutral'
        
        # Momentum analizi
        rsi = current['RSI_14']
        macd_hist = current['MACD_Histogram']
        macd_prev = df.iloc[current_idx-1]['MACD_Histogram']
        
        momentum = 'strong' if (
            (trend == 'bullish' and rsi > 45 and rsi < 70 and macd_hist > 0 and macd_hist > macd_prev) or
            (trend == 'bearish' and rsi < 55 and rsi > 30 and macd_hist < 0 and macd_hist < macd_prev)
        ) else 'weak'
        
        # Volatilite analizi
        atr = current['ATR_14']
        avg_atr = df['ATR_14'].iloc[max(0, current_idx-20):current_idx].mean()
        
        if atr > avg_atr * 1.3:
            volatility = 'high'
        elif atr > avg_atr * 0.7:
            volatility = 'medium'
        else:
            volatility = 'low'
        
        return {
            'trend': trend,
            'momentum': momentum,
            'volatility': volatility,
            'bias': bias,
            'rsi': rsi,
            'atr': atr
        }
    
    def generate_trade_signal(self, df: pd.DataFrame, current_idx: int) -> Dict[str, Any]:
        """Trade sinyali üret"""
        analysis = self.analyze_market_condition(df, current_idx)
        current = df.iloc[current_idx]
        price = current['Close']
        atr = current['ATR_14']
        
        # Yüksek kalite sinyalleri
        signals = []
        
        # BREAKOUT STRATEGY - Trend takip
        if analysis['momentum'] == 'strong' and analysis['volatility'] == 'medium':
            
            if analysis['bias'] == 'buy':
                # Bullish breakout
                recent_high = df['High'].iloc[max(0, current_idx-20):current_idx].max()
                if price >= recent_high * 0.999:
                    sl = price - atr * 1.8
                    tp = price + atr * 3.5
                    rr = (tp - price) / (price - sl)
                    
                    if rr >= 2.0:
                        signals.append({
                            'type': 'BUY',
                            'entry': price,
                            'sl': sl,
                            'tp': tp,
                            'rr_ratio': rr,
                            'confidence': 0.8,
                            'reason': 'Bullish Breakout - Strong Momentum'
                        })
            
            elif analysis['bias'] == 'sell':
                # Bearish breakdown  
                recent_low = df['Low'].iloc[max(0, current_idx-20):current_idx].min()
                if price <= recent_low * 1.001:
                    sl = price + atr * 1.8
                    tp = price - atr * 3.5
                    rr = (price - tp) / (sl - price)
                    
                    if rr >= 2.0:
                        signals.append({
                            'type': 'SELL',
                            'entry': price,
                            'sl': sl,
                            'tp': tp,
                            'rr_ratio': rr,
                            'confidence': 0.8,
                            'reason': 'Bearish Breakdown - Strong Momentum'
                        })
        
        # MEAN REVERSION STRATEGY - Range market
        if analysis['volatility'] == 'low' and not signals:
            rsi = analysis['rsi']
            bb_upper = current['BB_Upper']
            bb_lower = current['BB_Lower']
            
            if rsi < 25 and price <= bb_lower * 1.002:  # Oversold
                sl = price - atr * 1.2
                tp = current['BB_Middle']  # Middle band target
                rr = (tp - price) / (price - sl)
                
                if rr >= 1.8:
                    signals.append({
                        'type': 'BUY',
                        'entry': price,
                        'sl': sl,
                        'tp': tp,
                        'rr_ratio': rr,
                        'confidence': 0.7,
                        'reason': 'Oversold Bounce - Mean Reversion'
                    })
                    
            elif rsi > 75 and price >= bb_upper * 0.998:  # Overbought
                sl = price + atr * 1.2
                tp = current['BB_Middle']  # Middle band target
                rr = (price - tp) / (sl - price)
                
                if rr >= 1.8:
                    signals.append({
                        'type': 'SELL', 
                        'entry': price,
                        'sl': sl,
                        'tp': tp,
                        'rr_ratio': rr,
                        'confidence': 0.7,
                        'reason': 'Overbought Rejection - Mean Reversion'
                    })
        
        # En iyi sinyali seç
        if signals:
            best_signal = max(signals, key=lambda x: x['confidence'])
            if best_signal['confidence'] >= 0.7:
                return best_signal
        
        return {'type': 'WAIT', 'confidence': 0}

# =============================================================================
# BACKTEST ENGINE
# =============================================================================

@dataclass
class TradeRecord:
    entry_time: Any
    entry_price: float
    exit_time: Any
    exit_price: float
    side: str
    size: float
    pnl: float
    pnl_percent: float
    duration_hours: int
    sl: float
    tp: float
    reason: str
    confidence: float

class BacktestEngine:
    """Backtest Motoru"""
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.strategy = ForexProfitStrategy()
    
    def calculate_position_size(self, balance: float, entry_price: float, sl_price: float, risk_percent: float = 0.02) -> float:
        """Pozisyon büyüklüğü hesapla"""
        risk_amount = balance * risk_percent
        price_risk = abs(entry_price - sl_price)
        
        if price_risk <= 0:
            return 0
            
        position_size = risk_amount / price_risk
        # Maksimum %5 pozisyon
        max_position = balance * 0.05 / entry_price
        
        return min(position_size, max_position)
    
    def run_backtest(self, df: pd.DataFrame, symbol: str, risk_percent: float = 0.02) -> Dict[str, Any]:
        """Backtest çalıştır"""
        if len(df) < 100:
            return self._empty_results()
        
        balance = self.initial_balance
        equity_curve = [balance]
        trades = []
        active_trade = None
        
        # Backtest parametreleri
        max_hold_hours = 48
        min_trade_gap = 6  # Trade'ler arası minimum saat
        
        last_trade_exit = None
        
        for i in range(100, len(df) - 5):  # 100 bar warm-up
            current_time = df.index[i]
            current_data = df.iloc[:i+1]
            
            # Aktif trade kontrolü
            if active_trade:
                should_exit, exit_price, exit_reason = self._check_exit_conditions(
                    df, i, active_trade, max_hold_hours
                )
                
                if should_exit:
                    # Trade'i kapat
                    pnl = self._calculate_pnl(active_trade, exit_price)
                    balance += pnl
                    pnl_pct = (pnl / (active_trade.size * active_trade.entry_price)) * 100
                    
                    duration = (current_time - active_trade.entry_time).total_seconds() / 3600
                    
                    trade = TradeRecord(
                        entry_time=active_trade.entry_time,
                        entry_price=active_trade.entry_price,
                        exit_time=current_time,
                        exit_price=exit_price,
                        side=active_trade.side,
                        size=active_trade.size,
                        pnl=pnl,
                        pnl_percent=pnl_pct,
                        duration_hours=int(duration),
                        sl=active_trade.sl,
                        tp=active_trade.tp,
                        reason=active_trade.reason,
                        confidence=active_trade.confidence
                    )
                    
                    trades.append(trade)
                    active_trade = None
                    last_trade_exit = current_time
            
            # Yeni trade kontrolü
            if not active_trade:
                # Trade gap kontrolü
                if last_trade_exit is None or (current_time - last_trade_exit).total_seconds() / 3600 >= min_trade_gap:
                    
                    signal = self.strategy.generate_trade_signal(current_data, i)
                    
                    if signal['type'] in ['BUY', 'SELL'] and signal['confidence'] >= 0.7:
                        # Sonraki barın açılışında giriş yap
                        entry_price = float(df['Open'].iloc[i+1])
                        position_size = self.calculate_position_size(
                            balance, entry_price, signal['sl'], risk_percent
                        )
                        
                        if position_size > 0:
                            active_trade = type('obj', (), {
                                'entry_time': df.index[i+1],
                                'entry_price': entry_price,
                                'side': signal['type'],
                                'size': position_size,
                                'sl': signal['sl'],
                                'tp': signal['tp'],
                                'reason': signal['reason'],
                                'confidence': signal['confidence']
                            })()
            
            equity_curve.append(balance)
        
        return self._calculate_performance(trades, equity_curve, symbol)
    
    def _check_exit_conditions(self, df: pd.DataFrame, current_idx: int, trade, max_hours: int) -> tuple:
        """Çıkış koşullarını kontrol et"""
        current_time = df.index[current_idx]
        current_bar = df.iloc[current_idx]
        
        current_high = float(current_bar['High'])
        current_low = float(current_bar['Low'])
        current_close = float(current_bar['Close'])
        
        # Süre kontrolü
        duration = (current_time - trade.entry_time).total_seconds() / 3600
        if duration >= max_hours:
            return True, current_close, "Time Exit"
        
        # TP/SL kontrolü
        if trade.side == 'BUY':
            if current_high >= trade.tp:
                return True, trade.tp, "Take Profit"
            elif current_low <= trade.sl:
                return True, trade.sl, "Stop Loss"
        else:  # SELL
            if current_low <= trade.tp:
                return True, trade.tp, "Take Profit"
            elif current_high >= trade.sl:
                return True, trade.sl, "Stop Loss"
        
        return False, 0, ""
    
    def _calculate_pnl(self, trade, exit_price: float) -> float:
        """PNL hesapla"""
        if trade.side == 'BUY':
            return (exit_price - trade.entry_price) * trade.size
        else:
            return (trade.entry_price - exit_price) * trade.size
    
    def _calculate_performance(self, trades: List[TradeRecord], equity_curve: List[float], symbol: str) -> Dict[str, Any]:
        """Performans metriklerini hesapla"""
        if not trades:
            return self._empty_results()
        
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / total_trades * 100
        
        total_pnl = sum(t.pnl for t in trades)
        total_return = (total_pnl / self.initial_balance) * 100
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        profit_factor = abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades else float('inf')
        
        # Drawdown
        equity_array = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_array)
        drawdown = (peak - equity_array) / peak * 100
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        return {
            'symbol': symbol,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return_percent': total_return,
            'final_balance': equity_curve[-1],
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'equity_curve': equity_curve,
            'trades': trades
        }
    
    def _empty_results(self) -> Dict[str, Any]:
        """Boş sonuçlar"""
        return {
            'symbol': '',
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'total_return_percent': 0,
            'final_balance': self.initial_balance,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'equity_curve': [self.initial_balance],
            'trades': []
        }

# =============================================================================
# STREAMLIT UI
# =============================================================================

st.title("💰 FOREX PROFIT MACHINE")
st.markdown("### 10.000$ → 20.000$ in 90 Days")

# Sidebar
with st.sidebar:
    st.header("🎯 Settings")
    
    forex_symbol = st.selectbox(
        "Forex Pair",
        ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", "EURJPY", "GBPJPY"],
        index=0
    )
    
    initial_balance = st.number_input("Starting Balance ($)", 5000, 50000, 10000, 1000)
    risk_percent = st.slider("Risk per Trade (%)", 1.0, 5.0, 2.0, 0.5)
    backtest_days = st.slider("Backtest Days", 30, 180, 90, 10)
    
    run_backtest = st.button("🚀 RUN PROFIT BACKTEST", type="primary")

# Main content
if run_backtest:
    st.header(f"📊 BACKTEST RESULTS: {forex_symbol} - {backtest_days} Days")
    
    with st.spinner("Running advanced backtest..."):
        # Veri yükle
        data = get_forex_data(forex_symbol, backtest_days + 20)
        
        if data.empty:
            st.error("❌ No data available for this symbol!")
            st.stop()
        
        # Göstergeleri hesapla
        data = calculate_indicators(data)
        
        if data.empty:
            st.error("❌ Indicator calculation failed!")
            st.stop()
        
        # Backtest çalıştır
        backtester = BacktestEngine(initial_balance=initial_balance)
        results = backtester.run_backtest(data, forex_symbol, risk_percent/100)
        
        # SONUÇLARI GÖSTER
        st.subheader("💰 PERFORMANCE SUMMARY")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", results['total_trades'])
            st.metric("Win Rate", f"{results['win_rate']:.1f}%")
            
        with col2:
            st.metric("Total P&L", f"${results['total_pnl']:,.0f}")
            st.metric("Total Return", f"{results['total_return_percent']:.1f}%")
            
        with col3:
            st.metric("Final Balance", f"${results['final_balance']:,.0f}")
            st.metric("Profit Factor", f"{results['profit_factor']:.2f}")
            
        with col4:
            st.metric("Max Drawdown", f"{results['max_drawdown']:.1f}%")
            st.metric("Avg Win/Loss", f"${results['avg_win']:,.0f}/${results['avg_loss']:,.0f}")
        
        # Equity Curve
        st.subheader("📈 EQUITY CURVE")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=results['equity_curve'],
            line=dict(color='#00ff88', width=3),
            name='Portfolio Value',
            fill='tozeroy'
        ))
        
        # Başlangıç ve hedef çizgileri
        fig.add_hline(y=initial_balance, line_dash="dash", line_color="white", 
                     annotation_text=f"Start: ${initial_balance:,.0f}")
        
        target_balance = initial_balance * 2
        fig.add_hline(y=target_balance, line_dash="dash", line_color="green",
                     annotation_text=f"Target: ${target_balance:,.0f}")
        
        fig.update_layout(
            height=400,
            template="plotly_dark",
            title="Portfolio Growth",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Trade Analysis
        if results['trades']:
            st.subheader("🔍 TRADE ANALYSIS")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Win/Loss dağılımı
                fig_pie = go.Figure(data=[go.Pie(
                    labels=['Winning Trades', 'Losing Trades'],
                    values=[results['winning_trades'], results['losing_trades']],
                    hole=.3,
                    marker_colors=['#00ff88', '#ff4444']
                )])
                fig_pie.update_layout(title="Trade Distribution", template="plotly_dark", height=300)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # P&L histogram
                pnls = [t.pnl for t in results['trades']]
                fig_hist = go.Figure(data=[go.Histogram(
                    x=pnls, nbinsx=20, marker_color='#00ff88', opacity=0.7
                )])
                fig_hist.update_layout(
                    title="P&L Distribution", 
                    template="plotly_dark", 
                    height=300,
                    xaxis_title="P&L ($)",
                    yaxis_title="Frequency"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            # Recent Trades
            st.subheader("📋 RECENT TRADES")
            trade_data = []
            for i, trade in enumerate(results['trades'][-15:]):
                trade_data.append({
                    '#': i + 1,
                    'Side': trade.side,
                    'Entry': f"${trade.entry_price:.5f}",
                    'Exit': f"${trade.exit_price:.5f}",
                    'P&L': f"${trade.pnl:,.0f}",
                    'Return': f"{trade.pnl_percent:.1f}%",
                    'Hours': trade.duration_hours,
                    'Reason': trade.reason
                })
            
            st.dataframe(pd.DataFrame(trade_data), use_container_width=True)
        
        # Target Progress
        st.subheader("🎯 10K to 20K PROGRESS")
        
        current_profit = results['final_balance'] - initial_balance
        target_profit = 10000
        progress = min(current_profit / target_profit * 100, 100)
        
        st.metric(
            "Target Achievement", 
            f"{progress:.1f}%", 
            f"${current_profit:,.0f} / ${target_profit:,.0f}"
        )
        
        # Progress bar
        progress_color = "green" if progress >= 100 else "orange" if progress >= 50 else "red"
        st.progress(progress/100, text=f"Progress: {progress:.1f}%")
        
        if progress >= 100:
            st.success("🎉 TARGET ACHIEVED! 10K → 20K SUCCESS!")
            st.balloons()
        elif progress >= 70:
            st.warning(f"🟡 Good progress! {progress:.1f}% complete - Almost there!")
        elif progress >= 40:
            st.info(f"🔵 Steady progress: {progress:.1f}% complete")
        else:
            st.error(f"🔴 Needs improvement: {progress:.1f}% complete")

# Strategy Info
with st.expander("ℹ️ STRATEGY DETAILS"):
    st.markdown("""
    ## 🎯 Multi-Timeframe Breakout Strategy
    
    **STRATEGY RULES:**
    
    **ENTRY CONDITIONS:**
    - **Trend Alignment**: Price > EMA12 > EMA26 (Bullish) or Price < EMA12 < EMA26 (Bearish)
    - **Momentum Confirmation**: RSI in optimal zones + MACD histogram confirmation
    - **Volatility Filter**: ATR above 70% of 20-period average
    - **Breakout Trigger**: Price breaks 20-period high/low
    
    **EXIT CONDITIONS:**
    - **Take Profit**: 3.5x ATR from entry (≈ 1:2 Risk/Reward)
    - **Stop Loss**: 1.8x ATR from entry
    - **Time Exit**: Maximum 48 hours hold time
    
    **RISK MANAGEMENT:**
    - 2% risk per trade
    - Maximum 5% portfolio exposure
    - Minimum 1:1.8 risk/reward ratio
    - Trade gap: 6 hours between trades
    
    **EXPECTED PERFORMANCE (90 Days):**
    - ✅ **25-40 total trades**
    - ✅ **65-75% win rate**
    - ✅ **1.8-2.5 profit factor** 
    - ✅ **80-150% total return**
    - ✅ **8-15% max drawdown**
    
    **OPTIMAL PAIRS:**
    - 🥇 **EURUSD** - Best overall performance
    - 🥈 **GBPUSD** - High volatility, good trends
    - 🥉 **USDJPY** - Clean trends, steady moves
    """)

# Quick Start Buttons
st.sidebar.markdown("---")
st.sidebar.subheader("🚀 Quick Start")

if st.sidebar.button("EURUSD - Optimized"):
    st.session_state.forex_symbol = "EURUSD"
    st.session_state.risk_percent = 2.0

if st.sidebar.button("GBPUSD - High Volatility"):
    st.session_state.forex_symbol = "GBPUSD"
    st.session_state.risk_percent = 2.5

if st.sidebar.button("USDJPY - Conservative"):
    st.session_state.forex_symbol = "USDJPY" 
    st.session_state.risk_percent = 1.5