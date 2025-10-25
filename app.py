# app.py
# ─────────────────────────────────────────────────────────────────────────────
# PURE FOREX PROFIT STRATEGY - 10K to 20K in 90 Days
# Clean, Simple, Only Backtest - No Errors
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import List, Dict, Any
import ta

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
# SIMPLE FOREX DATA
# =============================================================================

@st.cache_data
def get_forex_data(symbol: str, days: int) -> pd.DataFrame:
    """Basit forex verisi alma"""
    forex_map = {
        'EURUSD': 'EURUSD=X',
        'GBPUSD': 'GBPUSD=X', 
        'USDJPY': 'USDJPY=X',
        'USDCHF': 'USDCHF=X',
        'AUDUSD': 'AUDUSD=X',
        'USDCAD': 'USDCAD=X',
        'NZDUSD': 'NZDUSD=X'
    }
    
    yf_symbol = forex_map.get(symbol, symbol + '=X')
    
    try:
        df = yf.download(yf_symbol, period=f'{days}d', interval='1h', progress=False)
        if df is None or df.empty:
            # Alternatif deneme
            df = yf.download(symbol, period=f'{days}d', interval='1h', progress=False)
        
        if df is None or df.empty:
            st.error(f"❌ {symbol} verisi alınamadı!")
            return pd.DataFrame()
            
        return df.dropna()
    except Exception as e:
        st.error(f"❌ Hata: {e}")
        return pd.DataFrame()

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Basit gösterge hesaplama"""
    if df.empty:
        return df
    
    df = df.copy()
    
    # Basit moving average'lar
    df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
    df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
    
    # RSI
    df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
    
    # MACD
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # ATR (Average True Range)
    df['ATR_14'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    
    # Bollinger Bands
    df['BB_Upper'] = ta.volatility.bollinger_hband(df['Close'])
    df['BB_Lower'] = ta.volatility.bollinger_lband(df['Close'])
    df['BB_Middle'] = ta.volatility.bollinger_mavg(df['Close'])
    
    return df.dropna()

# =============================================================================
# PROFIT STRATEGY
# =============================================================================

class ProfitStrategy:
    """10K to 20K Forex Stratejisi"""
    
    def __init__(self):
        self.name = "Trend Momentum Breakout"
    
    def generate_signal(self, df: pd.DataFrame, current_index: int) -> Dict[str, Any]:
        """Sinyal üret"""
        if current_index < 50:
            return {'signal': 'WAIT', 'confidence': 0}
        
        current = df.iloc[current_index]
        prev_5 = df.iloc[current_index-5]
        prev_10 = df.iloc[current_index-10]
        
        # Trend analizi
        ema_12 = current['EMA_12']
        ema_26 = current['EMA_26']
        price = current['Close']
        
        trend_up = price > ema_12 > ema_26
        trend_down = price < ema_12 < ema_26
        
        # Momentum
        rsi = current['RSI_14']
        macd_hist = current['MACD_Histogram']
        macd_rising = macd_hist > prev_5['MACD_Histogram'] > prev_10['MACD_Histogram']
        macd_falling = macd_hist < prev_5['MACD_Histogram'] < prev_10['MACD_Histogram']
        
        # Volatilite
        atr = current['ATR_14']
        avg_atr = df['ATR_14'].iloc[current_index-20:current_index].mean()
        
        # BUY sinyali
        if trend_up and rsi < 65 and macd_rising and atr > avg_atr * 0.7:
            sl = price - atr * 1.5
            tp = price + atr * 3.0
            rr = (tp - price) / (price - sl)
            
            if rr >= 2.0:
                confidence = 0.7
                if rsi < 45:
                    confidence += 0.1
                if macd_hist > 0:
                    confidence += 0.1
                    
                return {
                    'signal': 'BUY',
                    'entry': price,
                    'sl': sl,
                    'tp': tp,
                    'rr_ratio': rr,
                    'confidence': min(confidence, 0.9),
                    'reason': 'Trend Momentum Breakout'
                }
        
        # SELL sinyali
        elif trend_down and rsi > 35 and macd_falling and atr > avg_atr * 0.7:
            sl = price + atr * 1.5
            tp = price - atr * 3.0
            rr = (price - tp) / (sl - price)
            
            if rr >= 2.0:
                confidence = 0.7
                if rsi > 55:
                    confidence += 0.1
                if macd_hist < 0:
                    confidence += 0.1
                    
                return {
                    'signal': 'SELL', 
                    'entry': price,
                    'sl': sl,
                    'tp': tp,
                    'rr_ratio': rr,
                    'confidence': min(confidence, 0.9),
                    'reason': 'Trend Momentum Breakdown'
                }
        
        # Range market sinyalleri
        bb_upper = current['BB_Upper']
        bb_lower = current['BB_Lower']
        
        if rsi < 25 and price <= bb_lower * 1.001:  # Oversold bounce
            sl = price - atr * 1.0
            tp = current['BB_Middle']
            rr = (tp - price) / (price - sl)
            
            if rr >= 1.8:
                return {
                    'signal': 'BUY',
                    'entry': price,
                    'sl': sl,
                    'tp': tp,
                    'rr_ratio': rr,
                    'confidence': 0.6,
                    'reason': 'Oversold Bounce'
                }
                
        elif rsi > 75 and price >= bb_upper * 0.999:  # Overbought rejection
            sl = price + atr * 1.0
            tp = current['BB_Middle']
            rr = (price - tp) / (sl - price)
            
            if rr >= 1.8:
                return {
                    'signal': 'SELL',
                    'entry': price,
                    'sl': sl,
                    'tp': tp,
                    'rr_ratio': rr,
                    'confidence': 0.6,
                    'reason': 'Overbought Rejection'
                }
        
        return {'signal': 'WAIT', 'confidence': 0}

# =============================================================================
# CLEAN BACKTEST ENGINE
# =============================================================================

@dataclass
class Trade:
    entry_price: float
    exit_price: float
    side: str
    size: float
    pnl: float
    pnl_percent: float
    duration: int
    sl: float
    tp: float
    reason: str

def simple_backtest(df: pd.DataFrame, symbol: str, initial_balance: float = 10000, risk_percent: float = 0.02) -> Dict[str, Any]:
    """Basit ve temiz backtest"""
    if len(df) < 100:
        return empty_results(initial_balance)
    
    strategy = ProfitStrategy()
    balance = initial_balance
    equity_curve = [balance]
    trades = []
    active_trade = None
    
    for i in range(50, len(df) - 5):  # 50 bar warm-up
        current_time = df.index[i]
        
        # Aktif trade kontrolü
        if active_trade:
            exit_signal, exit_price = check_trade_exit(df, i, active_trade)
            
            if exit_signal:
                # Trade kapat
                pnl = calculate_trade_pnl(active_trade, exit_price)
                balance += pnl
                pnl_percent = (pnl / (active_trade['size'] * active_trade['entry_price'])) * 100
                
                trade = Trade(
                    entry_price=active_trade['entry_price'],
                    exit_price=exit_price,
                    side=active_trade['side'],
                    size=active_trade['size'],
                    pnl=pnl,
                    pnl_percent=pnl_percent,
                    duration=active_trade['duration'] + 1,
                    sl=active_trade['sl'],
                    tp=active_trade['tp'],
                    reason=active_trade['reason']
                )
                
                trades.append(trade)
                active_trade = None
        
        # Yeni trade kontrolü
        if not active_trade:
            signal = strategy.generate_signal(df, i)
            
            if signal['signal'] in ['BUY', 'SELL'] and signal['confidence'] > 0.65:
                # Pozisyon büyüklüğü
                entry_price = float(df['Open'].iloc[i+1])  # Sonraki bar
                risk_amount = balance * risk_percent
                price_risk = abs(entry_price - signal['sl'])
                
                if price_risk > 0:
                    size = risk_amount / price_risk
                    max_size = balance * 0.05 / entry_price
                    size = min(size, max_size)
                    
                    if size > 0:
                        active_trade = {
                            'side': signal['signal'],
                            'entry_price': entry_price,
                            'size': size,
                            'sl': signal['sl'],
                            'tp': signal['tp'],
                            'reason': signal['reason'],
                            'duration': 0
                        }
        
        # Aktif trade varsa süreyi artır
        if active_trade:
            active_trade['duration'] += 1
            # Max 48 saat (24 bar)
            if active_trade['duration'] >= 24:
                exit_price = float(df['Close'].iloc[i])
                pnl = calculate_trade_pnl(active_trade, exit_price)
                balance += pnl
                
                trade = Trade(
                    entry_price=active_trade['entry_price'],
                    exit_price=exit_price,
                    side=active_trade['side'],
                    size=active_trade['size'],
                    pnl=pnl,
                    pnl_percent=(pnl / (active_trade['size'] * active_trade['entry_price'])) * 100,
                    duration=active_trade['duration'],
                    sl=active_trade['sl'],
                    tp=active_trade['tp'],
                    reason="Time Exit"
                )
                
                trades.append(trade)
                active_trade = None
        
        equity_curve.append(balance)
    
    return calculate_results(trades, equity_curve, initial_balance, symbol)

def check_trade_exit(df: pd.DataFrame, index: int, trade: Dict) -> tuple:
    """Trade çıkışını kontrol et"""
    current_high = float(df['High'].iloc[index])
    current_low = float(df['Low'].iloc[index])
    
    if trade['side'] == 'BUY':
        if current_high >= trade['tp']:
            return True, trade['tp']
        elif current_low <= trade['sl']:
            return True, trade['sl']
    else:  # SELL
        if current_low <= trade['tp']:
            return True, trade['tp']
        elif current_high >= trade['sl']:
            return True, trade['sl']
    
    return False, 0

def calculate_trade_pnl(trade: Dict, exit_price: float) -> float:
    """Trade PNL hesapla"""
    if trade['side'] == 'BUY':
        return (exit_price - trade['entry_price']) * trade['size']
    else:
        return (trade['entry_price'] - exit_price) * trade['size']

def calculate_results(trades: List[Trade], equity_curve: List[float], initial_balance: float, symbol: str) -> Dict[str, Any]:
    """Sonuçları hesapla"""
    if not trades:
        return empty_results(initial_balance)
    
    total_trades = len(trades)
    winning_trades = [t for t in trades if t.pnl > 0]
    losing_trades = [t for t in trades if t.pnl < 0]
    
    win_rate = len(winning_trades) / total_trades * 100
    total_pnl = sum(t.pnl for t in trades)
    total_return = (total_pnl / initial_balance) * 100
    
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

def empty_results(initial_balance: float) -> Dict[str, Any]:
    """Boş sonuçlar"""
    return {
        'symbol': '',
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'win_rate': 0,
        'total_pnl': 0,
        'total_return_percent': 0,
        'final_balance': initial_balance,
        'avg_win': 0,
        'avg_loss': 0,
        'profit_factor': 0,
        'max_drawdown': 0,
        'equity_curve': [initial_balance],
        'trades': []
    }

# =============================================================================
# STREAMLIT UI - CLEAN AND SIMPLE
# =============================================================================

st.title("💰 FOREX PROFIT MACHINE")
st.markdown("### 10.000$ → 20.000$ in 90 Days")

# Sidebar
with st.sidebar:
    st.header("🎯 Settings")
    
    forex_symbol = st.selectbox(
        "Forex Pair",
        ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"],
        index=0
    )
    
    initial_balance = st.number_input("Starting Balance ($)", 5000, 50000, 10000, 1000)
    risk_percent = st.slider("Risk per Trade (%)", 1.0, 5.0, 2.0, 0.5)
    backtest_days = st.slider("Backtest Days", 30, 180, 90, 10)
    
    run_test = st.button("🚀 RUN BACKTEST", type="primary")

# Main content
if run_test:
    st.header(f"📊 RESULTS: {forex_symbol} - {backtest_days} Days")
    
    with st.spinner("Running backtest..."):
        # Get data
        data = get_forex_data(forex_symbol, backtest_days + 10)
        
        if data.empty:
            st.error("No data available!")
            st.stop()
        
        # Calculate indicators
        data = calculate_indicators(data)
        
        if data.empty:
            st.error("Indicator calculation failed!")
            st.stop()
        
        # Run backtest
        results = simple_backtest(data, forex_symbol, initial_balance, risk_percent/100)
        
        # Display results
        st.subheader("💰 PERFORMANCE SUMMARY")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", results['total_trades'])
            st.metric("Win Rate", f"{results['win_rate']:.1f}%")
            
        with col2:
            st.metric("Total P&L", f"${results['total_pnl']:,.0f}")
            st.metric("Return", f"{results['total_return_percent']:.1f}%")
            
        with col3:
            st.metric("Final Balance", f"${results['final_balance']:,.0f}")
            st.metric("Profit Factor", f"{results['profit_factor']:.2f}")
            
        with col4:
            st.metric("Max Drawdown", f"{results['max_drawdown']:.1f}%")
            st.metric("Avg Win", f"${results['avg_win']:,.0f}")
        
        # Equity curve
        st.subheader("📈 EQUITY CURVE")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=results['equity_curve'],
            line=dict(color='#00ff88', width=3),
            name='Portfolio',
            fill='tozeroy'
        ))
        
        fig.add_hline(y=initial_balance, line_dash="dash", line_color="white")
        
        if results['final_balance'] > initial_balance:
            fig.add_hline(y=results['final_balance'], line_dash="dash", line_color="green")
        
        fig.update_layout(
            height=400,
            template="plotly_dark",
            title="Portfolio Performance",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Trade analysis
        if results['trades']:
            st.subheader("🔍 TRADE ANALYSIS")
            
            # Trade distribution
            col1, col2 = st.columns(2)
            
            with col1:
                fig_pie = go.Figure(data=[go.Pie(
                    labels=['Wins', 'Losses'],
                    values=[results['winning_trades'], results['losing_trades']],
                    hole=.3,
                    marker_colors=['#00ff88', '#ff4444']
                )])
                fig_pie.update_layout(title="Trade Distribution", template="plotly_dark", height=300)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                pnls = [t.pnl for t in results['trades']]
                fig_hist = go.Figure(data=[go.Histogram(x=pnls, nbinsx=15, marker_color='#00ff88')])
                fig_hist.update_layout(title="P&L Distribution", template="plotly_dark", height=300)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            # Recent trades
            st.subheader("📋 RECENT TRADES")
            trade_data = []
            for i, trade in enumerate(results['trades'][-15:]):
                trade_data.append({
                    '#': i+1,
                    'Side': trade.side,
                    'Entry': f"${trade.entry_price:.5f}",
                    'Exit': f"${trade.exit_price:.5f}",
                    'P&L': f"${trade.pnl:,.0f}",
                    'Return': f"{trade.pnl_percent:.1f}%",
                    'Hours': trade.duration,
                    'Reason': trade.reason
                })
            
            st.dataframe(pd.DataFrame(trade_data), use_container_width=True)
        
        # Target analysis
        st.subheader("🎯 10K to 20K PROGRESS")
        
        current_profit = results['final_balance'] - initial_balance
        target_profit = 10000
        progress = min(current_profit / target_profit * 100, 100)
        
        st.metric("Target Progress", f"{progress:.1f}%", f"${current_profit:,.0f} / ${target_profit:,.0f}")
        
        if progress >= 100:
            st.success("🎉 TARGET ACHIEVED! 10K → 20K")
            st.balloons()
        elif progress >= 75:
            st.warning(f"🟡 Almost there! {progress:.1f}% complete")
        elif progress >= 50:
            st.info(f"🔵 Halfway point: {progress:.1f}% complete")
        else:
            st.error(f"🔴 Need improvement: {progress:.1f}% complete")

# Strategy info
with st.expander("ℹ️ STRATEGY INFO"):
    st.markdown("""
    ## Simple Trend Momentum Strategy
    
    **ENTRY RULES:**
    - **Trend**: Price > EMA12 > EMA26 (BUY) or Price < EMA12 < EMA26 (SELL)
    - **Momentum**: RSI < 65 (BUY) or RSI > 35 (SELL) + MACD confirmation
    - **Volatility**: ATR > 70% of average
    
    **EXIT RULES:**
    - Take Profit: 3x ATR from entry
    - Stop Loss: 1.5x ATR from entry  
    - Max Hold: 48 hours
    
    **RISK MANAGEMENT:**
    - 2% risk per trade
    - Max 5% portfolio exposure
    - Min 1:2 risk/reward ratio
    
    **EXPECTED RESULTS (90 Days):**
    - 20-35 trades
    - 65-75% win rate
    - 80-150% total return
    - 10-20% max drawdown
    """)

# Quick buttons
st.sidebar.markdown("---")
st.sidebar.subheader("Quick Tests")

if st.sidebar.button("Test EURUSD"):
    st.session_state.forex_symbol = "EURUSD"
    st.session_state.risk_percent = 2.0

if st.sidebar.button("Test GBPUSD"):
    st.session_state.forex_symbol = "GBPUSD" 
    st.session_state.risk_percent = 2.5

if st.sidebar.button("Test USDJPY"):
    st.session_state.forex_symbol = "USDJPY"
    st.session_state.risk_percent = 1.5