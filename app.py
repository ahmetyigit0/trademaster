import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# =========================
# ŞİFRE KORUMASI
# =========================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    
    def password_entered():
        st.session_state["password_correct"] = (st.session_state.get("password","") == "efe")
    
    if not st.session_state["password_correct"]:
        st.text_input("🔐 Şifre", type="password", on_change=password_entered, key="password")
        return False
    return True

if not check_password():
    st.stop()

# =========================
# BACKTEST MOTORU
# =========================
class SwingBacktest:
    def __init__(self, commission=0.001):
        self.commission = float(commission)
    
    @staticmethod
    def _num(s):
        return pd.to_numeric(s, errors="coerce")
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        # Zorunlu kolon kontrolü ve numerik dönüşüm
        need = ["Open","High","Low","Close","Volume"]
        for c in need:
            if c not in d.columns:
                raise ValueError(f"Girdi verisinde '{c}' kolonu yok.")
            d[c] = self._num(d[c])
        
        # EMA'lar
        d["EMA_20"] = d["Close"].ewm(span=20, min_periods=1, adjust=False).mean()
        d["EMA_50"] = d["Close"].ewm(span=50, min_periods=1, adjust=False).mean()
        
        # RSI (14)
        delta = d["Close"].diff()
        gain = delta.clip(lower=0).rolling(window=14, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-12)
        d["RSI"] = 100 - (100 / (1 + rs))
        
        # ATR (14)
        prev_close = d["Close"].shift(1)
        tr = pd.concat([
            (d["High"] - d["Low"]).abs(),
            (d["High"] - prev_close).abs(),
            (d["Low"] - prev_close).abs()
        ], axis=1).max(axis=1)
        d["ATR"] = tr.rolling(window=14, min_periods=1).mean()
        
        d.replace([np.inf, -np.inf], np.nan, inplace=True)
        d.fillna(method="bfill", inplace=True)
        d.fillna(method="ffill", inplace=True)
        return d
    
    def generate_signals(self, df: pd.DataFrame, rsi_oversold=40, atr_multiplier=2.0) -> pd.DataFrame:
        rows = []
        for i in range(len(df)):
            row = df.iloc[i]
            close_val = float(row["Close"])
            ema20 = float(row["EMA_20"])
            ema50 = float(row["EMA_50"])
            rsi_val = float(row["RSI"])
            atr_val = float(row["ATR"])
            
            trend_ok = ema20 > ema50
            rsi_ok = rsi_val < float(rsi_oversold)
            price_ok = close_val > ema20
            
            is_buy = bool(trend_ok and rsi_ok and price_ok)
            if is_buy:
                sl = close_val - atr_val * float(atr_multiplier)
                tp = close_val + atr_val * float(atr_multiplier) * 2.0
            else:
                sl = 0.0; tp = 0.0
            
            rows.append({
                "date": df.index[i],
                "action": "buy" if is_buy else "hold",
                "is_buy": is_buy,
                "price": close_val,
                "stop_loss": sl,
                "take_profit": tp
            })
        
        sig = pd.DataFrame(rows)
        if sig.empty:
            return sig
        sig = sig.set_index("date")
        # Aynı timestamp varsa son kaydı al → Series ambiguity önlenir
        sig = sig.groupby(level=0).last()
        return sig
    
    def run_backtest(self, data: pd.DataFrame, rsi_oversold=40, atr_multiplier=2.0, risk_per_trade=0.02):
        df = self.calculate_indicators(data)
        sigs = self.generate_signals(df, rsi_oversold, atr_multiplier)
        
        capital = 10000.0
        position = None
        trades = []
        equity_curve = []
        
        for date in df.index:
            price = float(df.loc[date, "Close"])
            # Mark-to-market
            current_equity = capital + (position["shares"] * price if position is not None else 0.0)
            equity_curve.append({"date": date, "equity": current_equity})
            
            # Güvenli sinyal çekme
            sig = None
            if date in sigs.index:
                row = sigs.loc[date]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]
                sig = row
            
            if position is None:
                is_buy = bool(sig["is_buy"]) if sig is not None and "is_buy" in sig else False
                if is_buy:
                    sl = float(sig["stop_loss"])
                    risk_per_share = price - sl
                    if risk_per_share <= 0:
                        continue
                    risk_amt = capital * float(risk_per_trade)
                    shares = max(risk_amt / risk_per_share, 0.0)
                    if shares > 0:
                        cost = shares * price
                        fee = cost * self.commission
                        capital -= (cost + fee)
                        position = {
                            "entry_date": date,
                            "entry_price": price,
                            "shares": shares,
                            "stop_loss": sl,
                            "take_profit": float(sig["take_profit"])
                        }
            else:
                exit_reason = None
                exit_price = None
                if price <= position["stop_loss"]:
                    exit_reason, exit_price = "SL", position["stop_loss"]
                elif price >= position["take_profit"]:
                    exit_reason, exit_price = "TP", position["take_profit"]
                
                if exit_reason:
                    exit_value = position["shares"] * float(exit_price)
                    fee = exit_value * self.commission
                    capital += (exit_value - fee)
                    
                    entry_value = position["shares"] * float(position["entry_price"])
                    entry_fee = entry_value * self.commission
                    pnl = (exit_value - fee) - (entry_value + entry_fee)
                    
                    trades.append({
                        "entry_date": position["entry_date"],
                        "exit_date": date,
                        "entry_price": float(position["entry_price"]),
                        "exit_price": float(exit_price),
                        "shares": float(position["shares"]),
                        "pnl": float(pnl),
                        "return_pct": float(pnl / (entry_value + entry_fee) * 100.0) if entry_value > 0 else 0.0,
                        "exit_reason": exit_reason
                    })
                    position = None
        
        # Son bar: açık pozisyonu kapat
        if position is not None:
            last_price = float(df["Close"].iloc[-1])
            exit_value = position["shares"] * last_price
            fee = exit_value * self.commission
            capital += (exit_value - fee)
            
            entry_value = position["shares"] * float(position["entry_price"])
            entry_fee = entry_value * self.commission
            pnl = (exit_value - fee) - (entry_value + entry_fee)
            
            trades.append({
                "entry_date": position["entry_date"],
                "exit_date": df.index[-1],
                "entry_price": float(position["entry_price"]),
                "exit_price": float(last_price),
                "shares": float(position["shares"]),
                "pnl": float(pnl),
                "return_pct": float(pnl / (entry_value + entry_fee) * 100.0) if entry_value > 0 else 0.0,
                "exit_reason": "OPEN"
            })
        
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        equity_df = pd.DataFrame(equity_curve)
        return trades_df, equity_df
    
    @staticmethod
    def calculate_metrics(trades_df: pd.DataFrame, equity_df: pd.DataFrame):
        if trades_df.empty or equity_df.empty:
            return {
                "total_return": "0.0%",
                "total_trades": "0",
                "win_rate": "0.0%",
                "avg_win": "$0.00",
                "avg_loss": "$0.00",
            }
        initial_equity = float(equity_df["equity"].iloc[0])
        final_equity = float(equity_df["equity"].iloc[-1])
        total_return = (final_equity - initial_equity) / initial_equity * 100.0
        
        total_trades = len(trades_df)
        wins = trades_df[trades_df["pnl"] > 0]
        win_rate = (len(wins) / total_trades * 100.0) if total_trades > 0 else 0.0
        avg_win = float(wins["pnl"].mean()) if not wins.empty else 0.0
        losses = trades_df[trades_df["pnl"] < 0]
        avg_loss = float(losses["pnl"].mean()) if not losses.empty else 0.0
        
        return {
            "total_return": f"{round(total_return, 2)}%",
            "total_trades": f"{total_trades}",
            "win_rate": f"{round(win_rate, 1)}%",
            "avg_win": f"${round(avg_win, 2)}",
            "avg_loss": f"${round(avg_loss, 2)}",
        }

# =========================
# STREAMLIT UYGULAMASI
# =========================
st.set_page_config(page_title="Swing Backtest", layout="wide")
st.title("🚀 Swing Trading Backtest (Stabil)")

# Sidebar
st.sidebar.header("⚙️ Ayarlar")
ticker = st.sidebar.selectbox("Sembol", ["AAPL", "GOOGL", "MSFT", "TSLA", "BTC-USD", "ETH-USD"], index=4)
start_date = st.sidebar.date_input("Başlangıç", datetime(2023, 1, 1))
end_date = st.sidebar.date_input("Bitiş", datetime(2023, 12, 31))

st.sidebar.header("📊 Parametreler")
rsi_oversold = st.sidebar.slider("RSI Aşırı Satım", 25, 50, 40)
atr_multiplier = st.sidebar.slider("ATR Çarpanı", 1.0, 3.0, 2.0)
risk_per_trade = st.sidebar.slider("Risk %", 1.0, 5.0, 2.0) / 100.0

# Ana içerik
if st.button("🎯 Backtest Çalıştır", type="primary"):
    try:
        with st.spinner("Veri yükleniyor..."):
            extended_start = start_date - timedelta(days=100)  # indikatör ısınma
            data = yf.download(ticker, start=extended_start, end=end_date, progress=False)
            if data.empty:
                st.error("❌ Veri bulunamadı"); st.stop()
            # Seçili aralığı kes
            data = data[(data.index >= pd.to_datetime(start_date)) & (data.index <= pd.to_datetime(end_date))]
            st.success(f"✅ {len(data)} günlük veri yüklendi")
            st.info(f"📈 Fiyat aralığı: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
        
        backtester = SwingBacktest(commission=0.001)
        with st.spinner("Backtest çalıştırılıyor..."):
            trades, equity = backtester.run_backtest(data, rsi_oversold, atr_multiplier, risk_per_trade)
            metrics = backtester.calculate_metrics(trades, equity)
        
        st.subheader("📊 Performans Özeti")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Toplam Getiri", metrics['total_return'])
            st.metric("Toplam İşlem", metrics['total_trades'])
        with col2:
            st.metric("Win Rate", metrics['win_rate'])
            st.metric("Ort. Kazanç", metrics['avg_win'])
        with col3:
            st.metric("Ort. Kayıp", metrics['avg_loss'])
        
        if not trades.empty and not equity.empty:
            st.subheader("📈 Performans Grafikleri")
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            ax.plot(equity['date'], equity['equity'], linewidth=2)
            ax.set_title('Portföy Değeri')
            ax.set_ylabel('Equity ($)')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            st.subheader("📋 İşlem Listesi")
            disp = trades.copy()
            for c in ["entry_date","exit_date"]:
                if c in disp and pd.api.types.is_datetime64_any_dtype(disp[c]):
                    disp[c] = disp[c].dt.strftime("%Y-%m-%d")
            st.dataframe(disp, use_container_width=True)
        else:
            st.info("🤷 Hiç işlem gerçekleşmedi. RSI eşiğini yükseltmeyi veya tarih aralığını genişletmeyi deneyin.")
            
    except Exception as e:
        st.error(f"❌ Hata: {str(e)}")

st.markdown("---")
st.caption("Backtest Sistemi (stabil sinyal erişimi + komisyon)")
