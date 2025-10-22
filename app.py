import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

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
    def _to_num(s):
        return pd.to_numeric(s, errors="coerce")
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """EMA(20/50), RSI(14), ATR(14) hesaplar ve veriyi temizler."""
        try:
            d = df.copy()
            # Zorunlu kolonlar
            need = ["Open","High","Low","Close","Volume"]
            for c in need:
                if c not in d.columns:
                    raise ValueError(f"Girdi verisinde '{c}' kolonu yok.")
                d[c] = self._to_num(d[c])
            
            # EMA'lar
            d["EMA_20"] = d["Close"].ewm(span=20, min_periods=1, adjust=False).mean()
            d["EMA_50"] = d["Close"].ewm(span=50, min_periods=1, adjust=False).mean()
            
            # RSI (basit rolling yaklaşım)
            delta = d["Close"].diff()
            gain = (delta.clip(lower=0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.clip(upper=0)).rolling(window=14, min_periods=1).mean()
            rs = gain / (loss + 1e-12)
            d["RSI"] = 100 - (100 / (1 + rs))
            
            # ATR (vektörel True Range)
            prev_close = d["Close"].shift(1)
            tr = pd.concat([
                (d["High"] - d["Low"]).abs(),
                (d["High"] - prev_close).abs(),
                (d["Low"] - prev_close).abs()
            ], axis=1).max(axis=1)
            d["ATR"] = tr.rolling(window=14, min_periods=1).mean()
            
            # Temizlik
            d.replace([np.inf, -np.inf], np.nan, inplace=True)
            d.fillna(method="bfill", inplace=True)
            d.fillna(method="ffill", inplace=True)
            return d
        except Exception as e:
            st.error(f"İndikatör hatası: {e}")
            return pd.DataFrame()
    
    def generate_signals(self, df: pd.DataFrame, rsi_oversold=40, atr_multiplier=2.0) -> pd.DataFrame:
        """Sinyal üret: Trend (EMA20>EMA50) + RSI < eşik + Fiyat > EMA20. Index tekilleştirilir."""
        try:
            rows = []
            for i in range(len(df)):
                row = df.iloc[i]
                close_val = float(row["Close"])
                ema_20 = float(row["EMA_20"])
                ema_50 = float(row["EMA_50"])
                rsi_val = float(row["RSI"])
                atr_val = float(row["ATR"])
                
                trend_ok = ema_20 > ema_50
                rsi_ok = rsi_val < float(rsi_oversold)
                price_ok = close_val > ema_20
                
                is_buy = bool(trend_ok and rsi_ok and price_ok)
                if is_buy:
                    stop_loss = close_val - (atr_val * float(atr_multiplier))
                    take_profit = close_val + (atr_val * float(atr_multiplier) * 2.0)
                else:
                    stop_loss = 0.0
                    take_profit = 0.0
                
                rows.append({
                    "date": df.index[i],
                    "action": "buy" if is_buy else "hold",
                    "is_buy": is_buy,
                    "price": close_val,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit
                })
            
            sig = pd.DataFrame(rows)
            if sig.empty:
                return sig
            sig = sig.set_index("date")
            # ⬇️ aynı timestamp varsa son kaydı al → tekilleştir
            sig = sig.groupby(level=0).last()
            return sig
        except Exception as e:
            st.error(f"Sinyal hatası: {e}")
            return pd.DataFrame()
    
    def run_backtest(self, data: pd.DataFrame, rsi_oversold=40, atr_multiplier=2.0, risk_per_trade=0.02):
        """Basit long-only backtest: tek pozisyon, SL/TP ile çıkış, equity curve."""
        try:
            df = self.calculate_indicators(data)
            if df.empty:
                return pd.DataFrame(), pd.DataFrame()
            
            signals = self.generate_signals(df, rsi_oversold, atr_multiplier)
            if signals.empty:
                return pd.DataFrame(), pd.DataFrame()
            
            capital = 10000.0
            position = None
            trades = []
            equity_curve = []
            
            for date in df.index:
                current_price = float(df.loc[date, "Close"])
                
                # Equity güncelle (mark-to-market)
                current_equity = capital + (position["shares"] * current_price if position is not None else 0.0)
                equity_curve.append({"date": date, "equity": current_equity})
                
                # Sinyali güvenli çek
                sig = None
                if date in signals.index:
                    row = signals.loc[date]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[-1]
                    sig = row
                
                if position is None:
                    is_buy = bool(sig["is_buy"]) if sig is not None and "is_buy" in sig else False
                    if is_buy:
                        stop_loss = float(sig["stop_loss"])
                        risk_per_share = current_price - stop_loss
                        if risk_per_share <= 0:
                            continue
                        risk_amount = capital * float(risk_per_trade)
                        shares = max(risk_amount / risk_per_share, 0.0)
                        if shares > 0:
                            cost = shares * current_price
                            fee = cost * self.commission
                            total_cost = cost + fee
                            position = {
                                "entry_date": date,
                                "entry_price": current_price,
                                "shares": shares,
                                "stop_loss": stop_loss,
                                "take_profit": float(sig["take_profit"])
                            }
                            capital -= total_cost
                else:
                    # Pozisyon açık → SL/TP kontrolü
                    exit_reason = None
                    exit_price = None
                    if current_price <= position["stop_loss"]:
                        exit_reason, exit_price = "SL", position["stop_loss"]
                    elif current_price >= position["take_profit"]:
                        exit_reason, exit_price = "TP", position["take_profit"]
                    
                    if exit_reason is not None:
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
                            "exit_reason": exit_reason,
                            "hold_days": (date - position["entry_date"]).days if hasattr(date, "to_pydatetime") else 0
                        })
                        position = None
            
            # Son bar: açık pozisyonu kapat (mark-to-market)
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
                    "exit_reason": "OPEN",
                    "hold_days": (df.index[-1] - position["entry_date"]).days if hasattr(df.index[-1], "to_pydatetime") else 0
                })
            
            trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
            equity_df = pd.DataFrame(equity_curve)
            return trades_df, equity_df
        except Exception as e:
            st.error(f"Backtest hatası: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    @staticmethod
    def calculate_metrics(trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> dict:
        """Basit performans metrikleri."""
        if trades_df.empty or equity_df.empty:
            return {
                "total_return_%": 0.0,
                "total_trades": 0,
                "win_rate_%": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "max_drawdown_%": 0.0
            }
        try:
            initial_equity = float(equity_df["equity"].iloc[0]) if "equity" in equity_df else 10000.0
            final_equity = float(equity_df["equity"].iloc[-1])
            total_return = (final_equity - initial_equity) / initial_equity * 100.0
            
            total_trades = len(trades_df)
            winning = trades_df[trades_df["pnl"] > 0]
            losing = trades_df[trades_df["pnl"] < 0]
            win_rate = (len(winning) / total_trades * 100.0) if total_trades > 0 else 0.0
            avg_win = float(winning["pnl"].mean()) if not winning.empty else 0.0
            avg_loss = float(losing["pnl"].mean()) if not losing.empty else 0.0
            
            eq = equity_df.set_index("date")["equity"].astype(float)
            roll_max = eq.expanding().max()
            drawdown = (eq - roll_max) / roll_max * 100.0
            max_dd = float(drawdown.min()) if len(drawdown) else 0.0
            
            return {
                "total_return_%": round(total_return, 2),
                "total_trades": int(total_trades),
                "win_rate_%": round(win_rate, 1),
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
                "max_drawdown_%": round(max_dd, 2)
            }
        except Exception:
            return {
                "total_return_%": 0.0,
                "total_trades": 0,
                "win_rate_%": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "max_drawdown_%": 0.0
            }

# =========================
# STREAMLIT UYGULAMASI
# =========================
st.set_page_config(page_title="Swing Backtest", layout="wide")
st.title("🚀 Swing Trading Backtest")
st.markdown("**Seri belirsizliği ve indeks çakışmaları giderildi**")

# Sidebar
st.sidebar.header("⚙️ Ayarlar")
ticker = st.sidebar.selectbox("Sembol", ["AAPL", "GOOGL", "MSFT", "TSLA", "BTC-USD", "ETH-USD", "NVDA", "AMZN"], index=4)
start_date = st.sidebar.date_input("Başlangıç", datetime(2022, 1, 1))
end_date = st.sidebar.date_input("Bitiş", datetime(2023, 12, 31))

st.sidebar.header("📊 Parametreler")
rsi_oversold = st.sidebar.slider("RSI Aşırı Satım", 25, 50, 40)
atr_multiplier = st.sidebar.slider("ATR Çarpanı", 1.0, 3.0, 2.0)
risk_per_trade = st.sidebar.slider("Risk %", 1.0, 5.0, 2.0) / 100.0  # % → orana çevrildi

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
            st.metric("Toplam Getiri", f"{metrics['total_return_%']}%")
            st.metric("Toplam İşlem", f"{metrics['total_trades']}")
        with col2:
            st.metric("Win Rate", f"{metrics['win_rate_%']}%")
            st.metric("Ort. Kazanç", f"${metrics['avg_win']:.2f}")
        with col3:
            st.metric("Ort. Kayıp", f"${metrics['avg_loss']:.2f}")
            st.metric("Max Drawdown", f"{metrics['max_drawdown_%']}%")
        
        if not trades.empty and not equity.empty:
            st.subheader("📈 Performans Grafikleri")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            ax1.plot(equity['date'], equity['equity'], linewidth=2)
            ax1.set_title('Portföy Değeri')
            ax1.set_ylabel('Equity ($)')
            ax1.grid(True, alpha=0.3)
            
            eq = equity.set_index('date')['equity'].astype(float)
            roll_max = eq.expanding().max()
            drawdown = (eq - roll_max) / roll_max * 100.0
            ax2.fill_between(eq.index, drawdown.values, 0, alpha=0.3, color='red')
            ax2.set_title('Drawdown (%)')
            ax2.set_ylabel('Drawdown %')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.subheader("📋 İşlem Listesi")
            disp = trades.copy()
            for col in ["entry_date","exit_date"]:
                if pd.api.types.is_datetime64_any_dtype(disp[col]):
                    disp[col] = disp[col].dt.strftime("%Y-%m-%d")
            st.dataframe(disp, use_container_width=True)
        else:
            st.warning("""**🤔 Hala işlem yok! Şunları deneyin:**
- RSI eşik değerini 45-50 yapın
- BTC-USD / TSLA gibi volatil semboller deneyin
- Tarih aralığını 2020-2023 seçin
""")
    except Exception as e:
        st.error(f"❌ Hata: {str(e)}")

st.markdown("---")
st.caption("Swing Backtest v4.1 — indeks tekilleştirme + güvenli sinyal erişimi + komisyonlu PnL")