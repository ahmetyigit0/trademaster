# -*- coding: utf-8 -*-
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
        # Şifre: "efe"
        st.session_state["password_correct"] = (st.session_state.get("password","") == "efe")
    
    if not st.session_state["password_correct"]:
        st.text_input("🔐 Lütfen Şifreyi Girin", type="password", on_change=password_entered, key="password")
        return False
    return True

if not check_password():
    st.stop()

# =========================
# YARDIMCI: güvenli sayı/sayı-format (Bu kısım ana kod bloğuna taşındı)
# =========================
def to_scalar(x):
    try:
        if hasattr(x, "iloc"):
            if len(x) > 0:
                return float(x.iloc[-1])
            return float("nan")
        if isinstance(x, (np.ndarray, list, tuple)):
            return float(x[-1]) if len(x) > 0 else float("nan")
        return float(x)
    except Exception:
        try:
            return float(pd.to_numeric(x, errors="coerce"))
        except Exception:
            return float("nan")

def fmt(x, nd=2, prefix="", suffix=""):
    v = to_scalar(x)
    if pd.isna(v):
        return "-"
    return f"{prefix}{v:.{nd}f}{suffix}"
# =========================
# BACKTEST MOTORU
# =========================
class CleanSwingBacktest:
    def __init__(self, commission=0.001):
        self.commission = float(commission)
    
    @staticmethod
    def _num(s):
        # Veriyi sayısal formata dönüştürürken hataları NaN olarak kabul et
        return pd.to_numeric(s, errors="coerce")
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        
        # Veri Kontrolü
        required_cols = ["Open","High","Low","Close"]
        for col in required_cols:
            if col not in d.columns:
                raise ValueError(f"Girdi verisinde '{col}' kolonu eksik.")
            d[col] = self._num(d[col])
            
        # EMA
        d["EMA_20"] = d["Close"].ewm(span=20, min_periods=1, adjust=False).mean()
        d["EMA_50"] = d["Close"].ewm(span=50, min_periods=1, adjust=False).mean()
        
        # RSI
        delta = d["Close"].diff()
        gain = delta.clip(lower=0).rolling(window=14, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-12)
        d["RSI"] = 100 - (100 / (1 + rs))
        
        # ATR
        prev_close = d["Close"].shift(1)
        tr = pd.concat([
            (d["High"] - d["Low"]).abs(),
            (d["High"] - prev_close).abs(),
            (d["Low"] - prev_close).abs()
        ], axis=1).max(axis=1)
        d["ATR"] = tr.rolling(window=14, min_periods=1).mean()
        
        # Temizlik (NaN değerleri doldurma)
        d.replace([np.inf, -np.inf], np.nan, inplace=True)
        d.fillna(method="bfill", inplace=True) # Geriye dönük doldurma
        d.fillna(method="ffill", inplace=True) # Baştaki NaN'lar için ileriye doldurma
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
            
            # Basit Kesişim ve Trend Sinyali
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
                "stop_loss": sl,
                "take_profit": tp
            })
            
        sig = pd.DataFrame(rows)
        if sig.empty:
            return sig
        
        # İndeksi tarih yap ve aynı tarihteki son sinyali al (Tekil indeks sağlamak için)
        sig = sig.set_index("date").groupby(level=0).last()
        return sig
    
    def run_backtest(self, data: pd.DataFrame, rsi_oversold=40, atr_multiplier=2.0, risk_per_trade=0.02):
        df = self.calculate_indicators(data)
        sigs = self.generate_signals(df, rsi_oversold, atr_multiplier)
        
        capital = 10000.0
        position = None
        trades = []
        equity_curve = []
        
        # df zaten temiz, tek seviyeli bir indekse sahip (Tarih)
        for date in df.index:
            price = float(df.loc[date, "Close"])
            
            current_equity = capital + (position["shares"] * price if position is not None else 0.0)
            equity_curve.append({"date": date, "equity": current_equity})
            
            sig = None
            # Sinyal kontrolü: Sinyal DF'inde o tarih var mı?
            if date in sigs.index:
                sig = sigs.loc[date]
            
            if position is None:
                is_buy = bool(sig["is_buy"]) if sig is not None and "is_buy" in sig else False
                
                if is_buy:
                    sl = float(sig["stop_loss"])
                    risk_per_share = price - sl
                    
                    if risk_per_share <= 0: continue
                    
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
                # Pozisyon Çıkışı
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
        
        # Kapanış pozisyonu (Dönem sonu)
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
                "total_return": "0.0%", "total_trades": "0", "win_rate": "0.0%",
                "avg_win": "$0.00", "avg_loss": "$0.00",
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
st.set_page_config(page_title="Clean Backtest", layout="wide")
st.title("🆕 Yeni Uygulama: Temiz Backtest Motoru")

# Sidebar
st.sidebar.header("⚙️ Ayarlar")
ticker = st.sidebar.selectbox("Sembol", ["AAPL", "GOOGL", "MSFT", "TSLA", "BTC-USD", "ETH-USD"], index=4)
start_date = st.sidebar.date_input("Başlangıç", datetime(2023, 1, 1))
end_date = st.sidebar.date_input("Bitiş", datetime(2023, 12, 31))

st.sidebar.header("📊 Parametreler")
rsi_oversold = st.sidebar.slider("RSI Aşırı Satım Eşiği", 25, 50, 40)
atr_multiplier = st.sidebar.slider("ATR Çarpanı (SL/TP)", 1.0, 3.0, 2.0)
risk_per_trade = st.sidebar.slider("Risk % (Pozisyon Büyüklüğü)", 1.0, 5.0, 2.0) / 100.0

# Ana içerik
if st.button("🎯 Backtest Başlat", type="primary"):
    try:
        with st.spinner("Veri çekiliyor ve indikatörler hesaplanıyor..."):
            extended_start = start_date - timedelta(days=100)
            data = yf.download(ticker, start=extended_start, end=end_date, progress=False)
            
            if data.empty:
                st.error("❌ Veri çekilemedi veya tarih aralığı hatalı."); st.stop()
            
            data = data[(data.index >= pd.to_datetime(start_date)) & (data.index <= pd.to_datetime(end_date))]
            st.success(f"✅ {len(data)} günlük veri yüklendi.")
            st.info(f"📈 Fiyat aralığı: {fmt(data['Close'].min(),2,prefix='$')} - {fmt(data['Close'].max(),2,prefix='$')}")
            
        backtester = CleanSwingBacktest(commission=0.001)
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
            st.subheader("📈 Portföy Değeri")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(equity['date'], equity['equity'], linewidth=2, label='Equity')
            ax.set_title('Portföy Değeri Gelişimi')
            ax.set_ylabel('Equity ($)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            st.subheader("📋 İşlem Detayları")
            disp = trades.copy()
            for c in ["entry_date","exit_date"]:
                if c in disp and pd.api.types.is_datetime64_any_dtype(disp[c]):
                    disp[c] = disp[c].dt.strftime("%Y-%m-%d")
            st.dataframe(disp, use_container_width=True)
        else:
            st.warning("⚠️ Hiç işlem gerçekleşmedi. Lütfen parametreleri (özellikle RSI eşiğini) değiştirerek tekrar deneyin.")
            
    except Exception as e:
        st.error(f"❌ Uygulama Çalışma Hatası: {str(e)}")

st.markdown("---")
st.caption("Yeni ve Güvenli Backtest Sistemi")
