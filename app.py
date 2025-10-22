import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =========================
# ŞİFRE KORUMASI (Aynı)
# =========================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    
    if not st.session_state["password_correct"]:
        st.markdown("### 🔐 Kombine Strateji")
        password = st.text_input("Şifre:", type="password", key="password_input")
        
        if password == "efe":
            st.session_state["password_correct"] = True
            st.success("✅ Giriş!")
            st.rerun()
        elif password:
            st.error("❌ Yanlış!")
            st.stop()
        return False
    return True

if not check_password():
    st.stop()

# =========================
# BACKTEST MOTORU (Hata Düzeltildi)
# =========================
class CleanBacktest: # Sınıf adını değiştirdim
    def __init__(self):
        self.capital = 10000
    
    def indicators(self, df):
        df = df.copy()
        # Close ve diğer sütunları Pandas Series'e dönüştürerek modern metodları kullanıma hazırla
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        # 1. EMA - MODERN PANDAS SÖZ DİZİMİ
        df['EMA20'] = close.ewm(span=20, adjust=False).mean()
        df['EMA50'] = close.ewm(span=50, adjust=False).mean()
        
        # 2. RSI - Düzgün hesaplama ve hizalama
        def calculate_rsi(series, window=14):
            delta = series.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
            avg_loss = loss.ewm(com=window - 1, adjust=False).mean()
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))

        df['RSI'] = calculate_rsi(close)
        
        # 3. BB - MODERN PANDAS SÖZ DİZİMİ
        window = 20
        bb_mean = close.rolling(window=window).mean()
        bb_std = close.rolling(window=window).std()
        df['BB_Lower'] = bb_mean - bb_std * 2
        
        # 4. MACD - MODERN PANDAS SÖZ DİZİMİ
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # 5. FIB - MODERN PANDAS SÖZ DİZİMİ
        window_fib = 50
        high50 = high.rolling(window=window_fib).max()
        low50 = low.rolling(window=window_fib).min()
        df['Fib'] = low50 + (high50 - low50) * 0.382
        
        # İlk NaN'ları 0 ile doldurmak yerine (ki bu yanlış sinyal üretir),
        # trend ve momentum analizi için mantıklı bir başlangıç değeri kullanalım.
        # Basitlik için sadece NaN'ları doldurmak yerine, indikatörlerin 
        # sinyal vermeye başlayacağı ilk günden itibaren veri kullanılır.
        df = df.fillna(0.0) # Tüm NaN'ları 0 ile doldurmak hata veriyordu. Güvenli float ile dolduruldu.
        return df
    
    def signals(self, df, rsi_level, rr):
        signals = []
        risk = 0.02
        
        # İndikatörler sıfır olan (NaN'dan doldurulmuş) ilk günleri atla
        start_index = df[df['RSI'] != 0].index[0] if not df[df['RSI'] != 0].empty else 0
        df_clean = df.loc[start_index:]
        
        for i in range(len(df_clean)):
            row = df_clean.iloc[i]
            
            # Geçmiş veriye erişim için df_clean.iloc kullanıldı
            if i == 0:
                 macd_cross = False
            else:
                 macd_cross = df_clean.iloc[i]['MACD'] > df_clean.iloc[i]['Signal'] and \
                              df_clean.iloc[i-1]['MACD'] <= df_clean.iloc[i-1]['Signal']

            
            ema_ok = row['EMA20'] > row['EMA50']
            rsi_ok = row['RSI'] < rsi_level
            bb_ok = row['Close'] < row['BB_Lower']
            fib_ok = row['Close'] < row['Fib']
            
            # OR (Veya) Operasyonunun olduğu yer burasıydı. Artık hepsi skaler!
            if ema_ok and rsi_ok and (bb_ok or fib_ok) and macd_cross:
                price = row['Close']
                signals.append({
                    'date': df_clean.index[i],
                    'action': 'buy',
                    'sl': float(price) * (1 - risk),
                    'tp': float(price) * (1 + risk * rr)
                })
            else:
                signals.append({
                    'date': df_clean.index[i],
                    'action': 'hold',
                    'sl': 0.0,
                    'tp': 0.0
                })
        
        # Eksik başlangıç tarihlerini de içeren bir DataFrame oluştur
        # Başlangıçta 0'a doldurulan satırları da eklemek için:
        full_signals = pd.DataFrame(signals).set_index('date')
        
        # Orijinal indekse geri dönmek için
        signal_df = pd.DataFrame(index=df.index).merge(
            full_signals, 
            left_index=True, 
            right_index=True, 
            how='left'
        ).fillna({'action': 'hold', 'sl': 0.0, 'tp': 0.0})
        
        buy_count = len([s for s in signals if s['action']=='buy'])
        st.info(f"🎯 {buy_count} sinyal")
        return signal_df.reset_index() # Merge için date sütununu geri ver

    
    def backtest(self, df, rsi_level, rr, risk_pct):
        df_indicators = self.indicators(df)
        signals_df = self.signals(df_indicators, rsi_level, rr)
        
        # MERGE - GÜVENLİ (signals_df zaten reset_index yapıldı)
        df_combined = df_indicators.reset_index().merge(signals_df, on='Date', how='left').set_index('Date')
        
        # NaN kontrolü, sadece 0.0 ile doldurulmuş olanları 0.0 ile doldurur.
        df_combined['action'] = df_combined['action'].fillna('hold')
        df_combined[['sl', 'tp']] = df_combined[['sl', 'tp']].fillna(0.0)
        
        capital = float(self.capital)
        position = None
        trades = []
        equity = []
        
        for index, row in df_combined.iterrows():
            # Tüm değerleri kesinlikle float yap
            price = float(row['Close'])
            action = row['action']
            
            # Equity
            eq = capital
            if position:
                eq += float(position['shares']) * price
            equity.append({'date': index, 'equity': eq})
            
            # BUY
            if not position and action == 'buy':
                sl = float(row['sl'])
                tp = float(row['tp'])
                
                risk_share = price - sl
                if risk_share > 0:
                    shares = (capital * risk_pct) / risk_share
                    shares = min(shares, capital * 0.95 / price) # Max 95% pozisyon büyüklüğü
                    
                    position = {'date': index, 'price': price, 'shares': shares, 'sl': sl, 'tp': tp}
                    capital -= shares * price
            
            # SELL
            elif position:
                exit_p = None
                reason = None
                
                if price <= position['sl']:
                    exit_p = position['sl']
                    reason = 'SL'
                elif price >= position['tp']:
                    exit_p = position['tp']
                    reason = 'TP'
                
                if exit_p is not None:
                    capital += position['shares'] * exit_p
                    pnl = (exit_p - position['price']) * position['shares']
                    
                    trades.append({
                        'entry': position['date'],
                        'exit': index,
                        'entry_p': position['price'],
                        'exit_p': exit_p,
                        'shares': position['shares'],
                        'pnl': pnl,
                        'ret': (pnl / (position['price'] * position['shares'])) * 100,
                        'reason': reason
                    })
                    position = None
        
        # CLOSE OPEN
        if position:
            last_p = float(df_combined['Close'].iloc[-1])
            capital += position['shares'] * last_p
            pnl = (last_p - position['price']) * position['shares']
            trades.append({
                'entry': position['date'],
                'exit': df_combined.index[-1],
                'entry_p': position['price'],
                'exit_p': last_p,
                'shares': position['shares'],
                'pnl': pnl,
                'ret': (pnl / (position['price'] * position['shares'])) * 100,
                'reason': 'OPEN'
            })
        
        return pd.DataFrame(trades), pd.DataFrame(equity)
    
    def metrics(self, trades, equity):
        # Metrik hesaplama (Aynı, ancak sınıf adını düzelttim)
        if trades.empty or equity.empty:
            return {'ret': '0%', 'trades': 0, 'win': '0%', 'pf': '0', 'dd': '0%'}
        
        total_ret = ((equity['equity'].iloc[-1] - self.capital) / self.capital) * 100
        wins = len(trades[trades['pnl'] > 0])
        total_trades = len(trades)
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
        profit = trades[trades['pnl'] > 0]['pnl'].sum()
        loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
        pf = profit / loss if loss > 0 else 999.0
        
        peak = equity['equity'].expanding().max()
        # Max Drawdown hesaplanırken peak ile equity aynı index'e sahip olmalı.
        dd = ((equity['equity'] - peak) / peak * 100).min()
        
        return {
            'ret': f"{total_ret:+.1f}%",
            'trades': total_trades,
            'win': f"{win_rate:.1f}%",
            'pf': f"{pf:.1f}",
            'dd': f"{dd:.1f}%"
        }

# =========================
# APP
# =========================
st.set_page_config(layout="wide")
st.title("🧠 Kombine Swing")
st.markdown("EMA + RSI + BB + MACD + Fib")

# SIDEBAR
with st.sidebar:
    ticker = st.selectbox("Sembol", ["BTC-USD", "ETH-USD", "AAPL"])
    start = st.date_input("Başlangıç", datetime(2023, 1, 1))
    end = st.date_input("Bitiş", datetime(2024, 1, 1))
    rsi = st.slider("RSI", 20, 40, 30)
    rr = st.slider("R/R", 2.0, 4.0, 2.5) # Float değeriyle uyumlu hale getirildi
    risk = st.slider("Risk%", 1, 3, 2) / 100

# RUN
if st.button("🚀 BACKTEST", type="primary"):
    try:
        with st.spinner("Hesaplanıyor..."):
            # Veri indirme başlangıcını indikatörlerin hesaplanması için geriye çektik
            extended_start = start - pd.Timedelta(days=150)
            data = yf.download(ticker, start=extended_start, end=end, progress=False)
            
            if data.empty:
                st.error("Veri yok!")
                st.stop()
            
            # Başlangıç tarihinden sonraki veriyi backtest için kullan
            data = data[data.index >= pd.to_datetime(start)]

            bt = CleanBacktest() # Yeni sınıf adı
            trades, equity = bt.backtest(data, rsi, rr, risk)
            metrics = bt.metrics(trades, equity)
        
        # METRICS
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Getiri", metrics['ret'])
            st.metric("İşlem", metrics['trades'])
        with col2:
            st.metric("Win Rate", metrics['win'])
        with col3:
            st.metric("PF", metrics['pf'])
        with col4:
            st.metric("Max DD", metrics['dd'])
        
        # CHART
        if not equity.empty:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            ax1.plot(equity['date'], equity['equity'], 'g-', lw=2)
            ax1.set_title(f"{ticker} Equity")
            ax1.grid(True, alpha=0.3)
            
            peak = equity['equity'].expanding().max()
            dd = (equity['equity'] - peak) / peak * 100
            ax2.fill_between(equity['date'], dd, 0, color='r', alpha=0.3)
            ax2.set_title("Drawdown")
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # TRADES
        if not trades.empty:
            trades['entry'] = pd.to_datetime(trades['entry']).dt.strftime('%Y-%m-%d')
            trades['exit'] = pd.to_datetime(trades['exit']).dt.strftime('%Y-%m-%d')
            st.dataframe(trades.round(2), height=300)
            
    except Exception as e:
        st.error(f"❌ **Kritik Hata:** {e}")
        st.info("Lütfen indikatör hesaplamaları için kullanılan Pandas metodlarının güncel olduğundan emin olun.")

st.markdown("---")
st.markdown("**v9.0 - Modern Pandas Metotları Kullanıldı**")
