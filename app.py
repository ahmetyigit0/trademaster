import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sayfa ayarı
st.set_page_config(
    page_title="🚀 Yüksek Getirili Kripto Strateji",
    page_icon="💎",
    layout="wide"
)

# Başlık
st.title("💎 Yüksek Getirili Kripto Strateji - %300 Hedef")
st.markdown("---")

# OPTİMİZE ML Strateji sınıfı
class HighReturnMLStrategy:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.is_trained = False
        
    def create_features(self, df):
        """Özellikler oluştur"""
        try:
            features = pd.DataFrame(index=df.index)
            
            # RSI
            for period in [4, 6, 8]:
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=period, min_periods=1).mean()
                avg_loss = loss.rolling(window=period, min_periods=1).mean()
                rs = avg_gain / avg_loss.replace(0, np.nan)
                rs = rs.fillna(1)
                features[f'rsi_{period}h'] = 100 - (100 / (1 + rs))
            
            # EMA
            for span in [4, 8, 12]:
                features[f'ema_{span}h'] = df['Close'].ewm(span=span, adjust=False).mean()
            
            features['ema_cross'] = (features['ema_4h'] - features['ema_12h']) / df['Close']
            
            # Momentum
            features['momentum_2h'] = df['Close'] - df['Close'].shift(2)
            features['momentum_6h'] = df['Close'] - df['Close'].shift(6)
            
            # Volume
            volume_ema_8h = df['Volume'].ewm(span=8).mean()
            features['volume_ratio'] = df['Volume'] / volume_ema_8h.replace(0, 1)
            
            return features.fillna(0).replace([np.inf, -np.inf], 0)
            
        except Exception as e:
            features = pd.DataFrame(index=df.index)
            features['rsi_6h'] = 50
            features['ema_cross'] = 0
            features['volume_ratio'] = 1
            return features
    
    def create_target(self, df, lookahead=3):
        """Hedef değişken"""
        try:
            future_return = df['Close'].shift(-lookahead) / df['Close'] - 1
            
            target = np.zeros(len(df))
            bullish_threshold = 0.008  # %0.8
            bearish_threshold = -0.008
            
            target[future_return > bullish_threshold] = 1
            target[future_return < bearish_threshold] = -1
            
            return pd.Series(target, index=df.index, dtype=int)
            
        except Exception as e:
            return pd.Series(np.zeros(len(df)), index=df.index, dtype=int)
    
    def train_model(self, df):
        """Model eğit"""
        try:
            features = self.create_features(df)
            target = self.create_target(df)
            
            features = features.fillna(0)
            target = target.fillna(0)
            
            if len(features) < 100:
                return 0, pd.DataFrame()
            
            split_idx = int(len(features) * 0.8)
            X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
            y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]
            
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=8,
                min_samples_leaf=4,
                random_state=42
            )
            
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.is_trained = True
            self.feature_columns = features.columns.tolist()
            
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return accuracy, feature_importance
            
        except Exception as e:
            return 0, pd.DataFrame()
    
    def predict_signals(self, df):
        """Sinyal tahmini"""
        try:
            if not self.is_trained or self.model is None:
                return np.zeros(len(df))
            
            features = self.create_features(df)
            features = features.fillna(0)
            predictions = self.model.predict(features)
            
            return predictions
            
        except Exception as e:
            return np.zeros(len(df))

# OPTİMİZE Ana Strateji
class HighReturnStrategy:
    def __init__(self, initial_capital: float = 10000, enable_ml: bool = True):
        self.initial_capital = initial_capital
        self.results = {}
        self.enable_ml = enable_ml
        self.ml_strategy = HighReturnMLStrategy() if enable_ml else None
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Göstergeleri hesapla"""
        try:
            df = df.copy()
            
            # RSI
            for period in [4, 6, 8]:
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=period, min_periods=1).mean()
                avg_loss = loss.rolling(window=period, min_periods=1).mean()
                rs = avg_gain / avg_loss.replace(0, np.nan)
                rs = rs.fillna(1)
                df[f'RSI_{period}h'] = 100 - (100 / (1 + rs))
            
            # EMA
            for span in [4, 8, 12]:
                df[f'EMA_{span}h'] = df['Close'].ewm(span=span, adjust=False).mean()
            
            # Momentum
            for shift in [2, 4, 6]:
                df[f'Momentum_{shift}h'] = df['Close'] - df['Close'].shift(shift)
            
            # Volume
            volume_ema_8h = df['Volume'].ewm(span=8).mean()
            df['Volume_Ratio'] = df['Volume'] / volume_ema_8h.replace(0, 1)
            
            return df.fillna(0)
            
        except Exception as e:
            st.error(f"Göstergeler hesaplanırken hata: {e}")
            df['RSI_6h'] = 50
            df['EMA_4h'] = df['Close']
            df['Volume_Ratio'] = 1
            return df
    
    def generate_signals(self, df: pd.DataFrame, signal_threshold: float = 1.5) -> pd.DataFrame:
        """Yüksek getirili sinyal üret"""
        try:
            df = df.copy()
            df['Signal'] = 0
            
            # ML sinyalleri
            ml_signals = np.zeros(len(df))
            ml_accuracy = 0
            
            if self.enable_ml and self.ml_strategy:
                ml_accuracy, feature_importance = self.ml_strategy.train_model(df)
                if ml_accuracy > 0.45:
                    ml_signals = self.ml_strategy.predict_signals(df)
                    st.success(f"🤖 ML Doğruluğu: %{ml_accuracy:.1f}")
            
            # OPTİMİZE SİNYAL SİSTEMİ
            for i in range(12, len(df)):
                try:
                    rsi_4h = float(df['RSI_4h'].iloc[i])
                    rsi_6h = float(df['RSI_6h'].iloc[i])
                    ema_4h = float(df['EMA_4h'].iloc[i])
                    ema_8h = float(df['EMA_8h'].iloc[i])
                    ema_12h = float(df['EMA_12h'].iloc[i])
                    momentum_2h = float(df['Momentum_2h'].iloc[i])
                    volume_ratio = float(df['Volume_Ratio'].iloc[i])
                    
                    long_signals = 0
                    short_signals = 0
                    
                    # GÜÇLÜ LONG KOŞULLARI
                    if rsi_4h < 25 and rsi_6h < 30:  # Aşırı oversold
                        long_signals += 2.0
                    elif rsi_4h < 30:
                        long_signals += 1.0
                    
                    if ema_4h > ema_8h and ema_8h > ema_12h:  # Güçlü uptrend
                        long_signals += 2.0
                    elif ema_4h > ema_12h:
                        long_signals += 1.0
                    
                    if momentum_2h > 0:
                        long_signals += 0.5
                    
                    if volume_ratio > 1.8:  # Yüksek volume
                        long_signals += 1.0
                    
                    # GÜÇLÜ SHORT KOŞULLARI
                    if rsi_4h > 75 and rsi_6h > 70:  # Aşırı overbought
                        short_signals += 2.0
                    elif rsi_4h > 70:
                        short_signals += 1.0
                    
                    if ema_4h < ema_8h and ema_8h < ema_12h:  # Güçlü downtrend
                        short_signals += 2.0
                    elif ema_4h < ema_12h:
                        short_signals += 1.0
                    
                    if momentum_2h < 0:
                        short_signals += 0.5
                    
                    if volume_ratio > 1.8:
                        short_signals += 1.0
                    
                    # ML GÜÇLENDİRME
                    ml_signal = ml_signals[i]
                    if ml_signal == 1:
                        long_signals += 1.5
                    elif ml_signal == -1:
                        short_signals += 1.5
                    
                    # SİNYAL
                    if long_signals >= signal_threshold:
                        df.loc[df.index[i], 'Signal'] = 1
                    elif short_signals >= signal_threshold:
                        df.loc[df.index[i], 'Signal'] = -1
                        
                except:
                    continue
            
            total_signals = (df['Signal'] != 0).sum()
            st.info(f"**Sinyal Sayısı:** {total_signals}")
                    
            return df
            
        except Exception as e:
            st.error(f"Sinyal oluşturma hatası: {e}")
            df['Signal'] = 0
            return df
    
    def backtest_strategy(self, df: pd.DataFrame, progress_bar, 
                         position_size: float, stop_loss: float, 
                         take_profit: float, max_hold_hours: int = 6) -> dict:
        """Backtest"""
        try:
            capital = self.initial_capital
            position = 0
            entry_price = 0
            entry_capital = 0
            entry_time = None
            trades = []
            total_trades = 0
            winning_trades = 0
            
            for i in range(len(df)):
                if i % 100 == 0:
                    progress_bar.progress(min(i / len(df), 1.0))
                
                current_time = df.index[i]
                current_price = float(df['Close'].iloc[i])
                signal = int(df['Signal'].iloc[i])
                
                # POZİSYON AÇ
                if position == 0 and signal != 0:
                    position = signal
                    entry_price = current_price
                    entry_time = current_time
                    trade_size = min(capital * (position_size / 100), capital)
                    entry_capital = trade_size
                    total_trades += 1
                    
                    trades.append({
                        'entry_time': current_time,
                        'entry_price': entry_price,
                        'position': 'LONG' if position == 1 else 'SHORT',
                        'entry_capital': entry_capital,
                        'exit_time': None,
                        'exit_price': None,
                        'pnl': 0,
                        'pnl_percent': 0,
                        'hold_hours': 0,
                        'status': 'OPEN'
                    })
                
                # POZİSYON TAKİP
                elif position != 0:
                    current_trade = trades[-1]
                    hold_hours = (current_time - entry_time).total_seconds() / 3600
                    
                    if position == 1:  # Long
                        pnl_percent = (current_price - entry_price) / entry_price
                        
                        close_condition = (
                            pnl_percent <= -(stop_loss / 100) or
                            pnl_percent >= (take_profit / 100) or
                            signal == -1 or
                            hold_hours >= max_hold_hours or
                            pnl_percent >= 0.06  # %6 kar
                        )
                        
                        if close_condition:
                            pnl_amount = entry_capital * pnl_percent
                            capital += pnl_amount
                            
                            if pnl_amount > 0:
                                winning_trades += 1
                            
                            trades[-1].update({
                                'exit_time': current_time,
                                'exit_price': current_price,
                                'pnl': pnl_amount,
                                'pnl_percent': pnl_percent * 100,
                                'hold_hours': hold_hours,
                                'status': 'CLOSED'
                            })
                            position = 0
                            entry_price = 0
                            entry_time = None
                    
                    elif position == -1:  # Short
                        pnl_percent = (entry_price - current_price) / entry_price
                        
                        close_condition = (
                            pnl_percent <= -(stop_loss / 100) or
                            pnl_percent >= (take_profit / 100) or
                            signal == 1 or
                            hold_hours >= max_hold_hours or
                            pnl_percent >= 0.06  # %6 kar
                        )
                        
                        if close_condition:
                            pnl_amount = entry_capital * pnl_percent
                            capital += pnl_amount
                            
                            if pnl_amount > 0:
                                winning_trades += 1
                            
                            trades[-1].update({
                                'exit_time': current_time,
                                'exit_price': current_price,
                                'pnl': pnl_amount,
                                'pnl_percent': pnl_percent * 100,
                                'hold_hours': hold_hours,
                                'status': 'CLOSED'
                            })
                            position = 0
                            entry_price = 0
                            entry_time = None
            
            # Açık pozisyonları kapat
            if position != 0 and trades:
                last_price = float(df['Close'].iloc[-1])
                last_time = df.index[-1]
                hold_hours = (last_time - entry_time).total_seconds() / 3600
                
                if position == 1:
                    pnl_percent = (last_price - entry_price) / entry_price
                else:
                    pnl_percent = (entry_price - last_price) / entry_price
                
                pnl_amount = entry_capital * pnl_percent
                capital += pnl_amount
                
                if pnl_amount > 0:
                    winning_trades += 1
                
                trades[-1].update({
                    'exit_time': last_time,
                    'exit_price': last_price,
                    'pnl': pnl_amount,
                    'pnl_percent': pnl_percent * 100,
                    'hold_hours': hold_hours,
                    'status': 'CLOSED'
                })
            
            # SONUÇLAR
            final_capital = max(capital, 0)
            total_return = ((final_capital - self.initial_capital) / self.initial_capital) * 100
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            total_profit = sum(trade['pnl'] for trade in trades if trade['pnl'] > 0)
            total_loss = abs(sum(trade['pnl'] for trade in trades if trade['pnl'] < 0))
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            self.results = {
                'initial_capital': self.initial_capital,
                'final_capital': final_capital,
                'total_return': total_return,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'trades': trades
            }
            
            return self.results
            
        except Exception as e:
            st.error(f"Backtest hatası: {str(e)}")
            return {
                'initial_capital': self.initial_capital,
                'final_capital': self.initial_capital,
                'total_return': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'trades': []
            }

# Streamlit arayüzü
st.sidebar.header("💎 Yüksek Getiri Ayarları")

crypto_symbols = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD", 
    "Solana (SOL-USD)": "SOL-USD",
    "Cardano (ADA-USD)": "ADA-USD"
}

selected_crypto = st.sidebar.selectbox("Kripto Para Seçin:", list(crypto_symbols.keys()))
symbol = crypto_symbols[selected_crypto]

# AYARLAR
st.sidebar.subheader("⚡ Zaman Ayarları")
timeframe = st.sidebar.selectbox("Zaman Periyodu:", ["1h", "2h"], index=0)

end_date = st.sidebar.date_input("Bitiş Tarihi:", datetime.date.today() - datetime.timedelta(days=1))
period_months = st.sidebar.slider("Veri Süresi (Ay):", 1, 6, 3, 1)
start_date = end_date - datetime.timedelta(days=period_months*30)

st.sidebar.subheader("🎯 Risk Ayarları")
initial_capital = st.sidebar.number_input("Başlangıç Sermayesi (USD):", 1000, 100000, 10000, 1000)
position_size = st.sidebar.slider("İşlem Büyüklüğü (%):", 50, 100, 70, 5)

st.sidebar.subheader("🛡️ Stop & Take Profit")
stop_loss = st.sidebar.slider("Stop Loss (%):", 2.0, 8.0, 3.0, 0.5)
take_profit = st.sidebar.slider("Take Profit (%):", 4.0, 15.0, 8.0, 0.5)
max_hold_hours = st.sidebar.slider("Maksimum Bekleme (Saat):", 4, 12, 6, 1)

st.sidebar.subheader("🤖 ML Ayarları")
enable_ml = st.sidebar.checkbox("ML Modelini Etkinleştir", value=True)

# Ana içerik
st.subheader("💎 Yüksek Getirili Strateji - %300 Hedef")

st.success("""
**🎯 STRATEJİ ÖZELLİKLERİ:**
- **Position Size:** %70 (Yüksek)
- **Take Profit:** %8 (İyi)  
- **Stop Loss:** %3 (Orta)
- **Maksimum Bekleme:** 6 saat (Kısa)
- **Hedef:** 3 ayda %300+ getiri

**📈 OPTİMİZASYONLAR:**
- Aşırı oversold/overbought seviyeleri
- Güçlü trend onayı
- Yüksek volume filtresi
- ML destekli sinyaller
""")

# DÜZELTİLMİŞ veri yükleme
@st.cache_data
def load_data(symbol, start_date, end_date, timeframe='1h'):
    try:
        data = yf.download(symbol, start=start_date, end=end_date, interval=timeframe, progress=False)
        if data is None or data.empty:
            return None
        
        st.success(f"✅ {len(data)} adet {timeframe} verisi yüklendi")
        return data
        
    except Exception as e:
        st.error(f"Veri yüklenirken hata: {e}")
        return None

# DÜZELTİLMİŞ veri gösterimi
st.markdown("---")
st.subheader("📊 Veri Yükleme")

data = load_data(symbol, start_date, end_date, timeframe)

if data is not None and not data.empty:
    try:
        close_prices = data['Close']
        first_price = float(close_prices.iloc[0])
        last_price = float(close_prices.iloc[-1])
        price_change = ((last_price - first_price) / first_price) * 100
        
        # DÜZELTME: volatility hesaplamasını güvenli hale getir
        price_changes = close_prices.pct_change().dropna()
        if len(price_changes) > 0:
            volatility = price_changes.std() * np.sqrt(365 * 24) * 100
        else:
            volatility = 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("İlk Fiyat", f"${first_price:.2f}")
        with col2:
            st.metric("Son Fiyat", f"${last_price:.2f}")
        with col3:
            st.metric("Değişim", f"{price_change:+.2f}%")
        with col4:
            st.metric("Volatilite", f"{volatility:.1f}%" if volatility > 0 else "Hesaplanamadı")
            
    except Exception as e:
        st.error(f"Veri gösterilirken hata: {e}")
else:
    st.warning("⚠️ Veri yüklenemedi.")

# SİMÜLASYON BUTONU
st.markdown("---")
st.subheader("🚀 Backtest Başlat")

if st.button("💎 BACKTEST BAŞLAT", type="primary", use_container_width=True):
    if data is not None and not data.empty:
        with st.spinner("Backtest çalışıyor..."):
            start_time = time.time()
            progress_bar = st.progress(0)
            
            try:
                strategy = HighReturnStrategy(initial_capital, enable_ml=enable_ml)
                data_with_indicators = strategy.calculate_indicators(data)
                data_with_signals = strategy.generate_signals(data_with_indicators)
                results = strategy.backtest_strategy(
                    data_with_signals, progress_bar, position_size, stop_loss, take_profit, max_hold_hours
                )
                
                progress_bar.progress(1.0)
                end_time = time.time()
                
                st.success(f"✅ Backtest {end_time - start_time:.1f} saniyede tamamlandı!")
                
                # SONUÇLAR
                st.subheader("📈 Backtest Sonuçları")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Başlangıç", f"${results['initial_capital']:,.0f}")
                with col2:
                    st.metric("Son Sermaye", f"${results['final_capital']:,.0f}", 
                             f"{results['total_return']:+.1f}%")
                with col3:
                    st.metric("Win Rate", f"{results['win_rate']:.1f}%")
                with col4:
                    st.metric("İşlem Sayısı", results['total_trades'])
                
                # PERFORMANS
                if results['total_return'] >= 300:
                    st.success("🎉🎉🎉 HEDEFE ULAŞILDI! %300+ KAR! 🎉🎉🎉")
                    st.balloons()
                elif results['total_return'] >= 200:
                    st.success("🎉 MÜKEMMEL! %200+ KAR!")
                    st.balloons()
                elif results['total_return'] >= 100:
                    st.success("📈 ÇOK İYİ! %100+ KAR!")
                elif results['total_return'] >= 50:
                    st.info("✅ İYİ! %50+ KAR")
                elif results['total_return'] > 0:
                    st.warning("⚠️ DÜŞÜK KAR")
                else:
                    st.error("💥 KAYIP!")
                
                st.info(f"**Profit Factor:** {results['profit_factor']:.2f}")
                    
            except Exception as e:
                st.error(f"Backtest sırasında hata: {str(e)}")
    else:
        st.error("Veri yüklenemedi!")

st.markdown("---")
st.info("""
**💡 TAVSİYELER:**
- **Solana gibi yüksek volatilite coin'lerde deneyin**
- **ML her zaman aktif olsun**
- **2h timeframe daha iyi sonuç verebilir**
- **3 aylık veri ile başlayın**
""")
