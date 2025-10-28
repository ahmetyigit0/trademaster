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
    page_title="🤖 ML Destekli Kripto Strateji Simülasyonu",
    page_icon="🤖",
    layout="wide"
)

# Başlık
st.title("🤖 ML Destekli Kripto Vadeli İşlem Strateji Simülasyonu")
st.markdown("---")

# BASİT VE GÜVENLİ ML Strateji sınıfı
class MLTradingStrategy:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.is_trained = False
        
    def create_advanced_features(self, df):
        """ML için basit özellikler oluştur"""
        try:
            features = pd.DataFrame(index=df.index)
            
            # Sadece 3 temel özellik - çok güvenli
            features['rsi'] = df['RSI'].fillna(50)
            features['ema_cross'] = ((df['EMA_Short'] - df['EMA_Long']) / df['Close']).fillna(0)
            features['volume_ratio'] = df['Volume_Ratio'].fillna(1)
            
            # Sonsuz değerleri temizle
            features = features.replace([np.inf, -np.inf], 0)
            
            return features
            
        except Exception as e:
            # Hata durumunda boş DataFrame dönme, basit özelliklerle devam et
            features = pd.DataFrame(index=df.index)
            features['rsi'] = 50  # Default değer
            features['ema_cross'] = 0
            features['volume_ratio'] = 1
            return features
    
    def create_target_variable(self, df, horizon=2):
        """Basit hedef değişken oluştur"""
        try:
            # Çok basit: 2 gün sonra fiyat artışı
            future_prices = df['Close'].shift(-horizon)
            current_prices = df['Close']
            
            target = np.zeros(len(df))
            
            for i in range(len(df) - horizon):
                if (i + horizon) < len(df):
                    future_price = future_prices.iloc[i + horizon]
                    current_price = current_prices.iloc[i]
                    
                    if pd.notna(future_price) and pd.notna(current_price) and current_price > 0:
                        change = (future_price - current_price) / current_price
                        if change > 0.01:  # %1'den fazla artış
                            target[i] = 1
                        elif change < -0.01:  # %1'den fazla düşüş
                            target[i] = -1
            
            return pd.Series(target, index=df.index, dtype=int)
            
        except Exception as e:
            return pd.Series(np.zeros(len(df)), index=df.index, dtype=int)
    
    def train_model(self, df):
        """Model eğitimi - çok basit"""
        try:
            features = self.create_advanced_features(df)
            target = self.create_target_variable(df)
            
            # NaN'leri temizle
            features = features.fillna(0)
            target = target.fillna(0)
            
            if len(features) < 20:
                return 0, pd.DataFrame()
            
            # Basit split
            split_idx = max(10, int(len(features) * 0.7))
            if split_idx >= len(features):
                return 0, pd.DataFrame()
                
            X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
            y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]
            
            # Çok basit model
            self.model = RandomForestClassifier(
                n_estimators=30,
                max_depth=5,
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
            
            features = self.create_advanced_features(df)
            features = features.fillna(0)
            predictions = self.model.predict(features)
            
            return predictions
            
        except Exception as e:
            return np.zeros(len(df))

# Ana Strateji sınıfı
class CryptoStrategy:
    def __init__(self, initial_capital: float = 10000, enable_ml: bool = False):
        self.initial_capital = initial_capital
        self.results = {}
        self.enable_ml = enable_ml
        self.ml_strategy = MLTradingStrategy() if enable_ml else None
        
    def calculate_advanced_indicators(self, df: pd.DataFrame, rsi_period: int, ema_short: int, ema_long: int) -> pd.DataFrame:
        """Teknik göstergeleri hesapla"""
        try:
            df = df.copy()
            
            # RSI
            delta = df['Close'].diff().fillna(0)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            
            avg_gain = pd.Series(gain).rolling(window=rsi_period, min_periods=1).mean()
            avg_loss = pd.Series(loss).rolling(window=rsi_period, min_periods=1).mean()
            
            rs = avg_gain / np.where(avg_loss == 0, 1, avg_loss)
            rsi = 100 - (100 / (1 + rs))
            
            # EMA'lar
            ema_short_val = df['Close'].ewm(span=ema_short, adjust=False).mean()
            ema_long_val = df['Close'].ewm(span=ema_long, adjust=False).mean()
            
            # Volume
            volume_sma = df['Volume'].rolling(window=20, min_periods=1).mean()
            volume_ratio = df['Volume'] / np.where(volume_sma == 0, 1, volume_sma)
            
            # Sütunları ata
            df['RSI'] = rsi.fillna(50)
            df['EMA_Short'] = ema_short_val
            df['EMA_Long'] = ema_long_val
            df['Volume_Ratio'] = volume_ratio.fillna(1)
            
            return df.fillna(0)
            
        except Exception as e:
            st.error(f"Göstergeler hesaplanırken hata: {e}")
            return df
    
    def generate_advanced_signals(self, df: pd.DataFrame, rsi_oversold: float, rsi_overbought: float, signal_threshold: float) -> pd.DataFrame:
        """Sinyal oluştur"""
        try:
            df = df.copy()
            df['Signal'] = 0
            
            # ML sinyalleri
            ml_signals = np.zeros(len(df))
            if self.enable_ml and self.ml_strategy:
                try:
                    ml_accuracy, feature_importance = self.ml_strategy.train_model(df)
                    if ml_accuracy > 0.4:
                        ml_signals = self.ml_strategy.predict_signals(df)
                        st.success(f"🤖 ML Doğruluğu: %{ml_accuracy:.1f}")
                except:
                    pass
            
            # Geleneksel sinyaller
            for i in range(1, len(df)):
                try:
                    rsi = float(df['RSI'].iloc[i])
                    ema_short = float(df['EMA_Short'].iloc[i])
                    ema_long = float(df['EMA_Long'].iloc[i])
                    volume_ratio = float(df['Volume_Ratio'].iloc[i])
                    
                    long_signals = 0
                    short_signals = 0
                    
                    # LONG
                    if rsi < rsi_oversold:
                        long_signals += 1.0
                    if ema_short > ema_long:
                        long_signals += 1.0
                    if volume_ratio > 1.2:
                        long_signals += 0.5
                    
                    # SHORT
                    if rsi > rsi_overbought:
                        short_signals += 1.0
                    if ema_short < ema_long:
                        short_signals += 1.0
                    if volume_ratio > 1.2:
                        short_signals += 0.5
                    
                    # ML
                    if ml_signals[i] == 1:
                        long_signals += 1.0
                    elif ml_signals[i] == -1:
                        short_signals += 1.0
                    
                    if long_signals >= signal_threshold:
                        df.loc[df.index[i], 'Signal'] = 1
                    elif short_signals >= signal_threshold:
                        df.loc[df.index[i], 'Signal'] = -1
                        
                except:
                    continue
                    
            return df
            
        except Exception as e:
            st.error(f"Sinyal oluşturma hatası: {e}")
            df['Signal'] = 0
            return df
    
    def backtest_advanced_strategy(self, df: pd.DataFrame, progress_bar, position_size: float,
                                 stop_loss: float, take_profit: float) -> dict:
        """Backtest yap"""
        try:
            capital = self.initial_capital
            position = 0
            entry_price = 0
            entry_capital = 0
            trades = []
            total_trades = 0
            winning_trades = 0
            
            for i in range(len(df)):
                if i % 10 == 0:
                    progress_bar.progress(min(i / len(df), 1.0))
                
                current_date = df.index[i]
                current_price = float(df['Close'].iloc[i])
                signal = int(df['Signal'].iloc[i])
                
                # Pozisyon aç
                if position == 0 and signal != 0:
                    position = signal
                    entry_price = current_price
                    trade_size = min(capital * (position_size / 100), capital)
                    entry_capital = trade_size
                    total_trades += 1
                    
                    trades.append({
                        'entry_time': current_date,
                        'entry_price': entry_price,
                        'position': 'LONG' if position == 1 else 'SHORT',
                        'entry_capital': entry_capital,
                        'exit_time': None,
                        'exit_price': None,
                        'pnl': 0,
                        'status': 'OPEN'
                    })
                
                # Pozisyon kapat
                elif position != 0:
                    current_trade = trades[-1]
                    
                    if position == 1:  # Long
                        pnl_percent = (current_price - entry_price) / entry_price
                        
                        if pnl_percent <= -(stop_loss / 100) or pnl_percent >= (take_profit / 100) or signal == -1:
                            pnl_amount = entry_capital * pnl_percent
                            capital += pnl_amount
                            
                            if pnl_amount > 0:
                                winning_trades += 1
                            
                            trades[-1].update({
                                'exit_time': current_date,
                                'exit_price': current_price,
                                'pnl': pnl_amount,
                                'status': 'CLOSED'
                            })
                            position = 0
                    
                    elif position == -1:  # Short
                        pnl_percent = (entry_price - current_price) / entry_price
                        
                        if pnl_percent <= -(stop_loss / 100) or pnl_percent >= (take_profit / 100) or signal == 1:
                            pnl_amount = entry_capital * pnl_percent
                            capital += pnl_amount
                            
                            if pnl_amount > 0:
                                winning_trades += 1
                            
                            trades[-1].update({
                                'exit_time': current_date,
                                'exit_price': current_price,
                                'pnl': pnl_amount,
                                'status': 'CLOSED'
                            })
                            position = 0
            
            # Açık pozisyonları kapat
            if position != 0 and trades:
                last_price = float(df['Close'].iloc[-1])
                if position == 1:
                    pnl_percent = (last_price - entry_price) / entry_price
                else:
                    pnl_percent = (entry_price - last_price) / entry_price
                
                pnl_amount = entry_capital * pnl_percent
                capital += pnl_amount
                
                if pnl_amount > 0:
                    winning_trades += 1
                
                trades[-1].update({
                    'exit_time': df.index[-1],
                    'exit_price': last_price,
                    'pnl': pnl_amount,
                    'status': 'CLOSED'
                })
            
            # Sonuçlar
            final_capital = max(capital, 0)
            total_return = ((final_capital - self.initial_capital) / self.initial_capital) * 100
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            self.results = {
                'initial_capital': self.initial_capital,
                'final_capital': final_capital,
                'total_return': total_return,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
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
                'trades': []
            }

# Streamlit arayüzü
st.sidebar.header("⚙️ Simülasyon Ayarları")

# Kripto seçimi
crypto_symbols = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD", 
    "Binance Coin (BNB-USD)": "BNB-USD"
}

selected_crypto = st.sidebar.selectbox(
    "Kripto Para Seçin:",
    list(crypto_symbols.keys())
)

symbol = crypto_symbols[selected_crypto]

# ML Ayarları
st.sidebar.subheader("🤖 ML Ayarları")
enable_ml = st.sidebar.checkbox("Machine Learning Modelini Etkinleştir", value=True)

# Tarih ayarları
st.sidebar.subheader("📅 Tarih Ayarları")
end_date = st.sidebar.date_input(
    "Simülasyon Bitiş Tarihi:",
    datetime.date.today() - datetime.timedelta(days=1)
)

period_days = st.sidebar.slider(
    "Simülasyon Süresi (Gün):",
    min_value=60,
    max_value=180,
    value=90,
    step=30
)

start_date = end_date - datetime.timedelta(days=period_days)

# Diğer ayarlar
initial_capital = st.sidebar.number_input("Başlangıç Sermayesi (USD):", 1000, 100000, 10000)
position_size = st.sidebar.slider("İşlem Büyüklüğü (%):", 10, 100, 100)

st.sidebar.subheader("📊 Teknik Gösterge Ayarları")
rsi_period = st.sidebar.slider("RSI Periyodu:", 5, 30, 14)
ema_short = st.sidebar.slider("Kısa EMA Periyodu:", 5, 20, 9)
ema_long = st.sidebar.slider("Uzun EMA Periyodu:", 15, 50, 21)

st.sidebar.subheader("🎯 Sinyal Ayarları")
rsi_oversold = st.sidebar.slider("RSI Oversold Seviyesi:", 20, 45, 30)
rsi_overbought = st.sidebar.slider("RSI Overbought Seviyesi:", 55, 80, 70)
signal_threshold = st.sidebar.slider("Sinyal Eşik Değeri:", 1.0, 3.0, 1.5)

st.sidebar.subheader("🛡️ Risk Yönetimi")
stop_loss = st.sidebar.slider("Stop Loss (%):", 1, 10, 5)
take_profit = st.sidebar.slider("Take Profit (%):", 1, 20, 10)

# Ana içerik
st.subheader("🎯 Basit ve Etkili Strateji")

if enable_ml:
    st.success("🤖 **ML DESTEKLİ** - Basit ML ile geliştirilmiş")
else:
    st.info("📊 **GELENEKSEL** - Saf teknik analiz")

# Veri yükleme - ÇOK DAHA GÜVENLİ
@st.cache_data
def load_data(symbol, start_date, end_date):
    try:
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if data is None or data.empty:
            return None
            
        # Veri yapısını kontrol et
        if 'Close' not in data.columns:
            return None
            
        return data
        
    except Exception as e:
        st.error(f"Veri yüklenirken hata: {e}")
        return None

# Simülasyon
st.markdown("---")
st.subheader("🚀 Backtest Simülasyonu")

data = load_data(symbol, start_date, end_date)

if data is not None and not data.empty:
    try:
        # ÇOK GÜVENLİ veri erişimi
        close_prices = data['Close']
        if len(close_prices) > 0:
            first_price = float(close_prices.iloc[0])
            last_price = float(close_prices.iloc[-1])
            price_change = ((last_price - first_price) / first_price) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("İlk Fiyat", f"${first_price:.2f}")
            with col2:
                st.metric("Son Fiyat", f"${last_price:.2f}")
            with col3:
                st.metric("Değişim", f"{price_change:+.2f}%")
        else:
            st.warning("Veri yüklenemedi")
    except Exception as e:
        st.error(f"Veri gösterilirken hata: {e}")
else:
    st.warning("⚠️ Veri yüklenemedi. Lütfen tarih aralığını kontrol edin.")

# Simülasyon butonu
if st.button("🎯 Backtest Başlat", type="primary", use_container_width=True):
    if data is not None and not data.empty:
        with st.spinner("Simülasyon çalışıyor..."):
            progress_bar = st.progress(0)
            
            try:
                strategy = CryptoStrategy(initial_capital, enable_ml=enable_ml)
                
                # Göstergeleri hesapla
                data_with_indicators = strategy.calculate_advanced_indicators(
                    data, rsi_period, ema_short, ema_long
                )
                
                # Sinyalleri oluştur
                data_with_signals = strategy.generate_advanced_signals(
                    data_with_indicators, rsi_oversold, rsi_overbought, signal_threshold
                )
                
                # Backtest
                results = strategy.backtest_advanced_strategy(
                    data_with_signals, progress_bar, position_size, stop_loss, take_profit
                )
                
                progress_bar.progress(1.0)
                
                # Sonuçları göster
                st.subheader("📊 Simülasyon Sonuçları")
                
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
                
                if results['win_rate'] > 50:
                    st.success("🎉 Harika sonuç! Strateji başarılı.")
                elif results['win_rate'] > 0:
                    st.info("📈 İyi sonuç! Strateji karlı.")
                    
            except Exception as e:
                st.error(f"Simülasyon sırasında hata: {str(e)}")
    else:
        st.error("Veri yüklenemedi!")

st.markdown("---")
st.info("""
**⚠️ Uyarı:** Bu simülasyon sadece eğitim amaçlıdır. 
Gerçek trading için kullanmayın. Geçmiş performans gelecek sonuçların garantisi değildir.
""")
