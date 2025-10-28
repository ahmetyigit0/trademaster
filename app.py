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

# TAMAMEN DÜZELTİLMİŞ ML Strateji sınıfı
class MLTradingStrategy:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.is_trained = False
        
    def create_advanced_features(self, df):
        """ML için gelişmiş özellikler oluştur - BASİT VE GÜVENLİ"""
        try:
            # Önce boş bir DataFrame oluştur
            features = pd.DataFrame(index=df.index)
            
            # SADECE 5 TEMEL ÖZELLİK - hata riskini azalt
            features['rsi'] = df['RSI'].fillna(50)
            features['ema_cross'] = ((df['EMA_Short'] - df['EMA_Long']) / df['EMA_Long']).fillna(0)
            features['macd_hist'] = df['MACD_Histogram'].fillna(0)
            features['volume_ratio'] = df['Volume_Ratio'].fillna(1)
            features['momentum'] = df['Momentum'].fillna(0)
            
            # Sonsuz değerleri temizle
            features = features.replace([np.inf, -np.inf], 0)
            
            return features
            
        except Exception as e:
            st.error(f"Özellik oluşturma hatası: {e}")
            # Hata durumunda çok basit özelliklerle dön
            basic_features = pd.DataFrame(index=df.index)
            basic_features['rsi'] = df['RSI'].fillna(50)
            basic_features['ema_cross'] = 0
            return basic_features
    
    def create_target_variable(self, df, horizon=2, threshold=0.02):  # Daha kısa horizon
        """Hedef değişken oluştur - ÇOK DAHA BASİT"""
        try:
            # Çok basit hedef: 2 gün sonra fiyat yukarı mı aşağı mı?
            future_price = df['Close'].shift(-horizon)
            current_price = df['Close']
            
            # Gelecek fiyat NaN olmayanları bul
            valid_mask = future_price.notna() & (current_price > 0)
            
            # Basit hedef: 1=Yukarı, -1=Aşağı, 0=Değişim yok
            target = np.zeros(len(df))
            
            for i in range(len(df)):
                if valid_mask.iloc[i]:
                    price_change = (future_price.iloc[i] - current_price.iloc[i]) / current_price.iloc[i]
                    if price_change > threshold:
                        target[i] = 1    # AL
                    elif price_change < -threshold:
                        target[i] = -1   # SAT
                    # else 0 kalır (BEKLE)
            
            return pd.Series(target, index=df.index, dtype=int)
            
        except Exception as e:
            st.error(f"Hedef değişken hatası: {e}")
            return pd.Series(np.zeros(len(df)), index=df.index, dtype=int)
    
    def train_model(self, df):
        """Model eğitimi - ÇOK DAHA BASİT"""
        try:
            # Özellikler ve hedef
            features = self.create_advanced_features(df)
            target = self.create_target_variable(df)
            
            # NaN'leri temizle
            features = features.fillna(0)
            target = target.fillna(0)
            
            # Sadece target'ı -1,0,1 olanları al
            valid_targets = target[target.isin([-1, 0, 1])]
            features = features.loc[valid_targets.index]
            target = valid_targets
            
            if len(features) < 30:
                st.warning(f"Eğitim için yeterli veri yok: {len(features)} örnek")
                return 0, pd.DataFrame()
            
            # Sınıf dağılımını kontrol et
            unique, counts = np.unique(target, return_counts=True)
            class_dist = dict(zip(unique, counts))
            st.info(f"Sınıf dağılımı: {class_dist}")
            
            # Eğer sadece 1 sınıf varsa, model eğitme
            if len(unique) < 2:
                st.warning("Yeterli sınıf çeşitliliği yok, geleneksel yöntem kullanılacak")
                return 0, pd.DataFrame()
            
            # Basit train-test split
            split_idx = int(len(features) * 0.7)  # %70 train
            X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
            y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]
            
            # Daha basit model
            self.model = RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            
            # Eğitim
            self.model.fit(X_train, y_train)
            
            # Test
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.is_trained = True
            self.feature_columns = features.columns.tolist()
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return accuracy, feature_importance
            
        except Exception as e:
            st.warning(f"ML model eğitilemedi: {e}")
            return 0, pd.DataFrame()
    
    def predict_signals(self, df):
        """ML modeli ile sinyal tahmini"""
        try:
            if not self.is_trained or self.model is None:
                return np.zeros(len(df))
            
            features = self.create_advanced_features(df)
            if features.empty:
                return np.zeros(len(df))
            
            # Tahmin
            features_clean = features.fillna(0)
            predictions = self.model.predict(features_clean)
            
            return predictions
            
        except Exception as e:
            st.error(f"Tahmin hatası: {e}")
            return np.zeros(len(df))

# Gelişmiş Strateji sınıfı (ML entegreli)
class CryptoStrategy:
    def __init__(self, initial_capital: float = 10000, enable_ml: bool = False):
        self.initial_capital = initial_capital
        self.results = {}
        self.enable_ml = enable_ml
        self.ml_strategy = MLTradingStrategy() if enable_ml else None
        
    def calculate_advanced_indicators(self, df: pd.DataFrame, rsi_period: int, ema_short: int, ema_long: int, 
                                   macd_fast: int, macd_slow: int, macd_signal: int) -> pd.DataFrame:
        """Gelişmiş teknik göstergeleri hesapla"""
        try:
            df = df.copy()
            
            # RSI hesaplama - daha güvenli
            delta = df['Close'].diff().fillna(0)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            
            avg_gain = pd.Series(gain).rolling(window=rsi_period, min_periods=1).mean()
            avg_loss = pd.Series(loss).rolling(window=rsi_period, min_periods=1).mean()
            
            # Sıfıra bölme hatasını önle
            rs = avg_gain / np.where(avg_loss == 0, 1, avg_loss)
            rsi = 100 - (100 / (1 + rs))
            
            # EMA'lar
            ema_short_val = df['Close'].ewm(span=ema_short, adjust=False).mean()
            ema_long_val = df['Close'].ewm(span=ema_long, adjust=False).mean()
            
            # MACD
            exp1 = df['Close'].ewm(span=macd_fast, adjust=False).mean()
            exp2 = df['Close'].ewm(span=macd_slow, adjust=False).mean()
            macd = exp1 - exp2
            macd_signal_val = macd.ewm(span=macd_signal, adjust=False).mean()
            macd_histogram = macd - macd_signal_val
            
            # Momentum
            momentum = df['Close'] - df['Close'].shift(5)
            
            # Volume
            volume_sma = df['Volume'].rolling(window=20, min_periods=1).mean()
            volume_ratio = df['Volume'] / np.where(volume_sma == 0, 1, volume_sma)
            
            # Tüm sütunları ata
            df['RSI'] = rsi.fillna(50)
            df['EMA_Short'] = ema_short_val
            df['EMA_Long'] = ema_long_val
            df['MACD'] = macd
            df['MACD_Signal'] = macd_signal_val
            df['MACD_Histogram'] = macd_histogram
            df['Momentum'] = momentum.fillna(0)
            df['Volume_SMA'] = volume_sma
            df['Volume_Ratio'] = volume_ratio.fillna(1)
            
            return df.fillna(0)
            
        except Exception as e:
            st.error(f"Göstergeler hesaplanırken hata: {e}")
            return df
    
    def generate_advanced_signals(self, df: pd.DataFrame, rsi_oversold: float, rsi_overbought: float,
                                volume_threshold: float, signal_threshold: float) -> pd.DataFrame:
        """Gelişmiş alım-satım sinyalleri oluştur"""
        try:
            df = df.copy()
            df['Signal'] = 0  # 0: Bekle, 1: Long, -1: Short
            
            # ML sinyalleri - hata durumunda sessizce devam et
            ml_signals = np.zeros(len(df))
            ml_accuracy = 0
            
            if self.enable_ml and self.ml_strategy:
                try:
                    ml_accuracy, feature_importance = self.ml_strategy.train_model(df)
                    if ml_accuracy > 0.4:  # Düşük eşik
                        ml_signals = self.ml_strategy.predict_signals(df)
                        if ml_accuracy > 0:
                            st.success(f"🤖 ML Model Doğruluğu: %{ml_accuracy:.1f}")
                            
                            if not feature_importance.empty:
                                st.write("**Özellik Önem Sıralaması:**")
                                st.dataframe(feature_importance)
                except Exception as e:
                    # ML hatasını görmezden gel, geleneksel devam et
                    pass
            
            # GELENEKSEL SİNYAL ÜRETİMİ (ana odak)
            for i in range(1, len(df)):
                try:
                    if i < 20:  # Daha az bekle
                        continue
                        
                    rsi = float(df['RSI'].iloc[i])
                    ema_short = float(df['EMA_Short'].iloc[i])
                    ema_long = float(df['EMA_Long'].iloc[i])
                    macd = float(df['MACD'].iloc[i])
                    macd_prev = float(df['MACD'].iloc[i-1])
                    macd_signal = float(df['MACD_Signal'].iloc[i])
                    macd_signal_prev = float(df['MACD_Signal'].iloc[i-1])
                    momentum = float(df['Momentum'].iloc[i])
                    volume_ratio = float(df['Volume_Ratio'].iloc[i])
                    
                    # Geleneksel sinyal puanları
                    long_signals = 0
                    short_signals = 0
                    
                    # LONG sinyalleri (daha agresif)
                    if rsi < rsi_oversold:
                        long_signals += 1.0
                    if ema_short > ema_long:
                        long_signals += 1.0
                    if macd > macd_signal and macd_prev <= macd_signal_prev:
                        long_signals += 1.0
                    if momentum > 0:
                        long_signals += 0.5
                    if volume_ratio > volume_threshold:
                        long_signals += 0.5
                    
                    # SHORT sinyalleri
                    if rsi > rsi_overbought:
                        short_signals += 1.0
                    if ema_short < ema_long:
                        short_signals += 1.0
                    if macd < macd_signal and macd_prev >= macd_signal_prev:
                        short_signals += 1.0
                    if momentum < 0:
                        short_signals += 0.5
                    if volume_ratio > volume_threshold:
                        short_signals += 0.5
                    
                    # ML sinyali ekle (eğer varsa)
                    ml_signal = ml_signals[i]
                    if ml_signal == 1:
                        long_signals += 1.0
                    elif ml_signal == -1:
                        short_signals += 1.0
                    
                    # Sinyal belirleme
                    if long_signals >= signal_threshold:
                        df.loc[df.index[i], 'Signal'] = 1
                    elif short_signals >= signal_threshold:
                        df.loc[df.index[i], 'Signal'] = -1
                        
                except Exception:
                    continue
                    
            # Sinyal istatistiklerini göster
            signals_count = (df['Signal'] != 0).sum()
            st.info(f"Üretilen sinyal sayısı: {signals_count}")
                    
            return df
            
        except Exception as e:
            st.error(f"Sinyal oluşturma hatası: {e}")
            df['Signal'] = 0
            return df
    
    def backtest_advanced_strategy(self, df: pd.DataFrame, progress_bar, position_size: float,
                                 stop_loss: float, take_profit: float, max_profit: float) -> dict:
        """Gelişmiş stratejiyi backtest et"""
        try:
            capital = self.initial_capital
            position = 0
            entry_price = 0
            entry_capital = 0
            trades = []
            total_trades = 0
            winning_trades = 0
            
            total_rows = len(df)
            
            for i in range(len(df)):
                if i % 10 == 0 and total_rows > 0:
                    progress_bar.progress(min(i / total_rows, 1.0))
                
                current_date = df.index[i]
                current_price = float(df['Close'].iloc[i])
                signal = int(df['Signal'].iloc[i])
                
                # Pozisyon açma
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
                        'status': 'OPEN',
                        'pnl_percent': 0
                    })
                
                # Pozisyon kapatma
                elif position != 0:
                    current_trade = trades[-1]
                    
                    if position == 1:  # Long pozisyon
                        pnl_percent = (current_price - entry_price) / entry_price
                        
                        close_condition = (
                            pnl_percent <= -(stop_loss / 100) or
                            pnl_percent >= (take_profit / 100) or
                            signal == -1 or
                            pnl_percent >= (max_profit / 100)
                        )
                        
                        if close_condition:
                            pnl_amount = entry_capital * pnl_percent
                            capital += pnl_amount
                            
                            if pnl_amount > 0:
                                winning_trades += 1
                            
                            trades[-1].update({
                                'exit_time': current_date,
                                'exit_price': current_price,
                                'pnl': pnl_amount,
                                'status': 'CLOSED',
                                'pnl_percent': pnl_percent * 100
                            })
                            position = 0
                            entry_price = 0
                    
                    elif position == -1:  # Short pozisyon
                        pnl_percent = (entry_price - current_price) / entry_price
                        
                        close_condition = (
                            pnl_percent <= -(stop_loss / 100) or
                            pnl_percent >= (take_profit / 100) or
                            signal == 1 or
                            pnl_percent >= (max_profit / 100)
                        )
                        
                        if close_condition:
                            pnl_amount = entry_capital * pnl_percent
                            capital += pnl_amount
                            
                            if pnl_amount > 0:
                                winning_trades += 1
                            
                            trades[-1].update({
                                'exit_time': current_date,
                                'exit_price': current_price,
                                'pnl': pnl_amount,
                                'status': 'CLOSED',
                                'pnl_percent': pnl_percent * 100
                            })
                            position = 0
                            entry_price = 0
            
            # Açık pozisyonları kapat
            if position != 0 and trades:
                current_trade = trades[-1]
                last_price = float(df['Close'].iloc[-1])
                
                if position == 1:
                    pnl_percent = (last_price - entry_price) / entry_price
                else:
                    pnl_percent = (entry_price - last_price) / entry_price
                
                pnl_amount = current_trade['entry_capital'] * pnl_percent
                capital += pnl_amount
                
                if pnl_amount > 0:
                    winning_trades += 1
                
                trades[-1].update({
                    'exit_time': df.index[-1],
                    'exit_price': last_price,
                    'pnl': pnl_amount,
                    'status': 'CLOSED',
                    'pnl_percent': pnl_percent * 100
                })
            
            # Sonuçları hesapla
            final_capital = max(capital, 0)
            total_return = ((final_capital - self.initial_capital) / self.initial_capital) * 100
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Profit factor
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
                'sharpe_ratio': 0,  # Basit tut
                'max_drawdown': 0,  # Basit tut
                'trades': trades,
                'equity_curve': pd.DataFrame({'Date': [], 'Equity': []})
            }
            
            return self.results
            
        except Exception as e:
            st.error(f"Backtest sırasında hata: {str(e)}")
            return {
                'initial_capital': self.initial_capital,
                'final_capital': self.initial_capital,
                'total_return': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'trades': [],
                'equity_curve': pd.DataFrame({'Date': [], 'Equity': []})
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
    min_value=60,  # Daha kısa
    max_value=180,
    value=90,
    step=30
)

start_date = end_date - datetime.timedelta(days=period_days)

# Diğer ayarlar...
initial_capital = st.sidebar.number_input("Başlangıç Sermayesi (USD):", 1000, 100000, 10000, 1000)
position_size = st.sidebar.slider("İşlem Büyüklüğü (%):", 10, 100, 100, 5)

st.sidebar.subheader("📊 Teknik Gösterge Ayarları")
rsi_period = st.sidebar.slider("RSI Periyodu:", 5, 30, 14)
ema_short = st.sidebar.slider("Kısa EMA Periyodu:", 5, 20, 9)
ema_long = st.sidebar.slider("Uzun EMA Periyodu:", 15, 50, 21)

st.sidebar.subheader("🎯 Sinyal Ayarları")
rsi_oversold = st.sidebar.slider("RSI Oversold Seviyesi:", 20, 45, 30)  # Daha agresif
rsi_overbought = st.sidebar.slider("RSI Overbought Seviyesi:", 55, 80, 70)  # Daha agresif
signal_threshold = st.sidebar.slider("Sinyal Eşik Değeri:", 1.0, 3.0, 1.5, 0.1)

st.sidebar.subheader("🛡️ Risk Yönetimi")
stop_loss = st.sidebar.slider("Stop Loss (%):", 1, 10, 5)
take_profit = st.sidebar.slider("Take Profit (%):", 1, 20, 10)

# Ana içerik
st.subheader("🎯 Basit ve Etkili Strateji")

if enable_ml:
    st.success("🤖 **ML DESTEKLİ** - Basit ML ile geliştirilmiş")
else:
    st.info("📊 **GELENEKSEL** - Saf teknik analiz")

# Veri yükleme
@st.cache_data
def load_data(symbol, start_date, end_date):
    try:
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        return data if not data.empty else None
    except:
        return None

# Simülasyon
st.markdown("---")
st.subheader("🚀 Backtest Simülasyonu")

data = load_data(symbol, start_date, end_date)

if data is not None and not data.empty:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("İlk Fiyat", f"${data['Close'].iloc[0]:.2f}")
    with col2:
        st.metric("Son Fiyat", f"${data['Close'].iloc[-1]:.2f}")
    with col3:
        change = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
        st.metric("Değişim", f"{change:+.2f}%")

if st.button("🎯 Backtest Başlat", type="primary", use_container_width=True):
    if data is not None:
        with st.spinner("Simülasyon çalışıyor..."):
            progress_bar = st.progress(0)
            
            strategy = CryptoStrategy(initial_capital, enable_ml=enable_ml)
            data_with_indicators = strategy.calculate_advanced_indicators(
                data, rsi_period, ema_short, ema_long, 12, 26, 9
            )
            data_with_signals = strategy.generate_advanced_signals(
                data_with_indicators, rsi_oversold, rsi_overbought, 1.2, signal_threshold
            )
            results = strategy.backtest_advanced_strategy(
                data_with_signals, progress_bar, position_size, stop_loss, take_profit, 20
            )
            
            progress_bar.progress(1.0)
            
            # Sonuçları göster
            st.subheader("📊 Sonuçlar")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Son Sermaye", f"${results['final_capital']:,.2f}", 
                         f"{results['total_return']:+.2f}%")
            with col2:
                st.metric("Win Rate", f"{results['win_rate']:.1f}%")
            with col3:
                st.metric("Toplam İşlem", results['total_trades'])
            
            if results['win_rate'] > 0:
                st.balloons()
