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

# DÜZELTİLMİŞ ML Strateji sınıfı
class MLTradingStrategy:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.is_trained = False
        
    def create_advanced_features(self, df):
        """ML için gelişmiş özellikler oluştur"""
        try:
            features = pd.DataFrame(index=df.index)
            
            # Temel teknik göstergeler
            features['rsi'] = df['RSI']
            features['ema_cross'] = (df['EMA_Short'] - df['EMA_Long']) / df['EMA_Long']
            features['macd_hist'] = df['MACD_Histogram']
            features['volume_ratio'] = df['Volume_Ratio']
            features['momentum'] = df['Momentum']
            
            # Fiyat-based özellikler
            features['price_trend_5'] = df['Close'].pct_change(5)
            features['price_trend_10'] = df['Close'].pct_change(10)
            features['volatility_20'] = df['Close'].rolling(20).std()
            
            # ATR hesapla
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            true_range = np.maximum(np.maximum(high_low, high_close), low_close)
            features['atr'] = true_range.rolling(14).mean() / df['Close']
            
            # Support/resistance benzeri özellikler
            features['high_20_ratio'] = (df['High'].rolling(20).max() / df['Close']) - 1
            features['low_20_ratio'] = (df['Low'].rolling(20).min() / df['Close']) - 1
            
            # Volume-based özellikler
            features['volume_trend'] = df['Volume'].pct_change(5)
            features['volume_volatility'] = df['Volume'].rolling(10).std()
            
            # Mean reversion özellikleri
            features['price_vs_ema'] = (df['Close'] - df['EMA_Short']) / df['EMA_Short']
            features['rsi_deviation'] = (df['RSI'] - 50) / 50
            
            return features.fillna(0)
        except Exception as e:
            st.error(f"Özellik oluşturma hatası: {e}")
            return pd.DataFrame(index=df.index)
    
    def create_target_variable(self, df, horizon=3, threshold=0.015):
        """Hedef değişken oluştur"""
        try:
            # Horizon gün sonraki getiri
            future_return = (df['Close'].shift(-horizon) / df['Close']) - 1
            
            # Sınıflandırma: 1D array oluştur
            target = np.zeros(len(df))
            target[future_return > threshold] = 1      # AL
            target[future_return < -threshold] = -1    # SAT
            # 0: BEKLE (default)
            
            return pd.Series(target, index=df.index)  # 1 boyutlu Series döndür
            
        except Exception as e:
            st.error(f"Hedef değişken hatası: {e}")
            return pd.Series(np.zeros(len(df)), index=df.index)
    
    def train_model(self, df, test_size=0.2):
        """Model eğitimi"""
        try:
            # Özellikler ve hedef
            features = self.create_advanced_features(df)
            target = self.create_target_variable(df)
            
            # Aynı indekse sahip verileri hizala
            common_index = features.index.intersection(target.index)
            features = features.loc[common_index]
            target = target.loc[common_index]
            
            if len(features) < 100:
                st.warning("Eğitim için yeterli veri yok")
                return 0, pd.DataFrame()
            
            # Train-test split (zaman serisi uyumlu)
            split_idx = int(len(features) * (1 - test_size))
            X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
            y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]
            
            # Model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
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
            st.error(f"Model eğitim hatası: {e}")
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
            predictions = self.model.predict(features)
            
            # DataFrame ile aynı uzunlukta array oluştur
            signals = np.zeros(len(df))
            valid_indices = features.index
            df_indices = df.index
            
            for i, idx in enumerate(df_indices):
                if idx in valid_indices:
                    pos = np.where(valid_indices == idx)[0][0]
                    signals[i] = predictions[pos]
            
            return signals
            
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
            
            # RSI hesaplama
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=rsi_period, min_periods=1).mean()
            avg_loss = loss.rolling(window=rsi_period, min_periods=1).mean()
            rs = avg_gain / avg_loss
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
            volume_sma = df['Volume'].rolling(window=20).mean()
            volume_ratio = df['Volume'] / volume_sma
            
            # Tüm sütunları tek seferde ata
            df = df.assign(
                RSI=rsi,
                EMA_Short=ema_short_val,
                EMA_Long=ema_long_val,
                MACD=macd,
                MACD_Signal=macd_signal_val,
                MACD_Histogram=macd_histogram,
                Momentum=momentum,
                Volume_SMA=volume_sma,
                Volume_Ratio=volume_ratio
            )
            
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
            
            # ML sinyalleri
            ml_signals = np.zeros(len(df))
            ml_accuracy = 0
            feature_importance = pd.DataFrame()
            
            if self.enable_ml and self.ml_strategy:
                ml_accuracy, feature_importance = self.ml_strategy.train_model(df)
                if ml_accuracy > 0.55:  # Minimum doğruluk eşiği
                    ml_signals = self.ml_strategy.predict_signals(df)
                    st.success(f"🤖 ML Model Doğruluğu: %{ml_accuracy:.1f}")
                    
                    # Feature importance göster
                    if not feature_importance.empty:
                        st.write("**Özellik Önem Sıralaması:**")
                        st.dataframe(feature_importance.head(10))
            
            # Her satır için tek tek kontrol et
            for i in range(len(df)):
                try:
                    if i < 50:
                        continue
                        
                    rsi = float(df['RSI'].iloc[i])
                    ema_short = float(df['EMA_Short'].iloc[i])
                    ema_long = float(df['EMA_Long'].iloc[i])
                    macd = float(df['MACD'].iloc[i])
                    macd_signal = float(df['MACD_Signal'].iloc[i])
                    momentum = float(df['Momentum'].iloc[i])
                    volume_ratio = float(df['Volume_Ratio'].iloc[i])
                    
                    # Geleneksel sinyal puanları
                    long_signals = 0
                    short_signals = 0
                    
                    # LONG sinyalleri
                    if rsi < rsi_oversold and ema_short > ema_long:
                        long_signals += 1.5
                    if macd > macd_signal and df['MACD'].iloc[i-1] <= df['MACD_Signal'].iloc[i-1]:
                        long_signals += 1
                    if momentum > 0:
                        long_signals += 0.5
                    if volume_ratio > volume_threshold:
                        long_signals += 0.5
                    
                    # SHORT sinyalleri
                    if rsi > rsi_overbought and ema_short < ema_long:
                        short_signals += 1.5
                    if macd < macd_signal and df['MACD'].iloc[i-1] >= df['MACD_Signal'].iloc[i-1]:
                        short_signals += 1
                    if momentum < 0:
                        short_signals += 0.5
                    if volume_ratio > volume_threshold:
                        short_signals += 0.5
                    
                    # ML sinyali ekle (eğer güvenilirse)
                    ml_signal = ml_signals[i]
                    if ml_signal == 1 and self.ml_strategy.is_trained:
                        long_signals += 1.0
                    elif ml_signal == -1 and self.ml_strategy.is_trained:
                        short_signals += 1.0
                    
                    # Sinyal belirleme
                    if long_signals >= signal_threshold:
                        df.loc[df.index[i], 'Signal'] = 1
                    elif short_signals >= signal_threshold:
                        df.loc[df.index[i], 'Signal'] = -1
                        
                except Exception as e:
                    continue
                    
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
            
            # Sharpe Ratio
            if len(trades) > 1:
                returns = [trade['pnl_percent'] / 100 for trade in trades if trade['status'] == 'CLOSED']
                if returns and len(returns) > 1:
                    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
                else:
                    sharpe_ratio = 0
            else:
                sharpe_ratio = 0
            
            # Max drawdown
            equity_curve = self.calculate_equity_curve(trades)
            if not equity_curve.empty:
                equity_curve['Peak'] = equity_curve['Equity'].cummax()
                equity_curve['Drawdown'] = (equity_curve['Equity'] - equity_curve['Peak']) / equity_curve['Peak'] * 100
                max_drawdown = equity_curve['Drawdown'].min()
            else:
                max_drawdown = 0
            
            self.results = {
                'initial_capital': self.initial_capital,
                'final_capital': final_capital,
                'total_return': total_return,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'trades': trades,
                'equity_curve': equity_curve
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
    
    def calculate_equity_curve(self, trades: list) -> pd.DataFrame:
        """Equity curve hesapla"""
        try:
            if not trades:
                return pd.DataFrame({'Date': [], 'Equity': []})
            
            equity = [self.initial_capital]
            dates = [trades[0]['entry_time']]
            
            current_capital = self.initial_capital
            
            for trade in trades:
                if trade['status'] == 'CLOSED':
                    current_capital += trade['pnl']
                    equity.append(current_capital)
                    dates.append(trade['exit_time'])
            
            return pd.DataFrame({'Date': dates, 'Equity': equity})
        except:
            return pd.DataFrame({'Date': [], 'Equity': []})

# Sidebar - Tüm Ayarlar
st.sidebar.header("⚙️ Simülasyon Ayarları")

# Kripto seçimi
crypto_symbols = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD", 
    "Binance Coin (BNB-USD)": "BNB-USD",
    "Cardano (ADA-USD)": "ADA-USD",
    "Solana (SOL-USD)": "SOL-USD",
    "Ripple (XRP-USD)": "XRP-USD",
    "Dogecoin (DOGE-USD)": "DOGE-USD"
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
    min_value=90,
    max_value=365,
    value=180,
    step=30
)

start_date = end_date - datetime.timedelta(days=period_days)

st.sidebar.info(f"Simülasyon Aralığı: {start_date.strftime('%d.%m.%Y')} - {end_date.strftime('%d.%m.%Y')}")

# Sermaye ayarları
st.sidebar.subheader("💰 Sermaye Ayarları")
initial_capital = st.sidebar.number_input(
    "Başlangıç Sermayesi (USD):",
    min_value=1000,
    max_value=100000,
    value=10000,
    step=1000
)

position_size = st.sidebar.slider(
    "İşlem Büyüklüğü (%):",
    min_value=10,
    max_value=100,
    value=100,
    step=5
)

# Gösterge ayarları
st.sidebar.subheader("📊 Teknik Gösterge Ayarları")
rsi_period = st.sidebar.slider("RSI Periyodu:", 5, 30, 14)
ema_short = st.sidebar.slider("Kısa EMA Periyodu:", 5, 20, 9)
ema_long = st.sidebar.slider("Uzun EMA Periyodu:", 15, 50, 21)
macd_fast = st.sidebar.slider("MACD Hızlı Periyot:", 8, 20, 12)
macd_slow = st.sidebar.slider("MACD Yavaş Periyot:", 20, 35, 26)
macd_signal = st.sidebar.slider("MACD Sinyal Periyotu:", 5, 15, 9)

# Sinyal ayarları
st.sidebar.subheader("🎯 Sinyal Ayarları")
rsi_oversold = st.sidebar.slider("RSI Oversold Seviyesi:", 20, 45, 40)
rsi_overbought = st.sidebar.slider("RSI Overbought Seviyesi:", 55, 80, 60)
volume_threshold = st.sidebar.slider("Volume Eşik Değeri:", 0.5, 3.0, 1.2, 0.1)
signal_threshold = st.sidebar.slider("Sinyal Eşik Değeri:", 0.5, 3.0, 1.5, 0.1)

# Risk yönetimi ayarları
st.sidebar.subheader("🛡️ Risk Yönetimi")
stop_loss = st.sidebar.slider("Stop Loss (%):", 1, 10, 3)
take_profit = st.sidebar.slider("Take Profit (%):", 1, 20, 6)
max_profit = st.sidebar.slider("Maksimum Kar (%):", 5, 30, 15)

# Ana içerik
st.subheader("🎯 Gelişmiş Strateji Bilgileri")

if enable_ml:
    st.success("🤖 **ML DESTEKLİ MODE AKTİF** - Random Forest ile akıllı sinyal tahmini")
else:
    st.info("📊 **GELENEKSEL MODE** - Sadece teknik göstergeler")

st.markdown(f"""
**Strateji Detayları:**
- RSI ({rsi_period}) + EMA ({ema_short}/{ema_long}) + MACD ({macd_fast}/{macd_slow}/{macd_signal})
- Volume ve Momentum onayı
- {f"🤖 ML Model entegrasyonu" if enable_ml else "📊 Geleneksel analiz"}

**Risk Yönetimi:**
- %{stop_loss} Stop Loss
- %{take_profit} Take Profit  
- %{position_size} İşlem Büyüklüğü
- Maksimum %{max_profit} kar sınırı
""")

# Veri yükleme
@st.cache_data
def load_data(symbol, start_date, end_date):
    try:
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Veri yüklenirken hata oluştu: {e}")
        return None

# Simülasyon butonu
st.markdown("---")
st.subheader("🚀 Backtest Simülasyonu")

data = load_data(symbol, start_date, end_date)

if data is not None and not data.empty:
    try:
        first_price = float(data['Close'].iloc[0])
        last_price = float(data['Close'].iloc[-1])
        price_change = ((last_price - first_price) / first_price) * 100
        data_count = len(data)
        
        st.success(f"✅ {selected_crypto} verisi yüklendi: {data_count} günlük veri")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("İlk Fiyat", f"${first_price:.2f}")
        with col2:
            st.metric("Son Fiyat", f"${last_price:.2f}")
        with col3:
            st.metric("Dönem Değişim", f"{price_change:+.2f}%")
        with col4:
            st.metric("Veri Sayısı", data_count)
    except Exception as e:
        st.error(f"Veri gösterilirken hata: {e}")
else:
    st.warning("⚠️ Veri yüklenemedi. Lütfen tarih aralığını kontrol edin.")

# Simülasyon butonu
if st.button("🎯 Backtest Simülasyonunu Başlat", type="primary", use_container_width=True):
    if data is not None and not data.empty:
        with st.spinner("Simülasyon çalışıyor..."):
            start_time = time.time()
            
            # İlerleme çubuğu
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Stratejiyi çalıştır
                strategy = CryptoStrategy(initial_capital, enable_ml=enable_ml)
                
                # Göstergeleri hesapla
                status_text.text("Teknik göstergeler hesaplanıyor...")
                data_with_indicators = strategy.calculate_advanced_indicators(
                    data, rsi_period, ema_short, ema_long, macd_fast, macd_slow, macd_signal
                )
                
                # Sinyalleri oluştur
                status_text.text("Sinyal sistemi oluşturuluyor...")
                data_with_signals = strategy.generate_advanced_signals(
                    data_with_indicators, rsi_oversold, rsi_overbought, volume_threshold, signal_threshold
                )
                
                # Backtest yap
                status_text.text("Strateji backtest ediliyor...")
                results = strategy.backtest_advanced_strategy(
                    data_with_signals, progress_bar, position_size, stop_loss, take_profit, max_profit
                )
                
                end_time = time.time()
                calculation_time = end_time - start_time
                
                # İlerleme çubuğunu tamamla
                progress_bar.progress(1.0)
                status_text.empty()
                
                st.success(f"✅ Simülasyon {calculation_time:.2f} saniyede tamamlandı!")
                
                # Sonuçları göster
                st.subheader("📊 Simülasyon Sonuçları")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Başlangıç Sermayesi",
                        f"${results['initial_capital']:,.2f}"
                    )
                
                with col2:
                    st.metric(
                        "Son Sermaye", 
                        f"${results['final_capital']:,.2f}",
                        delta=f"{results['total_return']:+.2f}%"
                    )
                
                with col3:
                    st.metric(
                        "Toplam İşlem",
                        f"{results['total_trades']}",
                        delta=f"+{results['total_trades']}" if results['total_trades'] > 0 else "0"
                    )
                
                with col4:
                    st.metric(
                        "Win Rate",
                        f"{results['win_rate']:.1f}%"
                    )
                
                # Ek metrikler
                col5, col6, col7, col8 = st.columns(4)
                
                with col5:
                    st.metric(
                        "Karlı İşlem Sayısı",
                        f"{results['winning_trades']}"
                    )
                
                with col6:
                    profit_factor = results['profit_factor']
                    pf_display = f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞"
                    st.metric(
                        "Profit Factor",
                        pf_display
                    )
                
                with col7:
                    st.metric(
                        "Sharpe Ratio",
                        f"{results['sharpe_ratio']:.2f}"
                    )
                
                with col8:
                    st.metric(
                        "Max Drawdown",
                        f"{results['max_drawdown']:.1f}%"
                    )
                
                # Equity curve
                if not results['equity_curve'].empty:
                    st.subheader("📈 Portföy Performansı")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=results['equity_curve']['Date'],
                        y=results['equity_curve']['Equity'],
                        mode='lines+markers',
                        name='Portföy Değeri',
                        line=dict(color='blue', width=3),
                        marker=dict(size=4)
                    ))
                    
                    # Başlangıç sermayesi çizgisi
                    fig.add_hline(
                        y=initial_capital, 
                        line_dash="dash", 
                        line_color="red",
                        annotation_text="Başlangıç Sermayesi"
                    )
                    
                    fig.update_layout(
                        title="Portföy Performans Grafiği",
                        xaxis_title="Tarih",
                        yaxis_title="Portföy Değeri (USD)",
                        height=500,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # İşlem detayları
                if results['trades']:
                    closed_trades = [t for t in results['trades'] if t['status'] == 'CLOSED']
                    
                    if closed_trades:
                        st.subheader("📋 İşlem Detayları")
                        
                        trades_df = pd.DataFrame(closed_trades)
                        
                        # Renk fonksiyonu
                        def color_pnl(val):
                            color = 'color: green' if val > 0 else 'color: red' if val < 0 else 'color: black'
                            return color
                        
                        # DataFrame'i düzenle
                        display_df = trades_df[['entry_time', 'exit_time', 'position', 'entry_price', 'exit_price', 'entry_capital', 'pnl', 'pnl_percent']]
                        display_df = display_df.rename(columns={
                            'entry_time': 'Giriş Tarihi',
                            'exit_time': 'Çıkış Tarihi',
                            'position': 'Pozisyon',
                            'entry_price': 'Giriş Fiyatı',
                            'exit_price': 'Çıkış Fiyatı',
                            'entry_capital': 'İşlem Büyüklüğü',
                            'pnl': 'Kar/Zarar ($)',
                            'pnl_percent': 'Kar/Zarar (%)'
                        })
                        
                        # Sayısal sütunları formatla
                        styled_df = display_df.style.format({
                            'Giriş Fiyatı': '{:.2f}',
                            'Çıkış Fiyatı': '{:.2f}',
                            'İşlem Büyüklüğü': '{:.2f}',
                            'Kar/Zarar ($)': '{:.2f}',
                            'Kar/Zarar (%)': '{:.2f}%'
                        }).applymap(color_pnl, subset=['Kar/Zarar ($)', 'Kar/Zarar (%)'])
                        
                        st.dataframe(styled_df, use_container_width=True, height=400)
                        
            except Exception as e:
                st.error(f"Simülasyon sırasında hata: {str(e)}")
    else:
        st.error("Veri yüklenemedi. Lütfen önce kripto para ve tarih seçin.")

# Bilgi
st.markdown("---")
st.info("""
**⚠️ Uyarı:** Bu simülasyon sadece eğitim amaçlıdır. Gerçek trading için kullanmayın. 
Geçmiş performans gelecek sonuçların garantisi değildir.

**🤖 ML Entegrasyonu:** 
- Random Forest classifier ile sinyal tahmini
- 15+ teknik ve istatistiksel özellik
- Otomatik pattern tanıma
- Feature importance analizi
""")
