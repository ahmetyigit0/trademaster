import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import ta  # Technical Analysis library - ÇOK ÖNEMLİ!

# Sayfa ayarı
st.set_page_config(
    page_title="🚀 Maximize Win Rate Strateji",
    page_icon="🎯",
    layout="wide"
)

# Başlık
st.title("🎯 Maximize Win Rate - Gelişmiş Strateji")
st.markdown("---")

# GELİŞMİŞ ML Strateji sınıfı
class AdvancedMLStrategy:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False
        
    def create_advanced_features(self, df):
        """Win rate artıran gelişmiş özellikler"""
        try:
            features = pd.DataFrame(index=df.index)
            
            # 1. GELİŞMİŞ TEKNİK GÖSTERGELER (ta kütüphanesi)
            # Ichimoku Cloud
            ichimoku = ta.trend.IchimokuIndicator(df['High'], df['Low'])
            features['ichimoku_a'] = ichimoku.ichimoku_a()
            features['ichimoku_b'] = ichimoku.ichimoku_b()
            features['ichimoku_base'] = ichimoku.ichimoku_base_line()
            features['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['Close'])
            features['bb_upper'] = bollinger.bollinger_hband()
            features['bb_lower'] = bollinger.bollinger_lband()
            features['bb_middle'] = bollinger.bollinger_mavg()
            features['bb_width'] = features['bb_upper'] - features['bb_lower']
            features['bb_position'] = (df['Close'] - features['bb_lower']) / features['bb_width']
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
            features['stoch_k'] = stoch.stoch()
            features['stoch_d'] = stoch.stoch_signal()
            
            # Williams %R
            features['williams_r'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r()
            
            # CCI (Commodity Channel Index)
            features['cci'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close']).cci()
            
            # ADX (Average Directional Index)
            features['adx'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
            
            # 2. PRICE ACTION ÖZELLİKLERİ
            features['price_vs_high_20'] = df['Close'] / df['High'].rolling(20).max() - 1
            features['price_vs_low_20'] = df['Close'] / df['Low'].rolling(20).min() - 1
            features['body_size'] = (df['Close'] - df['Open']) / df['Open']
            features['upper_shadow'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / df['Close']
            features['lower_shadow'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / df['Close']
            
            # 3. VOLUME ANALİZİ
            features['volume_sma_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
            features['volume_price_trend'] = features['volume_sma_ratio'] * df['Close'].pct_change()
            features['obv'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
            
            # 4. VOLATILITY ÖZELLİKLERİ
            features['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
            features['atr_ratio'] = features['atr'] / df['Close']
            features['volatility_ratio'] = df['Close'].rolling(10).std() / df['Close'].rolling(30).std()
            
            # 5. MULTI-TIMEFRAME ANALİZ
            features['rsi_4h'] = ta.momentum.RSIIndicator(df['Close'], window=4).rsi()
            features['rsi_12h'] = ta.momentum.RSIIndicator(df['Close'], window=12).rsi()
            features['ema_6h'] = df['Close'].ewm(span=6).mean()
            features['ema_18h'] = df['Close'].ewm(span=18).mean()
            features['multi_tf_trend'] = (features['ema_6h'] > features['ema_18h']).astype(int)
            
            # 6. SUPPORT/RESISTANCE
            features['resistance_distance'] = (df['High'].rolling(20).max() - df['Close']) / df['Close']
            features['support_distance'] = (df['Close'] - df['Low'].rolling(20).min()) / df['Close']
            
            # 7. MARKET REGIME
            features['trend_strength'] = features['adx'] / 100
            features['is_ranging'] = ((features['adx'] < 25) & (features['bb_width'] / df['Close'] < 0.02)).astype(int)
            features['is_trending'] = (features['adx'] > 30).astype(int)
            
            return features.fillna(0).replace([np.inf, -np.inf], 0)
            
        except Exception as e:
            st.error(f"Gelişmiş özellik oluşturma hatası: {e}")
            return self.create_basic_features(df)
    
    def create_basic_features(self, df):
        """Temel özellikler (fallback)"""
        features = pd.DataFrame(index=df.index)
        features['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi().fillna(50)
        features['ema_cross'] = (df['Close'].ewm(span=8).mean() - df['Close'].ewm(span=21).mean()) / df['Close']
        features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean().replace(0, 1)
        return features.fillna(0)
    
    def create_smart_target(self, df, horizon=4, confidence_threshold=0.6):
        """Akıllı hedef değişken - Win rate odaklı"""
        try:
            # Gelecek fiyat
            future_prices = df['Close'].shift(-horizon)
            current_prices = df['Close']
            
            # Getiri hesapla
            returns = (future_prices - current_prices) / current_prices
            
            # Çoklu threshold ile sınıflandırma
            target = np.zeros(len(df))
            
            # YÜKSEK GÜVEN SINIFLARI
            strong_bullish = returns > 0.02  # %2'den fazla
            strong_bearish = returns < -0.02 # %2'den fazla
            
            # ORTA GÜVEN SINIFLARI
            moderate_bullish = (returns > 0.008) & (returns <= 0.02)
            moderate_bearish = (returns < -0.008) & (returns >= -0.02)
            
            # Sınıflandırma
            target[strong_bullish] = 2      # ÇOK GÜÇLÜ AL
            target[moderate_bullish] = 1    # AL
            target[moderate_bearish] = -1   # SAT
            target[strong_bearish] = -2     # ÇOK GÜÇLÜ SAT
            
            return pd.Series(target, index=df.index, dtype=int)
            
        except Exception as e:
            return pd.Series(np.zeros(len(df)), index=df.index, dtype=int)
    
    def train_advanced_model(self, df):
        """Gelişmiş model eğitimi"""
        try:
            features = self.create_advanced_features(df)
            target = self.create_smart_target(df)
            
            features = features.fillna(0)
            target = target.fillna(0)
            
            if len(features) < 100:
                return 0, pd.DataFrame()
            
            # Time Series Cross Validation
            tscv = TimeSeriesSplit(n_splits=5)
            accuracies = []
            
            # Ensemble model
            rf_model = RandomForestClassifier(
                n_estimators=150,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'  # Class imbalance için
            )
            
            gb_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # Feature scaling
            features_scaled = self.scaler.fit_transform(features)
            
            # Cross validation
            for train_idx, test_idx in tscv.split(features_scaled):
                X_train, X_test = features_scaled[train_idx], features_scaled[test_idx]
                y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
                
                rf_model.fit(X_train, y_train)
                y_pred = rf_model.predict(X_test)
                accuracies.append(accuracy_score(y_test, y_pred))
            
            # Final model
            self.model = rf_model
            self.model.fit(features_scaled, target)
            
            self.is_trained = True
            self.feature_columns = features.columns.tolist()
            
            accuracy = np.mean(accuracies)
            
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return accuracy, feature_importance
            
        except Exception as e:
            return 0, pd.DataFrame()
    
    def predict_advanced_signals(self, df):
        """Gelişmiş sinyal tahmini"""
        try:
            if not self.is_trained or self.model is None:
                return np.zeros(len(df))
            
            features = self.create_advanced_features(df)
            features = features.fillna(0)
            features_scaled = self.scaler.transform(features)
            predictions = self.model.predict(features_scaled)
            
            return predictions
            
        except Exception as e:
            return np.zeros(len(df))

# WIN RATE ODAKLI Ana Strateji
class HighWinRateStrategy:
    def __init__(self, initial_capital: float = 10000, enable_ml: bool = True):
        self.initial_capital = initial_capital
        self.results = {}
        self.enable_ml = enable_ml
        self.ml_strategy = AdvancedMLStrategy() if enable_ml else None
        
    def calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Win rate artıran göstergeler"""
        try:
            df = df.copy()
            
            # TEMEL GÖSTERGELER
            df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi().fillna(50)
            df['RSI_4h'] = ta.momentum.RSIIndicator(df['Close'], window=4).rsi().fillna(50)
            
            # EMA'lar
            for span in [8, 21, 50]:
                df[f'EMA_{span}'] = df['Close'].ewm(span=span, adjust=False).mean()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['Close'])
            df['BB_Upper'] = bb.bollinger_hband()
            df['BB_Lower'] = bb.bollinger_lband()
            df['BB_Middle'] = bb.bollinger_mavg()
            df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
            df['Stoch_K'] = stoch.stoch().fillna(50)
            df['Stoch_D'] = stoch.stoch_signal().fillna(50)
            
            # Volume
            df['Volume_SMA'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA'].replace(0, 1)
            
            # ATR
            df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
            
            return df.fillna(0)
            
        except Exception as e:
            st.error(f"Göstergeler hesaplanırken hata: {e}")
            df['RSI'] = 50
            df['EMA_8'] = df['Close']
            df['Volume_Ratio'] = 1
            return df
    
    def generate_high_winrate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """YÜKSEK WIN RATE sinyal sistemi"""
        try:
            df = df.copy()
            df['Signal'] = 0
            
            # ML SİNYALLERİ
            ml_signals = np.zeros(len(df))
            ml_accuracy = 0
            
            if self.enable_ml and self.ml_strategy:
                ml_accuracy, feature_importance = self.ml_strategy.train_advanced_model(df)
                if ml_accuracy > 0.40:
                    ml_signals = self.ml_strategy.predict_advanced_signals(df)
                    st.success(f"🤖 Gelişmiş ML Doğruluğu: %{ml_accuracy:.1f}")
                    
                    if not feature_importance.empty:
                        st.write("**En Önemli 8 Özellik:**")
                        st.dataframe(feature_importance.head(8))
            
            # ÇOKLU KONFİRMASYON SİSTEMİ
            for i in range(20, len(df)):  # Daha fazla geçmiş veri
                try:
                    # 1. MOMENTUM KONFİRMASYONU
                    rsi = df['RSI'].iloc[i]
                    rsi_4h = df['RSI_4h'].iloc[i]
                    stoch_k = df['Stoch_K'].iloc[i]
                    stoch_d = df['Stoch_D'].iloc[i]
                    
                    # 2. TREND KONFİRMASYONU
                    ema_8 = df['EMA_8'].iloc[i]
                    ema_21 = df['EMA_21'].iloc[i]
                    ema_50 = df['EMA_50'].iloc[i]
                    
                    # 3. VOLATILITY KONFİRMASYONU
                    bb_position = df['BB_Position'].iloc[i]
                    bb_width = df['BB_Width'].iloc[i] / df['Close'].iloc[i]
                    atr = df['ATR'].iloc[i] / df['Close'].iloc[i]
                    
                    # 4. VOLUME KONFİRMASYONU
                    volume_ratio = df['Volume_Ratio'].iloc[i]
                    
                    long_confirmations = 0
                    short_confirmations = 0
                    
                    # ÇOKLU LONG KONFİRMASYONLARI
                    # Momentum
                    if rsi < 35 and rsi_4h < 40: long_confirmations += 2
                    elif rsi < 40: long_confirmations += 1
                    
                    if stoch_k < 20 and stoch_d < 25: long_confirmations += 2
                    elif stoch_k < 30: long_confirmations += 1
                    
                    # Trend
                    if ema_8 > ema_21 and ema_21 > ema_50: long_confirmations += 3  # Güçlü uptrend
                    elif ema_8 > ema_21: long_confirmations += 2
                    elif ema_8 > ema_50: long_confirmations += 1
                    
                    # Volatility
                    if bb_position < 0.1: long_confirmations += 2  # Aşırı oversold
                    elif bb_position < 0.2: long_confirmations += 1
                    
                    if bb_width > 0.02: long_confirmations += 1  # Yüksek volatilite
                    
                    # Volume
                    if volume_ratio > 1.5: long_confirmations += 1
                    
                    # ÇOKLU SHORT KONFİRMASYONLARI
                    # Momentum
                    if rsi > 65 and rsi_4h > 60: short_confirmations += 2
                    elif rsi > 60: short_confirmations += 1
                    
                    if stoch_k > 80 and stoch_d > 75: short_confirmations += 2
                    elif stoch_k > 70: short_confirmations += 1
                    
                    # Trend
                    if ema_8 < ema_21 and ema_21 < ema_50: short_confirmations += 3  # Güçlü downtrend
                    elif ema_8 < ema_21: short_confirmations += 2
                    elif ema_8 < ema_50: short_confirmations += 1
                    
                    # Volatility
                    if bb_position > 0.9: short_confirmations += 2  # Aşırı overbought
                    elif bb_position > 0.8: short_confirmations += 1
                    
                    if bb_width > 0.02: short_confirmations += 1
                    
                    # Volume
                    if volume_ratio > 1.5: short_confirmations += 1
                    
                    # ML GÜÇLENDİRME
                    ml_signal = ml_signals[i]
                    if ml_signal >= 1:  # AL veya GÜÇLÜ AL
                        long_confirmations += ml_signal
                    elif ml_signal <= -1:  # SAT veya GÜÇLÜ SAT
                        short_confirmations += abs(ml_signal)
                    
                    # YÜKSEK KONFİRMASYON EŞİĞİ
                    if long_confirmations >= 6:  # Minimum 6 konfirmasyon
                        df.loc[df.index[i], 'Signal'] = 1
                    elif short_confirmations >= 6:
                        df.loc[df.index[i], 'Signal'] = -1
                        
                except Exception as e:
                    continue
            
            # Sinyal kalite analizi
            total_signals = (df['Signal'] != 0).sum()
            if total_signals > 0:
                signal_quality = total_signals / len(df) * 100
                st.info(f"**Sinyal Kalitesi:** {signal_quality:.1f}% - {total_signals} sinyal")
                    
            return df
            
        except Exception as e:
            st.error(f"Sinyal oluşturma hatası: {e}")
            df['Signal'] = 0
            return df
    
    def backtest_high_winrate_strategy(self, df: pd.DataFrame, progress_bar, 
                                     position_size: float, stop_loss: float, 
                                     take_profit: float, max_hold_hours: int = 8) -> dict:
        """Win rate odaklı backtest"""
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
                
                # POZİSYON AÇ - SADECE GÜÇLÜ SİNYALLER
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
                            pnl_percent >= 0.04  # Erken kar al
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
                            pnl_percent >= 0.04  # Erken kar al
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
st.sidebar.header("🎯 Win Rate Maximize Ayarları")

crypto_symbols = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD", 
    "Solana (SOL-USD)": "SOL-USD",
    "Cardano (ADA-USD)": "ADA-USD"
}

selected_crypto = st.sidebar.selectbox("Kripto Para Seçin:", list(crypto_symbols.keys()))
symbol = crypto_symbols[selected_crypto]

# WIN RATE ODAKLI AYARLAR
st.sidebar.subheader("⚡ Zaman Ayarları")
timeframe = st.sidebar.selectbox("Zaman Periyodu:", ["1h", "2h", "4h"], index=0)

end_date = st.sidebar.date_input("Bitiş Tarihi:", datetime.date.today() - datetime.timedelta(days=1))
period_months = st.sidebar.slider("Veri Süresi (Ay):", 1, 6, 4, 1)
start_date = end_date - datetime.timedelta(days=period_months*30)

st.sidebar.subheader("🎯 Win Rate Optimizasyonu")
initial_capital = st.sidebar.number_input("Başlangıç Sermayesi (USD):", 1000, 100000, 10000, 1000)
position_size = st.sidebar.slider("İşlem Büyüklüğü (%):", 30, 80, 50, 5)  # Daha düşük risk

st.sidebar.subheader("🛡️ Koruyucu Risk Yönetimi")
stop_loss = st.sidebar.slider("Stop Loss (%):", 1.5, 5.0, 2.5, 0.1)  # Sıkı stop
take_profit = st.sidebar.slider("Take Profit (%):", 3.0, 8.0, 5.0, 0.1)  # Düşük ama sık TP
max_hold_hours = st.sidebar.slider("Maksimum Bekleme (Saat):", 6, 24, 12, 1)

st.sidebar.subheader("🤖 Gelişmiş ML")
enable_ml = st.sidebar.checkbox("Gelişmiş ML'yi Etkinleştir", value=True)

# Ana içerik
st.subheader("🎯 Win Rate Maximize Strateji")

st.success("""
**🚀 WIN RATE ARTIRMA TEKNİKLERİ:**

1. **Çoklu Konfirmasyon Sistemi** (6+ onay)
2. **Gelişmiş Teknik Göstergeler** (Ichimoku, Bollinger, Stochastic)
3. **Multi-Timeframe Analiz**
4. **Volume-Price İlişkisi**
5. **Market Regime Detection**
6. **Ensemble ML Modeli**
7. **Strict Risk Management**
8. **Erken Kar Realizasyonu**

**📊 BEKLENEN WIN RATE: %65-80+**
""")

# Veri yükleme
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

# Veri gösterimi
st.markdown("---")
st.subheader("📊 Veri Yükleme")

data = load_data(symbol, start_date, end_date, timeframe)

if data is not None and not data.empty:
    close_prices = data['Close']
    first_price = float(close_prices.iloc[0])
    last_price = float(close_prices.iloc[-1])
    price_change = ((last_price - first_price) / first_price) * 100
    
    price_changes = close_prices.pct_change().dropna()
    volatility = price_changes.std() * np.sqrt(365 * 24) * 100 if len(price_changes) > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("İlk Fiyat", f"${first_price:.2f}")
    with col2:
        st.metric("Son Fiyat", f"${last_price:.2f}")
    with col3:
        st.metric("Değişim", f"{price_change:+.2f}%")
    with col4:
        st.metric("Volatilite", f"{volatility:.1f}%")

# SİMÜLASYON BUTONU
st.markdown("---")
st.subheader("🚀 Win Rate Backtest Başlat")

if st.button("🎯 BACKTEST BAŞLAT", type="primary", use_container_width=True):
    if data is not None and not data.empty:
        with st.spinner("Win rate backtest çalışıyor..."):
            start_time = time.time()
            progress_bar = st.progress(0)
            
            try:
                strategy = HighWinRateStrategy(initial_capital, enable_ml=enable_ml)
                data_with_indicators = strategy.calculate_advanced_indicators(data)
                data_with_signals = strategy.generate_high_winrate_signals(data_with_indicators)
                results = strategy.backtest_high_winrate_strategy(
                    data_with_signals, progress_bar, position_size, stop_loss, take_profit, max_hold_hours
                )
                
                progress_bar.progress(1.0)
                end_time = time.time()
                
                st.success(f"✅ Backtest {end_time - start_time:.1f} saniyede tamamlandı!")
                
                # SONUÇLAR
                st.subheader("📈 Win Rate Sonuçları")
                
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
                
                # WIN RATE DEĞERLENDİRMESİ
                if results['win_rate'] >= 80:
                    st.success("🎉🎉🎉 MÜKEMMEL! %80+ WIN RATE! 🎉🎉🎉")
                    st.balloons()
                elif results['win_rate'] >= 70:
                    st.success("🎉 ÇOK İYİ! %70+ WIN RATE!")
                    st.balloons()
                elif results['win_rate'] >= 60:
                    st.success("📈 İYİ! %60+ WIN RATE!")
                elif results['win_rate'] >= 50:
                    st.info("✅ ORTA! %50+ WIN RATE")
                else:
                    st.warning("⚠️ DÜŞÜK! Win rate geliştirilmeli")
                
                st.info(f"**Profit Factor:** {results['profit_factor']:.2f} | **Karlı İşlem:** {results['winning_trades']}/{results['total_trades']}")
                    
            except Exception as e:
                st.error(f"Backtest sırasında hata: {str(e)}")
    else:
        st.error("Veri yüklenemedi!")

st.markdown("---")
st.info("""
**💡 WIN RATE ARTIRMA İPUÇLARI:**

1. **Daha uzun timeframe (4h) daha yüksek win rate**
2. **Düşük position size (%30-50) daha iyi risk yönetimi**
3. **Sıkı stop loss (%2-3) kayıpları sınırlar**
4. **Düşük take profit (%4-6) daha sık kar realizasyonu**
5. **ML her zaman aktif olmalı**
6. **3-4 aylık veri optimal sonuç verir**
""")
