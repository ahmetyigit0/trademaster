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
    page_title="🚀 Yüksek Frekanslı Kripto Strateji Simülasyonu",
    page_icon="⚡",
    layout="wide"
)

# Başlık
st.title("🚀 Yüksek Frekanslı Kripto Strateji Simülasyonu")
st.markdown("---")

# DÜZELTİLMİŞ YÜKSEK FREKANS ML Strateji sınıfı
class HighFrequencyMLStrategy:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.is_trained = False
        
    def create_high_freq_features(self, df):
        """Yüksek frekanslı özellikler oluştur - TAMAMEN DÜZELTİLDİ"""
        try:
            features = pd.DataFrame(index=df.index)
            
            # TEK TEK sütun ekle - DataFrame atama hatasını önle
            # RSI - kısa periyot
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=6, min_periods=1).mean()
            avg_loss = loss.rolling(window=6, min_periods=1).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rs = rs.fillna(1)
            features['rsi_6h'] = 100 - (100 / (1 + rs))
            
            # EMA'lar - TEK TEK
            ema_4h = df['Close'].ewm(span=4, adjust=False).mean()
            ema_12h = df['Close'].ewm(span=12, adjust=False).mean()
            features['ema_4h'] = ema_4h
            features['ema_12h'] = ema_12h
            features['ema_cross'] = (ema_4h - ema_12h) / df['Close']  # TEK SÜTUN
            
            # Momentum - TEK TEK
            features['momentum_2h'] = df['Close'] - df['Close'].shift(2)
            features['momentum_6h'] = df['Close'] - df['Close'].shift(6)
            
            # Volatilite - TEK TEK
            features['volatility_8h'] = df['Close'].rolling(8).std() / df['Close']
            
            # ATR - TEK TEK
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            true_range = np.maximum(np.maximum(high_low, high_close), low_close)
            features['atr_6h'] = true_range.rolling(6).mean() / df['Close']
            
            # Volume - TEK TEK
            volume_ema_8h = df['Volume'].ewm(span=8).mean()
            features['volume_ema_8h'] = volume_ema_8h
            features['volume_ratio'] = df['Volume'] / volume_ema_8h.replace(0, 1)  # TEK SÜTUN
            features['volume_trend'] = df['Volume'].pct_change(4)
            
            # Price action - TEK TEK
            features['price_change_1h'] = df['Close'].pct_change(1)
            features['price_change_4h'] = df['Close'].pct_change(4)
            features['high_low_ratio'] = (df['High'] - df['Low']) / df['Close']
            
            # Support/Resistance - TEK TEK
            features['resistance_12h'] = (df['High'].rolling(12).max() / df['Close']) - 1
            features['support_12h'] = (df['Low'].rolling(12).min() / df['Close']) - 1
            
            # Mean reversion - TEK TEK
            features['price_vs_ema4'] = (df['Close'] - ema_4h) / ema_4h
            
            return features.fillna(0).replace([np.inf, -np.inf], 0)
            
        except Exception as e:
            st.error(f"Özellik oluşturma hatası: {e}")
            # Hata durumunda basit özellikler
            features = pd.DataFrame(index=df.index)
            features['rsi_6h'] = 50
            features['ema_cross'] = 0
            features['volume_ratio'] = 1
            return features
    
    def create_high_freq_target(self, df, lookahead=4):
        """Kısa vadeli hedef değişken"""
        try:
            # 4 saat sonraki fiyat değişimi
            future_return = df['Close'].shift(-lookahead) / df['Close'] - 1
            
            # Çoklu threshold'lar ile sınıflandırma
            target = np.zeros(len(df))
            
            # Daha agresif threshold'lar
            bullish_threshold = 0.008  # %0.8
            bearish_threshold = -0.008 # %0.8
            
            target[future_return > bullish_threshold] = 1      # GÜÇLÜ AL
            target[future_return < bearish_threshold] = -1     # GÜÇLÜ SAT
            # 0: BEKLE veya küçük hareketler
            
            return pd.Series(target, index=df.index, dtype=int)
            
        except Exception as e:
            return pd.Series(np.zeros(len(df)), index=df.index, dtype=int)
    
    def train_high_freq_model(self, df):
        """Yüksek frekans modeli eğit"""
        try:
            features = self.create_high_freq_features(df)
            target = self.create_high_freq_target(df)
            
            # Veri temizleme
            features = features.fillna(0)
            target = target.fillna(0)
            
            # Yeterli veri kontrolü
            if len(features) < 100:
                return 0, pd.DataFrame()
            
            # Zaman serisi split
            split_idx = int(len(features) * 0.8)  # %80 train
            X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
            y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]
            
            # Gelişmiş model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.is_trained = True
            self.feature_columns = features.columns.tolist()
            
            # Detaylı feature importance
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return accuracy, feature_importance
            
        except Exception as e:
            return 0, pd.DataFrame()
    
    def predict_high_freq_signals(self, df):
        """Yüksek frekanslı sinyal tahmini"""
        try:
            if not self.is_trained or self.model is None:
                return np.zeros(len(df))
            
            features = self.create_high_freq_features(df)
            features = features.fillna(0)
            predictions = self.model.predict(features)
            
            return predictions
            
        except Exception as e:
            return np.zeros(len(df))

# DÜZELTİLMİŞ YÜKSEK FREKANS Ana Strateji sınıfı
class HighFrequencyStrategy:
    def __init__(self, initial_capital: float = 10000, enable_ml: bool = True):
        self.initial_capital = initial_capital
        self.results = {}
        self.enable_ml = enable_ml
        self.ml_strategy = HighFrequencyMLStrategy() if enable_ml else None
        
    def calculate_high_freq_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Yüksek frekanslı göstergeleri hesapla - TAMAMEN DÜZELTİLDİ"""
        try:
            df = df.copy()
            
            # Çok kısa vadeli RSI'lar - TEK TEK EKLE
            for period in [4, 6, 8]:
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=period, min_periods=1).mean()
                avg_loss = loss.rolling(window=period, min_periods=1).mean()
                rs = avg_gain / avg_loss.replace(0, np.nan)
                rs = rs.fillna(1)
                df[f'RSI_{period}h'] = 100 - (100 / (1 + rs))  # TEK SÜTUN
            
            # Çoklu EMA'lar - TEK TEK EKLE
            for span in [2, 4, 6, 8, 12]:
                df[f'EMA_{span}h'] = df['Close'].ewm(span=span, adjust=False).mean()  # TEK SÜTUN
            
            # Momentum göstergeleri - TEK TEK EKLE
            for shift in [1, 2, 4, 6]:
                df[f'Momentum_{shift}h'] = df['Close'] - df['Close'].shift(shift)  # TEK SÜTUN
            
            # Volume göstergeleri - TEK TEK EKLE
            volume_ema_8h = df['Volume'].ewm(span=8).mean()
            df['Volume_EMA_8h'] = volume_ema_8h  # TEK SÜTUN
            df['Volume_Ratio'] = df['Volume'] / volume_ema_8h.replace(0, 1)  # TEK SÜTUN
            
            # Volatilite - TEK SÜTUN
            df['Volatility_12h'] = df['Close'].rolling(12).std() / df['Close']
            
            return df.fillna(0)
            
        except Exception as e:
            st.error(f"Göstergeler hesaplanırken hata: {e}")
            # Hata durumunda basit değerler
            df['RSI_6h'] = 50
            df['EMA_4h'] = df['Close']
            df['Volume_Ratio'] = 1
            return df
    
    def generate_high_freq_signals(self, df: pd.DataFrame, signal_threshold: float = 1.2) -> pd.DataFrame:
        """Yüksek frekanslı sinyal üret"""
        try:
            df = df.copy()
            df['Signal'] = 0
            
            # ML sinyalleri
            ml_signals = np.zeros(len(df))
            ml_accuracy = 0
            
            if self.enable_ml and self.ml_strategy:
                ml_accuracy, feature_importance = self.ml_strategy.train_high_freq_model(df)
                if ml_accuracy > 0.45:
                    ml_signals = self.ml_strategy.predict_high_freq_signals(df)
                    st.success(f"🤖 Yüksek Frekans ML Doğruluğu: %{ml_accuracy:.1f}")
                    
                    if not feature_importance.empty:
                        st.write("**Özellik Önem Sıralaması (İlk 10):**")
                        st.dataframe(feature_importance.head(10))
            
            # YÜKSEK FREKANS GELENEKSEL SİNYALLER
            for i in range(12, len(df)):  # İlk 12 saat atla
                try:
                    # Kısa vadeli göstergeler
                    rsi_4h = float(df[f'RSI_4h'].iloc[i])
                    rsi_6h = float(df[f'RSI_6h'].iloc[i])
                    ema_4h = float(df['EMA_4h'].iloc[i])
                    ema_12h = float(df['EMA_12h'].iloc[i])
                    momentum_2h = float(df['Momentum_2h'].iloc[i])
                    volume_ratio = float(df['Volume_Ratio'].iloc[i])
                    volatility = float(df['Volatility_12h'].iloc[i])
                    
                    long_signals = 0
                    short_signals = 0
                    
                    # ÇOK KISA VADELİ LONG SİNYALLERİ
                    if rsi_4h < 35: long_signals += 1.0
                    if rsi_6h < 40: long_signals += 0.5
                    if ema_4h > ema_12h: long_signals += 1.0
                    if momentum_2h > 0: long_signals += 0.5
                    if volume_ratio > 1.3: long_signals += 0.5
                    if volatility < 0.02: long_signals += 0.5  # Düşük volatilite
                    
                    # ÇOK KISA VADELİ SHORT SİNYALLERİ
                    if rsi_4h > 65: short_signals += 1.0
                    if rsi_6h > 60: short_signals += 0.5
                    if ema_4h < ema_12h: short_signals += 1.0
                    if momentum_2h < 0: short_signals += 0.5
                    if volume_ratio > 1.3: short_signals += 0.5
                    if volatility > 0.05: short_signals += 0.5  # Yüksek volatilite
                    
                    # ML SİNYAL GÜÇLENDİRME
                    ml_signal = ml_signals[i]
                    if ml_signal == 1:
                        long_signals += 1.5  # ML sinyali daha ağırlıklı
                    elif ml_signal == -1:
                        short_signals += 1.5
                    
                    # SİNYAL KARARI
                    if long_signals >= signal_threshold:
                        df.loc[df.index[i], 'Signal'] = 1
                    elif short_signals >= signal_threshold:
                        df.loc[df.index[i], 'Signal'] = -1
                        
                except Exception as e:
                    continue
            
            # Sinyal istatistikleri
            total_signals = (df['Signal'] != 0).sum()
            long_signals = (df['Signal'] == 1).sum()
            short_signals = (df['Signal'] == -1).sum()
            
            st.info(f"""
            **📊 Sinyal İstatistikleri:**
            - Toplam Sinyal: {total_signals}
            - Long Sinyal: {long_signals}
            - Short Sinyal: {short_signals}
            - Sinyal Yoğunluğu: {total_signals/len(df)*100:.1f}%
            """)
                    
            return df
            
        except Exception as e:
            st.error(f"Sinyal oluşturma hatası: {e}")
            df['Signal'] = 0
            return df
    
    def backtest_high_freq_strategy(self, df: pd.DataFrame, progress_bar, 
                                  position_size: float, stop_loss: float, 
                                  take_profit: float, max_hold_hours: int = 24) -> dict:
        """Yüksek frekanslı backtest"""
        try:
            capital = self.initial_capital
            position = 0
            entry_price = 0
            entry_capital = 0
            entry_time = None
            trades = []
            total_trades = 0
            winning_trades = 0
            total_pnl = 0
            
            for i in range(len(df)):
                if i % 100 == 0:  # Daha sık progress update
                    progress_bar.progress(min(i / len(df), 1.0))
                
                current_time = df.index[i]
                current_price = float(df['Close'].iloc[i])
                signal = int(df['Signal'].iloc[i])
                
                # POZİSYON AÇMA - Çok daha agresif
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
                
                # POZİSYON TAKİP - Çok daha sık kontrol
                elif position != 0:
                    current_trade = trades[-1]
                    hold_hours = (current_time - entry_time).total_seconds() / 3600
                    
                    if position == 1:  # Long pozisyon
                        pnl_percent = (current_price - entry_price) / entry_price
                        
                        # ÇOKLU KAPATMA KOŞULLARI
                        close_condition = (
                            pnl_percent <= -(stop_loss / 100) or
                            pnl_percent >= (take_profit / 100) or
                            signal == -1 or  # Zıt sinyal
                            hold_hours >= max_hold_hours or  # Maksimum bekleme
                            pnl_percent >= 0.05  # Hızlı kar realizasyonu
                        )
                        
                        if close_condition:
                            pnl_amount = entry_capital * pnl_percent
                            capital += pnl_amount
                            total_pnl += pnl_amount
                            
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
                    
                    elif position == -1:  # Short pozisyon
                        pnl_percent = (entry_price - current_price) / entry_price
                        
                        close_condition = (
                            pnl_percent <= -(stop_loss / 100) or
                            pnl_percent >= (take_profit / 100) or
                            signal == 1 or  # Zıt sinyal
                            hold_hours >= max_hold_hours or  # Maksimum bekleme
                            pnl_percent >= 0.05  # Hızlı kar realizasyonu
                        )
                        
                        if close_condition:
                            pnl_amount = entry_capital * pnl_percent
                            capital += pnl_amount
                            total_pnl += pnl_amount
                            
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
                total_pnl += pnl_amount
                
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
            
            # DETAYLI SONUÇLAR
            final_capital = max(capital, 0)
            total_return = ((final_capital - self.initial_capital) / self.initial_capital) * 100
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Profit factor
            total_profit = sum(trade['pnl'] for trade in trades if trade['pnl'] > 0)
            total_loss = abs(sum(trade['pnl'] for trade in trades if trade['pnl'] < 0))
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Ortalama işlem süresi
            closed_trades = [t for t in trades if t['status'] == 'CLOSED']
            avg_hold_hours = np.mean([t['hold_hours'] for t in closed_trades]) if closed_trades else 0
            
            # Sharpe Ratio (basit)
            if len(closed_trades) > 1:
                returns = [t['pnl_percent'] / 100 for t in closed_trades]
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(365 * 24) if np.std(returns) > 0 else 0
            else:
                sharpe_ratio = 0
            
            self.results = {
                'initial_capital': self.initial_capital,
                'final_capital': final_capital,
                'total_return': total_return,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'avg_hold_hours': avg_hold_hours,
                'total_pnl': total_pnl,
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
                'sharpe_ratio': 0,
                'avg_hold_hours': 0,
                'total_pnl': 0,
                'trades': []
            }

# Streamlit arayüzü
st.sidebar.header("⚡ Yüksek Frekans Ayarları")

# Kripto seçimi
crypto_symbols = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD", 
    "Binance Coin (BNB-USD)": "BNB-USD",
    "Solana (SOL-USD)": "SOL-USD"
}

selected_crypto = st.sidebar.selectbox(
    "Kripto Para Seçin:",
    list(crypto_symbols.keys())
)

symbol = crypto_symbols[selected_crypto]

# Zaman periyodu
st.sidebar.subheader("⏰ Zaman Ayarları")
timeframe = st.sidebar.selectbox(
    "Zaman Periyodu:",
    ["1h", "2h", "4h"],
    index=0
)

end_date = st.sidebar.date_input(
    "Bitiş Tarihi:",
    datetime.date.today() - datetime.timedelta(days=1)
)

period_months = st.sidebar.slider(
    "Veri Süresi (Ay):",
    min_value=1,
    max_value=6,
    value=3,
    step=1
)

start_date = end_date - datetime.timedelta(days=period_months*30)

# Risk yönetimi
st.sidebar.subheader("🎯 Risk Yönetimi")
initial_capital = st.sidebar.number_input("Başlangıç Sermayesi (USD):", 1000, 1000000, 10000, 1000)
position_size = st.sidebar.slider("İşlem Büyüklüğü (%):", 10, 100, 50, 5)

st.sidebar.subheader("🛡️ Stop & Take Profit")
stop_loss = st.sidebar.slider("Stop Loss (%):", 0.5, 5.0, 1.5, 0.1)
take_profit = st.sidebar.slider("Take Profit (%):", 0.5, 10.0, 3.0, 0.1)
max_hold_hours = st.sidebar.slider("Maksimum Bekleme (Saat):", 1, 48, 12, 1)

st.sidebar.subheader("🤖 ML Ayarları")
enable_ml = st.sidebar.checkbox("Yüksek Frekans ML Modelini Etkinleştir", value=True)
signal_threshold = st.sidebar.slider("Sinyal Eşik Değeri:", 0.5, 3.0, 1.5, 0.1)

# Ana içerik
st.subheader("⚡ Yüksek Frekanslı Trading Stratejisi")

st.success("""
**🚀 ÖZELLİKLER:**
- **Saatlik/2-saatlik/4-saatlik** veriler
- **1000+ işlem** backtest potansiyeli  
- **Kısa vadeli** teknik göstergeler (1-12 saat)
- **ML destekli** yüksek frekans sinyalleri
- **Gerçekçi** risk yönetimi
""")

# Veri yükleme - YÜKSEK FREKANS
@st.cache_data
def load_high_freq_data(symbol, start_date, end_date, timeframe='1h'):
    try:
        data = yf.download(symbol, start=start_date, end=end_date, interval=timeframe, progress=False)
        if data is None or data.empty:
            return None
        
        st.success(f"✅ {len(data)} adet {timeframe} verisi yüklendi")
        return data
        
    except Exception as e:
        st.error(f"Veri yüklenirken hata: {e}")
        return None

# Simülasyon
st.markdown("---")
st.subheader("📊 Veri Yükleme")

data = load_high_freq_data(symbol, start_date, end_date, timeframe)

if data is not None and not data.empty:
    try:
        close_prices = data['Close']
        first_price = float(close_prices.iloc[0])
        last_price = float(close_prices.iloc[-1])
        price_change = ((last_price - first_price) / first_price) * 100
        volatility = close_prices.pct_change().std() * np.sqrt(365 * 24) * 100  # Yıllık volatilite
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("İlk Fiyat", f"${first_price:.2f}")
        with col2:
            st.metric("Son Fiyat", f"${last_price:.2f}")
        with col3:
            st.metric("Değişim", f"{price_change:+.2f}%")
        with col4:
            st.metric("Volatilite", f"{volatility:.1f}%")
            
        st.info(f"**Veri Aralığı:** {data.index[0].strftime('%d.%m.%Y')} - {data.index[-1].strftime('%d.%m.%Y')}")
        
    except Exception as e:
        st.error(f"Veri gösterilirken hata: {e}")
else:
    st.warning("⚠️ Veri yüklenemedi. Lütfen tarih aralığını kontrol edin.")

# SİMÜLASYON BUTONU
st.markdown("---")
st.subheader("🚀 Yüksek Frekans Backtest Başlat")

if st.button("⚡ BACKTEST BAŞLAT", type="primary", use_container_width=True):
    if data is not None and not data.empty:
        with st.spinner("Yüksek frekans backtest çalışıyor... Bu biraz zaman alabilir..."):
            start_time = time.time()
            progress_bar = st.progress(0)
            
            try:
                # Stratejiyi başlat
                strategy = HighFrequencyStrategy(initial_capital, enable_ml=enable_ml)
                
                # Göstergeleri hesapla
                data_with_indicators = strategy.calculate_high_freq_indicators(data)
                
                # Sinyalleri oluştur
                data_with_signals = strategy.generate_high_freq_signals(data_with_indicators, signal_threshold)
                
                # Backtest
                results = strategy.backtest_high_freq_strategy(
                    data_with_signals, progress_bar, position_size, stop_loss, take_profit, max_hold_hours
                )
                
                progress_bar.progress(1.0)
                end_time = time.time()
                calculation_time = end_time - start_time
                
                st.success(f"✅ Backtest {calculation_time:.2f} saniyede tamamlandı!")
                
                # DETAYLI SONUÇLAR
                st.subheader("📈 Detaylı Backtest Sonuçları")
                
                # Ana metrikler
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Başlangıç", f"${results['initial_capital']:,.0f}")
                with col2:
                    st.metric("Son Sermaye", f"${results['final_capital']:,.0f}", 
                             f"{results['total_return']:+.1f}%")
                with col3:
                    st.metric("Win Rate", f"{results['win_rate']:.1f}%")
                with col4:
                    st.metric("Toplam İşlem", f"{results['total_trades']}")
                
                # İkinci sıra metrikler
                col5, col6, col7, col8 = st.columns(4)
                with col5:
                    st.metric("Profit Factor", f"{results['profit_factor']:.2f}")
                with col6:
                    st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
                with col7:
                    st.metric("Ort. Bekleme", f"{results['avg_hold_hours']:.1f} saat")
                with col8:
                    st.metric("Toplam PnL", f"${results['total_pnl']:,.0f}")
                
                # Performans değerlendirme
                if results['win_rate'] > 60:
                    st.success("🎉 MÜKEMMEL! Strateji çok başarılı!")
                elif results['win_rate'] > 50:
                    st.success("📈 ÇOK İYİ! Strateji karlı.")
                elif results['win_rate'] > 40:
                    st.info("✅ İYİ! Strateji çalışıyor.")
                else:
                    st.warning("⚠️ GELİŞTİRİLMESİ GEREKİYOR!")
                
                # İşlem detayları - DÜZELTİLDİ
                if results['trades']:
                    closed_trades = [t for t in results['trades'] if t['status'] == 'CLOSED']
                    if closed_trades:
                        st.subheader("📋 Son 20 İşlem Detayı")
                        trades_df = pd.DataFrame(closed_trades[-20:])  # Son 20 işlem
                        
                        # Basit gösterim - format hatasını önle
                        display_df = trades_df[['entry_time', 'position', 'entry_price', 'exit_price', 'pnl', 'hold_hours']]
                        display_df = display_df.rename(columns={
                            'entry_time': 'Giriş', 'position': 'Pozisyon',
                            'entry_price': 'Giriş Fiyatı', 'exit_price': 'Çıkış Fiyatı',
                            'pnl': 'Kar/Zarar ($)', 'hold_hours': 'Süre (saat)'
                        })
                        
                        # Basit dataframe gösterimi
                        st.dataframe(display_df, use_container_width=True, height=400)
                
                # Başarı kutlaması
                if results['total_trades'] > 100 and results['win_rate'] > 50:
                    st.balloons()
                    
            except Exception as e:
                st.error(f"Backtest sırasında hata: {str(e)}")
    else:
        st.error("Veri yüklenemedi!")

st.markdown("---")
st.info("""
**⚡ Yüksek Frekans Trading Avantajları:**
- **Daha fazla işlem** → daha istatistiksel olarak anlamlı sonuçlar
- **Kısa vadeli trendler** → daha yüksek win rate potansiyeli  
- **Hızlı kar realizasyonu** → düşük drawdown
- **ML etkinliği** → kısa vadeli pattern'lerde daha başarılı

**⚠️ Riskler:**
- İşlem maliyetleri (commission, spread)
- Slippage riski
- Aşırı işlem (overtrading) riski
""")
