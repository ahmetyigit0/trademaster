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
    page_title="🚀 ASTRO KAR Strateji - %300 Hedef",
    page_icon="💎",
    layout="wide"
)

# Başlık
st.title("💎 ASTRO KAR Strateji - 3 Ayda %300 Hedef")
st.markdown("---")

# YÜKSEK RİSK ML Strateji sınıfı
class AstroProfitMLStrategy:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.is_trained = False
        
    def create_aggresive_features(self, df):
        """Aşırı agresif özellikler"""
        try:
            features = pd.DataFrame(index=df.index)
            
            # AŞIRI KISA VADELİ göstergeler
            # RSI - çok kısa
            for period in [2, 3, 4]:
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=period, min_periods=1).mean()
                avg_loss = loss.rolling(window=period, min_periods=1).mean()
                rs = avg_gain / avg_loss.replace(0, np.nan)
                rs = rs.fillna(1)
                features[f'rsi_{period}h'] = 100 - (100 / (1 + rs))
            
            # EMA - çok hızlı
            for span in [2, 3, 5]:
                features[f'ema_{span}h'] = df['Close'].ewm(span=span, adjust=False).mean()
            
            features['ema_cross_aggressive'] = (features['ema_2h'] - features['ema_5h']) / df['Close']
            
            # MOMENTUM - aşırı kısa
            for shift in [1, 2]:
                features[f'momentum_{shift}h'] = df['Close'] - df['Close'].shift(shift)
                features[f'roc_{shift}h'] = (df['Close'] / df['Close'].shift(shift) - 1) * 100
            
            # VOLUME - aşırı hassas
            volume_ema_4h = df['Volume'].ewm(span=4).mean()
            features['volume_spike'] = df['Volume'] / volume_ema_4h.replace(0, 1)
            features['volume_acceleration'] = features['volume_spike'].pct_change()
            
            # PRICE ACTION - ultra kısa
            features['price_acceleration_1h'] = df['Close'].pct_change(1)
            features['price_acceleration_2h'] = df['Close'].pct_change(2)
            features['high_low_pressure'] = (df['High'] - df['Low']) / df['Close'] * 100
            
            # VOLATILITY - anlık
            features['instant_volatility'] = df['Close'].rolling(4).std() / df['Close'] * 100
            features['volatility_expansion'] = features['instant_volatility'].pct_change()
            
            return features.fillna(0).replace([np.inf, -np.inf], 0)
            
        except Exception as e:
            features = pd.DataFrame(index=df.index)
            features['rsi_3h'] = 50
            features['ema_cross_aggressive'] = 0
            features['volume_spike'] = 1
            return features
    
    def create_aggresive_target(self, df, lookahead=1):  # SADECE 1 SAAT!
        """Aşırı agresif hedef"""
        try:
            # 1 saat sonraki getiri - çok riskli!
            future_return = df['Close'].shift(-lookahead) / df['Close'] - 1
            
            target = np.zeros(len(df))
            
            # ÇOK YÜKSEK THRESHOLD'lar - büyük hareketleri yakala
            bullish_threshold = 0.015  # %1.5 - ÇOK YÜKSEK
            bearish_threshold = -0.015 # %1.5 - ÇOK YÜKSEK
            
            target[future_return > bullish_threshold] = 1      # GÜÇLÜ AL
            target[future_return < bearish_threshold] = -1     # GÜÇLÜ SAT
            
            return pd.Series(target, index=df.index, dtype=int)
            
        except Exception as e:
            return pd.Series(np.zeros(len(df)), index=df.index, dtype=int)
    
    def train_aggresive_model(self, df):
        """Aşırı agresif model"""
        try:
            features = self.create_aggresive_features(df)
            target = self.create_aggresive_target(df)
            
            features = features.fillna(0)
            target = target.fillna(0)
            
            if len(features) < 50:
                return 0, pd.DataFrame()
            
            split_idx = int(len(features) * 0.75)
            X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
            y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]
            
            # DAHA AGRESİF MODEL
            self.model = RandomForestClassifier(
                n_estimators=150,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
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
    
    def predict_aggresive_signals(self, df):
        """Aşırı agresif sinyal"""
        try:
            if not self.is_trained or self.model is None:
                return np.zeros(len(df))
            
            features = self.create_aggresive_features(df)
            features = features.fillna(0)
            predictions = self.model.predict(features)
            
            return predictions
            
        except Exception as e:
            return np.zeros(len(df))

# AŞIRI AGRESİF Ana Strateji
class AstroProfitStrategy:
    def __init__(self, initial_capital: float = 10000, enable_ml: bool = True):
        self.initial_capital = initial_capital
        self.results = {}
        self.enable_ml = enable_ml
        self.ml_strategy = AstroProfitMLStrategy() if enable_ml else None
        
    def calculate_aggresive_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aşırı agresif göstergeler"""
        try:
            df = df.copy()
            
            # ULTRA KISA VADELİ RSI
            for period in [2, 3, 4]:
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=period, min_periods=1).mean()
                avg_loss = loss.rolling(window=period, min_periods=1).mean()
                rs = avg_gain / avg_loss.replace(0, np.nan)
                rs = rs.fillna(1)
                df[f'RSI_{period}h'] = 100 - (100 / (1 + rs))
            
            # ULTRA HIZLI EMA
            for span in [2, 3, 5, 8]:
                df[f'EMA_{span}h'] = df['Close'].ewm(span=span, adjust=False).mean()
            
            # ULTRA HIZLI MOMENTUM
            for shift in [1, 2]:
                df[f'Momentum_{shift}h'] = (df['Close'] / df['Close'].shift(shift) - 1) * 100
            
            # VOLUME SPİKE
            volume_ema_4h = df['Volume'].ewm(span=4).mean()
            df['Volume_Spike'] = df['Volume'] / volume_ema_4h.replace(0, 1)
            
            # ANLIK VOLATILITE
            df['Volatility_4h'] = df['Close'].rolling(4).std() / df['Close'] * 100
            
            return df.fillna(0)
            
        except Exception as e:
            df['RSI_3h'] = 50
            df['EMA_3h'] = df['Close']
            df['Volume_Spike'] = 1
            return df
    
    def generate_aggresive_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """AŞIRI AGRESİF sinyal üretimi"""
        try:
            df = df.copy()
            df['Signal'] = 0
            
            # ML SİNYALLERİ
            ml_signals = np.zeros(len(df))
            ml_accuracy = 0
            
            if self.enable_ml and self.ml_strategy:
                ml_accuracy, feature_importance = self.ml_strategy.train_aggresive_model(df)
                if ml_accuracy > 0.40:  # Daha düşük eşik - daha fazla sinyal
                    ml_signals = self.ml_strategy.predict_aggresive_signals(df)
                    st.success(f"🤖 AGRESİF ML Doğruluğu: %{ml_accuracy:.1f}")
            
            # AŞIRI AGRESİF SİNYAL SİSTEMİ
            for i in range(6, len(df)):  # Sadece 6 saat bekle
                try:
                    # ULTRA KISA VADELİ GÖSTERGELER
                    rsi_2h = float(df['RSI_2h'].iloc[i])
                    rsi_3h = float(df['RSI_3h'].iloc[i])
                    ema_2h = float(df['EMA_2h'].iloc[i])
                    ema_3h = float(df['EMA_3h'].iloc[i])
                    ema_5h = float(df['EMA_5h'].iloc[i])
                    momentum_1h = float(df['Momentum_1h'].iloc[i])
                    volume_spike = float(df['Volume_Spike'].iloc[i])
                    volatility = float(df['Volatility_4h'].iloc[i])
                    
                    long_signals = 0
                    short_signals = 0
                    
                    # AŞIRI AGRESİF LONG KOŞULLARI
                    if rsi_2h < 25: long_signals += 2.0  # AŞIRI OVERSOLD
                    if rsi_3h < 30: long_signals += 1.5
                    
                    if ema_2h > ema_3h and ema_3h > ema_5h: long_signals += 2.0
                    elif ema_2h > ema_5h: long_signals += 1.0
                    
                    if momentum_1h > 1.0: long_signals += 1.5  # GÜÇLÜ MOMENTUM
                    elif momentum_1h > 0.5: long_signals += 1.0
                    
                    if volume_spike > 2.0: long_signals += 1.5  # AŞIRI VOLUME
                    elif volume_spike > 1.5: long_signals += 1.0
                    
                    if volatility > 3.0: long_signals += 1.0  # YÜKSEK VOLATILITE
                    
                    # AŞIRI AGRESİF SHORT KOŞULLARI
                    if rsi_2h > 75: short_signals += 2.0  # AŞIRI OVERBOUGHT
                    if rsi_3h > 70: short_signals += 1.5
                    
                    if ema_2h < ema_3h and ema_3h < ema_5h: short_signals += 2.0
                    elif ema_2h < ema_5h: short_signals += 1.0
                    
                    if momentum_1h < -1.0: short_signals += 1.5
                    elif momentum_1h < -0.5: short_signals += 1.0
                    
                    if volume_spike > 2.0: short_signals += 1.5
                    elif volume_spike > 1.5: short_signals += 1.0
                    
                    if volatility > 3.0: short_signals += 1.0
                    
                    # ML GÜÇLENDİRME - ÇOK AĞIRLIKLI
                    ml_signal = ml_signals[i]
                    if ml_signal == 1:
                        long_signals += 3.0  # ÇOK YÜKSEK AĞIRLIK
                    elif ml_signal == -1:
                        short_signals += 3.0
                    
                    # AŞIRI AGRESİF SİNYAL - ÇOK DÜŞÜK EŞİK
                    if long_signals >= 1.5:  # ÇOK DÜŞÜK EŞİK
                        df.loc[df.index[i], 'Signal'] = 1
                    elif short_signals >= 1.5:
                        df.loc[df.index[i], 'Signal'] = -1
                        
                except:
                    continue
            
            total_signals = (df['Signal'] != 0).sum()
            st.success(f"🚀 **AŞIRI AGRESİF Sinyal Sayısı:** {total_signals}")
                    
            return df
            
        except Exception as e:
            st.error(f"Sinyal oluşturma hatası: {e}")
            df['Signal'] = 0
            return df
    
    def backtest_aggresive_strategy(self, df: pd.DataFrame, progress_bar, 
                                  position_size: float, stop_loss: float, 
                                  take_profit: float, max_hold_hours: int = 4) -> dict:  # SADECE 4 SAAT!
        """AŞIRI AGRESİF backtest"""
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
                if i % 50 == 0:  # Daha sık progress
                    progress_bar.progress(min(i / len(df), 1.0))
                
                current_time = df.index[i]
                current_price = float(df['Close'].iloc[i])
                signal = int(df['Signal'].iloc[i])
                
                # AŞIRI AGRESİF POZİSYON AÇMA
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
                
                # AŞIRI AGRESİF POZİSYON TAKİP
                elif position != 0:
                    current_trade = trades[-1]
                    hold_hours = (current_time - entry_time).total_seconds() / 3600
                    
                    if position == 1:  # Long
                        pnl_percent = (current_price - entry_price) / entry_price
                        
                        # AŞIRI AGRESİF KAPATMA
                        close_condition = (
                            pnl_percent <= -(stop_loss / 100) or
                            pnl_percent >= (take_profit / 100) or
                            signal == -1 or
                            hold_hours >= max_hold_hours or
                            pnl_percent >= 0.08  # ÇOK YÜKSEK KAR - %8
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
                            pnl_percent >= 0.08  # ÇOK YÜKSEK KAR - %8
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

# Streamlit arayüzü - AŞIRI AGRESİF
st.sidebar.header("💎 ASTRO KAR Ayarları - %300 Hedef")

crypto_symbols = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD", 
    "Solana (SOL-USD)": "SOL-USD",
    "Cardano (ADA-USD)": "ADA-USD",
    "Dogecoin (DOGE-USD)": "DOGE-USD"  # Yüksek volatilite
}

selected_crypto = st.sidebar.selectbox("Kripto Para Seçin:", list(crypto_symbols.keys()))
symbol = crypto_symbols[selected_crypto]

# AŞIRI AGRESİF AYARLAR
st.sidebar.subheader("⚡ AŞIRI AGRESİF Ayarlar")
timeframe = st.sidebar.selectbox("Zaman Periyodu:", ["1h", "30m"], index=0)  # 30m eklendi!

end_date = st.sidebar.date_input("Bitiş Tarihi:", datetime.date.today() - datetime.timedelta(days=1))
period_months = st.sidebar.slider("Veri Süresi (Ay):", 1, 6, 3, 1)
start_date = end_date - datetime.timedelta(days=period_months*30)

st.sidebar.subheader("🎯 AŞIRI AGRESİF Risk")
initial_capital = st.sidebar.number_input("Başlangıç Sermayesi (USD):", 1000, 100000, 10000, 1000)
position_size = st.sidebar.slider("İşlem Büyüklüğü (%):", 50, 100, 80, 5)  # %80 - ÇOK YÜKSEK!

st.sidebar.subheader("💥 AŞIRI AGRESİF Stop & Take Profit")
stop_loss = st.sidebar.slider("Stop Loss (%):", 2.0, 10.0, 5.0, 0.5)  # YÜKSEK STOP
take_profit = st.sidebar.slider("Take Profit (%):", 5.0, 20.0, 12.0, 0.5)  # ÇOK YÜKSEK TAKE PROFIT
max_hold_hours = st.sidebar.slider("Maksimum Bekleme (Saat):", 1, 6, 4, 1)  # ÇOK KISA

st.sidebar.subheader("🤖 AŞIRI AGRESİF ML")
enable_ml = st.sidebar.checkbox("AŞIRI AGRESİF ML'yi Etkinleştir", value=True)

# Ana içerik
st.subheader("💎 ASTRO KAR Strateji - 3 Ayda %300 Hedef")

st.warning("""
**🚨 ULTRA YÜKSEK RİSK UYARISI!**
- **Position Size:** %80 (ÇOK YÜKSEK RİSK)
- **Stop Loss:** %5 (YÜKSEK)  
- **Take Profit:** %12 (ÇOK YÜKSEK)
- **Maksimum Bekleme:** 4 saat (ÇOK KISA)
- **Hedef:** 3 ayda %300+ getiri

**⚠️ BU STRATEJİ ÇOK YÜKSEK RİSK İÇERİR! POTANSİYEL %100 KAYIP RİSKİ!**
""")

# Veri yükleme
@st.cache_data
def load_aggresive_data(symbol, start_date, end_date, timeframe='1h'):
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

data = load_aggresive_data(symbol, start_date, end_date, timeframe)

if data is not None and not data.empty:
    close_prices = data['Close']
    first_price = float(close_prices.iloc[0])
    last_price = float(close_prices.iloc[-1])
    price_change = ((last_price - first_price) / first_price) * 100
    volatility = close_prices.pct_change().std() * np.sqrt(365 * 24) * 100
    
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
st.subheader("🚀 ASTRO KAR Backtest Başlat")

if st.button("💥 ASTRO KAR BACKTEST BAŞLAT", type="primary", use_container_width=True):
    if data is not None and not data.empty:
        with st.spinner("AŞIRI AGRESİF backtest çalışıyor..."):
            start_time = time.time()
            progress_bar = st.progress(0)
            
            try:
                strategy = AstroProfitStrategy(initial_capital, enable_ml=enable_ml)
                data_with_indicators = strategy.calculate_aggresive_indicators(data)
                data_with_signals = strategy.generate_aggresive_signals(data_with_indicators)
                results = strategy.backtest_aggresive_strategy(
                    data_with_signals, progress_bar, position_size, stop_loss, take_profit, max_hold_hours
                )
                
                progress_bar.progress(1.0)
                end_time = time.time()
                
                st.success(f"✅ Backtest {end_time - start_time:.1f} saniyede tamamlandı!")
                
                # SONUÇLAR
                st.subheader("📈 ASTRO KAR Sonuçları")
                
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
                
                # PERFORMANS DEĞERLENDİRME
                if results['total_return'] >= 300:
                    st.success("🎉🎉🎉 HEDEFE ULAŞILDI! 3 AYDA %300+ KAR! 🎉🎉🎉")
                    st.balloons()
                elif results['total_return'] >= 200:
                    st.success("🎉 MÜKEMMEL! %200+ KAR - HEDEFE YAKINSINIZ!")
                    st.balloons()
                elif results['total_return'] >= 100:
                    st.success("📈 ÇOK İYİ! %100+ KAR - HEDEF MÜMKÜN!")
                elif results['total_return'] >= 50:
                    st.info("✅ İYİ! %50+ KAR - DEVAM EDİN!")
                elif results['total_return'] > 0:
                    st.warning("⚠️ DÜŞÜK KAR - AYARLARI DEĞİŞTİRİN")
                else:
                    st.error("💥 KAYIP! - STRATEJİ YENİDEN GÖZDEN GEÇİRİLMELİ")
                
                # Detaylar
                st.info(f"**Profit Factor:** {results['profit_factor']:.2f} | **Karlı İşlem:** {results['winning_trades']}")
                    
            except Exception as e:
                st.error(f"Backtest sırasında hata: {str(e)}")
    else:
        st.error("Veri yüklenemedi!")

st.markdown("---")
st.error("""
**🚨 ÖNEMLİ UYARILAR:**
1. **Bu strateji ÇOK YÜKSEK RİSK içerir**
2. **%100 kayıp riski bulunmaktadır**
3. **Sadece deneyimli trader'lar için**
4. **Gerçek parayla asla test etmeyin**
5. **Küçük pozisyonlarla başlayın**

**💎 TAVSİYELER:**
- **Solana, Dogecoin gibi yüksek volatilite coin'lerde daha iyi sonuç verebilir**
- **30m timeframe daha fazla işlem üretebilir**
- **ML her zaman aktif olmalı**
""")
