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
        """Yüksek frekanslı özellikler oluştur"""
        try:
            features = pd.DataFrame(index=df.index)
            
            # RSI - kısa periyot
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=6, min_periods=1).mean()
            avg_loss = loss.rolling(window=6, min_periods=1).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rs = rs.fillna(1)
            features['rsi_6h'] = 100 - (100 / (1 + rs))
            
            # EMA'lar
            ema_4h = df['Close'].ewm(span=4, adjust=False).mean()
            ema_12h = df['Close'].ewm(span=12, adjust=False).mean()
            features['ema_4h'] = ema_4h
            features['ema_12h'] = ema_12h
            features['ema_cross'] = (ema_4h - ema_12h) / df['Close']
            
            # Momentum
            features['momentum_2h'] = df['Close'] - df['Close'].shift(2)
            features['momentum_6h'] = df['Close'] - df['Close'].shift(6)
            
            # Volatilite
            features['volatility_8h'] = df['Close'].rolling(8).std() / df['Close']
            
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
    
    def create_high_freq_target(self, df, lookahead=2):  # Daha kısa tahmin
        """Kısa vadeli hedef değişken"""
        try:
            # 2 saat sonraki fiyat değişimi
            future_return = df['Close'].shift(-lookahead) / df['Close'] - 1
            
            target = np.zeros(len(df))
            bullish_threshold = 0.005  # %0.5
            bearish_threshold = -0.005 # %0.5
            
            target[future_return > bullish_threshold] = 1
            target[future_return < bearish_threshold] = -1
            
            return pd.Series(target, index=df.index, dtype=int)
            
        except Exception as e:
            return pd.Series(np.zeros(len(df)), index=df.index, dtype=int)
    
    def train_high_freq_model(self, df):
        """Yüksek frekans modeli eğit"""
        try:
            features = self.create_high_freq_features(df)
            target = self.create_high_freq_target(df)
            
            features = features.fillna(0)
            target = target.fillna(0)
            
            if len(features) < 100:
                return 0, pd.DataFrame()
            
            split_idx = int(len(features) * 0.8)
            X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
            y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]
            
            self.model = RandomForestClassifier(
                n_estimators=80,
                max_depth=12,
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

# İYİLEŞTİRİLMİŞ Ana Strateji sınıfı
class HighFrequencyStrategy:
    def __init__(self, initial_capital: float = 10000, enable_ml: bool = True):
        self.initial_capital = initial_capital
        self.results = {}
        self.enable_ml = enable_ml
        self.ml_strategy = HighFrequencyMLStrategy() if enable_ml else None
        
    def calculate_high_freq_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Yüksek frekanslı göstergeleri hesapla"""
        try:
            df = df.copy()
            
            # RSI'lar
            for period in [4, 6, 8]:
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=period, min_periods=1).mean()
                avg_loss = loss.rolling(window=period, min_periods=1).mean()
                rs = avg_gain / avg_loss.replace(0, np.nan)
                rs = rs.fillna(1)
                df[f'RSI_{period}h'] = 100 - (100 / (1 + rs))
            
            # EMA'lar
            for span in [4, 8, 12]:
                df[f'EMA_{span}h'] = df['Close'].ewm(span=span, adjust=False).mean()
            
            # Momentum
            for shift in [2, 4, 6]:
                df[f'Momentum_{shift}h'] = df['Close'] - df['Close'].shift(shift)
            
            # Volume
            volume_ema_8h = df['Volume'].ewm(span=8).mean()
            df['Volume_Ratio'] = df['Volume'] / volume_ema_8h.replace(0, 1)
            
            # Volatilite
            df['Volatility_12h'] = df['Close'].rolling(12).std() / df['Close']
            
            return df.fillna(0)
            
        except Exception as e:
            st.error(f"Göstergeler hesaplanırken hata: {e}")
            df['RSI_6h'] = 50
            df['EMA_4h'] = df['Close']
            df['Volume_Ratio'] = 1
            return df
    
    def generate_high_freq_signals(self, df: pd.DataFrame, signal_threshold: float = 1.2) -> pd.DataFrame:
        """DAHA İYİ sinyal üret - KARLILIK ODAKLI"""
        try:
            df = df.copy()
            df['Signal'] = 0
            
            # ML sinyalleri
            ml_signals = np.zeros(len(df))
            ml_accuracy = 0
            
            if self.enable_ml and self.ml_strategy:
                ml_accuracy, feature_importance = self.ml_strategy.train_high_freq_model(df)
                if ml_accuracy > 0.48:  # Daha yüksek eşik
                    ml_signals = self.ml_strategy.predict_high_freq_signals(df)
                    st.success(f"🤖 ML Doğruluğu: %{ml_accuracy:.1f}")
            
            # İYİLEŞTİRİLMİŞ SİNYAL SİSTEMİ
            for i in range(12, len(df)):
                try:
                    rsi_4h = float(df[f'RSI_4h'].iloc[i])
                    rsi_6h = float(df[f'RSI_6h'].iloc[i])
                    ema_4h = float(df['EMA_4h'].iloc[i])
                    ema_8h = float(df['EMA_8h'].iloc[i])
                    ema_12h = float(df['EMA_12h'].iloc[i])
                    momentum_2h = float(df['Momentum_2h'].iloc[i])
                    volume_ratio = float(df['Volume_Ratio'].iloc[i])
                    volatility = float(df['Volatility_12h'].iloc[i])
                    
                    long_signals = 0
                    short_signals = 0
                    
                    # İYİLEŞTİRİLMİŞ LONG KOŞULLARI
                    if rsi_4h < 30 and rsi_6h < 35:  # Daha oversold
                        long_signals += 2.0
                    elif rsi_4h < 35:
                        long_signals += 1.0
                    
                    if ema_4h > ema_8h and ema_8h > ema_12h:  # Güçlü trend
                        long_signals += 1.5
                    elif ema_4h > ema_12h:
                        long_signals += 1.0
                    
                    if momentum_2h > 0:
                        long_signals += 0.5
                    
                    if volume_ratio > 1.5:  # Daha yüksek volume eşiği
                        long_signals += 0.5
                    
                    if volatility < 0.015:  # Düşük volatilite
                        long_signals += 0.5
                    
                    # İYİLEŞTİRİLMİŞ SHORT KOŞULLARI
                    if rsi_4h > 70 and rsi_6h > 65:  # Daha overbought
                        short_signals += 2.0
                    elif rsi_4h > 65:
                        short_signals += 1.0
                    
                    if ema_4h < ema_8h and ema_8h < ema_12h:  # Güçlü downtrend
                        short_signals += 1.5
                    elif ema_4h < ema_12h:
                        short_signals += 1.0
                    
                    if momentum_2h < 0:
                        short_signals += 0.5
                    
                    if volume_ratio > 1.5:
                        short_signals += 0.5
                    
                    if volatility > 0.04:  # Yüksek volatilite
                        short_signals += 0.5
                    
                    # ML GÜÇLENDİRME
                    ml_signal = ml_signals[i]
                    if ml_signal == 1:
                        long_signals += 1.5
                    elif ml_signal == -1:
                        short_signals += 1.5
                    
                    # DAHA SEÇİCİ SİNYAL
                    if long_signals >= max(signal_threshold, 2.0):  # Minimum 2.0
                        df.loc[df.index[i], 'Signal'] = 1
                    elif short_signals >= max(signal_threshold, 2.0):
                        df.loc[df.index[i], 'Signal'] = -1
                        
                except:
                    continue
            
            total_signals = (df['Signal'] != 0).sum()
            st.info(f"**Üretilen sinyal sayısı:** {total_signals}")
                    
            return df
            
        except Exception as e:
            st.error(f"Sinyal oluşturma hatası: {e}")
            df['Signal'] = 0
            return df
    
    def backtest_high_freq_strategy(self, df: pd.DataFrame, progress_bar, 
                                  position_size: float, stop_loss: float, 
                                  take_profit: float, max_hold_hours: int = 8) -> dict:  # Daha kısa hold
        """İYİLEŞTİRİLMİŞ backtest"""
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
                
                # POZİSYON AÇMA
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
                            pnl_percent >= 0.03  # Daha erken kar al
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
                            pnl_percent >= 0.03  # Daha erken kar al
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
st.sidebar.header("⚡ Yüksek Frekans Ayarları")

crypto_symbols = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD", 
    "Binance Coin (BNB-USD)": "BNB-USD"
}

selected_crypto = st.sidebar.selectbox("Kripto Para Seçin:", list(crypto_symbols.keys()))
symbol = crypto_symbols[selected_crypto]

# OPTİMİZE AYARLAR
st.sidebar.subheader("⏰ Zaman Ayarları")
timeframe = st.sidebar.selectbox("Zaman Periyodu:", ["1h", "2h"], index=0)

end_date = st.sidebar.date_input("Bitiş Tarihi:", datetime.date.today() - datetime.timedelta(days=1))
period_months = st.sidebar.slider("Veri Süresi (Ay):", 1, 6, 3, 1)
start_date = end_date - datetime.timedelta(days=period_months*30)

st.sidebar.subheader("🎯 OPTİMİZE Risk Yönetimi")
initial_capital = st.sidebar.number_input("Başlangıç Sermayesi (USD):", 1000, 100000, 10000, 1000)
position_size = st.sidebar.slider("İşlem Büyüklüğü (%):", 20, 100, 60, 5)  # Daha yüksek position

st.sidebar.subheader("🛡️ OPTİMİZE Stop & Take Profit")
stop_loss = st.sidebar.slider("Stop Loss (%):", 1.0, 3.0, 1.2, 0.1)  # Daha sıkı stop
take_profit = st.sidebar.slider("Take Profit (%):", 2.0, 6.0, 3.5, 0.1)  # Daha yüksek take profit
max_hold_hours = st.sidebar.slider("Maksimum Bekleme (Saat):", 4, 24, 8, 1)  # Daha kısa

st.sidebar.subheader("🤖 ML Ayarları")
enable_ml = st.sidebar.checkbox("ML Modelini Etkinleştir", value=True)
signal_threshold = st.sidebar.slider("Sinyal Eşik Değeri:", 1.5, 3.0, 2.0, 0.1)  # Daha seçici

# Ana içerik
st.subheader("⚡ OPTİMİZE Yüksek Frekans Stratejisi")

st.success("""
**🎯 İYİLEŞTİRİLMİŞ STRATEJİ:**
- **Daha seçici sinyaller** (daha az ama daha kaliteli işlem)
- **Daha sıkı risk yönetimi** (Stop: 1.2%, Take Profit: 3.5%)
- **Daha kısa pozisyon süreleri** (max 8 saat)
- **Gelişmiş ML entegrasyonu**
- **Daha yüksek position büyüklüğü** (%60)
""")

# Veri yükleme
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

# DÜZELTİLMİŞ veri gösterimi
st.markdown("---")
st.subheader("📊 Veri Yükleme")

data = load_high_freq_data(symbol, start_date, end_date, timeframe)

if data is not None and not data.empty:
    try:
        close_prices = data['Close']
        first_price = float(close_prices.iloc[0])
        last_price = float(close_prices.iloc[-1])
        price_change = ((last_price - first_price) / first_price) * 100
        
        # DÜZELTİLMİŞ - format hatası olmayan gösterim
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("İlk Fiyat", f"${first_price:.2f}")
        with col2:
            st.metric("Son Fiyat", f"${last_price:.2f}")
        with col3:
            st.metric("Değişim", f"{price_change:+.2f}%")
            
        st.info(f"**Veri Aralığı:** {data.index[0].strftime('%d.%m.%Y')} - {data.index[-1].strftime('%d.%m.%Y')}")
        
    except Exception as e:
        st.error(f"Veri gösterilirken hata: {e}")
else:
    st.warning("⚠️ Veri yüklenemedi.")

# SİMÜLASYON BUTONU
st.markdown("---")
st.subheader("🚀 Backtest Başlat")

if st.button("⚡ BACKTEST BAŞLAT", type="primary", use_container_width=True):
    if data is not None and not data.empty:
        with st.spinner("Backtest çalışıyor..."):
            start_time = time.time()
            progress_bar = st.progress(0)
            
            try:
                strategy = HighFrequencyStrategy(initial_capital, enable_ml=enable_ml)
                data_with_indicators = strategy.calculate_high_freq_indicators(data)
                data_with_signals = strategy.generate_high_freq_signals(data_with_indicators, signal_threshold)
                results = strategy.backtest_high_freq_strategy(
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
                
                col5, col6 = st.columns(2)
                with col5:
                    st.metric("Profit Factor", f"{results['profit_factor']:.2f}")
                with col6:
                    st.metric("Karlı İşlem", f"{results['winning_trades']}")
                
                # PERFORMANS DEĞERLENDİRME
                if results['total_return'] > 20:
                    st.success("🎉 MÜKEMMEL! Strateji çok başarılı!")
                elif results['total_return'] > 10:
                    st.success("📈 ÇOK İYİ! Strateji karlı.")
                elif results['total_return'] > 0:
                    st.info("✅ İYİ! Strateji çalışıyor.")
                else:
                    st.warning("⚠️ Strateji geliştirilmeli.")
                
                # İşlem detayları - BASİT GÖSTERİM
                if results['trades']:
                    closed_trades = [t for t in results['trades'] if t['status'] == 'CLOSED']
                    if closed_trades:
                        st.subheader("📋 İşlem Özeti")
                        
                        # Basit dataframe - format hatası yok
                        trades_df = pd.DataFrame(closed_trades)
                        summary_df = trades_df[['position', 'entry_price', 'exit_price', 'pnl', 'hold_hours']].tail(10)
                        
                        st.dataframe(summary_df, use_container_width=True)
                
                if results['total_return'] > 10:
                    st.balloons()
                    
            except Exception as e:
                st.error(f"Backtest sırasında hata: {str(e)}")
    else:
        st.error("Veri yüklenemedi!")

st.markdown("---")
st.info("""
**🎯 OPTİMİZASYONLAR:**
1. **Daha seçici sinyaller** - Daha az false signal
2. **Sıkı risk yönetimi** - Stop: 1.2%, Take Profit: 3.5%
3. **Kısa pozisyon süreleri** - Daha hızlı kar realizasyonu
4. **Yüksek position büyüklüğü** - %60 ile daha fazla kazanç
5. **Gelişmiş ML** - Daha yüksek doğruluk eşiği

**⚠️ NOT:** Bu ayarlar Bitcoin ve Ethereum için optimize edilmiştir.
""")
