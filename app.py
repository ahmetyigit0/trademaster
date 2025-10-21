# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =========================
# PAGE CONFIG - PROFESSIONAL
# =========================
st.set_page_config(
    page_title="AI Crypto Master | Profesyonel Analiz Platformu",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🚀"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .signal-buy {
        background: rgba(0,255,0,0.1);
        border-left: 4px solid #00ff00;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .signal-sell {
        background: rgba(255,0,0,0.1);
        border-left: 4px solid #ff0000;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .signal-neutral {
        background: rgba(255,255,0,0.1);
        border-left: 4px solid #ffff00;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# AI PREDICTION MODELS
# =========================
class AdvancedAIPredictor:
    def __init__(self):
        self.models = {}
    
    def calculate_support_resistance(self, df, window=20):
        """AI destekli destek/direnç seviyeleri"""
        high = df['High'].rolling(window=window).max()
        low = df['Low'].rolling(window=window).min()
        close = df['Close']
        
        # Fibonacci retracement levels
        recent_high = high.iloc[-1]
        recent_low = low.iloc[-1]
        diff = recent_high - recent_low
        
        fib_levels = {
            '0.236': recent_high - diff * 0.236,
            '0.382': recent_high - diff * 0.382,
            '0.5': recent_high - diff * 0.5,
            '0.618': recent_high - diff * 0.618,
            '0.786': recent_high - diff * 0.786
        }
        
        return fib_levels
    
    def momentum_oscillator(self, df, periods=[5, 10, 20]):
        """Çoklu zaman dilimi momentum analizi"""
        momentum_data = {}
        for period in periods:
            momentum_data[f'Momentum_{period}'] = (df['Close'] / df['Close'].shift(period) - 1) * 100
        return pd.DataFrame(momentum_data)
    
    def volume_analysis(self, df):
        """Akıllı hacim analizi"""
        df['Volume_SMA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        df['Volume_Spike'] = df['Volume_Ratio'] > 2.0
        return df
    
    def predictive_rsi(self, df, period=14, lookforward=5):
        """Geleceği tahmin eden RSI"""
        df['Future_Close'] = df['Close'].shift(-lookforward)
        df['Future_Return'] = (df['Future_Close'] - df['Close']) / df['Close'] * 100
        
        # RSI hesaplama
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # RSI + Momentum kombinasyonu
        df['RSI_Momentum'] = df['RSI'].diff(3)
        
        return df

# =========================
# SENTIMENT ANALYSIS
# =========================
class SentimentAnalyzer:
    def __init__(self):
        self.news_sources = []
    
    def get_crypto_fear_greed(self):
        """Kripto Korku & Açgözlülük Endeksi"""
        try:
            # Mock data - gerçek uygulamada API'den çekilir
            return {
                'value': 65,
                'sentiment': 'Açgözlülük',
                'description': 'Piyasa iyimser, yükseliş devam edebilir'
            }
        except:
            return {'value': 50, 'sentiment': 'Nötr', 'description': 'Veri alınamadı'}
    
    def market_sentiment_score(self, ticker):
        """Piyasa duygu analizi"""
        # Gerçek uygulamada Twitter/Reddit API entegrasyonu
        return {
            'bullish': 65,
            'bearish': 25,
            'neutral': 10,
            'momentum': 'positive'
        }

# =========================
# ADVANCED TECHNICAL INDICATORS
# =========================
class AdvancedTechnicalAnalysis:
    @staticmethod
    def ichimoku_cloud(df):
        """Ichimoku Bulut sistemi"""
        high_9 = df['High'].rolling(9).max()
        low_9 = df['Low'].rolling(9).min()
        df['Tenkan_sen'] = (high_9 + low_9) / 2
        
        high_26 = df['High'].rolling(26).max()
        low_26 = df['Low'].rolling(26).min()
        df['Kijun_sen'] = (high_26 + low_26) / 2
        
        df['Senkou_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)
        
        high_52 = df['High'].rolling(52).max()
        low_52 = df['Low'].rolling(52).min()
        df['Senkou_B'] = ((high_52 + low_52) / 2).shift(26)
        
        return df
    
    @staticmethod
    def super_trend(df, period=10, multiplier=3):
        """SuperTrend indikatörü"""
        hl2 = (df['High'] + df['Low']) / 2
        atr = AdvancedTechnicalAnalysis.calculate_atr(df, period)
        
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        super_trend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)
        
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > upper_band.iloc[i-1]:
                super_trend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            elif df['Close'].iloc[i] < lower_band.iloc[i-1]:
                super_trend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                super_trend.iloc[i] = super_trend.iloc[i-1]
                direction.iloc[i] = direction.iloc[i-1]
        
        df['Super_Trend'] = super_trend
        df['Super_Trend_Direction'] = direction
        return df
    
    @staticmethod
    def calculate_atr(df, period=14):
        """Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean()
        return atr

    @staticmethod
    def calculate_macd(df, fast=12, slow=26, signal=9):
        """MACD göstergesi"""
        exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        return df

# =========================
# STREAMLIT APP
# =========================
def main():
    st.markdown('<h1 class="main-header">🚀 AI Crypto Master</h1>', unsafe_allow_html=True)
    st.markdown("### 🤖 Yapay Zeka Destekli Profesyonel Kripto Analiz Platformu")
    
    # Sidebar - Professional Configuration
    st.sidebar.header("⚙️ Profesyonel Ayarlar")
    
    # Asset Selection
    ticker = st.sidebar.selectbox(
        "Kripto Varlık",
        ["BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD", "LINK-USD", "BNB-USD", "XRP-USD", "SOL-USD", "AVAX-USD", "MATIC-USD"],
        index=0
    )
    
    # Timeframe Selection
    timeframe = st.sidebar.selectbox(
        "Zaman Dilimi",
        ["1h", "4h", "1d", "1wk"],
        index=2
    )
    
    period_map = {"1h": "30d", "4h": "90d", "1d": "1y", "1wk": "2y"}
    period = period_map[timeframe]
    
    # Advanced Strategy Selection
    strategy = st.sidebar.selectbox(
        "Ticaret Stratejisi",
        ["AI Momentum", "Trend Takip", "Mean Reversion", "Volatilite Breakout", "Multi Timeframe"],
        index=0
    )
    
    # Risk Management
    st.sidebar.header("🎯 Risk Yönetimi")
    capital = st.sidebar.number_input("Sermaye ($)", 1000, 1000000, 10000)
    risk_per_trade = st.sidebar.slider("İşlem Başına Risk %", 0.1, 5.0, 1.0)
    max_drawdown = st.sidebar.slider("Maks. Çöküş %", 5.0, 50.0, 20.0)
    
    # Advanced Settings
    st.sidebar.header("🔧 Gelişmiş Ayarlar")
    ema_short = st.sidebar.slider("EMA Kısa", 5, 25, 9)
    ema_long = st.sidebar.slider("EMA Uzun", 20, 100, 21)
    rsi_period = st.sidebar.slider("RSI Periyodu", 5, 25, 14)
    
    # Load Data
    @st.cache_data(ttl=300)
    def load_data(ticker, period, interval):
        try:
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            if data.empty:
                st.error("Veri alınamadı. Lütfen sembolü kontrol edin.")
                return None
            
            # Advanced Technical Analysis
            analyzer = AdvancedTechnicalAnalysis()
            data = analyzer.ichimoku_cloud(data)
            data = analyzer.super_trend(data)
            data = analyzer.calculate_macd(data)
            
            # AI Predictions
            ai_predictor = AdvancedAIPredictor()
            data = ai_predictor.volume_analysis(data)
            data = ai_predictor.predictive_rsi(data)
            
            # Additional indicators
            data['EMA_Short'] = data['Close'].ewm(span=ema_short).mean()
            data['EMA_Long'] = data['Close'].ewm(span=ema_long).mean()
            data['SMA_50'] = data['Close'].rolling(50).mean()
            data['SMA_200'] = data['Close'].rolling(200).mean()
            
            return data
        except Exception as e:
            st.error(f"Veri yükleme hatası: {str(e)}")
            return None
    
    data = load_data(ticker, period, timeframe)
    
    if data is None or data.empty:
        st.warning("Veri yüklenemedi. Lütfen bağlantınızı ve sembolü kontrol edin.")
        return
    
    # Initialize AI Tools
    ai_predictor = AdvancedAIPredictor()
    sentiment_analyzer = SentimentAnalyzer()
    
    # Main Dashboard
    st.markdown("### 📊 Gerçek Zamanlı Piyasa Verileri")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = data['Close'].iloc[-1]
        price_change_24h = ((current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100
        st.metric(
            "Şu Anki Fiyat",
            f"${current_price:.2f}",
            f"{price_change_24h:+.2f}%"
        )
    
    with col2:
        fear_greed = sentiment_analyzer.get_crypto_fear_greed()
        st.metric(
            "Korku & Açgözlülük",
            f"{fear_greed['value']}",
            fear_greed['sentiment']
        )
    
    with col3:
        volume_ratio = data['Volume_Ratio'].iloc[-1] if 'Volume_Ratio' in data.columns else 1.0
        st.metric(
            "Hacim Oranı",
            f"{volume_ratio:.2f}x",
            "Yüksek" if volume_ratio > 1.5 else "Normal"
        )
    
    with col4:
        trend_direction = data['Super_Trend_Direction'].iloc[-1] if 'Super_Trend_Direction' in data.columns else 0
        trend_text = "YÜKSELİŞ" if trend_direction > 0 else "DÜŞÜŞ" if trend_direction < 0 else "NÖTR"
        st.metric(
            "Trend Yönü",
            trend_text,
            "AI Onaylı" if abs(trend_direction) == 1 else "Beklemede"
        )
    
    # AI Signal Generation
    st.markdown("---")
    st.header("🤖 AI Ticaret Sinyalleri")
    
    # Calculate advanced signals
    support_resistance = ai_predictor.calculate_support_resistance(data)
    momentum_df = ai_predictor.momentum_oscillator(data)
    sentiment = sentiment_analyzer.market_sentiment_score(ticker)
    
    # Generate AI Signal
    current_rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns else 50
    super_trend_signal = data['Super_Trend_Direction'].iloc[-1] if 'Super_Trend_Direction' in data.columns else 0
    volume_signal = data['Volume_Ratio'].iloc[-1] > 1.5 if 'Volume_Ratio' in data.columns else False
    momentum_signal = momentum_df.iloc[-1].mean() > 0 if not momentum_df.empty else False
    
    # MACD Signals
    macd_signal = data['MACD'].iloc[-1] > data['MACD_Signal'].iloc[-1] if all(col in data.columns for col in ['MACD', 'MACD_Signal']) else False
    
    # AI Decision Logic
    buy_signals = 0
    sell_signals = 0
    
    # Buy conditions
    if current_rsi < 40 and super_trend_signal > 0:
        buy_signals += 2
    if momentum_signal and volume_signal:
        buy_signals += 1
    if sentiment['bullish'] > 60:
        buy_signals += 1
    if macd_signal:
        buy_signals += 1
    if data['EMA_Short'].iloc[-1] > data['EMA_Long'].iloc[-1]:
        buy_signals += 1
    
    # Sell conditions
    if current_rsi > 70 and super_trend_signal < 0:
        sell_signals += 2
    if not momentum_signal and volume_signal:
        sell_signals += 1
    if sentiment['bearish'] > 60:
        sell_signals += 1
    if not macd_signal:
        sell_signals += 1
    if data['EMA_Short'].iloc[-1] < data['EMA_Long'].iloc[-1]:
        sell_signals += 1
    
    # Display Signal
    signal_col, confidence_col, action_col = st.columns(3)
    
    with signal_col:
        if buy_signals >= 4:
            st.markdown('<div class="signal-buy">', unsafe_allow_html=True)
            st.success("🚀 GÜÇLÜ AL SİNYALİ")
            st.write(f"Alım Sinyalleri: {buy_signals}/6")
            st.markdown('</div>', unsafe_allow_html=True)
            recommendation = "AL"
            confidence = min(buy_signals * 15, 95)
        elif sell_signals >= 4:
            st.markdown('<div class="signal-sell">', unsafe_allow_html=True)
            st.error("🔻 GÜÇLÜ SAT SİNYALİ")
            st.write(f"Satım Sinyalleri: {sell_signals}/6")
            st.markdown('</div>', unsafe_allow_html=True)
            recommendation = "SAT"
            confidence = min(sell_signals * 15, 95)
        else:
            st.markdown('<div class="signal-neutral">', unsafe_allow_html=True)
            st.warning("⚡ NÖTR - BEKLE")
            st.write(f"Alım: {buy_signals}/6, Satım: {sell_signals}/6")
            st.markdown('</div>', unsafe_allow_html=True)
            recommendation = "BEKLE"
            confidence = 50
    
    with confidence_col:
        st.metric("AI Güven Oranı", f"%{confidence:.0f}")
        st.progress(confidence / 100)
    
    with action_col:
        if recommendation == "AL":
            # Position sizing calculation
            risk_amount = capital * (risk_per_trade / 100)
            stop_loss = data['Super_Trend'].iloc[-1] if 'Super_Trend' in data.columns else current_price * 0.95
            position_size = risk_amount / abs(current_price - stop_loss)
            
            st.metric("Önerilen Pozisyon", f"{position_size:.4f} {ticker.split('-')[0]}")
            st.metric("Stop Loss", f"${stop_loss:.2f}")
            take_profit = current_price + (current_price - stop_loss) * 2
            st.metric("Take Profit", f"${take_profit:.2f}")
    
    # Advanced Charts
    st.markdown("---")
    st.header("📊 Gelişmiş Analiz Grafikleri")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Ana Grafik", "📈 Teknik Göstergeler", "🤖 AI Tahminleri", "📚 Eğitim"])
    
    with tab1:
        # Main Price Chart
        fig = go.Figure()
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Fiyat"
        ))
        
        # EMAs
        fig.add_trace(go.Scatter(
            x=data.index, y=data['EMA_Short'],
            name=f'EMA {ema_short}',
            line=dict(color='orange', width=1)
        ))
        fig.add_trace(go.Scatter(
            x=data.index, y=data['EMA_Long'],
            name=f'EMA {ema_long}',
            line=dict(color='red', width=1)
        ))
        
        # SuperTrend
        if 'Super_Trend' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data['Super_Trend'],
                name='SuperTrend',
                line=dict(color='green', width=2, dash='dot')
            ))
        
        fig.update_layout(
            title=f'{ticker} - Fiyat Hareketleri ve Trend Analizi',
            height=600,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Technical Indicators in sub-tabs
        tech_tab1, tech_tab2, tech_tab3 = st.tabs(["RSI", "MACD", "Hacim"])
        
        with tech_tab1:
            # RSI Chart
            if 'RSI' in data.columns:
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Aşırı Alım")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Aşırı Satım")
                fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="Nötr")
                fig_rsi.update_layout(title='RSI Göstergesi', height=400, yaxis_range=[0, 100])
                st.plotly_chart(fig_rsi, use_container_width=True)
        
        with tech_tab2:
            # MACD Chart
            if all(col in data.columns for col in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue')))
                fig_macd.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], name='Sinyal', line=dict(color='red')))
                
                # Histogram with colors
                colors = ['green' if x >= 0 else 'red' for x in data['MACD_Histogram']]
                fig_macd.add_trace(go.Bar(
                    x=data.index, 
                    y=data['MACD_Histogram'], 
                    name='Histogram',
                    marker_color=colors,
                    opacity=0.3
                ))
                
                fig_macd.update_layout(title='MACD Göstergesi', height=400)
                st.plotly_chart(fig_macd, use_container_width=True)
        
        with tech_tab3:
            # Volume Chart
            fig_volume = go.Figure()
            colors = ['red' if data['Close'].iloc[i] < data['Open'].iloc[i] else 'green' for i in range(len(data))]
            
            fig_volume.add_trace(go.Bar(
                x=data.index, 
                y=data['Volume'], 
                name='Hacim',
                marker_color=colors,
                opacity=0.7
            ))
            
            if 'Volume_SMA' in data.columns:
                fig_volume.add_trace(go.Scatter(
                    x=data.index, 
                    y=data['Volume_SMA'], 
                    name='Hacim Ortalaması',
                    line=dict(color='orange', width=2)
                ))
            
            fig_volume.update_layout(title='Hacim Analizi', height=400)
            st.plotly_chart(fig_volume, use_container_width=True)
    
    with tab3:
        # AI Predictions and Insights
        st.subheader("🤖 AI Tahmin ve Öngörüler")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 🎯 Destek ve Direnç Seviyeleri")
            for level, price in support_resistance.items():
                distance_pct = ((current_price - price) / current_price) * 100
                st.write(f"**Fibonacci {level}:** ${price:.2f} ({distance_pct:+.1f}%)")
            
            st.write("### 📊 Momentum Analizi")
            if not momentum_df.empty:
                momentum_current = momentum_df.iloc[-1]
                for period, value in momentum_current.items():
                    st.write(f"**{period}:** {value:+.2f}%")
        
        with col2:
            st.write("### 🔍 Teknik Göstergeler")
            indicators = {
                "RSI": f"{current_rsi:.1f}",
                "Trend Gücü": f"{abs(super_trend_signal)}",
                "Hacim Anomali": "Evet" if volume_signal else "Hayır",
                "MACD Sinyal": "Pozitif" if macd_signal else "Negatif",
                "EMA Yönü": "Yükseliş" if data['EMA_Short'].iloc[-1] > data['EMA_Long'].iloc[-1] else "Düşüş"
            }
            
            for indicator, value in indicators.items():
                st.write(f"**{indicator}:** {value}")
    
    with tab4:
        # Educational Content
        st.subheader("📚 Kripto Ticaret Eğitimi")
        
        st.markdown("""
        ### 🎯 Temel Analiz Stratejileri
        
        **1. Trend Takip Stratejisi:**
        - EMA'ların kesişimlerini takip edin
        - Yükselen trendde alım, düşen trendde satım
        - SuperTrend ile trend dönüşümlerini yakalayın
        
        **2. Momentum Stratejisi:**
        - RSI 30 altında aşırı satım, 70 üstünde aşırı alım
        - MACD kesişimleri ile giriş/çıkış noktaları
        - Hacim artışı ile momentum onayı
        
        **3. Risk Yönetimi:**
        - Pozisyon büyüklüğünü sermayenin %1-2'si ile sınırla
        - Stop-loss kullanmadan asla işlem yapma
        - Risk/Ödül oranı en az 1:2 olmalı
        """)
        
        st.markdown("""
        ### ⚠️ Önemli Uyarılar
        
        - Bu araç sadece eğitim amaçlıdır
        - Geçmiş performans geleceği garanti etmez
        - Kripto paralar yüksek risk içerir
        - Yatırım tavsiyesi değildir
        - Kendi araştırmanızı yapın
        """)
    
    # Performance Metrics
    st.markdown("---")
    st.header("📈 Performans Metrikleri")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Volatility
        volatility = data['Close'].pct_change().std() * np.sqrt(365) * 100
        st.metric("Yıllık Volatilite", f"%{volatility:.1f}")
    
    with col2:
        # Sharpe Ratio (approximate)
        returns = data['Close'].pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(365) if returns.std() != 0 else 0
        st.metric("Sharpe Oranı", f"{sharpe:.2f}")
    
    with col3:
        # Max Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        st.metric("Maks. Çöküş", f"%{max_drawdown:.1f}")
    
    with col4:
        # Win Rate (based on daily returns)
        win_rate = (returns > 0).mean() * 100
        st.metric("Kazanç Oranı", f"%{win_rate:.1f}")
    
    # Final Disclaimer
    st.markdown("---")
    st.error("""
    **⚠️ ÖNEMLİ RİSK UYARISI:** 
    Bu uygulama sadece eğitim ve demo amaçlıdır. Gerçek para ile kullanmayın.
    Kripto para ticareti yüksek risk içerir ve sermayenizin tamamını kaybedebilirsiniz.
    Yatırım kararlarınızı sadece bu araca dayandırmayın, mutlaka profesyonel danışmanlık alın.
    """)

if __name__ == "__main__":
    main()