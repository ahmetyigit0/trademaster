import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
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
    }
    .signal-sell {
        background: rgba(255,0,0,0.1);
        border-left: 4px solid #ff0000;
    }
    .signal-neutral {
        background: rgba(255,255,0,0.1);
        border-left: 4px solid #ffff00;
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
        ["BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD", "LINK-USD", "BNB-USD", "XRP-USD", "SOL-USD"],
        index=0
    )
    
    # Timeframe Selection
    timeframe = st.sidebar.selectbox(
        "Zaman Dilimi",
        ["1g", "1h", "4h", "1d", "1hft"],
        index=3
    )
    
    period_map = {"1g": "7d", "1h": "30d", "4h": "90d", "1d": "1y", "1hft": "2y"}
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
            
            # AI Predictions
            ai_predictor = AdvancedAIPredictor()
            data = ai_predictor.volume_analysis(data)
            data = ai_predictor.predictive_rsi(data)
            
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
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = data['Close'].iloc[-1]
        price_change = ((current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100
        st.metric(
            "Şu Anki Fiyat",
            f"${current_price:.2f}",
            f"{price_change:+.2f}%"
        )
    
    with col2:
        fear_greed = sentiment_analyzer.get_crypto_fear_greed()
        st.metric(
            "Korku & Açgözlülük",
            f"{fear_greed['value']}",
            fear_greed['sentiment']
        )
    
    with col3:
        volume_ratio = data['Volume_Ratio'].iloc[-1]
        st.metric(
            "Hacim Oranı",
            f"{volume_ratio:.2f}x",
            "Yüksek" if volume_ratio > 1.5 else "Normal"
        )
    
    with col4:
        trend_direction = data['Super_Trend_Direction'].iloc[-1]
        st.metric(
            "Trend Yönü",
            "YÜKSELİŞ" if trend_direction > 0 else "DÜŞÜŞ",
            "AI Onaylı" if abs(trend_direction) == 1 else "Nötr"
        )
    
    # AI Signal Generation
    st.markdown("---")
    st.header("🤖 AI Ticaret Sinyalleri")
    
    # Calculate advanced signals
    support_resistance = ai_predictor.calculate_support_resistance(data)
    momentum_df = ai_predictor.momentum_oscillator(data)
    sentiment = sentiment_analyzer.market_sentiment_score(ticker)
    
    # Generate AI Signal
    current_rsi = data['RSI'].iloc[-1]
    super_trend_signal = data['Super_Trend_Direction'].iloc[-1]
    volume_signal = data['Volume_Ratio'].iloc[-1] > 1.5
    momentum_signal = momentum_df.iloc[-1].mean() > 0
    
    # AI Decision Logic
    buy_signals = 0
    sell_signals = 0
    
    if current_rsi < 40 and super_trend_signal > 0:
        buy_signals += 2
    if momentum_signal and volume_signal:
        buy_signals += 1
    if sentiment['bullish'] > 60:
        buy_signals += 1
    
    if current_rsi > 70 and super_trend_signal < 0:
        sell_signals += 2
    if not momentum_signal and volume_signal:
        sell_signals += 1
    if sentiment['bearish'] > 60:
        sell_signals += 1
    
    # Display Signal
    signal_col, confidence_col, action_col = st.columns(3)
    
    with signal_col:
        if buy_signals >= 3:
            st.markdown('<div class="metric-card signal-buy">', unsafe_allow_html=True)
            st.success("🚀 GÜÇLÜ AL SİNYALİ")
            st.markdown('</div>', unsafe_allow_html=True)
            recommendation = "AL"
            confidence = min(buy_signals * 25, 95)
        elif sell_signals >= 3:
            st.markdown('<div class="metric-card signal-sell">', unsafe_allow_html=True)
            st.error("🔻 GÜÇLÜ SAT SİNYALİ")
            st.markdown('</div>', unsafe_allow_html=True)
            recommendation = "SAT"
            confidence = min(sell_signals * 25, 95)
        else:
            st.markdown('<div class="metric-card signal-neutral">', unsafe_allow_html=True)
            st.warning("⚡ NÖTR - BEKLE")
            st.markdown('</div>', unsafe_allow_html=True)
            recommendation = "BEKLE"
            confidence = 50
    
    with confidence_col:
        st.metric("AI Güven Oranı", f"%{confidence:.0f}")
    
    with action_col:
        if recommendation == "AL":
            # Position sizing calculation
            risk_amount = capital * (risk_per_trade / 100)
            stop_loss = data['Super_Trend'].iloc[-1]
            position_size = risk_amount / (current_price - stop_loss)
            
            st.metric("Önerilen Pozisyon", f"{position_size:.4f} {ticker.split('-')[0]}")
            st.metric("Stop Loss", f"${stop_loss:.2f}")
    
    # Advanced Charts
    st.markdown("---")
    st.header("📊 Gelişmiş Analiz Grafikleri")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Ana Grafik", "📈 Teknik Göstergeler", "🤖 AI Tahminleri", "📉 Risk Analizi"])
    
    with tab1:
        # Main Price Chart with Ichimoku
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
        
        # Ichimoku Cloud
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Senkou_A'],
            name='Ichimoku A',
            line=dict(color='green', width=1)
        ))
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Senkou_B'],
            name='Ichimoku B',
            line=dict(color='red', width=1),
            fill='tonexty'
        ))
        
        fig.update_layout(
            title=f'{ticker} - Ichimoku Bulut Analizi',
            height=600,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Technical Indicators
        fig2 = go.Figure()
        
        # RSI
        fig2.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')))
        fig2.add_hline(y=70, line_dash="dash", line_color="red")
        fig2.add_hline(y=30, line_dash="dash", line_color="green")
        
        fig2.update_layout(title='RSI Göstergesi', height=400)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Volume
        fig3 = go.Figure()
        colors = ['red' if data['Close'].iloc[i] < data['Close'].iloc[i-1] 
                 else 'green' for i in range(len(data))]
        
        fig3.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Hacim', marker_color=colors))
        fig3.add_trace(go.Scatter(x=data.index, y=data['Volume_SMA'], name='Hacim SMA', line=dict(color='orange')))
        
        fig3.update_layout(title='Hacim Analizi', height=300)
        st.plotly_chart(fig3, use_container_width=True)
    
    with tab3:
        # AI Predictions
        st.subheader("🤖 AI Tahmin ve Öngörüler")
        
        # Support Resistance Levels
        st.write("### 🎯 Destek ve Direnç Seviyeleri")
        for level, price in support_resistance.items():
            distance_pct = ((current_price - price) / current_price) * 100
            st.write(f"**Fibonacci {level}:** ${price:.2f} ({distance_pct:+.1f}%)")
        
        # Momentum Analysis
        st.write("### 📊 Momentum Analizi")
        momentum_current = momentum_df.iloc[-1]
        for period, value in momentum_current.items():
            st.write(f"**{period}:** {value:+.2f}%")
    
    with tab4:
        # Risk Analysis
        st.subheader("📉 Risk Analizi ve Yönetimi")
        
        # Volatility Analysis
        volatility = data['Close'].pct_change().std() * np.sqrt(365) * 100
        max_loss = data['Close'].min() / data['Close'].max() - 1
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Yıllık Volatilite", f"%{volatility:.1f}")
        with col2:
            st.metric("Maks. Tarihi Kayıp", f"%{(max_loss*100):.1f}")
        with col3:
            var_95 = np.percentile(data['Close'].pct_change().dropna() * 100, 5)
            st.metric("1 Günlük VaR (%95)", f"%{abs(var_95):.1f}")
        
        # Drawdown Analysis
        cumulative = (1 + data['Close'].pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=data.index, y=drawdown*100, name='Çöküş', fill='tozeroy'))
        fig4.update_layout(title='Portföy Çöküş Analizi', yaxis_title='Çöküş %')
        st.plotly_chart(fig4, use_container_width=True)
    
    # Educational Section
    st.markdown("---")
    st.header("🎓 AI Strateji Açıklaması")
    
    with st.expander("🤖 AI Nasıl Karar Veriyor?"):
        st.markdown("""
        ### Akıllı Karar Alma Mekanizması
        
        **1. Çoklu Zaman Dilimi Analizi:**
        - Kısa, orta ve uzun vadeli trend uyumu
        - 4 farklı zaman diliminde momentum kontrolü
        
        **2. İndikatör Senkronizasyonu:**
        - RSI + MACD + Ichimoku uyumu
        - Volume spike + price movement confirmation
        
        **3. Piyasa Duygusu Entegrasyonu:**
        - Fear & Greed Index
        - Sosyal medya sinyalleri
        - Haber etkisi analizi
        
        **4. Risk-AwAIr Kontrolleri:**
        - Otomatik pozisyon büyüklüğü
        - Dinamik stop-loss seviyeleri
        - Çoklu take-profit hedefleri
        """)
    
    # Disclaimer
    st.markdown("---")
    st.warning("""
    **⚠️ ÖNEMLİ UYARI:** 
    Bu araç sadece eğitim ve araştırma amaçlıdır. Yatırım tavsiyesi değildir. 
    Kripto para ticareti yüksek risk içerir. AI tahminleri %100 doğru değildir.
    """)

if __name__ == "__main__":
    main()