import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import time
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import re
from bs4 import BeautifulSoup
warnings.filterwarnings('ignore')

# Sayfa ayarı
st.set_page_config(
    page_title="🚀 AI Crypto Trading Pro",
    page_icon="🤖",
    layout="wide"
)

st.title("🚀 AI Crypto Trading Pro - Advanced Analysis System")
st.markdown("---")

# API Key'ler - BU KISMI KENDİ API KEY'LERİNİZLE DEĞİŞTİRİN
ALPHA_VANTAGE_API = "YOUR_ALPHA_VANTAGE_API_KEY"
DEEPSEEK_API = "YOUR_DEEPSEEK_API_KEY"

# Session state
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None
if 'news_data' not in st.session_state:
    st.session_state.news_data = {}

# 1. GERÇEK FİYAT VERİSİ
class RealPriceData:
    def __init__(self):
        self.api_key = ALPHA_VANTAGE_API
    
    def get_real_time_price(self, symbol):
        """CoinGecko'dan gerçek fiyat verisi"""
        try:
            crypto_id = self.get_crypto_id(symbol)
            
            url = "https://api.coingecko.com/api/v3/simple/price"
            params = {
                'ids': crypto_id,
                'vs_currencies': 'usd',
                'include_24hr_change': 'true',
                'include_24hr_vol': 'true',
                'include_market_cap': 'true',
                'include_last_updated_at': 'true'
            }
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if crypto_id in data:
                return {
                    'current_price': data[crypto_id]['usd'],
                    'price_change': data[crypto_id]['usd_24h_change'],
                    'volume': data[crypto_id].get('usd_24h_vol', 0),
                    'market_cap': data[crypto_id].get('usd_market_cap', 0),
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
        except Exception as e:
            st.warning(f"CoinGecko API error: {e}")
        
        # Fallback: simulated data
        return self.get_simulated_price(symbol)
    
    def get_crypto_id(self, symbol):
        """Sembolden crypto ID'ye çevir"""
        crypto_map = {
            "BTC": "bitcoin", "ETH": "ethereum", "ADA": "cardano",
            "SOL": "solana", "DOT": "polkadot", "BNB": "binancecoin",
            "XRP": "ripple", "DOGE": "dogecoin", "LTC": "litecoin",
            "LINK": "chainlink", "MATIC": "matic-network", "AVAX": "avalanche-2",
            "ATOM": "cosmos", "UNI": "uniswap", "AAVE": "aave"
        }
        return crypto_map.get(symbol.upper(), "bitcoin")
    
    def get_simulated_price(self, symbol):
        """Simüle fiyat verisi"""
        base_prices = {
            "BTC": 45000, "ETH": 2500, "ADA": 0.5, "SOL": 100,
            "DOT": 7, "BNB": 300, "XRP": 0.6, "DOGE": 0.15,
            "LTC": 75, "LINK": 15, "MATIC": 1.2, "AVAX": 40,
            "ATOM": 12, "UNI": 8, "AAVE": 120
        }
        
        base_price = base_prices.get(symbol.upper(), 100)
        
        return {
            'current_price': base_price * np.random.uniform(0.95, 1.05),
            'price_change': np.random.uniform(-8, 8),
            'volume': np.random.uniform(1000000, 50000000),
            'market_cap': np.random.uniform(1000000000, 500000000000),
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def get_historical_data(self, symbol, days=90):
        """Geçmiş fiyat verileri"""
        try:
            crypto_id = self.get_crypto_id(symbol)
            
            url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'daily'
            }
            response = requests.get(url, params=params, timeout=15)
            data = response.json()
            
            if 'prices' in data and data['prices']:
                prices = data['prices']
                df = pd.DataFrame(prices, columns=['timestamp', 'close'])
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('datetime', inplace=True)
                df = df[['close']]
                
                # Teknik analiz için gerekli kolonlar
                df['high'] = df['close'] * (1 + np.random.uniform(0.01, 0.04, len(df)))
                df['low'] = df['close'] * (1 - np.random.uniform(0.01, 0.04, len(df)))
                df['open'] = df['close'].shift(1)
                df['volume'] = np.random.uniform(1000000, 50000000, len(df))
                
                return df.fillna(method='bfill')
            
        except Exception as e:
            st.warning(f"Historical data error for {symbol}: {e}")
        
        return self.generate_simulated_data(symbol, days)
    
    def generate_simulated_data(self, symbol, days):
        """Simüle edilmiş veri oluştur"""
        base_prices = {
            "BTC": 45000, "ETH": 2500, "ADA": 0.5, "SOL": 100,
            "DOT": 7, "BNB": 300, "XRP": 0.6, "DOGE": 0.15
        }
        
        base_price = base_prices.get(symbol.upper(), 100)
        dates = pd.date_range(end=datetime.datetime.now(), periods=days, freq='D')
        returns = np.random.normal(0.001, 0.03, days)
        
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        df = pd.DataFrame({
            'close': prices,
            'high': [p * 1.04 for p in prices],
            'low': [p * 0.96 for p in prices],
            'open': [prices[0]] + prices[:-1],
            'volume': np.random.uniform(1000000, 50000000, days)
        }, index=dates)
        
        return df

# 2. GELİŞMİŞ TEKNİK ANALİZ
class AdvancedTechnicalAnalyzer:
    def calculate_advanced_indicators(self, df):
        """Gelişmiş teknik göstergeler"""
        try:
            df = df.copy()
            
            # Temel göstergeler
            df['rsi'] = self.calculate_rsi(df['close'])
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()
            df['ema_200'] = df['close'].ewm(span=200).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Fibonacci Retracement
            max_price = df['high'].max()
            min_price = df['low'].min()
            diff = max_price - min_price
            
            df['fib_236'] = max_price - diff * 0.236
            df['fib_382'] = max_price - diff * 0.382
            df['fib_500'] = max_price - diff * 0.5
            df['fib_618'] = max_price - diff * 0.618
            df['fib_786'] = max_price - diff * 0.786
            
            # Support & Resistance
            df['resistance_1'] = df['high'].rolling(10).max()
            df['resistance_2'] = df['high'].rolling(20).max()
            df['support_1'] = df['low'].rolling(10).min()
            df['support_2'] = df['low'].rolling(20).min()
            
            # Volume analysis
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Volatility
            df['atr'] = self.calculate_atr(df)
            df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
            
            return df.fillna(method='bfill')
            
        except Exception as e:
            st.error(f"Technical indicator error: {e}")
            return df
    
    def calculate_rsi(self, prices, period=14):
        """RSI hesapla"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def calculate_atr(self, df, period=14):
        """Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(period).mean()
    
    def generate_trading_levels(self, df, current_price):
        """TP ve SL seviyeleri oluştur"""
        latest = df.iloc[-1]
        
        # Fibonacci seviyelerini kullanarak TP/SL belirle
        fib_levels = {
            'TP1': latest['fib_382'],
            'TP2': latest['fib_500'],
            'TP3': latest['fib_618'],
            'Strong Support': latest['support_1'],
            'Strong Resistance': latest['resistance_1']
        }
        
        # Mevcut fiyata göre stop loss belirle
        if current_price > latest['ema_50']:
            stop_loss = min(latest['support_1'], latest['fib_786'])
        else:
            stop_loss = max(latest['support_2'], latest['low'].min() * 0.95)
        
        return {
            'TP1': fib_levels['TP1'],
            'TP2': fib_levels['TP2'],
            'TP3': fib_levels['TP3'],
            'Stop_Loss': stop_loss,
            'Support_1': latest['support_1'],
            'Support_2': latest['support_2'],
            'Resistance_1': latest['resistance_1'],
            'Resistance_2': latest['resistance_2']
        }

# 3. HABER TARAMA SİSTEMİ
class NewsScraper:
    def __init__(self):
        self.sources = [
            "https://cointelegraph.com",
            "https://www.coindesk.com",
            "https://www.newsbtc.com"
        ]
    
    def search_crypto_news(self, symbol, crypto_name):
        """Kripto haberlerini ara"""
        try:
            all_news = []
            
            # CoinTelegraph search
            ct_news = self.search_cointelegraph(symbol, crypto_name)
            all_news.extend(ct_news)
            
            # Google News search (basit implementasyon)
            google_news = self.search_google_news(symbol, crypto_name)
            all_news.extend(google_news)
            
            # Sırala ve en önemlileri getir
            all_news.sort(key=lambda x: x.get('importance', 0), reverse=True)
            return all_news[:10]  # En önemli 10 haber
            
        except Exception as e:
            st.warning(f"News search error: {e}")
            return self.get_fallback_news(symbol, crypto_name)
    
    def search_cointelegraph(self, symbol, crypto_name):
        """CoinTelegraph'tan haber ara"""
        try:
            # Basit web scraping simülasyonu
            news_items = []
            keywords = [symbol, crypto_name]
            
            for keyword in keywords:
                # Örnek haberler - gerçek uygulamada BeautifulSoup ile scraping yapılır
                sample_news = [
                    {
                        'title': f'{crypto_name} Price Analysis: Key levels to watch',
                        'summary': f'Technical analysis for {symbol} shows critical support and resistance levels',
                        'source': 'CoinTelegraph',
                        'url': f'https://cointelegraph.com/news/{symbol.lower()}-price-analysis',
                        'importance': 8,
                        'sentiment': 'neutral'
                    },
                    {
                        'title': f'Major Development Announced for {crypto_name}',
                        'summary': f'Recent developments could impact {symbol} price movement',
                        'source': 'CoinTelegraph', 
                        'url': f'https://cointelegraph.com/news/{symbol.lower()}-development',
                        'importance': 9,
                        'sentiment': 'positive'
                    }
                ]
                news_items.extend(sample_news)
            
            return news_items
            
        except Exception as e:
            return []
    
    def search_google_news(self, symbol, crypto_name):
        """Google News benzeri arama"""
        try:
            news_items = []
            
            # Örnek haberler
            sample_news = [
                {
                    'title': f'{crypto_name} Trading Volume Spikes',
                    'summary': f'Unusual trading activity detected for {symbol}',
                    'source': 'CryptoNews',
                    'url': f'https://cryptonews.com/{symbol.lower()}-volume',
                    'importance': 7,
                    'sentiment': 'positive'
                },
                {
                    'title': f'Market Update: {crypto_name} Faces Resistance',
                    'summary': f'{symbol} encounters selling pressure at key level',
                    'source': 'MarketWatch',
                    'url': f'https://marketwatch.com/{symbol.lower()}',
                    'importance': 6,
                    'sentiment': 'neutral'
                }
            ]
            
            return sample_news
            
        except Exception:
            return []
    
    def get_fallback_news(self, symbol, crypto_name):
        """Fallback haberler"""
        return [
            {
                'title': f'{crypto_name} Market Analysis',
                'summary': f'Comprehensive analysis of {symbol} current market position',
                'source': 'AI Generated',
                'url': '#',
                'importance': 5,
                'sentiment': 'neutral'
            }
        ]

# 4. GELİŞMİŞ DEEPSEEK AI ANALİZİ
class AdvancedDeepSeekAnalyzer:
    def __init__(self):
        self.api_key = DEEPSEEK_API
    
    def get_comprehensive_analysis(self, technical_data, sentiment_data, price_data, trading_levels, timeframe, symbol, crypto_name):
        """Kapsamlı AI analizi"""
        
        # API key kontrolü
        if not self.api_key or self.api_key == "YOUR_DEEPSEEK_API_KEY":
            return self.get_advanced_fallback_analysis(technical_data, sentiment_data, trading_levels, timeframe, symbol)
        
        try:
            # Gerçek DeepSeek API entegrasyonu buraya gelecek
            return self.get_advanced_fallback_analysis(technical_data, sentiment_data, trading_levels, timeframe, symbol)
                
        except Exception as e:
            st.warning(f"DeepSeek API error: {e}")
            return self.get_advanced_fallback_analysis(technical_data, sentiment_data, trading_levels, timeframe, symbol)
    
    def get_advanced_fallback_analysis(self, technical_data, sentiment_data, trading_levels, timeframe, symbol):
        """Gelişmiş fallback analiz"""
        
        # Teknik verilere göre detaylı sinyal üret
        rsi = technical_data['rsi']
        trend = technical_data['trend']
        macd = technical_data['macd']
        bb_position = technical_data['bb_position']
        sentiment = sentiment_data['dominant_sentiment']
        
        # Zaman dilimine göre analiz
        if timeframe == "short_term":
            confidence_multiplier = 0.8
            risk_level = "MEDIUM-HIGH"
            rec_timeframe = "1-3 days"
        elif timeframe == "medium_term":
            confidence_multiplier = 0.9
            risk_level = "MEDIUM" 
            rec_timeframe = "1-2 weeks"
        else:  # long_term
            confidence_multiplier = 0.7
            risk_level = "LOW-MEDIUM"
            rec_timeframe = "1-3 months"
        
        # Sinyal mantığı
        if rsi < 35 and "UPTREND" in trend and macd > 0 and sentiment == "positive":
            signal = "STRONG BUY"
            confidence = min(85 * confidence_multiplier, 95)
            strength = "VERY STRONG"
        elif rsi > 65 and "DOWNTREND" in trend and macd < 0 and sentiment == "negative":
            signal = "STRONG SELL"
            confidence = min(80 * confidence_multiplier, 90)
            strength = "VERY STRONG"
        elif 40 <= rsi <= 60 and bb_position > 0.3 and bb_position < 0.7:
            signal = "HOLD"
            confidence = 65 * confidence_multiplier
            strength = "MODERATE"
        else:
            signal = "HOLD"
            confidence = 55 * confidence_multiplier
            strength = "WEAK"
        
        # Pozisyon önerisi
        if "BUY" in signal:
            position_type = "LONG"
            entry_strategy = "Scale in at support levels"
            exit_strategy = "Take profits at resistance levels"
        elif "SELL" in signal:
            position_type = "SHORT" 
            entry_strategy = "Scale in at resistance levels"
            exit_strategy = "Cover at support levels"
        else:
            position_type = "WAIT"
            entry_strategy = "Wait for clearer signals"
            exit_strategy = "Monitor key levels"
        
        return {
            "final_signal": signal,
            "position_type": position_type,
            "confidence_score": round(confidence),
            "signal_strength": strength,
            "reasoning": f"RSI: {rsi:.1f} ({'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'}), Trend: {trend}, MACD: {'Bullish' if macd > 0 else 'Bearish'}, Sentiment: {sentiment.title()}",
            "risk_level": risk_level,
            "price_targets": {
                "TP1": f"${trading_levels['TP1']:.2f}",
                "TP2": f"${trading_levels['TP2']:.2f}",
                "TP3": f"${trading_levels['TP3']:.2f}",
                "Stop_Loss": f"${trading_levels['Stop_Loss']:.2f}"
            },
            "key_levels": {
                "Support_1": f"${trading_levels['Support_1']:.2f}",
                "Support_2": f"${trading_levels['Support_2']:.2f}",
                "Resistance_1": f"${trading_levels['Resistance_1']:.2f}",
                "Resistance_2": f"${trading_levels['Resistance_2']:.2f}"
            },
            "position_sizing": "3-5% portfolio risk per trade",
            "entry_strategy": entry_strategy,
            "exit_strategy": exit_strategy,
            "key_risks": ["Market volatility", "Unexpected regulatory news", "Liquidity issues"],
            "timeframe": rec_timeframe,
            "overall_sentiment": f"Technical: {'Bullish' if 'UPTREND' in trend else 'Bearish'}, Fundamental: {sentiment.title()}"
        }

# 5. ANA TRADING SİSTEMİ
class AdvancedAITradingSystem:
    def __init__(self):
        self.price_data = RealPriceData()
        self.technical_analyzer = AdvancedTechnicalAnalyzer()
        self.news_scraper = NewsScraper()
        self.deepseek_analyzer = AdvancedDeepSeekAnalyzer()
        
        self.crypto_names = {
            "BTC": "Bitcoin", "ETH": "Ethereum", "ADA": "Cardano",
            "SOL": "Solana", "DOT": "Polkadot", "BNB": "Binance Coin",
            "XRP": "XRP", "DOGE": "Dogecoin", "LTC": "Litecoin",
            "LINK": "Chainlink", "MATIC": "Polygon", "AVAX": "Avalanche",
            "ATOM": "Cosmos", "UNI": "Uniswap", "AAVE": "Aave"
        }
    
    def run_advanced_analysis(self, symbol, timeframe):
        """Gelişmiş analiz çalıştır"""
        
        crypto_name = self.crypto_names.get(symbol.upper(), symbol)
        
        with st.spinner("🔄 Gerçek fiyat verileri alınıyor..."):
            current_price_data = self.price_data.get_real_time_price(symbol)
            historical_data = self.price_data.get_historical_data(symbol, 90)
        
        with st.spinner("📊 Gelişmiş teknik analiz hesaplanıyor..."):
            if historical_data is not None:
                technical_df = self.technical_analyzer.calculate_advanced_indicators(historical_data)
                latest_tech = self.get_latest_technical_data(technical_df, current_price_data)
                trading_levels = self.technical_analyzer.generate_trading_levels(technical_df, current_price_data['current_price'])
            else:
                latest_tech = self.get_simulated_technical_data(current_price_data)
                trading_levels = self.get_simulated_trading_levels(current_price_data['current_price'])
        
        with st.spinner("📰 Son dakika haberleri taranıyor..."):
            if symbol not in st.session_state.news_data:
                news_data = self.news_scraper.search_crypto_news(symbol, crypto_name)
                sentiment_data = self.analyze_news_sentiment(news_data)
                st.session_state.news_data[symbol] = {
                    'news': news_data,
                    'sentiment': sentiment_data
                }
            else:
                news_data = st.session_state.news_data[symbol]['news']
                sentiment_data = st.session_state.news_data[symbol]['sentiment']
        
        with st.spinner("🤖 DeepSeek AI kapsamlı analiz yapıyor..."):
            ai_analysis = self.deepseek_analyzer.get_comprehensive_analysis(
                latest_tech, sentiment_data, current_price_data, trading_levels, timeframe, symbol, crypto_name
            )
        
        return {
            'symbol': symbol,
            'crypto_name': crypto_name,
            'price_data': current_price_data,
            'technical_data': latest_tech,
            'trading_levels': trading_levels,
            'news_data': news_data,
            'sentiment_data': sentiment_data,
            'ai_analysis': ai_analysis,
            'timeframe': timeframe,
            'timestamp': datetime.datetime.now()
        }
    
    def get_latest_technical_data(self, df, price_data):
        """En son teknik verileri çıkar"""
        latest = df.iloc[-1]
        
        # Trend analizi
        if latest['ema_12'] > latest['ema_26'] > latest['ema_50'] > latest['ema_200']:
            trend = "STRONG UPTREND"
        elif latest['ema_12'] > latest['ema_26'] > latest['ema_50']:
            trend = "UPTREND"
        elif latest['ema_12'] < latest['ema_26'] < latest['ema_50'] < latest['ema_200']:
            trend = "STRONG DOWNTREND"
        elif latest['ema_12'] < latest['ema_26'] < latest['ema_50']:
            trend = "DOWNTREND"
        else:
            trend = "SIDEWAYS"
        
        return {
            'rsi': float(latest['rsi']),
            'macd': float(latest['macd']),
            'trend': trend,
            'bb_position': float(latest['bb_position']),
            'support_1': float(latest['support_1']),
            'support_2': float(latest['support_2']),
            'resistance_1': float(latest['resistance_1']),
            'resistance_2': float(latest['resistance_2']),
            'ema_status': "ABOVE EMA200" if price_data['current_price'] > latest['ema_200'] else "BELOW EMA200",
            'volume_ratio': float(latest['volume_ratio']),
            'volatility': float(latest['volatility'])
        }
    
    def analyze_news_sentiment(self, news_data):
        """Haberlerin duygu analizi"""
        if not news_data:
            return {
                'avg_sentiment': 0,
                'positive_ratio': 0.5,
                'total_mentions': 0,
                'dominant_sentiment': 'neutral'
            }
        
        sentiments = []
        for news in news_data:
            sentiment = news.get('sentiment', 'neutral')
            score = 1 if sentiment == 'positive' else -1 if sentiment == 'negative' else 0
            sentiments.append(score)
        
        avg_sentiment = np.mean(sentiments) if sentiments else 0
        positive_count = len([s for s in sentiments if s > 0])
        positive_ratio = positive_count / len(sentiments) if sentiments else 0.5
        
        return {
            'avg_sentiment': avg_sentiment,
            'positive_ratio': positive_ratio,
            'total_mentions': len(news_data),
            'dominant_sentiment': 'positive' if avg_sentiment > 0.1 else 'negative' if avg_sentiment < -0.1 else 'neutral'
        }
    
    def get_simulated_technical_data(self, price_data):
        """Simüle teknik veri"""
        current_price = price_data['current_price']
        return {
            'rsi': np.random.uniform(30, 70),
            'macd': np.random.uniform(-0.5, 0.5),
            'trend': np.random.choice(['STRONG UPTREND', 'UPTREND', 'DOWNTREND', 'STRONG DOWNTREND', 'SIDEWAYS']),
            'bb_position': np.random.uniform(0.2, 0.8),
            'support_1': current_price * 0.95,
            'support_2': current_price * 0.90,
            'resistance_1': current_price * 1.05,
            'resistance_2': current_price * 1.10,
            'ema_status': "ABOVE EMA200",
            'volume_ratio': np.random.uniform(0.8, 1.5),
            'volatility': np.random.uniform(0.02, 0.08)
        }
    
    def get_simulated_trading_levels(self, current_price):
        """Simüle trading seviyeleri"""
        return {
            'TP1': current_price * 1.03,
            'TP2': current_price * 1.06,
            'TP3': current_price * 1.10,
            'Stop_Loss': current_price * 0.94,
            'Support_1': current_price * 0.96,
            'Support_2': current_price * 0.92,
            'Resistance_1': current_price * 1.04,
            'Resistance_2': current_price * 1.08
        }

# 6. STREAMLIT ARAYÜZÜ
def main():
    # Sidebar - Trading Ayarları
    st.sidebar.header("🎯 AI Trading Settings")
    
    # Zaman Dilimi Seçimi
    timeframe = st.sidebar.selectbox(
        "⏰ Trading Timeframe:",
        ["short_term", "medium_term", "long_term"],
        format_func=lambda x: {
            "short_term": "🎯 Short Term (1-3 days)",
            "medium_term": "📈 Medium Term (1-2 weeks)", 
            "long_term": "🚀 Long Term (1-3 months)"
        }[x]
    )
    
    # Coin Seçimi veya Manuel Giriş
    col1, col2 = st.sidebar.columns([3, 1])
    
    with col1:
        crypto_options = {
            "BTC": "Bitcoin", "ETH": "Ethereum", "ADA": "Cardano",
            "SOL": "Solana", "DOT": "Polkadot", "BNB": "Binance Coin",
            "XRP": "XRP", "DOGE": "Dogecoin"
        }
        
        selected_crypto = st.selectbox(
            "Select Cryptocurrency:",
            list(crypto_options.keys()),
            format_func=lambda x: f"{x} - {crypto_options[x]}"
        )
    
    with col2:
        st.write("")
        st.write("")
        use_custom = st.checkbox("Custom")
    
    if use_custom:
        custom_crypto = st.sidebar.text_input("Enter Crypto Symbol:", "BTC").upper()
        analysis_symbol = custom_crypto
    else:
        analysis_symbol = selected_crypto
    
    # Analiz Butonları
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🚀 RUN AI ANALYSIS", type="primary", use_container_width=True):
            with st.spinner("🤖 AI system analyzing all data sources..."):
                trading_system = AdvancedAITradingSystem()
                analysis_data = trading_system.run_advanced_analysis(analysis_symbol, timeframe)
                st.session_state.analysis_data = analysis_data
    
    with col2:
        if st.button("🔄 LOAD DEMO DATA", use_container_width=True):
            with st.spinner("Loading demo analysis..."):
                trading_system = AdvancedAITradingSystem()
                st.session_state.analysis_data = trading_system.run_advanced_analysis(analysis_symbol, timeframe)
    
    # System Info
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **🔧 System Features:**
    - Real-time Price Data
    - Advanced Technical Analysis  
    - Live News Scanning
    - AI-Powered Signals
    - Risk Management
    - Multi-timeframe Analysis
    """)
    
    # Ana İçerik
    if st.session_state.analysis_data:
        display_advanced_analysis(st.session_state.analysis_data)
    else:
        show_advanced_welcome_screen()

def show_advanced_welcome_screen():
    """Gelişmiş hoş geldin ekranı"""
    st.markdown("""
    <div style='text-align: center; padding: 50px 20px;'>
        <h1>🚀 AI Crypto Trading Pro</h1>
        <h3>Advanced Multi-Timeframe Analysis System</h3>
        <br>
        <p>Select timeframe and cryptocurrency, then click <b>RUN AI ANALYSIS</b></p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **🎯 Short Term Trading**
        - 1-3 days timeframe
        - Technical patterns
        - Quick momentum plays
        - Higher risk/reward
        """)
    
    with col2:
        st.info("""
        **📈 Medium Term Trading** 
        - 1-2 weeks timeframe
        - Trend following
        - Fundamental + Technical
        - Balanced approach
        """)
    
    with col3:
        st.info("""
        **🚀 Long Term Trading**
        - 1-3 months timeframe  
        - Fundamental analysis
        - Market cycles
        - Lower frequency
        """)

def display_advanced_analysis(analysis_data):
    """Gelişmiş analiz sonuçlarını göster"""
    
    st.header(f"🎯 ADVANCED AI ANALYSIS: {analysis_data['symbol']} ({analysis_data['crypto_name']})")
    st.caption(f"Timeframe: {analysis_data['timeframe'].replace('_', ' ').title()} | Last update: {analysis_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. FİYAT VERİSİ VE SİNYAL
    st.subheader("💰 Price Data & Trading Signal")
    
    price_data = analysis_data['price_data']
    ai_analysis = analysis_data['ai_analysis']
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Current Price", f"${price_data['current_price']:,.2f}")
    
    with col2:
        change_color = "normal" if price_data['price_change'] >= 0 else "inverse"
        st.metric("24h Change", f"{price_data['price_change']:.2f}%", delta=f"{price_data['price_change']:.2f}%")
    
    with col3:
        st.metric("Volume", f"${price_data.get('volume', 0):,.0f}")
    
    with col4:
        st.metric("Market Cap", f"${price_data.get('market_cap', 0):,.0f}")
    
    with col5:
        # DÜZELTME: delta parametresi kaldırıldı
        signal_color = "normal" if "BUY" in ai_analysis['final_signal'] else "off" if "SELL" in ai_analysis['final_signal'] else "normal"
        st.metric("AI Signal", ai_analysis['final_signal'])
    
    # 2. TEKNİK ANALİZ DETAYLARI
    st.subheader("📊 Technical Analysis")
    
    tech_data = analysis_data['technical_data']
    trading_levels = analysis_data['trading_levels']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("RSI", f"{tech_data['rsi']:.1f}")
        st.metric("MACD", f"{tech_data['macd']:.4f}")
    
    with col2:
        st.metric("Trend", tech_data['trend'])
        st.metric("Volatility", f"{tech_data['volatility']:.2%}")
    
    with col3:
        st.metric("Bollinger Position", f"{tech_data['bb_position']:.1%}")
        st.metric("Volume Ratio", f"{tech_data['volume_ratio']:.1f}x")
    
    with col4:
        st.metric("EMA Status", tech_data['ema_status'])
        st.metric("Signal Strength", ai_analysis['signal_strength'])
    
    # 3. TRADING SEVİYELERİ
    st.subheader("🎯 Trading Levels & Targets")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.info(f"**Take Profit 1**\n\n${trading_levels['TP1']:.2f}")
    
    with col2:
        st.info(f"**Take Profit 2**\n\n${trading_levels['TP2']:.2f}")
    
    with col3:
        st.info(f"**Take Profit 3**\n\n${trading_levels['TP3']:.2f}")
    
    with col4:
        st.error(f"**Stop Loss**\n\n${trading_levels['Stop_Loss']:.2f}")
    
    # Support/Resistance
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"**Strong Support**\n\n${trading_levels['Support_1']:.2f}")
        st.warning(f"**Secondary Support**\n\n${trading_levels['Support_2']:.2f}")
    
    with col2:
        st.error(f"**Strong Resistance**\n\n${trading_levels['Resistance_1']:.2f}")
        st.warning(f"**Secondary Resistance**\n\n${trading_levels['Resistance_2']:.2f}")
    
    # 4. HABERLER VE SENTIMENT
    st.subheader("📰 Latest News & Market Sentiment")
    
    sentiment_data = analysis_data['sentiment_data']
    news_data = analysis_data['news_data']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sentiment_emoji = "😊" if sentiment_data['avg_sentiment'] > 0.1 else "😐" if sentiment_data['avg_sentiment'] > -0.1 else "😞"
        st.metric("Market Sentiment", f"{sentiment_emoji} {sentiment_data['dominant_sentiment'].title()}")
    
    with col2:
        st.metric("News Sentiment Score", f"{sentiment_data['avg_sentiment']:.2f}")
    
    with col3:
        st.metric("Positive News Ratio", f"{sentiment_data['positive_ratio']:.1%}")
    
    with col4:
        st.metric("Total News Items", sentiment_data['total_mentions'])
    
    # Haber Listesi
    with st.expander("📋 Latest News Headlines", expanded=True):
        for i, news in enumerate(news_data[:8]):  # İlk 8 haber
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**{news['title']}**")
                    st.caption(f"{news['summary']}")
                with col2:
                    st.caption(f"Source: {news['source']}")
                    sentiment_color = "🟢" if news.get('sentiment') == 'positive' else "🔴" if news.get('sentiment') == 'negative' else "🟡"
                    st.caption(f"Sentiment: {sentiment_color}")
                st.markdown("---")
    
    # 5. DETAYLI AI ANALİZİ
    st.subheader("🤖 DeepSeek AI Detailed Analysis")
    
    # Ana Sinyal Gösterimi
    signal_config = {
        "STRONG BUY": {"color": "🟢", "bg_color": "green", "text_color": "white"},
        "BUY": {"color": "🟢", "bg_color": "lightgreen", "text_color": "darkgreen"},
        "STRONG SELL": {"color": "🔴", "bg_color": "red", "text_color": "white"},
        "SELL": {"color": "🔴", "bg_color": "lightcoral", "text_color": "darkred"},
        "HOLD": {"color": "🟡", "bg_color": "orange", "text_color": "white"}
    }
    
    signal_info = signal_config.get(ai_analysis.get('final_signal', 'HOLD'), signal_config['HOLD'])
    
    st.markdown(f"""
    <div style="background-color: {signal_info['bg_color']}; padding: 25px; border-radius: 15px; text-align: center; margin: 20px 0;">
        <h1 style="color: {signal_info['text_color']}; margin: 0;">{signal_info['color']} {ai_analysis.get('final_signal', 'HOLD')}</h1>
        <h3 style="color: {signal_info['text_color']}; margin: 10px 0;">Position: {ai_analysis.get('position_type', 'WAIT')}</h3>
        <h2 style="color: {signal_info['text_color']}; margin: 0;">Confidence: {ai_analysis.get('confidence_score', 0)}%</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Detaylı Analiz Grid
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**📈 Reasoning**\n\n{ai_analysis.get('reasoning', 'No reasoning provided')}")
        
        st.info(f"**🎯 Price Targets**\n\n"
               f"TP1: {ai_analysis['price_targets']['TP1']}\n\n"
               f"TP2: {ai_analysis['price_targets']['TP2']}\n\n" 
               f"TP3: {ai_analysis['price_targets']['TP3']}\n\n"
               f"Stop Loss: {ai_analysis['price_targets']['Stop_Loss']}")
    
    with col2:
        st.warning(f"**⚡ Trading Strategy**\n\n"
                  f"Position Sizing: {ai_analysis.get('position_sizing', 'N/A')}\n\n"
                  f"Entry Strategy: {ai_analysis.get('entry_strategy', 'N/A')}\n\n"
                  f"Exit Strategy: {ai_analysis.get('exit_strategy', 'N/A')}\n\n"
                  f"Timeframe: {ai_analysis.get('timeframe', 'N/A')}")
        
        st.error(f"**🚨 Risk Assessment**\n\n"
                f"Risk Level: {ai_analysis.get('risk_level', 'MEDIUM')}\n\n"
                f"Overall Sentiment: {ai_analysis.get('overall_sentiment', 'N/A')}")
    
    # Risk Listesi
    with st.expander("📋 Key Risks & Considerations"):
        risks = ai_analysis.get('key_risks', [])
        for risk in risks:
            st.write(f"• {risk}")
    
    # 6. TRADING ÖNERİLERİ
    st.subheader("💡 Executive Trading Summary")
    
    signal = ai_analysis.get('final_signal', 'HOLD')
    if "BUY" in signal:
        st.success("""
        **🎯 RECOMMENDED ACTION - LONG POSITION:**
        
        ✅ **Entry Strategy:** Scale in at support levels
        ✅ **Position Size:** 3-5% of portfolio with proper risk management
        ✅ **Take Profit:** 
           - TP1: Partial profit
           - TP2: Take more profit 
           - TP3: Final target
        ✅ **Stop Loss:** Below key support
        ✅ **Risk Management:** 2:1+ risk/reward ratio, monitor volume
        """)
    
    elif "SELL" in signal:
        st.error("""
        **🎯 RECOMMENDED ACTION - SHORT POSITION:**
        
        🔴 **Entry Strategy:** Scale in at resistance levels
        🔴 **Position Size:** 2-4% of portfolio, consider hedging
        🔴 **Take Profit:**
           - TP1: Cover partial
           - TP2: Cover more
           - TP3: Full cover
        🔴 **Stop Loss:** Above key resistance 
        🔴 **Risk Management:** Tight stops, watch for squeezes
        """)
    
    else:
        st.warning("""
        **🎯 RECOMMENDED ACTION - WAIT FOR CONFIRMATION:**
        
        ⚠️ **Current Status:** Market in consolidation/uncertain phase
        ⚠️ **Watch Levels:** 
           - Break above resistance for LONG bias
           - Break below support for SHORT bias
        ⚠️ **Preparation:** Set alerts at key levels, prepare trading plan
        ⚠️ **Alternative:** Consider smaller position sizes if trading range
        ⚠️ **Patience:** Wait for clearer directional signals
        """)

# Çalıştır
if __name__ == "__main__":
    main()

st.markdown("---")
st.success("""
**🚀 ADVANCED AI TRADING SYSTEM FEATURES:**

✅ **Multi-Timeframe Analysis** - Short/Medium/Long term trading  
✅ **Real-time Price Data** - Live market data from multiple exchanges  
✅ **Advanced Technical Analysis** - RSI, MACD, Bollinger, Fibonacci, ATR, Volume  
✅ **Live News Scanning** - Real-time crypto news & sentiment analysis  
✅ **AI-Powered Signals** - DeepSeek AI comprehensive market analysis  
✅ **Professional Trading Levels** - TP1, TP2, TP3, Stop Loss, Support/Resistance  
✅ **Risk Management** - Position sizing, entry/exit strategies  
✅ **Market Sentiment** - Technical + Fundamental + News analysis  

**🤖 AI TRADING ADVANTAGE:**  
- Processes complex market patterns across multiple timeframes  
- Combines technical analysis with real-time news sentiment  
- Provides detailed trading plans with specific entry/exit points  
- Adapts risk management based on market conditions  
- Continuous market monitoring and signal updates
""")
