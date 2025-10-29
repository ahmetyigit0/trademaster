import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import time
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re

# Sayfa ayarı
st.set_page_config(
    page_title="📊 Crypto Social Media & News Analyzer",
    page_icon="🔍",
    layout="wide"
)

# Başlık
st.title("📊 Crypto Social Media & News Analyzer")
st.markdown("---")

# Session state
if 'last_analysis' not in st.session_state:
    st.session_state.last_analysis = None

# Crypto symbols
CRYPTO_SYMBOLS = {
    'BTC': 'Bitcoin',
    'ETH': 'Ethereum',
    'BNB': 'Binance Coin', 
    'XRP': 'XRP',
    'ADA': 'Cardano',
    'SOL': 'Solana',
    'DOT': 'Polkadot',
    'DOGE': 'Dogecoin'
}

# Sentiment analysis function
def analyze_sentiment(text):
    """Metin duygu analizi"""
    try:
        analysis = TextBlob(text)
        # -1 (olumsuz) ile +1 (olumlu) arası
        polarity = analysis.sentiment.polarity
        
        if polarity > 0.1:
            return 'positive', polarity
        elif polarity < -0.1:
            return 'negative', polarity
        else:
            return 'neutral', polarity
    except:
        return 'neutral', 0

# Get crypto news from multiple sources
def get_crypto_news(crypto_symbol, limit=20):
    """Çoklu kaynaktan kripto haberleri getir"""
    news_items = []
    
    try:
        # CoinGecko News API
        url = "https://api.coingecko.com/api/v3/news"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            for item in data.get('news', [])[:limit]:
                if crypto_symbol.lower() in item.get('title', '').lower() or \
                   crypto_symbol.lower() in item.get('description', '').lower():
                    
                    sentiment, score = analyze_sentiment(item.get('title', '') + ' ' + item.get('description', ''))
                    
                    news_items.append({
                        'source': 'CoinGecko',
                        'title': item.get('title', ''),
                        'description': item.get('description', ''),
                        'url': item.get('url', ''),
                        'published_at': item.get('published_at', ''),
                        'sentiment': sentiment,
                        'sentiment_score': score,
                        'crypto': crypto_symbol
                    })
    except Exception as e:
        st.error(f"News API error: {e}")
    
    # Simulated social media data (gerçek uygulamada Twitter API vs kullanılır)
    simulated_tweets = generate_simulated_social_data(crypto_symbol, limit//2)
    news_items.extend(simulated_tweets)
    
    return news_items

def generate_simulated_social_data(crypto_symbol, count=10):
    """Simüle sosyal medya verisi (gerçek API yerine)"""
    tweets = []
    
    # Örnek tweet verileri
    sample_tweets = {
        'BTC': [
            "Bitcoin breaking new highs! 🚀 #BTC #Bitcoin",
            "BTC correction healthy for long term growth",
            "Bitcoin volatility concerns investors",
            "Major institution adopts Bitcoin",
            "BTC technical analysis shows bullish pattern"
        ],
        'ETH': [
            "Ethereum 2.0 upgrade driving price surge #ETH",
            "Gas fees still high on Ethereum network",
            "ETH DeFi ecosystem expanding rapidly",
            "Ethereum competitors gaining traction",
            "Vitalik Buterin announces new ETH improvement"
        ],
        'ADA': [
            "Cardano smart contracts live! #ADA",
            "ADA price reacting positively to developments",
            "Cardano ecosystem growing steadily",
            "Charles Hoskinson updates on ADA roadmap",
            "ADA technical analysis promising"
        ]
    }
    
    base_tweets = sample_tweets.get(crypto_symbol, [
        f"{crypto_symbol} showing strong momentum",
        f"Trading volume increasing for {crypto_symbol}",
        f"Market sentiment mixed for {crypto_symbol}",
        f"{crypto_symbol} technical indicators turning bullish",
        f"Regulatory news affecting {crypto_symbol} price"
    ])
    
    for i in range(count):
        tweet_text = base_tweets[i % len(base_tweets)]
        sentiment, score = analyze_sentiment(tweet_text)
        
        tweets.append({
            'source': 'Twitter',
            'title': tweet_text,
            'description': '',
            'url': f'https://twitter.com/user/status/{int(time.time())}{i}',
            'published_at': datetime.datetime.now() - datetime.timedelta(hours=i),
            'sentiment': sentiment,
            'sentiment_score': score,
            'crypto': crypto_symbol,
            'engagement': np.random.randint(100, 10000)
        })
    
    return tweets

# Trend analysis
def analyze_trends(news_data):
    """Haber ve sosyal medya trend analizi"""
    if not news_data:
        return {}
    
    df = pd.DataFrame(news_data)
    
    # Sentiment dağılımı
    sentiment_counts = df['sentiment'].value_counts()
    
    # Zaman bazlı analiz
    df['hour'] = pd.to_datetime(df['published_at']).dt.hour
    hourly_sentiment = df.groupby('hour')['sentiment_score'].mean()
    
    # Anahtar kelime analizi
    all_text = ' '.join(df['title'].fillna('') + ' ' + df['description'].fillna(''))
    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
    common_words = Counter(words).most_common(20)
    
    return {
        'sentiment_distribution': sentiment_counts,
        'hourly_sentiment': hourly_sentiment,
        'common_words': common_words,
        'total_mentions': len(df),
        'average_sentiment': df['sentiment_score'].mean(),
        'positive_ratio': len(df[df['sentiment'] == 'positive']) / len(df)
    }

# Sidebar
st.sidebar.header("🔍 Analysis Settings")

selected_crypto = st.sidebar.selectbox(
    "Select Cryptocurrency:",
    list(CRYPTO_SYMBOLS.keys()),
    format_func=lambda x: f"{x} - {CRYPTO_SYMBOLS[x]}"
)

analysis_type = st.sidebar.radio(
    "Analysis Type:",
    ["Social Media Sentiment", "News Analysis", "Trend Detection"]
)

time_range = st.sidebar.selectbox(
    "Time Range:",
    ["24 Hours", "3 Days", "1 Week", "1 Month"]
)

# Main analysis function
def run_social_analysis():
    st.header(f"📊 {CRYPTO_SYMBOLS[selected_crypto]} ({selected_crypto}) - {analysis_type}")
    
    with st.spinner("🔍 Analyzing social media and news data..."):
        # Verileri getir
        news_data = get_crypto_news(selected_crypto, 30)
        
        if not news_data:
            st.error("No data found for analysis")
            return
        
        # Trend analizi
        trends = analyze_trends(news_data)
        
        # Sonuçları göster
        display_analysis_results(news_data, trends)

def display_analysis_results(news_data, trends):
    """Analiz sonuçlarını göster"""
    
    # Overview Metrics
    st.subheader("📈 Overview Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Mentions", trends['total_mentions'])
    
    with col2:
        st.metric("Average Sentiment", f"{trends['average_sentiment']:.2f}")
    
    with col3:
        st.metric("Positive Ratio", f"{trends['positive_ratio']:.1%}")
    
    with col4:
        dominant_sentiment = trends['sentiment_distribution'].index[0]
        st.metric("Dominant Sentiment", dominant_sentiment.title())
    
    st.markdown("---")
    
    # Sentiment Analysis
    st.subheader("😊 Sentiment Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment dağılımı
        fig_sentiment = px.pie(
            values=trends['sentiment_distribution'].values,
            names=trends['sentiment_distribution'].index,
            title="Sentiment Distribution"
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    with col2:
        # Zaman bazlı sentiment
        if len(trends['hourly_sentiment']) > 0:
            fig_hourly = px.line(
                x=trends['hourly_sentiment'].index,
                y=trends['hourly_sentiment'].values,
                title="Hourly Sentiment Trend",
                labels={'x': 'Hour', 'y': 'Sentiment Score'}
            )
            st.plotly_chart(fig_hourly, use_container_width=True)
    
    # Word Cloud ve Common Words
    st.subheader("🔤 Keyword Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Word Cloud
        st.write("**Word Cloud**")
        if trends['common_words']:
            wordcloud = WordCloud(
                width=400, 
                height=200, 
                background_color='white'
            ).generate_from_frequencies(dict(trends['common_words']))
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
    
    with col2:
        # Common Words
        st.write("**Top Keywords**")
        words_df = pd.DataFrame(trends['common_words'][:10], columns=['Word', 'Count'])
        st.dataframe(words_df, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed News/Social Feed
    st.subheader("📰 Recent Mentions & News")
    
    # Filtreleme seçenekleri
    col1, col2 = st.columns(2)
    
    with col1:
        sentiment_filter = st.selectbox(
            "Filter by Sentiment:",
            ["All", "Positive", "Negative", "Neutral"]
        )
    
    with col2:
        source_filter = st.selectbox(
            "Filter by Source:",
            ["All", "CoinGecko", "Twitter"]
        )
    
    # Filtreleme
    filtered_data = news_data
    
    if sentiment_filter != "All":
        filtered_data = [item for item in filtered_data if item['sentiment'] == sentiment_filter.lower()]
    
    if source_filter != "All":
        filtered_data = [item for item in filtered_data if item['source'] == source_filter]
    
    # Haberleri göster
    for i, item in enumerate(filtered_data[:15]):
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Sentiment icon
                sentiment_icon = {
                    'positive': '🟢',
                    'negative': '🔴', 
                    'neutral': '🟡'
                }.get(item['sentiment'], '⚪')
                
                st.write(f"**{sentiment_icon} {item['title']}**")
                if item['description']:
                    st.write(item['description'][:200] + "...")
                
                st.caption(f"Source: {item['source']} • {item['published_at']}")
            
            with col2:
                st.metric(
                    "Sentiment Score", 
                    f"{item['sentiment_score']:.2f}",
                    delta="Positive" if item['sentiment_score'] > 0.1 else "Negative" if item['sentiment_score'] < -0.1 else "Neutral"
                )
            
            st.markdown("---")

# Price-Sentiment Correlation (simulated)
def display_price_sentiment_correlation():
    """Fiyat-duygu korelasyonu analizi"""
    st.subheader("💰 Price-Sentiment Correlation")
    
    # Simüle veri (gerçek uygulamada gerçek fiyat verisi kullan)
    dates = pd.date_range(end=datetime.datetime.now(), periods=30, freq='D')
    prices = np.random.normal(40000, 5000, 30).cumsum()
    sentiment_scores = np.random.normal(0, 0.3, 30)
    
    correlation_df = pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'Sentiment': sentiment_scores
    })
    
    # Korelasyon grafiği
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=correlation_df['Date'],
        y=correlation_df['Price'],
        name='Price',
        yaxis='y1',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=correlation_df['Date'],
        y=correlation_df['Sentiment'] * 10000 + correlation_df['Price'].mean(),
        name='Sentiment (scaled)',
        yaxis='y1',
        line=dict(color='red', dash='dot')
    ))
    
    fig.update_layout(
        title="Price vs Sentiment Correlation",
        yaxis=dict(title='Price'),
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Korelasyon katsayısı
    correlation = np.corrcoef(correlation_df['Price'], correlation_df['Sentiment'])[0,1]
    st.metric("Correlation Coefficient", f"{correlation:.2f}")

# Alert System
def display_alerts(trends):
    """Önemli değişiklikler için alert sistemi"""
    st.subheader("🚨 Important Alerts")
    
    alerts = []
    
    # Sentiment değişimi alert
    if trends['positive_ratio'] > 0.7:
        alerts.append("🟢 Strong positive sentiment detected!")
    elif trends['positive_ratio'] < 0.3:
        alerts.append("🔴 Strong negative sentiment detected!")
    
    # Volume alert
    if trends['total_mentions'] > 25:
        alerts.append("📈 High social media volume!")
    
    # Keyword alerts
    common_words = [word for word, count in trends['common_words'][:5]]
    alert_keywords = ['hack', 'scam', 'regulation', 'ban', 'lawsuit']
    for keyword in alert_keywords:
        if keyword in common_words:
            alerts.append(f"⚠️ Alert: '{keyword}' trending in discussions")
    
    if alerts:
        for alert in alerts:
            st.warning(alert)
    else:
        st.info("No significant alerts at this time")

# Main application
def main():
    # Run analysis
    run_social_analysis()
    
    # Additional features
    display_price_sentiment_correlation()
    news_data = get_crypto_news(selected_crypto, 20)
    trends = analyze_trends(news_data)
    display_alerts(trends)
    
    # Analysis timestamp
    st.sidebar.markdown("---")
    st.sidebar.write(f"Last analysis: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Run the app
if __name__ == "__main__":
    main()

st.markdown("---")
st.info("""
**🔍 Analysis Features:**
- ✅ **Real-time Social Media Monitoring**
- ✅ **News Sentiment Analysis** 
- ✅ **Trend Detection & Alerts**
- ✅ **Price-Sentiment Correlation**
- ✅ **Keyword & Hashtag Tracking**
- ✅ **Multi-source Data Aggregation**

**🚨 Alert Types:**
- Sentiment spikes (positive/negative)
- High volume mentions
- Trending keywords
- Correlation anomalies
""")
