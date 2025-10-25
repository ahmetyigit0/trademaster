import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import time
from typing import Tuple, List, Dict

# Sayfa ayarı
st.set_page_config(
    page_title="Kripto Vadeli İşlem Strateji Simülasyonu",
    page_icon="📈",
    layout="wide"
)

# Başlık
st.title("📈 Kripto Vadeli İşlem Strateji Simülasyonu")
st.markdown("---")

# Strateji sınıfı
class CryptoStrategy:
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.results = {}
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Teknik göstergeleri hesapla"""
        df = df.copy()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bantları
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # EMA'lar
        df['EMA_9'] = df['Close'].ewm(span=9).mean()
        df['EMA_21'] = df['Close'].ewm(span=21).mean()
        df['EMA_50'] = df['Close'].ewm(span=50).mean()
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Alım-satım sinyalleri oluştur"""
        df = df.copy()
        df['Signal'] = 0  # 0: Bekle, 1: Long, -1: Short
        
        # Strateji kuralları
        long_condition = (
            (df['RSI'] < 35) &  # RSI oversold
            (df['MACD'] > df['MACD_Signal']) &  # MACD yukarı kesiyor
            (df['Close'] < df['BB_Lower']) &  # Fiyat Bollinger alt bandında
            (df['EMA_9'] > df['EMA_21'])  # Kısa EMA uzun EMA'nın üstünde
        )
        
        short_condition = (
            (df['RSI'] > 65) &  # RSI overbought
            (df['MACD'] < df['MACD_Signal']) &  # MACD aşağı kesiyor
            (df['Close'] > df['BB_Upper']) &  # Fiyat Bollinger üst bandında
            (df['EMA_9'] < df['EMA_21'])  # Kısa EMA uzun EMA'nın altında
        )
        
        df.loc[long_condition, 'Signal'] = 1
        df.loc[short_condition, 'Signal'] = -1
        
        return df
    
    def backtest_strategy(self, df: pd.DataFrame, progress_bar) -> Dict:
        """Stratejiyi backtest et"""
        capital = self.initial_capital
        position = 0  # 0: Pozisyon yok, 1: Long, -1: Short
        entry_price = 0
        trades = []
        total_trades = 0
        winning_trades = 0
        
        # İlerleme çubuğu için
        total_rows = len(df)
        
        for i, (index, row) in enumerate(df.iterrows()):
            # İlerleme çubuğunu güncelle
            if i % 10 == 0:  # Her 10 işlemde bir güncelle
                progress_bar.progress(i / total_rows)
            
            current_price = row['Close']
            signal = row['Signal']
            
            # Pozisyon açma/kapama
            if position == 0 and signal != 0:
                # Yeni pozisyon aç
                position = signal
                entry_price = current_price
                trade_size = capital * 0.1  # %10 risk
                entry_capital = trade_size
                total_trades += 1
                
                trades.append({
                    'entry_time': index,
                    'entry_price': entry_price,
                    'position': position,
                    'entry_capital': entry_capital,
                    'exit_time': None,
                    'exit_price': None,
                    'pnl': 0
                })
                
            elif position != 0:
                current_trade = trades[-1]
                
                # Kar alma/zarar kesme
                if position == 1:  # Long pozisyon
                    pnl_percent = (current_price - entry_price) / entry_price
                    stop_loss = -0.05  # %5 stop loss
                    take_profit = 0.10  # %10 take profit
                    
                    if pnl_percent <= stop_loss or pnl_percent >= take_profit or signal == -1:
                        # Pozisyonu kapat
                        pnl_amount = current_trade['entry_capital'] * pnl_percent
                        capital += pnl_amount
                        
                        if pnl_amount > 0:
                            winning_trades += 1
                            
                        trades[-1].update({
                            'exit_time': index,
                            'exit_price': current_price,
                            'pnl': pnl_amount
                        })
                        position = 0
                        
                elif position == -1:  # Short pozisyon
                    pnl_percent = (entry_price - current_price) / entry_price
                    stop_loss = -0.05  # %5 stop loss
                    take_profit = 0.10  # %10 take profit
                    
                    if pnl_percent <= stop_loss or pnl_percent >= take_profit or signal == 1:
                        # Pozisyonu kapat
                        pnl_amount = current_trade['entry_capital'] * pnl_percent
                        capital += pnl_amount
                        
                        if pnl_amount > 0:
                            winning_trades += 1
                            
                        trades[-1].update({
                            'exit_time': index,
                            'exit_price': current_price,
                            'pnl': pnl_amount
                        })
                        position = 0
        
        # Açık pozisyonları kapat
        if position != 0 and trades:
            current_trade = trades[-1]
            if current_trade['exit_time'] is None:
                last_price = df['Close'].iloc[-1]
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
                    'pnl': pnl_amount
                })
        
        # Sonuçları hesapla
        final_capital = capital
        total_return = ((final_capital - self.initial_capital) / self.initial_capital) * 100
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        self.results = {
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'trades': trades,
            'equity_curve': self.calculate_equity_curve(trades, df)
        }
        
        return self.results
    
    def calculate_equity_curve(self, trades: List[Dict], df: pd.DataFrame) -> pd.DataFrame:
        """Equity curve hesapla"""
        equity = [self.initial_capital]
        dates = [df.index[0]]
        
        current_capital = self.initial_capital
        
        for trade in trades:
            if trade['exit_time'] is not None:
                current_capital += trade['pnl']
                equity.append(current_capital)
                dates.append(trade['exit_time'])
        
        return pd.DataFrame({'Date': dates, 'Equity': equity})

# Sidebar
st.sidebar.header("⚙️ Simülasyon Ayarları")

# Kripto seçimi
crypto_symbols = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD", 
    "Binance Coin (BNB-USD)": "BNB-USD",
    "Cardano (ADA-USD)": "ADA-USD",
    "Solana (SOL-USD)": "SOL-USD"
}

selected_crypto = st.sidebar.selectbox(
    "Kripto Para Seçin:",
    list(crypto_symbols.keys())
)

symbol = crypto_symbols[selected_crypto]

# Tarih seçimi
end_date = st.sidebar.date_input(
    "Simülasyon Bitiş Tarihi:",
    datetime.date.today() - datetime.timedelta(days=1)
)

start_date = end_date - datetime.timedelta(days=90)

st.sidebar.info(f"Simülasyon Aralığı: {start_date.strftime('%d.%m.%Y')} - {end_date.strftime('%d.%m.%Y')}")

# Başlangıç sermayesi
initial_capital = st.sidebar.number_input(
    "Başlangıç Sermayesi (USD):",
    min_value=1000,
    max_value=100000,
    value=10000,
    step=1000
)

# Ana içerik
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Fiyat Grafiği ve Sinyaller")
    
    # Veri yükleme
    @st.cache_data
    def load_data(symbol, start_date, end_date):
        try:
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if data.empty:
                st.error("Veri çekilemedi. Lütfen farklı bir tarih aralığı deneyin.")
                return None
            return data
        except Exception as e:
            st.error(f"Veri yüklenirken hata oluştu: {e}")
            return None

    data = load_data(symbol, start_date, end_date)
    
    if data is not None and not data.empty:
        try:
            # Strateji uygula
            strategy = CryptoStrategy(initial_capital)
            data_with_indicators = strategy.calculate_indicators(data)
            data_with_signals = strategy.generate_signals(data_with_indicators)
            
            # Grafik oluştur
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Fiyat ve Sinyaller', 'RSI'),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )
            
            # Fiyat grafiği
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Fiyat'
                ),
                row=1, col=1
            )
            
            # Bollinger Bantları
            fig.add_trace(
                go.Scatter(
                    x=data_with_indicators.index,
                    y=data_with_indicators['BB_Upper'],
                    line=dict(color='rgba(255, 0, 0, 0.3)'),
                    name='BB Upper'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data_with_indicators.index,
                    y=data_with_indicators['BB_Lower'],
                    line=dict(color='rgba(0, 255, 0, 0.3)'),
                    name='BB Lower',
                    fill='tonexty'
                ),
                row=1, col=1
            )
            
            # Alım sinyalleri
            long_signals = data_with_signals[data_with_signals['Signal'] == 1]
            if not long_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=long_signals.index,
                        y=long_signals['Close'],
                        mode='markers',
                        marker=dict(color='green', size=10, symbol='triangle-up'),
                        name='Long Sinyal'
                    ),
                    row=1, col=1
                )
            
            # Satım sinyalleri
            short_signals = data_with_signals[data_with_signals['Signal'] == -1]
            if not short_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=short_signals.index,
                        y=short_signals['Close'],
                        mode='markers',
                        marker=dict(color='red', size=10, symbol='triangle-down'),
                        name='Short Sinyal'
                    ),
                    row=1, col=1
                )
            
            # RSI
            fig.add_trace(
                go.Scatter(
                    x=data_with_indicators.index,
                    y=data_with_indicators['RSI'],
                    line=dict(color='purple'),
                    name='RSI'
                ),
                row=2, col=1
            )
            
            # RSI seviyeleri
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
            
            fig.update_layout(
                height=600,
                showlegend=True,
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Grafik oluşturulurken hata oluştu: {e}")
    else:
        st.warning("Veri yüklenemedi. Lütfen tarih aralığını kontrol edin.")

with col2:
    st.subheader("🎯 Strateji Bilgileri")
    
    st.markdown("""
    **Kullanılan Strateji:**
    - RSI + MACD + Bollinger Bantları
    - Çoklu zaman periyodu analizi
    
    **Long Koşulları:**
    - RSI < 35 (Oversold)
    - MACD > MACD Signal
    - Fiyat BB Alt Bandı altında
    - EMA(9) > EMA(21)
    
    **Short Koşulları:**
    - RSI > 65 (Overbought)  
    - MACD < MACD Signal
    - Fiyat BB Üst Bandı üstünde
    - EMA(9) < EMA(21)
    
    **Risk Yönetimi:**
    - %5 Stop Loss
    - %10 Take Profit
    - %10 Pozisyon Büyüklüğü
    """)

# Simülasyon butonu - HER ZAMAN GÖRÜNÜR
st.markdown("---")
st.subheader("🚀 Backtest Simülasyonu")

# Buton her zaman görünsün
if st.button("🎯 Backtest Simülasyonunu Başlat", type="primary", key="backtest_button"):
    if data is not None and not data.empty:
        with st.spinner("Simülasyon çalışıyor..."):
            start_time = time.time()
            
            # İlerleme çubuğu
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Stratejiyi çalıştır
                strategy = CryptoStrategy(initial_capital)
                data_with_indicators = strategy.calculate_indicators(data)
                data_with_signals = strategy.generate_signals(data_with_indicators)
                
                status_text.text("Strateji backtest ediliyor...")
                results = strategy.backtest_strategy(data_with_signals, progress_bar)
                
                end_time = time.time()
                calculation_time = end_time - start_time
                
                # İlerleme çubuğunu tamamla
                progress_bar.progress(1.0)
                
                st.success(f"✅ Simülasyon hesaplaması {calculation_time:.2f} saniye içinde tamamlandı!")
                
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
                        f"{results['total_trades']}"
                    )
                
                with col4:
                    st.metric(
                        "Win Rate",
                        f"{results['win_rate']:.1f}%"
                    )
                
                # Equity curve
                if not results['equity_curve'].empty:
                    st.subheader("📈 Equity Curve")
                    equity_fig = go.Figure()
                    
                    equity_fig.add_trace(
                        go.Scatter(
                            x=results['equity_curve']['Date'],
                            y=results['equity_curve']['Equity'],
                            line=dict(color='blue'),
                            name='Portföy Değeri'
                        )
                    )
                    
                    equity_fig.add_hline(
                        y=initial_capital,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Başlangıç Sermayesi"
                    )
                    
                    equity_fig.update_layout(
                        height=400,
                        showlegend=True,
                        xaxis_title="Tarih",
                        yaxis_title="Portföy Değeri (USD)"
                    )
                    
                    st.plotly_chart(equity_fig, use_container_width=True)
                
                # İşlem detayları
                if results['trades']:
                    st.subheader("📋 İşlem Detayları")
                    
                    trades_df = pd.DataFrame(results['trades'])
                    # Sadece kapanan işlemleri göster
                    closed_trades = trades_df[trades_df['exit_time'].notna()]
                    
                    if not closed_trades.empty:
                        closed_trades['pnl_percent'] = (closed_trades['pnl'] / closed_trades['entry_capital']) * 100
                        
                        # Renkli PNL sütunu
                        def color_pnl(val):
                            color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
                            return f'color: {color}'
                        
                        styled_trades = closed_trades.style.format({
                            'entry_price': '{:.2f}',
                            'exit_price': '{:.2f}', 
                            'entry_capital': '{:.2f}',
                            'pnl': '{:.2f}',
                            'pnl_percent': '{:.2f}%'
                        }).applymap(color_pnl, subset=['pnl', 'pnl_percent'])
                        
                        st.dataframe(styled_trades, use_container_width=True)
                    else:
                        st.info("Kapanan işlem bulunamadı.")
                else:
                    st.info("Hiç işlem yapılmadı.")
                    
            except Exception as e:
                st.error(f"Simülasyon sırasında hata oluştu: {e}")
    else:
        st.error("Veri yüklenemedi. Lütfen önce kripto para ve tarih seçin.")

# Bilgi
st.markdown("---")
st.info("""
**⚠️ Uyarı:** Bu simülasyon sadece eğitim amaçlıdır. Gerçek trading için kullanmayın. 
Geçmiş performans gelecek sonuçların garantisi değildir.
""")