import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import time
from typing import Dict, List

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
        
        # EMA'lar
        df['EMA_9'] = df['Close'].ewm(span=9).mean()
        df['EMA_21'] = df['Close'].ewm(span=21).mean()
        
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
            if i % 10 == 0:
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
                    'position': 'LONG' if position == 1 else 'SHORT',
                    'entry_capital': entry_capital,
                    'exit_time': None,
                    'exit_price': None,
                    'pnl': 0,
                    'status': 'OPEN'
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
                            'pnl': pnl_amount,
                            'status': 'CLOSED',
                            'pnl_percent': pnl_percent * 100
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
                            'pnl': pnl_amount,
                            'status': 'CLOSED',
                            'pnl_percent': pnl_percent * 100
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
                    'pnl': pnl_amount,
                    'status': 'CLOSED',
                    'pnl_percent': pnl_percent * 100
                })
        
        # Sonuçları hesapla
        final_capital = capital
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
            if trade['status'] == 'CLOSED':
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

# Simülasyon butonu
st.markdown("---")
st.subheader("🚀 Backtest Simülasyonu")

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

if st.button("🎯 Backtest Simülasyonunu Başlat", type="primary", use_container_width=True):
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
                status_text.empty()
                
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
                
                # Ek metrikler
                col5, col6 = st.columns(2)
                
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
                
                # Equity curve
                if not results['equity_curve'].empty:
                    st.subheader("📈 Equity Curve")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=results['equity_curve']['Date'],
                        y=results['equity_curve']['Equity'],
                        mode='lines',
                        name='Portföy Değeri',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig.add_hline(
                        y=initial_capital, 
                        line_dash="dash", 
                        line_color="red",
                        annotation_text="Başlangıç Sermayesi"
                    )
                    
                    fig.update_layout(
                        title="Portföy Performansı",
                        xaxis_title="Tarih",
                        yaxis_title="Portföy Değeri (USD)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # İşlem detayları
                if results['trades']:
                    st.subheader("📋 İşlem Detayları")
                    
                    trades_df = pd.DataFrame(results['trades'])
                    # Sadece kapanan işlemleri göster
                    closed_trades = trades_df[trades_df['status'] == 'CLOSED']
                    
                    if not closed_trades.empty:
                        # DataFrame'i düzenle
                        display_df = closed_trades[['entry_time', 'exit_time', 'position', 'entry_price', 'exit_price', 'entry_capital', 'pnl', 'pnl_percent']]
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
                        
                        # Renkli gösterim
                        def color_pnl(val):
                            if 'Kar/Zarar' in val.name:
                                if val > 0:
                                    return 'color: green'
                                elif val < 0:
                                    return 'color: red'
                            return ''
                        
                        styled_df = display_df.style.format({
                            'Giriş Fiyatı': '{:.2f}',
                            'Çıkış Fiyatı': '{:.2f}',
                            'İşlem Büyüklüğü': '{:.2f}',
                            'Kar/Zarar ($)': '{:.2f}',
                            'Kar/Zarar (%)': '{:.2f}%'
                        }).apply(color_pnl, subset=['Kar/Zarar ($)', 'Kar/Zarar (%)'])
                        
                        st.dataframe(styled_df, use_container_width=True)
                        
                        # İstatistikler
                        st.subheader("📈 İşlem İstatistikleri")
                        avg_profit = closed_trades['pnl'].mean()
                        max_profit = closed_trades['pnl'].max()
                        max_loss = closed_trades['pnl'].min()
                        
                        stat_col1, stat_col2, stat_col3 = st.columns(3)
                        with stat_col1:
                            st.metric("Ortalama Kar/Zarar", f"${avg_profit:.2f}")
                        with stat_col2:
                            st.metric("Maksimum Kar", f"${max_profit:.2f}")
                        with stat_col3:
                            st.metric("Maksimum Zarar", f"${max_loss:.2f}")
                        
                    else:
                        st.info("Kapanan işlem bulunamadı.")
                else:
                    st.info("Hiç işlem yapılmadı.")
                    
            except Exception as e:
                st.error(f"Simülasyon sırasında hata oluştu: {str(e)}")
    else:
        st.error("Veri yüklenemedi. Lütfen önce kripto para ve tarih seçin.")

# Bilgi
st.markdown("---")
st.info("""
**⚠️ Uyarı:** Bu simülasyon sadece eğitim amaçlıdır. Gerçek trading için kullanmayın. 
Geçmiş performans gelecek sonuçların garantisi değildir.
""")