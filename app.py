import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import time

# Sayfa ayarı
st.set_page_config(
    page_title="Kripto Vadeli İşlem Strateji Simülasyonu",
    page_icon="📈",
    layout="wide"
)

# Başlık
st.title("📈 Kripto Vadeli İşlem Strateji Simülasyonu")
st.markdown("---")

# Gelişmiş Strateji sınıfı
class CryptoStrategy:
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.results = {}
        
    def calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gelişmiş teknik göstergeleri hesapla"""
        try:
            df = df.copy()
            
            # RSI hesaplama
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14, min_periods=1).mean()
            avg_loss = loss.rolling(window=14, min_periods=1).mean()
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # EMA'lar
            df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
            df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
            df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
            
            # MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # DÜZELTİLMİŞ Bollinger Bantları - DataFrame problemi çözüldü
            bb_middle = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            
            # Tek tek sütun ataması yap
            df = df.assign(
                BB_Middle=bb_middle,
                BB_Upper=bb_middle + (bb_std * 2),
                BB_Lower=bb_middle - (bb_std * 2)
            )
            
            # Bollinger Band pozisyonu
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
            # Momentum
            df['Momentum'] = df['Close'] - df['Close'].shift(5)
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            
            return df.fillna(0)
        except Exception as e:
            st.error(f"Göstergeler hesaplanırken hata: {e}")
            return df
    
    def generate_advanced_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gelişmiş alım-satım sinyalleri oluştur"""
        try:
            df = df.copy()
            df['Signal'] = 0  # 0: Bekle, 1: Long, -1: Short
            
            # Her satır için tek tek kontrol et
            for i in range(len(df)):
                try:
                    if i < 50:  # İlk 50 gün yeterli veri yok
                        continue
                        
                    rsi = float(df['RSI'].iloc[i])
                    ema_9 = float(df['EMA_9'].iloc[i])
                    ema_21 = float(df['EMA_21'].iloc[i])
                    ema_50 = float(df['EMA_50'].iloc[i])
                    macd = float(df['MACD'].iloc[i])
                    macd_signal = float(df['MACD_Signal'].iloc[i])
                    bb_position = float(df['BB_Position'].iloc[i])
                    momentum = float(df['Momentum'].iloc[i])
                    volume_ratio = float(df['Volume_Ratio'].iloc[i])
                    
                    # LONG sinyalleri (daha esnek koşullar)
                    long_signals = 0
                    
                    # 1. RSI oversold + EMA trend
                    if rsi < 45 and ema_9 > ema_21:
                        long_signals += 1
                    
                    # 2. MACD bullish crossover
                    if macd > macd_signal and df['MACD'].iloc[i-1] <= df['MACD_Signal'].iloc[i-1]:
                        long_signals += 1
                    
                    # 3. Bollinger Band bounce
                    if bb_position < 0.2 and momentum > 0:
                        long_signals += 1
                    
                    # 4. Volume confirmation
                    if volume_ratio > 1.2:
                        long_signals += 0.5
                    
                    # SHORT sinyalleri (daha esnek koşullar)
                    short_signals = 0
                    
                    # 1. RSI overbought + EMA trend
                    if rsi > 55 and ema_9 < ema_21:
                        short_signals += 1
                    
                    # 2. MACD bearish crossover
                    if macd < macd_signal and df['MACD'].iloc[i-1] >= df['MACD_Signal'].iloc[i-1]:
                        short_signals += 1
                    
                    # 3. Bollinger Band rejection
                    if bb_position > 0.8 and momentum < 0:
                        short_signals += 1
                    
                    # 4. Volume confirmation
                    if volume_ratio > 1.2:
                        short_signals += 0.5
                    
                    # Sinyal belirleme (eşik değerleri düşürüldü)
                    if long_signals >= 1.5:  # Daha düşük eşik
                        df.loc[df.index[i], 'Signal'] = 1
                    elif short_signals >= 1.5:  # Daha düşük eşik
                        df.loc[df.index[i], 'Signal'] = -1
                        
                except Exception:
                    continue
                    
            return df
        except Exception as e:
            st.error(f"Sinyal oluşturma hatası: {e}")
            df['Signal'] = 0
            return df
    
    def backtest_advanced_strategy(self, df: pd.DataFrame, progress_bar) -> dict:
        """Gelişmiş stratejiyi backtest et"""
        try:
            capital = self.initial_capital
            position = 0  # 0: Pozisyon yok, 1: Long, -1: Short
            entry_price = 0
            trades = []
            total_trades = 0
            winning_trades = 0
            
            total_rows = len(df)
            
            for i in range(len(df)):
                # İlerleme çubuğunu güncelle
                if i % 10 == 0 and total_rows > 0:
                    progress_bar.progress(min(i / total_rows, 1.0))
                
                current_date = df.index[i]
                current_price = float(df['Close'].iloc[i])
                signal = int(df['Signal'].iloc[i])
                
                # Pozisyon açma (daha agresif)
                if position == 0 and signal != 0:
                    position = signal
                    entry_price = current_price
                    trade_size = min(capital * 0.15, capital)  # %15 risk - artırıldı
                    entry_capital = trade_size
                    total_trades += 1
                    
                    trades.append({
                        'entry_time': current_date,
                        'entry_price': entry_price,
                        'position': 'LONG' if position == 1 else 'SHORT',
                        'entry_capital': entry_capital,
                        'exit_time': None,
                        'exit_price': None,
                        'pnl': 0,
                        'status': 'OPEN',
                        'pnl_percent': 0
                    })
                
                # Pozisyon kapatma (daha esnek)
                elif position != 0:
                    current_trade = trades[-1]
                    
                    if position == 1:  # Long pozisyon
                        pnl_percent = (current_price - entry_price) / entry_price
                        
                        # Daha esnek kapatma koşulları
                        close_condition = (
                            pnl_percent <= -0.03 or  # %3 stop loss
                            pnl_percent >= 0.06 or   # %6 take profit
                            signal == -1 or          # Zıt sinyal
                            pnl_percent >= 0.15      # Maksimum kar
                        )
                        
                        if close_condition:
                            pnl_amount = entry_capital * pnl_percent
                            capital += pnl_amount
                            
                            if pnl_amount > 0:
                                winning_trades += 1
                            
                            trades[-1].update({
                                'exit_time': current_date,
                                'exit_price': current_price,
                                'pnl': pnl_amount,
                                'status': 'CLOSED',
                                'pnl_percent': pnl_percent * 100
                            })
                            position = 0
                            entry_price = 0
                    
                    elif position == -1:  # Short pozisyon
                        pnl_percent = (entry_price - current_price) / entry_price
                        
                        # Daha esnek kapatma koşulları
                        close_condition = (
                            pnl_percent <= -0.03 or  # %3 stop loss
                            pnl_percent >= 0.06 or   # %6 take profit
                            signal == 1 or           # Zıt sinyal
                            pnl_percent >= 0.15      # Maksimum kar
                        )
                        
                        if close_condition:
                            pnl_amount = entry_capital * pnl_percent
                            capital += pnl_amount
                            
                            if pnl_amount > 0:
                                winning_trades += 1
                            
                            trades[-1].update({
                                'exit_time': current_date,
                                'exit_price': current_price,
                                'pnl': pnl_amount,
                                'status': 'CLOSED',
                                'pnl_percent': pnl_percent * 100
                            })
                            position = 0
                            entry_price = 0
            
            # Açık pozisyonları kapat
            if position != 0 and trades:
                current_trade = trades[-1]
                last_price = float(df['Close'].iloc[-1])
                
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
            final_capital = max(capital, 0)
            total_return = ((final_capital - self.initial_capital) / self.initial_capital) * 100
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Profit factor
            total_profit = sum(trade['pnl'] for trade in trades if trade['pnl'] > 0)
            total_loss = abs(sum(trade['pnl'] for trade in trades if trade['pnl'] < 0))
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Sharpe Ratio (basit)
            if len(trades) > 1:
                returns = [trade['pnl_percent'] / 100 for trade in trades if trade['status'] == 'CLOSED']
                if returns:
                    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
                else:
                    sharpe_ratio = 0
            else:
                sharpe_ratio = 0
            
            # Equity curve
            equity_curve = self.calculate_equity_curve(trades)
            
            self.results = {
                'initial_capital': self.initial_capital,
                'final_capital': final_capital,
                'total_return': total_return,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'trades': trades,
                'equity_curve': equity_curve
            }
            
            return self.results
            
        except Exception as e:
            st.error(f"Backtest sırasında hata: {str(e)}")
            return {
                'initial_capital': self.initial_capital,
                'final_capital': self.initial_capital,
                'total_return': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'trades': [],
                'equity_curve': pd.DataFrame({'Date': [], 'Equity': []})
            }
    
    def calculate_equity_curve(self, trades: list) -> pd.DataFrame:
        """Equity curve hesapla"""
        try:
            if not trades:
                return pd.DataFrame({'Date': [], 'Equity': []})
            
            equity = [self.initial_capital]
            dates = [trades[0]['entry_time']]
            
            current_capital = self.initial_capital
            
            for trade in trades:
                if trade['status'] == 'CLOSED':
                    current_capital += trade['pnl']
                    equity.append(current_capital)
                    dates.append(trade['exit_time'])
            
            return pd.DataFrame({'Date': dates, 'Equity': equity})
        except:
            return pd.DataFrame({'Date': [], 'Equity': []})

# Sidebar
st.sidebar.header("⚙️ Simülasyon Ayarları")

# Kripto seçimi
crypto_symbols = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD", 
    "Binance Coin (BNB-USD)": "BNB-USD",
    "Cardano (ADA-USD)": "ADA-USD",
    "Solana (SOL-USD)": "SOL-USD",
    "Ripple (XRP-USD)": "XRP-USD",
    "Dogecoin (DOGE-USD)": "DOGE-USD"
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

start_date = end_date - datetime.timedelta(days=180)  # 180 gün - daha uzun periyot

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
st.subheader("🎯 Gelişmiş Strateji Bilgileri")
    
st.markdown("""
**Gelişmiş Çoklu Gösterge Stratejisi:**
- RSI + EMA + MACD + Bollinger Bantları kombinasyonu
- Volume ve Momentum onayı
- Çoklu zaman periyodu analizi

**Long Sinyal Koşulları:**
- RSI < 45 (Oversold bölgesi) + EMA yükselişi
- MACD bullish crossover
- Bollinger Band alt seviyesinden bounce
- Volume artışı onayı

**Short Sinyal Koşulları:**
- RSI > 55 (Overbought bölgesi) + EMA düşüşü  
- MACD bearish crossover
- Bollinger Band üst seviyesinden rejection
- Volume artışı onayı

**Risk Yönetimi:**
- %3 Stop Loss (esnek)
- %6 Take Profit (esnek)
- %15 Pozisyon Büyüklüğü (artırıldı)
- Maksimum %15 kar sınırı
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
            return None
        return data
    except Exception as e:
        st.error(f"Veri yüklenirken hata oluştu: {e}")
        return None

# Uygulama başladığında veriyi yükle
data = load_data(symbol, start_date, end_date)

if data is not None and not data.empty:
    # Temel bilgileri göster
    try:
        first_price = float(data['Close'].iloc[0])
        last_price = float(data['Close'].iloc[-1])
        price_change = ((last_price - first_price) / first_price) * 100
        data_count = len(data)
        
        st.success(f"✅ {selected_crypto} verisi yüklendi: {data_count} günlük veri ({start_date} - {end_date})")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("İlk Fiyat", f"${first_price:.2f}")
        with col2:
            st.metric("Son Fiyat", f"${last_price:.2f}")
        with col3:
            st.metric("Dönem Değişim", f"{price_change:+.2f}%")
        with col4:
            st.metric("Veri Sayısı", data_count)
    except Exception as e:
        st.error(f"Veri gösterilirken hata: {e}")
else:
    st.warning("⚠️ Veri yüklenemedi. Lütfen tarih aralığını kontrol edin.")

# Simülasyon butonu
if st.button("🎯 Gelişmiş Backtest Simülasyonunu Başlat", type="primary", use_container_width=True):
    if data is not None and not data.empty:
        with st.spinner("Gelişmiş simülasyon çalışıyor..."):
            start_time = time.time()
            
            # İlerleme çubuğu
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Stratejiyi çalıştır
                strategy = CryptoStrategy(initial_capital)
                
                # Göstergeleri hesapla
                status_text.text("Gelişmiş teknik göstergeler hesaplanıyor...")
                data_with_indicators = strategy.calculate_advanced_indicators(data)
                
                # Sinyalleri oluştur
                status_text.text("Çoklu sinyal sistemi oluşturuluyor...")
                data_with_signals = strategy.generate_advanced_signals(data_with_indicators)
                
                # Backtest yap
                status_text.text("Gelişmiş strateji backtest ediliyor...")
                results = strategy.backtest_advanced_strategy(data_with_signals, progress_bar)
                
                end_time = time.time()
                calculation_time = end_time - start_time
                
                # İlerleme çubuğunu tamamla
                progress_bar.progress(1.0)
                status_text.empty()
                
                st.success(f"✅ Gelişmiş simülasyon hesaplaması {calculation_time:.2f} saniye içinde tamamlandı!")
                
                # Sonuçları göster
                st.subheader("📊 Gelişmiş Simülasyon Sonuçları")
                
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
                        f"{results['total_trades']}",
                        delta=f"+{results['total_trades']}" if results['total_trades'] > 0 else "0"
                    )
                
                with col4:
                    st.metric(
                        "Win Rate",
                        f"{results['win_rate']:.1f}%"
                    )
                
                # Ek metrikler
                col5, col6, col7 = st.columns(3)
                
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
                
                with col7:
                    st.metric(
                        "Sharpe Ratio",
                        f"{results['sharpe_ratio']:.2f}"
                    )
                
                # Equity curve
                if not results['equity_curve'].empty:
                    st.subheader("📈 Portföy Performansı")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=results['equity_curve']['Date'],
                        y=results['equity_curve']['Equity'],
                        mode='lines+markers',
                        name='Portföy Değeri',
                        line=dict(color='blue', width=3),
                        marker=dict(size=4)
                    ))
                    
                    # Başlangıç sermayesi çizgisi
                    fig.add_hline(
                        y=initial_capital, 
                        line_dash="dash", 
                        line_color="red",
                        annotation_text="Başlangıç Sermayesi"
                    )
                    
                    fig.update_layout(
                        title="Gelişmiş Strateji - Portföy Performans Grafiği",
                        xaxis_title="Tarih",
                        yaxis_title="Portföy Değeri (USD)",
                        height=500,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Equity curve verisi bulunamadı.")
                
                # İşlem detayları
                if results['trades']:
                    closed_trades = [t for t in results['trades'] if t['status'] == 'CLOSED']
                    
                    if closed_trades:
                        st.subheader("📋 İşlem Detayları")
                        
                        trades_df = pd.DataFrame(closed_trades)
                        
                        # Renk fonksiyonu
                        def color_pnl(val):
                            color = 'color: green' if val > 0 else 'color: red' if val < 0 else 'color: black'
                            return color
                        
                        # DataFrame'i düzenle
                        display_df = trades_df[['entry_time', 'exit_time', 'position', 'entry_price', 'exit_price', 'entry_capital', 'pnl', 'pnl_percent']]
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
                        
                        # Sayısal sütunları formatla
                        styled_df = display_df.style.format({
                            'Giriş Fiyatı': '{:.2f}',
                            'Çıkış Fiyatı': '{:.2f}',
                            'İşlem Büyüklüğü': '{:.2f}',
                            'Kar/Zarar ($)': '{:.2f}',
                            'Kar/Zarar (%)': '{:.2f}%'
                        }).applymap(color_pnl, subset=['Kar/Zarar ($)', 'Kar/Zarar (%)'])
                        
                        st.dataframe(styled_df, use_container_width=True, height=400)
                        
                        # İstatistikler
                        st.subheader("📈 Detaylı İşlem İstatistikleri")
                        avg_profit = trades_df['pnl'].mean()
                        max_profit = trades_df['pnl'].max()
                        max_loss = trades_df['pnl'].min()
                        total_pnl = trades_df['pnl'].sum()
                        avg_hold_time = (trades_df['exit_time'] - trades_df['entry_time']).mean()
                        
                        stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)
                        with stat_col1:
                            st.metric("Ortalama Kar/Zarar", f"${avg_profit:.2f}")
                        with stat_col2:
                            st.metric("Maksimum Kar", f"${max_profit:.2f}")
                        with stat_col3:
                            st.metric("Maksimum Zarar", f"${max_loss:.2f}")
                        with stat_col4:
                            st.metric("Toplam Kar/Zarar", f"${total_pnl:.2f}")
                        with stat_col5:
                            if pd.notna(avg_hold_time):
                                st.metric("Ort. Bekleme Süresi", f"{avg_hold_time.days} gün")
                            else:
                                st.metric("Ort. Bekleme Süresi", "N/A")
                        
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

**📊 Gelişmiş Strateji Özellikleri:**
- Çoklu gösterge kombinasyonu (RSI, EMA, MACD, Bollinger)
- Volume ve momentum onayı
- Daha esnek giriş/çıkış kuralları
- Artırılmış işlem frekansı
- Gelişmiş risk yönetimi
""")