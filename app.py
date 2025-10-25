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
                results =Strateji backtest ediliyor...")
                results = strategy strategy.backtest_strategy(data.backtest_strategy(data_with_with_signals, progress_bar)
_signals, progress_bar)
                
                               
                end_time = time.time()
 end_time = time.time()
                calculation_time = end                calculation_time = end_time_time - start_time
                
                - start_time
                
                # İ # İlerlelerlememe çubuğunu tamamla
                çubuğunu tamamla
                progress_bar.progress(1.0)
                status_text.empty()
                
 progress_bar.progress(1.0)
                status_text.empty()
                
                st.success                st.success(f"✅ Simülasyon hesaplamas(f"✅ Simülasyon hesaplaması {calculation_time:.2f} sı {calculation_time:.2f} sanianiye içinde tamamlandı!")
ye içinde tamamlandı!")
                
                               
                # Sonuç # Sonuçları göları göster
                stster
                st.subheader("📊 Sim.subheader("📊 Simülülasyon Sonuçları")
                
asyon Sonuçları")
                
                col                col1, col2,1, col2, col col3, col4 = st3, col4 = st.columns.columns(4)
                
               (4)
                
                with with col1:
                    col1:
                    st.metric st.metric(
                        "Baş(
                        "Başlanglangıç Sermayesiıç Sermayesi",
                        f",
                        f"${results"${results['initial_c['initial_capital']:,.2apital']:,.2f}"
                    )
                
f}"
                    )
                
                with                with col2:
                    col2:
                    st.metric st.metric(
                        "(
                        "Son SermayeSon Sermaye", 
                        f"", 
                        f"${results['final_capital']:${results['final_capital']:,.2f},.2f}",
                       ",
                        delta=f"{results delta=f"{results['total_return['total_return']:+.2f']:+.2f}%}%"
                    )
                
"
                    )
                
                with                with col3:
                    col3:
                    st.metric(
                        "Toplam İş st.metric(
                        "Toplam İşlem",
lem",
                        f"{                        f"{results['total_tradesresults['total_trades']}"
']}"
                    )
                
                                   )
                
                with with col4:
                    col4:
                    st.metric(
                        st.metric(
                        "Win "Win Rate",
                        f Rate"{results['win_rate']:.",
                        f"{results['win1_rate']:.1f}%f}%"
                    )
"
                    )
                
                # Ek                
                # Ek met metrikler
               rikler
                col5 col5, col6 = st, col6 = st.columns(2.columns(2)
                
)
                
                with col5                with col5:
                    st:
                    st.metric.metric(
                        "Karl(
                        "Karlı İşlemı İşlem Say Sayısısı",
                        f"{results['ı",
                        f"{results['winningwinning_trades']}"
_trades']}"
                    )
                
                with col6:
                    profit_factor = results['profit                    )
                
                with col6:
                    profit_factor_factor']
                    pf_display = f"{profit_factor:.2 = results['profit_factor']
                    pf_display = f"{profit_factor:.2ff}" if profit}" if profit_factor_factor != != float('inf') else "∞"
                    st float('inf') else "∞"
                    st.m.metric(
                        "Profit Factor",
                        pf_displayetric(
                        "Profit Factor",
                        pf_display
                    )
                
                # Equity
                    )
                
                # Equity curve
 curve
                if not results                if not results['equity_curve'].empty:
                    st.subheader("📈 Equity Curve")
                    
                    fig = go.Figure()
                    fig.add['equity_curve'].empty:
                    st.subheader("📈 Equity Curve")
                    
                    fig = go.Figure_trace(()
                    fig.add_trace(go.Scatter(
go.Scatter(
                        x=results['equity                        x=results['equity_curve']['Date'],
_curve']['Date'],
                        y=                        y=results['equresults['equity_ity_curve']['Equitycurve']['Equity'],
                        mode=''],
                        mode='lines',
                        namelines',
                        name='Portfö='Portföy Değeri',
                        line=dicty Değeri',
                        line=dict(color='blue', width=2)
(color='blue', width=2)
                    ))
                    
                    fig.add                    ))
                    
                    fig.add_hline_hline(
                        y=initial_capital, 
                        line(
                        y=initial_capital, 
                        line_dash="_dash="dashdash", 
                        line", 
                        line_color="red",
_color="red",
                        annotation_text                        annotation_text="Baş="Başlangıç Sermayesi"
                    )
                    
langıç Sermayesi"
                    )
                    
                    fig.update_layout(
                                           fig.update_layout(
                        title title="Portföy Performans="Portföy Performansıı",
                        xaxis_title="",
                        xaxis_title="TariTarih",
h",
                        yaxis_title="                        yaxis_title="PortföyPortföy Değeri (USD)",
                        Değeri (USD)",
                        height= height=400
                    )
400
                    )
                    
                                       
                    st.plotly_ch st.plotly_chart(fart(fig, use_containerig, use_container_width=True)
                
_width=True)
                
                # İş                # İşlem detlem detayları
               ayları
                if results if results['trades']['trades']:
                   :
                    st.subheader("📋 st.subheader("📋 İş İşlem Detaylarılem Detayları")
                    
")
                    
                    trades_df =                    trades_df = pd.DataFrame pd.DataFrame(results['trades(results['trades'])
                    # Sadece'])
                    # Sadece kapanan i kapanan işlemleri göşlemleri göster
ster
                    closed_trades                    closed_trades = trades_df = trades_df[trades_df[trades_df['status'] ==['status'] == 'CLOSED 'CLOSED']
                    
']
                    
                    if not closed                    if not closed_trades_trades.empty:
                        # DataFrame'i dü.empty:
                        # DataFrame'i düzenzenlele
                        display_df = closed_trades[
                        display_df = closed_trades[['entry['entry_time', 'exit_time', 'position', 'entry_time', 'exit_time', 'position', 'entry_price', 'exit_price', '_price', 'exit_price', 'entry_capital', 'entry_capital', 'pnl', 'pnl_perpnl', 'pnl_percent']]
                        display_dfcent']]
                        display_df = display = display_df.rename(_df.rename(columns={
                           columns={
                            'entry_time': 'Giriş Tarihi',
                            'entry_time': 'Giriş Tarihi',
                            'exit_time 'exit_time': '': 'Çıkış Tarihi',
                            'Çıkış Tarihi',
                            'position': 'Pozisposition': 'Pozisyonyon',
                            'entry_price':',
                            'entry_price': ' 'Giriş FiyatGiriş Fiyatı',
                            'exit_priceı',
                            'exit_price':': 'Çıkış F 'Çıkış Fiyatı',
iyatı',
                            '                            'entry_capital': 'İentry_capital':şlem Büyüklüğ 'İşlem Büyüklüü',
                            'pnlğü',
                            'pnl': 'Kar/Zarar ($': 'Kar/Zarar)',
                            'pnl ($)',
                            'pnl_percent': 'Kar/Z_percent': 'Kar/Zarar (%)'
                        })
arar (%)'
                        })
                        
                                               
                        # Renkli gö # Renkli göstersterim
                        def colorim
                        def color_p_pnl(val):
                           nl(val):
                            if ' if 'Kar/ZarKar/Zarar'ar' in val.name:
                                if val > in val.name:
                                if val > 0:
                                    0:
                                    return 'color: green'
 return 'color: green'
                                elif                                elif val < 0 val < 0:
                                   :
                                    return 'color: red'
 return 'color: red'
                            return                            return ''
                        
                        ''
                        
                        styled_df = display_df.style.format styled_df = display_df.style.format({
                           ({
                            'Giriş Fiyatı': '{:.2f}',
                            ' 'Giriş Fiyatı': '{:.2f}',
                            'Çıkış Fiyatı': '{:.2f}Çıkış Fiyatı': '{:.2f}',
                            'İş',
                            'İşlem Bülem Büyüklüğüyüklüğü': '{:.2f}': '{:.2f}',
',
                            'Kar/Zar                            'Kar/Zararar ($)': '{:.2 ($)': '{:.2ff}',
                            'Kar}',
                            'Kar/Z/Zarar (%)': '{arar (%)': '{:.2:.2f}%'
                       f}%'
                        }).apply }).apply(color_pnl, subset(color_pnl, subset=['Kar/Zarar ($=['Kar/Zarar ($)', 'Kar/Z)', 'Kar/Zararar (%))'])
                        
                        st.dataar (%))'])
                        
                       frame(styled_df, use_container_width=True)
                        
                        st.dataframe(styled_df, use_container_width=True # İstatistikler
                        st.subheader)
                        
                        # İstatistikler
                        st("📈 İşlem İstatistikleri")
                       .subheader("📈 İşlem İstatistikleri avg_profit = closed_t")
                        avg_profit = closed_trades['pnl'].meanrades['pnl'].mean()
                        max_profit = closed_t()
                        max_profit = closed_trades['pnl'].rades['pnl'].max()
                        max_loss = closed_tmax()
                        max_loss = closedrades['pnl'].min()
_trades['pnl'].                        
                        stat_col1min()
                        
                        stat_col1, stat_col2, stat, stat_col2, stat_col3 = st.columns(3)
                        with stat_col1:
_col3 = st.columns(3)
                        with stat_col1:
                                                       st st.metric.metric("Ortalama Kar/Zar("Ortalama Kar/Zarar",ar", f"${avg_profit f"${avg_profit:.2f}")
                        with:.2f}")
                        with stat_col stat_col2:
                            st.m2:
                            st.metric("Maksimum Kar", fetric("Maksimum Kar", f"${max"${max_profit:.2f_profit:.2f}")
                        with stat_col}")
                        with stat_col3:
                            st.metric3:
                            st.metric("M("Maksimum Zararaksimum Zarar", f", f"${max_loss:.2f}")
                        
                    else"${max_loss:.2f}")
                        
                    else:
                        st.info:
                        st.info("Kapanan işlem bulunamad("Kapanan işlem bulunamadı.")
               ı.")
                else:
                    st.info(" else:
                    st.info("Hiç işlem yapıHiç işlem yaplmadı.")
                    
            exceptılmadı.")
                    
            Exception as e:
                st.error except Exception as e:
               (f"Simülasyon s st.error(f"Simülasyon sırasında hata oırasında hata oluştu: {str(e)}luştu: {str")
    else:
        st(e)}")
    else:
.error("Veri yük        st.error("Veri yüklenemedi. Lütlenemedi. Lütfen önce kripto para vefen önce kripto tarih seçin.")

 para ve tarih seçin.")

# Bilgi
st.mark# Bilgi
st.markdown("down("---")
st.info(""---")
st.info("""
**"
**⚠️⚠️ U Uyarı:** Bu simülasyon syarı:** Bu simülasyon sadeceadece eğitim eğitim amaçlıdır. Gerç amaçlıdır. Gerçek tradingek trading için kullanmay için kullanmayın.ın. 
 
GeGeçmiş performans gelecek sonuçların garantisi değildir.
çmiş performans gelecek sonuçların garantisi değildir.
""")