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
    
    if data is not None:
        # Strateji uygula
        strategy = CryptoStrategy(initial_capital)
        data_with_indicators = strategy.calculate_indicators çekilemedi. Lütfen farklı bir tarih aralığı deneyin.")
                return None
            return data
        except Exception as e:
            st.error(f"Veri yüklenirken hata oluştu: {e}")
            return None

    data = load_data(symbol, start_date, end_date)
    
    if data is not None:
        # Strateji uygula
        strategy = CryptoStrategy(initial_capital)
        data_with_indicators = strategy.calculate_(data)
        data_with_signindicators(data)
als = strategy.generate_signals        data_with_signals = strategy.generate_signals(data_with_indic(data_with_indicators)
ators)
        
               
        # Grafik o # Grafik oluştluştur
        figur
        fig = make = make_subplots(
           _subplots(
            rows= rows=2, cols=1,
           2, cols=1,
            sub subplotplot_titles=_titles=('Fiyat ve S('Fiyat ve Sinyaller',inyaller', 'RS 'RSI'),
            verticalI'),
            vertical_spacing=_spacing=00.1,
            row_heights.1,
            row_heights=[0.7, =[0.7, 0.30.3]
       ]
        )
        
        # F )
        
        # Fiyat graiyat grafiğifiği
        fig
        fig.add_trace.add_trace(
            go(
            go.Candlest.Candlestick(
                xick(
                x=data.index=data.index,
                open=data,
                open=data['Open['Open'],
               '],
                high=data['High'],
 high=data['High'],
                low=data                low=data['Low'],
['Low'],
                close                close=data['=data['Close'],
               Close'],
                name='Fiy name='Fiyat'
at'
            ),
            row            ),
            row=1=1, col=1, col=1
       
        )
        
        )
        
        # Bollinger # Bollinger Bantları Bantları
        fig.add_trace(

        fig.add_trace(
            go.Scatter(
                           go.Scatter(
                x=data_with_ x=data_with_indicators.index,
indicators.index,
                y                y=data_with_indic=data_with_indicators['ators['BBBB_Upper'],
_Upper'],
                line=dict                line=dict(color='(color='rgba(255rgba(255, , 0, 00, 0,, 0.3) 0.3)'),
               '),
                name='BB Upper name='BB Upper'
           '
            ),
            row= ),
            row=1, col1, col=1
        )
        
=1
        )
        
        fig.add_trace        fig.add_trace(
           (
            go.Scatter(
                x=data_with go.Scatter(
                x=data_with_indicators_indicators.index,
                y=data_with_.index,
                y=data_with_indicators['indicators['BB_Lower'],
                line=dictBB_Lower'],
                line=dict(color='rgba(color='rgba(0, 255, (0, 255, 0, 00, 0.3)'),
.3)'),
                name='BB Lower',
                fill='                name='BB Lower',
                fill='tonexty'
           tonexty'
            ),
            row=1, col=1
 ),
            row=1, col=1
        )
        
               )
        
        # Alım sin # Alım sinyyallalleri
        long_signeri
        long_signalsals = data_with_signals = data_with_signals[data[data_with_signals['Signal_with_signals['Signal'] =='] == 1]
        if not 1]
        if not long_signals.empty:
            long_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x= fig.add_trace(
                go.Scatter(
                    x=long_signlong_signals.index,
                    y=long_signals['Closeals.index,
                    y=long_signals['Close'],
                    mode='markers'],
                    mode='markers',
                    marker=dict(color',
                    marker=dict(color='green', size=10='green', size=10,, symbol='triangle-up'),
                    name symbol='triangle-up'),
                    name='Long Sinyal'
='Long Sinyal'
                               ),
                row=1 ),
                row=1, col, col=1
           =1
            )
        
        # Satım )
        
        # Satım sin sinyalleri
       yalleri
        short_sign short_signals = data_with_signalsals = data_with_signals[data_with_signals['[data_with_signals['SignalSignal'] == -1]
'] == -1]
        if        if not short_signals not short_signals.empty:
.empty:
            fig.add_t            fig.add_trace(
race(
                go.Scatter                go.Scatter(
                    x=short_sign(
                    x=short_signals.index,
                    y=als.index,
                    y=shortshort_signals['Close'],
_signals['Close'],
                    mode='markers',
                    marker                    mode='markers',
=dict(color='red',                    marker=dict(color='red', size=10, symbol size=10, symbol='='triangle-down'),
                    nametriangle-down'),
                    name='='Short Sinyal'
                ),
Short Sinyal'
                ),
                row=1,                row=1, col col=1
            )
=1
            )
        
        # RSI
        
        # RSI
        fig.add_trace(
        fig.add_trace(
                       go.Scatter(
                x=data go.Scatter(
                x=data_with_indic_with_indicators.index,
                y=data_with_indicators['RSI'],
                line=dict(color='purple'),
                name='RSators.index,
                y=data_with_indicators['RSI'],
                line=dict(color='purple'),
                name='RSI'
            ),
            rowI'
            ),
            row=2, col=1=2, col=1
        )
        

        )
        
        #        # RSI sevi RSI seviyeleryeleri
       i
        fig.add_ fig.add_hline(y=hline(y=70, line70, line_dash="d_dash="dash",ash", line_color="red line_color="red", row=2", row=2, col, col=1)
       =1)
        fig.add fig.add_hline(y=30,_hline(y=30, line_dash=" line_dash="dash", linedash", line_color="_color="green",green", row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=_color="gray", row=2, col=2, col=1)
        
        fig.update_layout(
            height1)
        
        fig.update_layout(
           =600,
            showlegend height=600,
            showlegend=True,
            xaxis_r=True,
            xaxis_rangeslider_visible=Falseangeslider_visible=False
        )
        
        st.plot
        )
        
        stly_chart(fig, use.plotly_chart(fig, use_container_width=True_container_width=True)

)

with col2:
    stwith col2:
    st.sub.subheader("🎯 Strateheader("🎯 Stratejiji Bilgileri")
 Bilgileri")
    
    
    st.markdown(""    st.markdown("""
    **Kullanı"
    **Kullanılan Strateji:**
lan Strateji:**
       - RSI + MAC - RSI + MACDD + Bollinger + Bollinger Bant Bantları
    - Çokluları
    - Çoklu zaman zaman periyodu anal periyodu analizi
    
    **Long Koizi
    
    **Long Koşullşulları:**
   arı:**
    - - RSI < 35 (Oversold)
    - MACD > MACD Signal
    - Fiy RSI < 35 (Oversold)
    - MACD > MACD Signal
at BB Alt Bandı altında
    - Fiyat BB Alt Bandı altında
    -    - EMA( EMA(99) > EMA(21) > EMA(21)
)
    
    
    **Short    **Short Koşulları:**
    - RSI > 65 (Overbought)  
    - Koşulları:**
    - RSI > 65 (Overbought)  
 MACD < MACD Signal
    - Fiy    - MACD < MACD Signal
    - Fat BB Üst Bandı üiyat BB Üst Bandı üstünde
    -stünde
    - EMA(9) EMA(9) < EMA(21)
    
 < EMA(21)
    
       **Risk Yönet **Risk Yönetimiimi:**
    - %:**
    - %55 Stop Loss
    - Stop Loss
    - %10 %10 Take Profit
    - Take Profit
    - %10 Pozisyon Büyüklüğü
 %10 Pozisyon Büyüklüğü
    ""    """)

# Simül")

# Simülasyon butasyon butonu
st.markdownonu
st.markdown("---("---")
st.sub")
st.subheader("🚀 Backtest Simülasyonu")

if st.button("header("🚀 Backtest Simülasyonu")

if st.button("🎯 Backtest Simülasyonunu🎯 Backtest Simülasyonunu Başlat Başlat", type", type="primary"):
    if data is not None:
="primary"):
    if data is not None:
        with st.spinner("Simülasyon çalış        with st.spinner("Simülasyon çalışıyor..."):
            start_time =ıyor..."):
            start_time = time.time()
            
            # time.time()
            
            # İlerleme çubuğu İlerleme çubuğu

            progress_bar = st            progress_bar = st.pro.progress(0)
           gress(0)
            status_text = st.empty()
 status_text = st.empty()
            
            # Stratejiyi ç            
            # Stratejialıştır
yi çalıştır
            strategy = CryptoStrategy            strategy = CryptoStrategy(initial(initial_capital_capital)
)
            data_with_indicators = strategy            data_with_indicators = strategy.calculate.calculate_indicators(data)
_indicators(data)
            data            data_with_signals =_with_signals = strategy.generate strategy.generate_signals(data_with_signals(data_with_indicators)
_indicators)
            
            status_text.text("Strateji            
            status_text.text("St backtest edrateji backtest ediliyor...")
            resultsiliyor...")
            results = strategy.back = strategy.backtest_strategy(data_withtest_strategy(data_with_signals_signals, progress_bar)
, progress_bar)
            
            end_time = time.time()
            calculation_time = end_time            
            end_time = time.time()
            calculation_time = end_time - start - start_time
            
            # İ_time
            
            # İlerlerlemeleme ç çubuğunu tamamla
ubuğunu tamamla
            progress_bar            progress_bar.progress(1.0)
            
.progress(1.0)
            
            st.success            st.success(f"✅ Simülasyon(f"✅ Simülasyon hesa hesaplaması {plaması {calculation_timecalculation_time:.2f}:.2f} sani saniye içinde tamye içinde tamamlandamlandı!")
            
           ı!")
            
            # Son # Sonuçları göuçları göster
ster
            st.subheader            st.subheader("📊 Simülasyon Son("📊 Simülasyon Sonuuçları")
            
            colçları")
            
            col1,1, col2 col2, col3,, col3, col4 col4 = st.columns( = st.columns(4)
4)
            
            with col            
            with col1:
                st1:
                st.metric(
.metric(
                    "Ba                    "Başlangıçşlangıç Sermay Sermayesi",
                    fesi",
                    f"${results['initial"${results['_capital']:,.2finitial_capital']:,.2f}"
                )
            
            with col}"
                )
            
            with col22:
                st.m:
                st.metric(
etric(
                    "Son Serm                    "Son Sermaye",aye", 
                    f"${results['final_capital']: 
                    f"${results['final_capital']:,.2f}",
                   ,.2f}",
                    delta=f"{results['total delta=f"{results['total_return']_return']:+.2f}%"
               :+.2f}%"
                )
 )
            
            with col3:
            
            with col3:
                st.metric(
                    "Toplam İş                st.metric(
                    "Toplam İlem",
                    f"{resultsşlem",
                    f"{results['total_trades']}"
['total_trades']}"
                )
            
            with col4:
                )
            
            with col                st.metric(
                   4:
                st.metric(
                    "Win Rate",
                    f "Win Rate",
                    f"{results['win_rate']"{results['win_rate']:.1f}%"
:.1f}%"
                )
            
                           )
            
            # Equity curve
            if # Equity curve
            if not not results['equity_ results['equity_curve'].empty:
                stcurve'].empty:
                st.subheader("📈 Equity.subheader("📈 Equity Curve")
                equity_fig Curve")
                equity_fig = go.Figure()
                
                = go.Figure()
                
 equity_fig.add_trace(
                equity_fig.add_t                    go.Scatterrace(
                    go.Scatter(
(
                                               x=results x=results['equ['equity_curve']['Date'],
ity_curve']['Date'],
                        y=results['equity_curve                        y=results['equity_curve']['Equity'],
                       ']['Equity'],
                        line=dict(color='blue'),
 line=dict(color='blue'),
                        name='Portföy De                        name='Portföy Değğeri'
                    )
eri'
                    )
                )
                
                equity                )
                
                equity_fig.add_fig.add_hline(
                    y=initial_capital,
                    line_hline(
                    y=initial_capital,
                    line_dash="dash_dash="dash",
                    line_color="red",
                    annotation",
                    line_color="red",
                   _text="Başlang annotation_text="Başlangıç Sermayesiıç Sermayesi"
                )
                
                equity"
                )
                
                equity_fig.update_layout(
                   _fig.update_layout(
                    height=400,
                    show height=400,
                    showlegend=True,
                    xaxis_title="Tlegend=True,
                    xaxis_title="Tarih",
                    yaxis_title="Portföyarih",
                    yaxis_title="Portföy Değeri (USD)"
 Değeri (USD)"
                )
                )
                
                st.plot                
                st.plotly_chly_chart(equityart(equity_fig, use_fig, use_container_width=True)
            
            # İ_container_width=True)
            
           şlem detayları
 # İşlem detayları
            if results['trades']:
                st            if results['trades']:
                st.subheader("📋 İ.subheader("📋 İşlem Detşlem Detayları")
                
                trades_df = pd.DataFrameayları")
                
                trades_df = pd.DataFrame(results['trades'])
                # Sadece kapan(results['trades'])
                # Sadece kapanan işan işlemlerilemleri göster
 göster
                closed_trades =                closed_trades = trades_df trades_df[trades[trades_df['exit_time'].notna()]
_df['exit_time'].notna()]
                
                               
                if not closed_trades.empty:
                    closed_t if not closed_trades.empty:
                    closed_trades['rades['pnl_percent']pnl_percent'] = (closed_trades['pnl'] = (closed_trades['pn / closed_trades['entryl'] / closed_trades['entry_capital']) * _capital']) * 100100

                    
                                       
                    # Renkli P # Renkli PNL sNL sütunuütunu
                    def color
                    def color_pnl(val):
                        color = 'green' if val_pnl(val):
                        color > 0 else 'red = 'green' if val > 0 else 'red' if' if val < val < 0 0 else else 'gray'
 'gray'
                                               return f'color: {color}'
                    
                    styled_trades = return f'color: {color}'
                    
                    styled closed_trades.style.format({
                        'entry_price': '{:.2_trades = closed_trades.style.format({
                        'entry_price': '{:.2f}',
                        'exit_price': '{f}',
                        'exit_price': '{:.2f:.2f}', 
                        '}', 
                        'entry_centry_capital': '{:.2apital': '{:.2ff}',
                        'pnl}',
                        'pnl': '{': '{:.2f}',
                        'pnl_per:.2f}',
                        'pnl_percentcent': '{:.': '{:.2f}%2f}%'
                    }).'
                    }).applymap(color_pnlapplymap(color_pnl, subset=['pnl', 'pn, subset=['pnl', 'pnl_percent'])
                    
l_percent'])
                    
                                       st.dataframe(styled st.dataframe(styled_t_trades, use_container_width=Truerades, use_container_width=True)
                else:
                    st.info)
                else:
                    st("Kapanan iş.info("Kapanan ilem bulunamadı.")
şlem bulunamadı.")
            else            else:
                st.info:
                st.info("Hiç i("Hiç işşlem yaplem yapılmadı.")

    else:
       ılmadı.")

    else:
        st st.error("Veri y.error("Veri yüklenemediüklenemedi. Lüt. Lütfen simfen simülasyon ayarlarınıülasyon ayarlarını kontrol edin.")

 kontrol edin.")

# Bilgi
# Bilgi
st.markdown("---")
st.infost.markdown("---")
st.info("""
**⚠️("""
**⚠️ Uyar Uyarı:**ı:** Bu simülasyon Bu simülasyon sade sadece ece eğitim amaçğitim amaçlıdır. Gerçek trading için klıdır. Gerullanmayın. 
Geçek trading için kullanmayın. 
Geçmiş performansçmiş performans gelecek gelecek sonuç sonuçların garantların garantisi değildisi değildir.
""")
```ir.
""")