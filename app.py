import yfinance as yf
import pandas as pd

# 1. Hisse Senedi Seçimi ve Veri Çekme
ticker_symbol = "AAPL"
start_date = "2022-01-01"
end_date = "2024-01-01" # Son 2 yıl

try:
    # yfinance ile veriyi çekme
    data = yf.download(ticker_symbol, start=start_date, end=end_date)

    if data.empty:
        print(f"{ticker_symbol} için veri bulunamadı.")
    else:
        # 2. Strateji Göstergelerinin Hesaplanması (EMA 20 ve SMA 50)
        
        # Kısa dönem: 20 günlük Üstel Hareketli Ortalama (EMA)
        data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
        
        # Uzun dönem: 50 günlük Basit Hareketli Ortalama (SMA)
        data['SMA_50'] = data['Close'].rolling(window=50).mean()

        # İlk birkaç satırı gösterelim
        print(f"\n--- {ticker_symbol} Verileri ve Hesaplanan Göstergeler (İlk 5 Satır) ---")
        print(data[['Close', 'EMA_20', 'SMA_50']].head())
        
        # Hesaplamalar yapıldı! Şimdi sinyal üretimine geçebiliriz.

except Exception as e:
    print(f"Veri çekilirken bir hata oluştu: {e}")
