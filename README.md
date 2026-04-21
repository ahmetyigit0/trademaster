# TradeVault — Trading Journal

Profesyonel bir trading journal ve pozisyon hesaplayıcı.

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma

```bash
streamlit run app.py
```

Uygulama `http://localhost:8501` adresinde açılır.

## Özellikler

- 📊 **Stats Bar** — Win rate, toplam PnL, profit factor
- ➕ **Pozisyon Ekleme** — Tek veya parçalı entry, risk hesaplama
- 📐 **Risk Yönetimi** — Otomatik pozisyon büyüklüğü önerisi
- 📈 **Aktif Pozisyonlar** — Detaylı görüntüleme, kapatma/silme
- 📁 **Kapalı İşlemler** — LONG/SHORT ve WIN/LOSS filtresi
- 💾 **JSON Kalıcı Depolama** — Sayfa yenilemede veri korunur

## Veri

Veriler `trades_data.json` dosyasında saklanır.
