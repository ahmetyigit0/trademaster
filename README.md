# ⬡ TRADEVAULT — Trading Journal & Position Calculator

Professional-grade Streamlit trading journal with dark theme, position calculator, and persistent JSON storage.

## 🚀 Kurulum & Çalıştırma

```bash
# 1. Gerekli paketi kur
pip install streamlit

# 2. Uygulamayı başlat
streamlit run app.py
```

Tarayıcında otomatik olarak `http://localhost:8501` açılır.

## 📁 Dosya Yapısı

```
trading_journal/
├── app.py            → Ana uygulama (sayfa yapısı, stats, routing)
├── components.py     → UI bileşenleri (formlar, kartlar)
├── utils.py          → Hesaplama ve veri yönetimi
├── style.css         → Dark theme CSS
├── requirements.txt  → Bağımlılıklar
└── trades_data.json  → Otomatik oluşturulur (verileriniz burada saklanır)
```

## ✨ Özellikler

| # | Özellik |
|---|---------|
| 1 | Dark theme, kart tabanlı modern UI |
| 2 | Pozisyon ekleme (symbol, yön, sermaye, risk%) |
| 3 | Tek / parçalı giriş (3 entry + ağırlık) |
| 4 | Otomatik risk & pozisyon büyüklüğü hesaplama |
| 5 | Tek / parçalı TP sistemi |
| 6 | Aktif pozisyonlar (expand/collapse) |
| 7 | Pozisyon kapatma (PnL + yorum) |
| 8 | Kapalı işlemler geçmişi |
| 9 | LONG/SHORT ve WIN/LOSS filtreleri |
| 10 | JSON ile kalıcı veri saklama |

## 💡 Risk Hesaplama Mantığı

- Kullanıcı: Sermaye, Risk%, Entry, Stop Loss girer
- Sistem hesaplar: `Risk Tutarı = Sermaye × Risk%`
- `Risk/Unit = |Entry - Stop|`
- `Önerilen Boyut = Risk Tutarı / Risk/Unit × Entry`
- Full sermaye riski ≤ hedef risk ise → "Full ile girebilirsin" mesajı
- Aksi halde → önerilen boyutu gösterir + "Uygula" butonu

## 🎨 Tema

- **Font**: Space Mono (başlıklar) + DM Sans (metin)
- **Renk**: `#0a0d12` arka plan, `#00d4ff` accent, `#00e676` yeşil, `#ff4757` kırmızı
- **Mobil uyumlu** responsive grid
