# Price Action – Streamlit App

Bu repo, **destek/direnç**, **mum formasyonları**, **price acceptance / rejection** ve hacim “spike + wick” yorumunu bir araya getirerek (swing & trend‐following) basit girişler üretir.

## Özellikler
- S/R bantları: yerel tepe/dip kümeleri ile otomatik yakınsayan bantlar
- Candlestick taraması: engulfing, hammer/hanging man, harami, morning/evening star, doji, pin bar
- Acceptance vs Rejection algısı (bant içinde iğne + bant dışı kapanış veya üst/alt bölgede kalıcılık)
- Hacim “spike + wick” sınıflaması (bull/bear/neutral)
- Basit R/R filtresiyle bar-by-bar hedef/stop backtesti
- Plotly candlestick + S/R bölgeleri + işaretler

## Hızlı Başlangıç
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Parametreler
- `Pivot order`: S/R tespiti için tepe/dip mesafesi
- `band %`: yakın seviyelerin tek banda birleştirme toleransı
- `tol %`: banda yakın sayılma toleransı
- `vol_win / vol_z`: hacim spike penceresi ve z-skor eşiği
- `min_rr`: minimum risk/ödül

## Notlar
- Bu çalışma eğitim amaçlıdır; al/sat tavsiyesi değildir.
- S/R ve hacim analizi sezgisel/heuristic kurallar içerir; gerçek para riske etmeden önce kapsamlı test yapın.
