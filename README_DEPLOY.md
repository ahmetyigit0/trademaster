# TradeVault — Streamlit Cloud Deploy Kılavuzu

## Neden Streamlit Cloud?
Türkiye'den Binance API'ye erişim kısıtlıdır (451 hatası).
Streamlit Cloud, ABD/EU IP kullandığı için bu kısıtlama geçerli değildir.

---

## Adım 1: GitHub Repository Oluştur

```bash
# Projeyi GitHub'a yükle
git init
git add .
git commit -m "TradeVault initial commit"
git branch -M main
git remote add origin https://github.com/KULLANICI_ADI/tradevault.git
git push -u origin main
```

> ⚠️ `.gitignore` dosyası `secrets.toml` ve `*.json` dosyalarını zaten hariç tutuyor.
> API anahtarların GitHub'a gitmez.

---

## Adım 2: Streamlit Cloud Hesabı

1. [share.streamlit.io](https://share.streamlit.io) adresine git
2. GitHub hesabınla giriş yap
3. **"New app"** butonuna tıkla

---

## Adım 3: App Ayarları

| Alan | Değer |
|------|-------|
| Repository | `kullanici_adi/tradevault` |
| Branch | `main` |
| Main file path | `app.py` |

**"Advanced settings"** → Python version: **3.11**

---

## Adım 4: Secrets Ayarla (ÖNEMLİ)

Deploy sayfasında **"Advanced settings"** → **"Secrets"** bölümüne şunu yapıştır:

```toml
[binance]
api_key = "BURAYA_DEMO_API_KEY"
api_secret = "BURAYA_DEMO_API_SECRET"
```

### Demo Trading API Key nasıl alınır?

1. [binance.com/en/my/settings/api-management](https://www.binance.com/en/my/settings/api-management) → **Create API**
2. **System Generated** seç → isim ver (örn: "TradeVault")
3. API Key sayfasında **"Enable Demo Trading"** kutucuğunu işaretle ✅
4. API Key ve Secret'ı kopyala → yukarıdaki secrets alanına yapıştır

---

## Adım 5: Deploy

**"Deploy!"** butonuna bas. ~2 dakika bekle.

Uygulama şu adreste yayınlanır:
`https://KULLANICI-tradevault-app-XXXXX.streamlit.app`

---

## Adım 6: TradeBot'u Test Et

1. **🤖 TradeBot** sekmesine git
2. **API Key bölümünü** aç → secrets'tan otomatik doldurulacak
3. Sembol seç → Strateji ayarla → **Botu Başlat**

---

## Veri Kalıcılığı

> ⚠️ Streamlit Cloud'da JSON dosyaları her deploy'da sıfırlanır.
>
> Trade journal verilerini korumak için:
> - **Yedek / Arşiv** sekmesinden düzenli JSON export al
> - Her büyük güncelleme öncesi backup indir

---

## Sorun Giderme

| Hata | Çözüm |
|------|-------|
| `ModuleNotFoundError` | requirements.txt'i kontrol et |
| `451 Restricted location` | Secrets'taki API key'i kontrol et |
| `AuthenticationError` | Demo Trading enabled mi? API key doğru mu? |
| Uygulama yavaş | Normal — ilk açılış ~30s sürebilir |

---

## Güncelleme

```bash
# Değişiklikleri push et → Streamlit Cloud otomatik redeploylar
git add .
git commit -m "update"
git push
```
