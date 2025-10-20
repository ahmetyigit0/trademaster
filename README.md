# Streamlit Strategy Starter

Bu depo **EMA/MACD/RSI/Bollinger** içeren basit bir strateji uygulamasıdır.

## Yerelde Çalıştırma
```bash
pip install -r requirements.txt
streamlit run app.py
```
Açılır: http://localhost:8501

## Streamlit Community Cloud (ücretsiz)
1. Bu klasörü bir GitHub reposu yap.
2. https://share.streamlit.io adresinden **New app** → repo → branch → `app.py`
3. Deploy et. (requirements.txt otomatik kurulacak)
4. Sorun olursa Logs sekmesine bak.

## Render.com (ücretsiz plan)
- Repo bağla → "Web Service"
- Runtime: Python
- Start Command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
- `Procfile` ekleyebilirsin (Render otomatik algılar).

## Hugging Face Spaces
- Space Type: **Streamlit**
- Dosyalar: `app.py`, `requirements.txt`
- Deploy → otomatik.

## İpuçları
- YFinance çağrılarını azaltmak için kısa period seç (3mo/6mo).
- `@st.cache_data` veri çekiminde timeout sorunlarını hafifletir.
- CSV ile manuel veri yüklemek her ortamda çalışır.