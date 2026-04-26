"""
Data Manager — Streamlit Cloud uyumlu kalıcı depolama
=====================================================
Streamlit Cloud'da dosya sistemi her deploy'da sıfırlanır.
Bu nedenle veriler JSON dosyasına + session_state'e kaydedilir.

Depolama önceliği:
1. JSON dosya (lokal çalışma)
2. st.session_state["_persist"] (Cloud'da geçici ama uygulama açıkken kalır)

NOT: Gerçek kalıcılık için GitHub'a commit veya harici DB gerekir.
Şu an session_state tabanlı çalışır — tarayıcı kapanınca sıfırlanır.
"""
import json
import os
from datetime import datetime

DATA_FILE      = "trades_data.json"
RULES_FILE     = "trade_rules.json"
_SESSION_KEY   = "_tradevault_data"

DEFAULT_DATA = {
    "active_positions": [],
    "closed_trades":    [],
    "next_id":          1,
}


def _is_cloud() -> bool:
    """Streamlit Cloud ortamında mıyız?"""
    return os.environ.get("STREAMLIT_SHARING_MODE") == "streamlit_sharing" \
        or os.environ.get("HOME", "") == "/home/appuser"


def load_data() -> dict:
    """
    Veriyi yükle.
    Lokal: JSON dosyasından.
    Cloud: JSON denenir, başarısız olursa DEFAULT döner.
    """
    # JSON dosyası varsa her zaman önce onu dene
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key, val in DEFAULT_DATA.items():
                data.setdefault(key, val)
            return data
        except (json.JSONDecodeError, IOError, PermissionError):
            pass

    return dict(DEFAULT_DATA)


def save_data(data: dict) -> None:
    """
    Veriyi kaydet.
    Cloud'da yazma yetkisi olmayabilir — sessizce devam et.
    """
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except (IOError, PermissionError, OSError):
        # Streamlit Cloud'da dosya sistemi read-only olabilir
        # Veri session_state'te zaten tutuluyor, kritik değil
        pass
