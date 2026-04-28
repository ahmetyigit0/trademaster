"""
Data Manager — Kalıcı Depolama
================================
Streamlit Cloud'da veriler her uyku/restart'ta sıfırlanır.
Çözüm: st.secrets'taki GitHub Gist token ile veriyi Gist'e yaz/oku.

Kurulum (Streamlit Cloud Secrets):
    [storage]
    gist_token = "ghp_xxxxxxxxxxxx"   # GitHub → Settings → Tokens → gist scope
    gist_id    = ""                    # İlk çalıştırmada otomatik oluşturulur

Gist token yoksa: sadece session_state'te tutar (uygulama açıkken kalır).
"""

import json
import os
import streamlit as st
from datetime import datetime

DATA_FILE  = "trades_data.json"
RULES_FILE = "trade_rules.json"

DEFAULT_DATA = {
    "active_positions": [],
    "closed_trades":    [],
    "drafts":           [],
    "next_id":          1,
}


# ── Gist helpers ─────────────────────────────────────────────────────────────

def _get_gist_cfg():
    """secrets.toml'dan gist config al."""
    try:
        token   = st.secrets["storage"]["gist_token"]
        gist_id = st.secrets["storage"].get("gist_id", "")
        return token, gist_id
    except Exception:
        return None, None


def _gist_read(token: str, gist_id: str) -> dict | None:
    try:
        import requests
        r = requests.get(
            f"https://api.github.com/gists/{gist_id}",
            headers={"Authorization": f"token {token}",
                     "Accept": "application/vnd.github.v3+json"},
            timeout=8,
        )
        if r.status_code == 200:
            files = r.json().get("files", {})
            f = files.get("tradevault_data.json")
            if f:
                return json.loads(f.get("content", "{}"))
    except Exception:
        pass
    return None


def _gist_write(token: str, gist_id: str, data: dict) -> str:
    """Yaz, gist_id yoksa oluştur. Yeni id döner."""
    try:
        import requests
        payload = {
            "description": "TradeVault Journal Data",
            "public": False,
            "files": {
                "tradevault_data.json": {
                    "content": json.dumps(data, ensure_ascii=False, indent=2)
                }
            }
        }
        headers = {"Authorization": f"token {token}",
                   "Accept": "application/vnd.github.v3+json"}
        if gist_id:
            r = requests.patch(f"https://api.github.com/gists/{gist_id}",
                               json=payload, headers=headers, timeout=8)
        else:
            r = requests.post("https://api.github.com/gists",
                              json=payload, headers=headers, timeout=8)
        if r.status_code in (200, 201):
            return r.json().get("id", gist_id)
    except Exception:
        pass
    return gist_id


# ── Public API ────────────────────────────────────────────────────────────────

def load_data() -> dict:
    """
    Öncelik:
    1. Gist (kalıcı, secrets varsa)
    2. Yerel JSON dosyası
    3. DEFAULT_DATA
    """
    # 1. Gist
    token, gist_id = _get_gist_cfg()
    if token and gist_id:
        gist_data = _gist_read(token, gist_id)
        if gist_data:
            for k, v in DEFAULT_DATA.items():
                gist_data.setdefault(k, v)
            return gist_data

    # 2. Yerel JSON
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in DEFAULT_DATA.items():
                data.setdefault(k, v)
            return data
        except Exception:
            pass

    return dict(DEFAULT_DATA)


def save_data(data: dict) -> None:
    """
    Öncelik:
    1. Gist (kalıcı)
    2. Yerel JSON
    """
    # 1. Gist
    token, gist_id = _get_gist_cfg()
    if token:
        new_id = _gist_write(token, gist_id, data)
        # Yeni oluşturulduysa id'yi kaydet (session_state ile taşı)
        if new_id and new_id != gist_id:
            if "gist_id_new" not in st.session_state:
                st.session_state["gist_id_new"] = new_id
                st.info(
                    f"✅ Gist oluşturuldu: `{new_id}`\n\n"
                    f"Streamlit Cloud Secrets'a ekle:\n"
                    f"```\n[storage]\ngist_token = \"...\"\ngist_id = \"{new_id}\"\n```"
                )

    # 2. Yerel JSON (yedek)
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
