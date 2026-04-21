import json
import os
from datetime import datetime

DATA_FILE = "trades_data.json"

DEFAULT_DATA = {
    "active_positions": [],
    "closed_trades": [],
    "next_id": 1,
}


def load_data() -> dict:
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Ensure keys exist for backwards compat
            for key, val in DEFAULT_DATA.items():
                data.setdefault(key, val)
            return data
        except (json.JSONDecodeError, IOError):
            pass
    return dict(DEFAULT_DATA)


def save_data(data: dict) -> None:
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
