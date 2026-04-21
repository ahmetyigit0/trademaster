import json
import os
import uuid
from datetime import datetime

DATA_FILE = os.path.join(os.path.dirname(__file__), "trades_data.json")

DEFAULT_DATA = {
    "active_positions": [],
    "closed_trades": []
}


def load_data() -> dict:
    """Load trading data from JSON file."""
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return DEFAULT_DATA.copy()
    return DEFAULT_DATA.copy()


def save_data(data: dict) -> None:
    """Persist trading data to JSON file."""
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def generate_id() -> str:
    return str(uuid.uuid4())[:8]


def calculate_avg_entry(entries: list[dict]) -> float:
    """
    entries: list of {"price": float, "weight": float}
    Returns weighted average entry price.
    """
    total_weight = sum(e["weight"] for e in entries if e.get("price") and e.get("weight"))
    if total_weight == 0:
        return 0.0
    return sum(e["price"] * e["weight"] for e in entries if e.get("price") and e.get("weight")) / total_weight


def calculate_position_size(capital: float, risk_pct: float, entry: float, stop: float, direction: str) -> dict:
    """
    Calculate recommended position size based on risk parameters.
    Returns dict with position_size, risk_amount, full_risk_pct, can_full.
    """
    if entry == 0 or stop == 0 or capital == 0:
        return {}

    risk_amount = capital * (risk_pct / 100)
    
    if direction == "LONG":
        risk_per_unit = entry - stop
    else:
        risk_per_unit = stop - entry

    if risk_per_unit <= 0:
        return {"error": "Stop loss yönü pozisyon yönüyle uyuşmuyor."}

    units = risk_amount / risk_per_unit
    position_size = units * entry

    full_risk_pct = (risk_per_unit / entry) * 100  # % risk if full capital used

    can_full = full_risk_pct <= risk_pct

    return {
        "position_size": position_size,
        "risk_amount": risk_amount,
        "full_risk_pct": full_risk_pct,
        "units": units,
        "risk_per_unit": risk_per_unit,
        "can_full": can_full,
    }


def calculate_rr(pos: dict, pnl: float) -> float:
    """Calculate Risk/Reward ratio from a closed position."""
    try:
        avg = pos.get("avg_entry", 0)
        stop = pos.get("stop_loss", 0)
        risk_amount = pos.get("risk_amount", 0)
        if risk_amount and risk_amount != 0:
            return round(pnl / risk_amount, 2)
        return 0.0
    except Exception:
        return 0.0


def get_position_index(active_positions: list, pos_id: str) -> int:
    for i, p in enumerate(active_positions):
        if p["id"] == pos_id:
            return i
    return -1


def format_trade_header(pos: dict, index: int) -> str:
    """Format the collapsible position header string."""
    symbol = pos.get("symbol", "?")
    direction = pos.get("direction", "?")
    rr = pos.get("rr", 0)
    pnl = pos.get("pnl", None)
    
    if pnl is not None:
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
    else:
        pnl_str = "Open"
    
    rr_str = f"1:{abs(rr):.1f}" if rr else "—"
    return f"#{index+1} · {symbol} · {direction} · RR {rr_str} · {pnl_str}"
