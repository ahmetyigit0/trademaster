"""Core financial calculations for the trading journal."""
from typing import Optional


def calculate_avg_entry(entries: list[dict]) -> float:
    total_weight = sum(e.get("weight", 0) for e in entries)
    if total_weight == 0:
        return 0.0
    return sum(e["price"] * e["weight"] for e in entries) / total_weight


def calculate_position_size(capital, risk_pct, avg_entry, stop_loss):
    if avg_entry <= 0 or stop_loss <= 0 or capital <= 0:
        return {}
    risk_amount = capital * (risk_pct / 100)
    price_diff = abs(avg_entry - stop_loss)
    if price_diff == 0:
        return {}
    risk_per_unit = price_diff / avg_entry
    full_capital_risk_pct = risk_per_unit * 100
    recommended_size = (risk_amount / price_diff) * avg_entry
    recommended_size = min(recommended_size, capital)
    can_use_full = full_capital_risk_pct <= risk_pct
    return {
        "risk_amount": risk_amount,
        "recommended_size": recommended_size,
        "full_capital_risk_pct": full_capital_risk_pct,
        "full_loss": risk_per_unit * capital,
        "can_use_full": can_use_full,
        "risk_per_unit": risk_per_unit,
    }


def calculate_rr(avg_entry, stop_loss, take_profits, direction="LONG"):
    if not take_profits or avg_entry <= 0 or stop_loss <= 0:
        return None
    risk = (avg_entry - stop_loss) if direction == "LONG" else (stop_loss - avg_entry)
    if risk <= 0:
        return None
    total_w = sum(t.get("weight", 1) for t in take_profits)
    if total_w == 0:
        return None
    weighted_rr = 0.0
    for t in take_profits:
        w = t.get("weight", 1) / total_w
        reward = (t["price"] - avg_entry) if direction == "LONG" else (avg_entry - t["price"])
        if reward > 0:
            weighted_rr += (reward / risk) * w
    return round(weighted_rr, 2) if weighted_rr > 0 else None


def calculate_r_multiple(pnl, risk_amount):
    if risk_amount <= 0:
        return 0.0
    return round(pnl / risk_amount, 2)


def calculate_position_heat(position_size, capital, risk_per_unit):
    if capital <= 0:
        return 0.0
    return round(position_size * risk_per_unit / capital * 100, 2)


def calculate_ev(win_rate, avg_win, avg_loss):
    return round((win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss)), 2)


def rr_color(rr):
    if rr is None:
        return "#64748b"
    if rr < 1:
        return "#ef4444"
    if rr < 2:
        return "#f59e0b"
    return "#10b981"


def format_pnl(pnl):
    if pnl > 0:
        return f"+${pnl:,.2f}"
    elif pnl < 0:
        return f"-${abs(pnl):,.2f}"
    return "$0.00"
