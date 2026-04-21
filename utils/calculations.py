"""Core financial calculations for the trading journal."""
from typing import Optional


def calculate_avg_entry(entries: list[dict]) -> float:
    """Weighted average entry price."""
    total_weight = sum(e.get("weight", 0) for e in entries)
    if total_weight == 0:
        return 0.0
    return sum(e["price"] * e["weight"] for e in entries) / total_weight


def calculate_position_size(
    capital: float,
    risk_pct: float,
    avg_entry: float,
    stop_loss: float,
) -> dict:
    """
    Returns:
        risk_amount: $ risked at current capital
        recommended_size: $ position size to stay within risk_pct
        full_capital_risk_pct: % risk if entire capital is deployed
        can_use_full: whether full capital fits within risk budget
        risk_per_unit: fraction risked per unit (e.g. 0.02 = 2%)
    """
    if avg_entry <= 0 or stop_loss <= 0 or capital <= 0:
        return {}

    risk_amount = capital * (risk_pct / 100)
    price_diff = abs(avg_entry - stop_loss)
    if price_diff == 0:
        return {}

    risk_per_unit = price_diff / avg_entry  # as a fraction
    full_capital_risk_pct = risk_per_unit * 100

    recommended_size = risk_amount / risk_per_unit  # in $
    recommended_size = min(recommended_size, capital)  # cap at capital

    can_use_full = full_capital_risk_pct <= risk_pct

    return {
        "risk_amount": risk_amount,
        "recommended_size": recommended_size,
        "full_capital_risk_pct": full_capital_risk_pct,
        "can_use_full": can_use_full,
        "risk_per_unit": risk_per_unit,
    }


def calculate_rr(avg_entry: float, stop_loss: float, take_profits: list[dict]) -> Optional[float]:
    """Risk-to-reward ratio based on first (or weighted avg) TP."""
    if not take_profits or avg_entry <= 0 or stop_loss <= 0:
        return None
    risk = abs(avg_entry - stop_loss)
    if risk == 0:
        return None
    # Weighted average TP
    total_w = sum(t.get("weight", 1) for t in take_profits)
    avg_tp = sum(t["price"] * t.get("weight", 1) for t in take_profits) / total_w
    reward = abs(avg_tp - avg_entry)
    return round(reward / risk, 2)


def format_pnl(pnl: float) -> str:
    if pnl > 0:
        return f"+${pnl:,.2f}"
    elif pnl < 0:
        return f"-${abs(pnl):,.2f}"
    return "$0.00"
