import streamlit as st
from utils.calculations import format_pnl


def render_stats_bar(data: dict):
    closed = data.get("closed_trades", [])
    active = data.get("active_positions", [])

    total_trades = len(closed)
    wins = [t for t in closed if t.get("pnl", 0) > 0]
    losses = [t for t in closed if t.get("pnl", 0) <= 0]
    win_rate = (len(wins) / total_trades * 100) if total_trades else 0
    total_pnl = sum(t.get("pnl", 0) for t in closed)
    avg_win = (sum(t["pnl"] for t in wins) / len(wins)) if wins else 0
    avg_loss = (sum(t["pnl"] for t in losses) / len(losses)) if losses else 0
    profit_factor = (sum(t["pnl"] for t in wins) / abs(sum(t["pnl"] for t in losses))) if losses and sum(t["pnl"] for t in losses) < 0 else float("inf") if wins else 0

    pnl_color = "#10b981" if total_pnl >= 0 else "#ef4444"
    pf_display = f"{profit_factor:.2f}" if profit_factor != float("inf") else "∞"

    st.markdown(f"""
    <div class="stats-container">
        <div class="stat-card">
            <div class="stat-label">Toplam PnL</div>
            <div class="stat-value" style="color:{pnl_color}">{format_pnl(total_pnl)}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Win Rate</div>
            <div class="stat-value">{win_rate:.1f}%</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Toplam İşlem</div>
            <div class="stat-value">{total_trades}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Aktif Pozisyon</div>
            <div class="stat-value">{len(active)}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Profit Factor</div>
            <div class="stat-value">{pf_display}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Avg Win / Loss</div>
            <div class="stat-value" style="font-size:0.85rem">
                <span style="color:#10b981">{format_pnl(avg_win)}</span>
                <span style="color:#4a5568"> / </span>
                <span style="color:#ef4444">{format_pnl(avg_loss)}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
