import streamlit as st
from utils.calculations import format_pnl, calculate_ev


def render_stats_bar(data: dict):
    closed = data.get("closed_trades", [])
    active = data.get("active_positions", [])

    wins   = [t for t in closed if t.get("pnl", 0) > 0]
    losses = [t for t in closed if t.get("pnl", 0) <= 0]
    n      = len(closed)

    win_rate    = len(wins) / n * 100 if n else 0
    total_pnl   = sum(t.get("pnl", 0) for t in closed)
    avg_win     = sum(t["pnl"] for t in wins)   / len(wins)   if wins   else 0
    avg_loss    = sum(t["pnl"] for t in losses) / len(losses) if losses else 0
    gross_win   = sum(t["pnl"] for t in wins)
    gross_loss  = abs(sum(t["pnl"] for t in losses))
    pf          = gross_win / gross_loss if gross_loss > 0 else float("inf")
    ev          = calculate_ev(win_rate / 100, avg_win, avg_loss) if n else 0

    pnl_c  = "#10b981" if total_pnl >= 0 else "#ef4444"
    ev_c   = "#10b981" if ev >= 0 else "#ef4444"
    pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"

    st.markdown(f"""
    <div class="stats-container">
      <div class="stat-card">
        <div class="stat-label">Toplam PnL</div>
        <div class="stat-value" style="color:{pnl_c}">{format_pnl(total_pnl)}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Win Rate</div>
        <div class="stat-value">{win_rate:.1f}%</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Toplam İşlem</div>
        <div class="stat-value">{n} <span style="font-size:0.7rem;color:#4a5568">({len(wins)}W / {len(losses)}L)</span></div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Aktif Pozisyon</div>
        <div class="stat-value">{len(active)}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Profit Factor</div>
        <div class="stat-value">{pf_str}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Expected Value</div>
        <div class="stat-value" style="color:{ev_c}">{format_pnl(ev)}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Avg Win / Loss</div>
        <div class="stat-value" style="font-size:0.82rem">
          <span style="color:#10b981">{format_pnl(avg_win)}</span>
          <span style="color:#4a5568"> / </span>
          <span style="color:#ef4444">{format_pnl(avg_loss)}</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
