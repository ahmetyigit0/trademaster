import streamlit as st
from utils.data_manager import save_data
from utils.calculations import format_pnl
from datetime import datetime


def render_closed_trades(data: dict):
    trades = data.get("closed_trades", [])

    if not trades:
        st.markdown("""
        <div style="text-align:center;padding:2.5rem;color:#2d3a4e;border:1px dashed #1e2530;border-radius:12px">
            <div style="font-size:2rem;margin-bottom:0.5rem">📂</div>
            <div style="font-family:'Space Mono',monospace;font-size:0.8rem">Henüz kapalı işlem yok</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Filters ──────────────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns([1, 1, 4])
    with fc1:
        dir_filter = st.selectbox("Yön", ["Tümü", "LONG", "SHORT"], key="filter_direction")
    with fc2:
        result_filter = st.selectbox("Sonuç", ["Tümü", "WIN", "LOSS"], key="filter_result")

    filtered = trades
    if dir_filter != "Tümü":
        filtered = [t for t in filtered if t.get("direction") == dir_filter]
    if result_filter != "Tümü":
        filtered = [t for t in filtered if t.get("result") == result_filter]

    # ── Summary row ──────────────────────────────────────────────────────────
    if filtered:
        total_pnl = sum(t.get("pnl", 0) for t in filtered)
        wins = sum(1 for t in filtered if t.get("result") == "WIN")
        wr = wins / len(filtered) * 100

        pnl_color = "#10b981" if total_pnl >= 0 else "#ef4444"
        st.markdown(f"""
        <div style="display:flex;gap:1.5rem;padding:0.75rem 0;margin-bottom:0.5rem;border-bottom:1px solid #1e2530">
            <span style="color:#4a5568;font-size:0.8rem">{len(filtered)} işlem</span>
            <span style="font-family:'Space Mono',monospace;font-size:0.8rem;color:{pnl_color}">{format_pnl(total_pnl)}</span>
            <span style="color:#4a5568;font-size:0.8rem">Win Rate: {wr:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Trade list (newest first) ─────────────────────────────────────────────
    for trade in reversed(filtered):
        _render_closed_trade(trade)


def _render_closed_trade(trade: dict):
    pnl = trade.get("pnl", 0)
    result = trade.get("result", "—")
    direction = trade.get("direction", "LONG")
    rr_display = trade.get("rr_display", "1:?")
    symbol = trade.get("symbol", "?")
    trade_id = trade.get("id", 0)

    pnl_str = format_pnl(pnl)
    pnl_color = "#10b981" if pnl >= 0 else "#ef4444"
    result_badge = "badge-win" if result == "WIN" else "badge-loss"
    dir_badge = "badge-long" if direction == "LONG" else "badge-short"

    title = f"#{trade_id} · {symbol} · {direction} · {rr_display} · {pnl_str}"

    with st.expander(title, expanded=False):
        avg_entry = trade.get("avg_entry", 0)
        stop_loss = trade.get("stop_loss", 0)
        entries = trade.get("entries", [])
        take_profits = trade.get("take_profits", [])
        comment = trade.get("comment", "")
        closed_at = trade.get("closed_at", "")
        pos_size = trade.get("position_size", 0)

        if closed_at:
            try:
                closed_at = datetime.fromisoformat(closed_at).strftime("%d %b %Y %H:%M")
            except ValueError:
                pass

        entry_display = " / ".join(
            f"${e['price']:,.4f} ({e['weight']:.0f}%)" for e in entries
        ) if entries else "—"

        tp_display = " / ".join(
            f"${t['price']:,.4f} ({t['weight']:.0f}%)" for t in take_profits
        ) if take_profits else "—"

        st.markdown(f"""
        <div class="detail-grid">
            <div class="detail-item">
                <div class="detail-label">Sonuç</div>
                <div class="detail-value">
                    <span class="badge {result_badge}">{result}</span>
                </div>
            </div>
            <div class="detail-item">
                <div class="detail-label">PnL</div>
                <div class="detail-value" style="color:{pnl_color}">{pnl_str}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Avg Entry</div>
                <div class="detail-value">${avg_entry:,.4f}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Stop Loss</div>
                <div class="detail-value" style="color:#ef4444">${stop_loss:,.4f}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">R/R</div>
                <div class="detail-value">{rr_display}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Pozisyon</div>
                <div class="detail-value">${pos_size:,.2f}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Take Profit</div>
                <div class="detail-value" style="font-size:0.75rem;color:#10b981">{tp_display}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Kapanış</div>
                <div class="detail-value" style="font-size:0.75rem">{closed_at}</div>
            </div>
        </div>
        <div class="detail-item" style="margin-top:0.6rem">
            <div class="detail-label">Entry Noktaları</div>
            <div class="detail-value" style="font-size:0.78rem">{entry_display}</div>
        </div>
        """, unsafe_allow_html=True)

        if comment:
            st.markdown(f"""
            <div class="detail-item" style="margin-top:0.5rem">
                <div class="detail-label">💬 Yorum</div>
                <div class="detail-value" style="font-size:0.8rem;color:#94a3b8">{comment}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")
        if st.button("🗑️ Sil", key=f"del_closed_{trade_id}", help="Bu işlemi kalıcı olarak sil"):
            _delete_closed(trade_id)
            st.rerun()


def _delete_closed(trade_id: int):
    data = st.session_state.data
    data["closed_trades"] = [t for t in data["closed_trades"] if t.get("id") != trade_id]
    save_data(data)
    st.session_state.data = data
