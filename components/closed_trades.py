import streamlit as st
from utils.data_manager import save_data
from utils.calculations import format_pnl, rr_color
from datetime import datetime

SETUP_TYPES = ["liquidity", "breakout", "trend", "range", "diğer"]


def render_closed_trades(data: dict):
    trades = data.get("closed_trades", [])
    if not trades:
        st.markdown("""
        <div style="text-align:center;padding:2.5rem;color:#2d3a4e;border:1px dashed #1e2530;border-radius:12px">
            <div style="font-size:2rem;margin-bottom:0.5rem">📂</div>
            <div style="font-family:'Space Mono',monospace;font-size:0.8rem">Henüz kapalı işlem yok</div>
        </div>""", unsafe_allow_html=True)
        return

    # ── Filters ──────────────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        dir_f = st.selectbox("Yön", ["Tümü", "LONG", "SHORT"], key="cf_dir")
    with fc2:
        res_f = st.selectbox("Sonuç", ["Tümü", "WIN", "LOSS"], key="cf_res")
    with fc3:
        setup_f = st.selectbox("Setup", ["Tümü"] + SETUP_TYPES, key="cf_setup")

    filtered = trades
    if dir_f   != "Tümü": filtered = [t for t in filtered if t.get("direction") == dir_f]
    if res_f   != "Tümü": filtered = [t for t in filtered if t.get("result")    == res_f]
    if setup_f != "Tümü": filtered = [t for t in filtered if t.get("setup_type") == setup_f]

    # ── Summary ───────────────────────────────────────────────────────────────
    if filtered:
        total_pnl = sum(t.get("pnl", 0) for t in filtered)
        wins = [t for t in filtered if t.get("result") == "WIN"]
        wr   = len(wins) / len(filtered) * 100
        pnl_c = "#10b981" if total_pnl >= 0 else "#ef4444"
        st.markdown(f"""
        <div style="display:flex;gap:1.5rem;padding:0.6rem 0;margin-bottom:0.5rem;border-bottom:1px solid #1e2530;flex-wrap:wrap">
          <span style="color:#64748b;font-size:0.8rem">{len(filtered)} işlem</span>
          <span style="font-family:'Space Mono',monospace;font-size:0.8rem;color:{pnl_c}">{format_pnl(total_pnl)}</span>
          <span style="color:#64748b;font-size:0.8rem">Win Rate: <b style="color:#e2e8f0">{wr:.1f}%</b></span>
        </div>""", unsafe_allow_html=True)

    for trade in reversed(filtered):
        _render_closed_trade(trade)


def _render_closed_trade(trade: dict):
    pnl       = trade.get("pnl", 0)
    result    = trade.get("result", "—")
    direction = trade.get("direction", "LONG")
    rr_str    = trade.get("rr_display", "1:?")
    symbol    = trade.get("symbol", "?")
    tid       = trade.get("id", 0)
    r_mult    = trade.get("r_multiple", 0)
    pnl_str   = format_pnl(pnl)

    pnl_color  = "#10b981" if pnl >= 0 else "#ef4444"
    dir_color  = "#10b981" if direction == "LONG" else "#ef4444"
    dir_bg     = "#064e3b" if direction == "LONG" else "#450a0a"
    res_color  = "#10b981" if result == "WIN" else "#ef4444"
    res_bg     = "#064e3b" if result == "WIN" else "#450a0a"
    r_sign     = "+" if r_mult >= 0 else ""

    title = f"#{tid} · {symbol} · {direction} · {rr_str} · {pnl_str}"

    with st.expander(title, expanded=False):
        avg_entry    = trade.get("avg_entry", 0)
        stop_loss    = trade.get("stop_loss", 0)
        entries      = trade.get("entries", [])
        take_profits = trade.get("take_profits", [])
        comment      = trade.get("comment", "")
        closed_at    = trade.get("closed_at", "")
        pos_size     = trade.get("position_size", 0)
        setup        = trade.get("setup_type", "—")
        exec_score   = trade.get("execution_score", "—")
        emotion      = trade.get("emotion", "—")
        mistakes     = trade.get("mistakes", [])
        market       = trade.get("market_condition", "—")
        plan         = "✅" if trade.get("plan_followed") else "❌"

        try:
            closed_at = datetime.fromisoformat(closed_at).strftime("%d %b %Y %H:%M")
        except Exception:
            pass

        entry_str = " / ".join(f"${e['price']:,.4f} ({e['weight']:.0f}%)" for e in entries) or "—"
        tp_str    = " / ".join(f"${t['price']:,.4f} ({t['weight']:.0f}%)" for t in take_profits) or "—"

        st.markdown(f"""
        <div class="detail-grid">
          <div class="detail-item">
            <div class="detail-label">Yön</div>
            <div class="detail-value">
              <span style="background:{dir_bg};color:{dir_color};padding:2px 8px;border-radius:4px;font-size:0.72rem;font-weight:700">{direction}</span>
            </div>
          </div>
          <div class="detail-item">
            <div class="detail-label">Sonuç</div>
            <div class="detail-value">
              <span style="background:{res_bg};color:{res_color};padding:2px 8px;border-radius:4px;font-size:0.72rem;font-weight:700">{result}</span>
            </div>
          </div>
          <div class="detail-item">
            <div class="detail-label">PnL</div>
            <div class="detail-value" style="color:{pnl_color}">{pnl_str}</div>
          </div>
          <div class="detail-item">
            <div class="detail-label">R Multiple</div>
            <div class="detail-value" style="color:{pnl_color}">{r_sign}{r_mult}R</div>
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
            <div class="detail-label">R:R</div>
            <div class="detail-value">{rr_str}</div>
          </div>
          <div class="detail-item">
            <div class="detail-label">Pozisyon</div>
            <div class="detail-value">${pos_size:,.2f}</div>
          </div>
          <div class="detail-item">
            <div class="detail-label">Setup</div>
            <div class="detail-value">{setup}</div>
          </div>
          <div class="detail-item">
            <div class="detail-label">Execution</div>
            <div class="detail-value">{exec_score}/10</div>
          </div>
          <div class="detail-item">
            <div class="detail-label">Psikoloji</div>
            <div class="detail-value">{emotion}</div>
          </div>
          <div class="detail-item">
            <div class="detail-label">Plana Uyum</div>
            <div class="detail-value">{plan}</div>
          </div>
          <div class="detail-item">
            <div class="detail-label">Piyasa</div>
            <div class="detail-value">{market}</div>
          </div>
          <div class="detail-item">
            <div class="detail-label">Kapanış</div>
            <div class="detail-value" style="font-size:0.72rem">{closed_at}</div>
          </div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;margin-top:0.5rem">
          <div class="detail-item">
            <div class="detail-label">Entries</div>
            <div class="detail-value" style="font-size:0.72rem">{entry_str}</div>
          </div>
          <div class="detail-item">
            <div class="detail-label">Take Profits</div>
            <div class="detail-value" style="font-size:0.72rem;color:#10b981">{tp_str}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if mistakes:
            badges = "".join(
                f"<span style='background:#1c0505;color:#f87171;padding:2px 7px;border-radius:4px;font-size:0.68rem;margin-right:4px'>{m}</span>"
                for m in mistakes
            )
            st.markdown(f"<div style='margin-top:0.4rem'><span style='font-size:0.68rem;color:#64748b'>HATALAR: </span>{badges}</div>", unsafe_allow_html=True)

        if comment:
            st.markdown(f"""<div class="detail-item" style="margin-top:0.5rem">
                <div class="detail-label">💬 Yorum</div>
                <div class="detail-value" style="font-size:0.8rem;color:#94a3b8">{comment}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")
        if st.button("🗑️ Sil", key=f"del_closed_{tid}"):
            _delete_closed(tid)
            st.rerun()


def _delete_closed(trade_id):
    data = st.session_state.data
    data["closed_trades"] = [t for t in data["closed_trades"] if t.get("id") != trade_id]
    save_data(data)
    st.session_state.data = data
