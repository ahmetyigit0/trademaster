import streamlit as st
from datetime import datetime
from utils import (
    generate_id, calculate_avg_entry, calculate_position_size,
    calculate_rr, save_data, format_trade_header
)


# ─── NEW POSITION FORM ──────────────────────────────────────────────────────────

def render_new_position_form():
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    st.markdown("### ＋ Yeni Pozisyon")

    col1, col2, col3 = st.columns(3)
    with col1:
        symbol = st.text_input("Symbol", placeholder="BTC, ETH, AAPL...", key="new_symbol").upper().strip()
    with col2:
        direction = st.radio("Yön", ["LONG", "SHORT"], horizontal=True, key="new_direction")
    with col3:
        capital = st.number_input("Toplam Sermaye ($)", min_value=0.0, step=100.0, key="new_capital")

    col4, col5 = st.columns(2)
    with col4:
        risk_pct = st.number_input("Risk (%)", min_value=0.1, max_value=100.0, value=2.0, step=0.1, key="new_risk")
    with col5:
        stop_loss = st.number_input("Stop Loss Fiyatı", min_value=0.0, step=0.01, format="%.4f", key="new_stop")

    # ── Entry System ──
    st.markdown("---")
    partial_entry = st.checkbox("Parçalı Giriş (3 Entry)", key="new_partial_entry")

    entries = []
    if partial_entry:
        st.markdown("**Entry Fiyatları ve Ağırlıklar**")
        e_col1, e_col2, e_col3 = st.columns(3)
        labels = ["Entry 1", "Entry 2", "Entry 3"]
        default_weights = [30.0, 40.0, 30.0]
        for i, (col, label, dw) in enumerate(zip([e_col1, e_col2, e_col3], labels, default_weights)):
            with col:
                price = st.number_input(f"{label} Fiyat", min_value=0.0, step=0.01,
                                        format="%.4f", key=f"new_entry_price_{i}")
                weight = st.number_input(f"{label} Ağırlık (%)", min_value=0.0, max_value=100.0,
                                         value=dw, step=1.0, key=f"new_entry_weight_{i}")
                entries.append({"price": price, "weight": weight})
    else:
        entry_price = st.number_input("Entry Fiyatı", min_value=0.0, step=0.01, format="%.4f", key="new_entry_single")
        entries = [{"price": entry_price, "weight": 100.0}]

    # ── Take Profit ──
    st.markdown("---")
    partial_tp = st.checkbox("Parçalı TP (3 Hedef)", key="new_partial_tp")
    take_profits = []
    if partial_tp:
        st.markdown("**Take Profit Hedefleri ve Ağırlıklar**")
        tp_col1, tp_col2, tp_col3 = st.columns(3)
        tp_labels = ["TP1", "TP2", "TP3"]
        for i, (col, label) in enumerate(zip([tp_col1, tp_col2, tp_col3], tp_labels)):
            with col:
                tp_price = st.number_input(f"{label} Fiyat", min_value=0.0, step=0.01,
                                           format="%.4f", key=f"new_tp_price_{i}")
                tp_weight = st.number_input(f"{label} Ağırlık (%)", min_value=0.0, max_value=100.0,
                                            value=33.3, step=1.0, key=f"new_tp_weight_{i}")
                take_profits.append({"price": tp_price, "weight": tp_weight})
    else:
        tp_single = st.number_input("Take Profit Fiyatı (opsiyonel)", min_value=0.0,
                                    step=0.01, format="%.4f", key="new_tp_single")
        take_profits = [{"price": tp_single, "weight": 100.0}] if tp_single > 0 else []

    # ── Position Size Calculator ──
    avg_entry = calculate_avg_entry(entries)
    calc_result = {}
    if avg_entry > 0 and stop_loss > 0 and capital > 0:
        calc_result = calculate_position_size(capital, risk_pct, avg_entry, stop_loss, direction)
        st.markdown("---")
        st.markdown("**📐 Pozisyon Hesaplama**")

        if "error" in calc_result:
            st.error(calc_result["error"])
        else:
            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                st.markdown(f'<div class="calc-box"><div class="calc-label">Ortalama Entry</div><div class="calc-value">${avg_entry:.4f}</div></div>', unsafe_allow_html=True)
            with rc2:
                st.markdown(f'<div class="calc-box"><div class="calc-label">Risk Tutarı</div><div class="calc-value">${calc_result["risk_amount"]:.2f}</div></div>', unsafe_allow_html=True)
            with rc3:
                st.markdown(f'<div class="calc-box"><div class="calc-label">Full Sermaye Riski</div><div class="calc-value">{calc_result["full_risk_pct"]:.2f}%</div></div>', unsafe_allow_html=True)

            if calc_result["can_full"]:
                st.success(f"✅ Full sermaye ({capital:,.2f}$) ile girilebilir. Risk: {calc_result['full_risk_pct']:.2f}% ≤ {risk_pct}%")
            else:
                rec_size = calc_result["position_size"]
                st.warning(f"⚠️ Önerilen pozisyon büyüklüğü: **${rec_size:,.2f}** (Full sermaye riski %{calc_result['full_risk_pct']:.2f} > %{risk_pct})")
                if st.button("📌 Öneriyi Uygula", key="apply_suggestion"):
                    st.session_state["applied_position_size"] = rec_size
                    st.rerun()

    # ── Applied position size display ──
    position_size_final = capital  # default full
    if not calc_result.get("can_full", True) and calc_result.get("position_size"):
        position_size_final = st.session_state.get("applied_position_size", calc_result["position_size"])

    st.markdown("---")
    btn_c1, btn_c2 = st.columns([1, 1])
    with btn_c1:
        if st.button("💾 Pozisyonu Kaydet", key="save_position", use_container_width=True):
            if not symbol:
                st.error("Symbol giriniz.")
            elif avg_entry == 0:
                st.error("En az bir entry fiyatı giriniz.")
            elif stop_loss == 0:
                st.error("Stop loss zorunludur.")
            elif capital == 0:
                st.error("Sermaye giriniz.")
            else:
                pos_id = generate_id()
                position_size = position_size_final if not calc_result.get("can_full", True) else capital
                risk_amount = calc_result.get("risk_amount", capital * risk_pct / 100)
                new_pos = {
                    "id": pos_id,
                    "symbol": symbol,
                    "direction": direction,
                    "capital": capital,
                    "risk_pct": risk_pct,
                    "risk_amount": risk_amount,
                    "entries": entries,
                    "avg_entry": avg_entry,
                    "stop_loss": stop_loss,
                    "take_profits": take_profits,
                    "position_size": position_size,
                    "partial_entry": partial_entry,
                    "partial_tp": partial_tp,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
                st.session_state.data["active_positions"].append(new_pos)
                save_data(st.session_state.data)
                # reset
                st.session_state.show_new_form = False
                if "applied_position_size" in st.session_state:
                    del st.session_state["applied_position_size"]
                st.rerun()

    with btn_c2:
        if st.button("✖ İptal", key="cancel_new_position", use_container_width=True):
            st.session_state.show_new_form = False
            if "applied_position_size" in st.session_state:
                del st.session_state["applied_position_size"]
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ─── ACTIVE POSITION CARD ──────────────────────────────────────────────────────

def render_active_position(pos: dict, index: int):
    pos_id = pos["id"]
    symbol = pos.get("symbol", "?")
    direction = pos.get("direction", "LONG")
    avg = pos.get("avg_entry", 0)
    stop = pos.get("stop_loss", 0)
    pos_size = pos.get("position_size", 0)
    risk_amount = pos.get("risk_amount", 0)
    entries = pos.get("entries", [])
    tps = pos.get("take_profits", [])
    created = pos.get("created_at", "")

    direction_class = "long-badge" if direction == "LONG" else "short-badge"
    header_text = f"#{index+1} · {symbol} · {direction} · Entry: ${avg:.4f} · Size: ${pos_size:,.0f}"

    is_expanded = pos_id in st.session_state.expanded_positions

    # Header card
    with st.container():
        st.markdown(f'<div class="position-card active-card">', unsafe_allow_html=True)

        h_col1, h_col2, h_col3, h_col4 = st.columns([4, 1, 1, 1])
        with h_col1:
            st.markdown(f"""
            <div class="pos-header-row">
                <span class="pos-number">#{index+1}</span>
                <span class="pos-symbol">{symbol}</span>
                <span class="badge {direction_class}">{direction}</span>
                <span class="pos-meta">Entry ${avg:.4f} · Size ${pos_size:,.0f}</span>
            </div>
            """, unsafe_allow_html=True)
        with h_col2:
            toggle_label = "▲ Kapat" if is_expanded else "▼ Detay"
            if st.button(toggle_label, key=f"toggle_{pos_id}", use_container_width=True):
                if is_expanded:
                    st.session_state.expanded_positions.discard(pos_id)
                else:
                    st.session_state.expanded_positions.add(pos_id)
                st.rerun()
        with h_col3:
            if st.button("🔒 Kapat", key=f"close_btn_{pos_id}", use_container_width=True):
                st.session_state.close_modal = pos_id
                st.rerun()
        with h_col4:
            if st.button("🗑 Sil", key=f"delete_{pos_id}", use_container_width=True):
                st.session_state.data["active_positions"] = [
                    p for p in st.session_state.data["active_positions"] if p["id"] != pos_id
                ]
                st.session_state.expanded_positions.discard(pos_id)
                save_data(st.session_state.data)
                st.rerun()

        # Expanded details
        if is_expanded:
            st.markdown('<div class="pos-details">', unsafe_allow_html=True)
            d_col1, d_col2, d_col3 = st.columns(3)

            with d_col1:
                st.markdown("**📥 Entry Detayları**")
                for i, e in enumerate(entries):
                    if e.get("price", 0) > 0:
                        st.markdown(f'<div class="detail-row"><span class="dl">Entry {i+1}</span><span class="dv">${e["price"]:.4f} <em>({e["weight"]:.0f}%)</em></span></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="detail-row highlight"><span class="dl">Ort. Entry</span><span class="dv">${avg:.4f}</span></div>', unsafe_allow_html=True)

            with d_col2:
                st.markdown("**🎯 TP / SL**")
                st.markdown(f'<div class="detail-row danger-row"><span class="dl">Stop Loss</span><span class="dv">${stop:.4f}</span></div>', unsafe_allow_html=True)
                if tps:
                    for i, tp in enumerate(tps):
                        if tp.get("price", 0) > 0:
                            st.markdown(f'<div class="detail-row success-row"><span class="dl">TP {i+1}</span><span class="dv">${tp["price"]:.4f} <em>({tp["weight"]:.0f}%)</em></span></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="detail-row"><span class="dl">TP</span><span class="dv">—</span></div>', unsafe_allow_html=True)

            with d_col3:
                st.markdown("**💰 Risk / Boyut**")
                st.markdown(f'<div class="detail-row"><span class="dl">Sermaye</span><span class="dv">${pos.get("capital",0):,.2f}</span></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="detail-row"><span class="dl">Pozisyon</span><span class="dv">${pos_size:,.2f}</span></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="detail-row danger-row"><span class="dl">Risk</span><span class="dv">${risk_amount:,.2f} ({pos.get("risk_pct",2)}%)</span></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="detail-row"><span class="dl">Tarih</span><span class="dv">{created}</span></div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


# ─── CLOSED POSITION CARD ──────────────────────────────────────────────────────

def render_closed_position(trade: dict, index: int):
    trade_id = trade["id"]
    symbol = trade.get("symbol", "?")
    direction = trade.get("direction", "LONG")
    pnl = trade.get("pnl", 0)
    rr = trade.get("rr", 0)
    avg = trade.get("avg_entry", 0)
    stop = trade.get("stop_loss", 0)
    tps = trade.get("take_profits", [])
    comment = trade.get("comment", "")
    closed_at = trade.get("closed_at", "")
    entries = trade.get("entries", [])

    is_win = pnl > 0
    pnl_class = "positive" if is_win else "negative"
    pnl_str = f"+${pnl:.2f}" if is_win else f"-${abs(pnl):.2f}"
    result_label = "WIN" if is_win else "LOSS"
    result_class = "win-badge" if is_win else "loss-badge"
    direction_class = "long-badge" if direction == "LONG" else "short-badge"
    rr_str = f"1:{abs(rr):.2f}" if rr else "—"

    is_expanded = trade_id in st.session_state.expanded_closed

    with st.container():
        st.markdown('<div class="position-card closed-card">', unsafe_allow_html=True)

        h_col1, h_col2 = st.columns([5, 1])
        with h_col1:
            st.markdown(f"""
            <div class="pos-header-row">
                <span class="pos-symbol">{symbol}</span>
                <span class="badge {direction_class}">{direction}</span>
                <span class="badge {result_class}">{result_label}</span>
                <span class="rr-label">RR {rr_str}</span>
                <span class="pnl-label {pnl_class}">{pnl_str}</span>
                <span class="pos-meta">{closed_at}</span>
            </div>
            """, unsafe_allow_html=True)
        with h_col2:
            toggle_label = "▲" if is_expanded else "▼ Detay"
            if st.button(toggle_label, key=f"toggle_closed_{trade_id}_{index}", use_container_width=True):
                if is_expanded:
                    st.session_state.expanded_closed.discard(trade_id)
                else:
                    st.session_state.expanded_closed.add(trade_id)
                st.rerun()

        if is_expanded:
            st.markdown('<div class="pos-details">', unsafe_allow_html=True)
            d_col1, d_col2, d_col3 = st.columns(3)

            with d_col1:
                st.markdown("**📥 Entry**")
                for i, e in enumerate(entries):
                    if e.get("price", 0) > 0:
                        st.markdown(f'<div class="detail-row"><span class="dl">Entry {i+1}</span><span class="dv">${e["price"]:.4f} ({e["weight"]:.0f}%)</span></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="detail-row highlight"><span class="dl">Ort. Entry</span><span class="dv">${avg:.4f}</span></div>', unsafe_allow_html=True)

            with d_col2:
                st.markdown("**🎯 TP / SL**")
                st.markdown(f'<div class="detail-row danger-row"><span class="dl">Stop</span><span class="dv">${stop:.4f}</span></div>', unsafe_allow_html=True)
                for i, tp in enumerate(tps):
                    if tp.get("price", 0) > 0:
                        st.markdown(f'<div class="detail-row success-row"><span class="dl">TP {i+1}</span><span class="dv">${tp["price"]:.4f}</span></div>', unsafe_allow_html=True)

            with d_col3:
                st.markdown("**📊 Sonuç**")
                st.markdown(f'<div class="detail-row"><span class="dl">PnL</span><span class="dv {pnl_class}">{pnl_str}</span></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="detail-row"><span class="dl">R:R</span><span class="dv">{rr_str}</span></div>', unsafe_allow_html=True)
                if comment:
                    st.markdown(f'<div class="detail-row"><span class="dl">Yorum</span><span class="dv">{comment}</span></div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
