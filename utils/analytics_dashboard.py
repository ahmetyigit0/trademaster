import streamlit as st
from analytics.engine import compute_analytics
from utils.calculations import format_pnl


def _table(headers: list[str], rows_html: str, col_aligns: list[str] | None = None) -> str:
    """Reusable HTML table inside a .card div."""
    ths = ""
    for i, h in enumerate(headers):
        align = (col_aligns[i] if col_aligns else "left") if i > 0 else "left"
        ths += f'<th style="padding:0.4rem 0.75rem;text-align:{align};color:#4a5568;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.08em">{h}</th>'
    return f"""
    <div class="card">
      <table style="width:100%;border-collapse:collapse">
        <thead><tr style="border-bottom:1px solid #1e2530">{ths}</tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""


def _bar_row(label: str, value: float, max_val: float, color: str, suffix: str = "") -> str:
    pct = min(value / max_val * 100, 100) if max_val > 0 else 0
    return f"""
    <div style="margin-bottom:0.65rem">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.25rem">
        <span style="font-size:0.8rem;color:#e2e8f0">{label}</span>
        <span style="font-family:'Space Mono',monospace;font-size:0.78rem;color:{color}">{suffix}{value:.1f}%</span>
      </div>
      <div style="background:#1e2530;border-radius:4px;height:7px">
        <div style="background:{color};width:{pct:.1f}%;height:7px;border-radius:4px;transition:width 0.4s"></div>
      </div>
    </div>"""


def render_analytics(data: dict):
    closed = data.get("closed_trades", [])

    if not closed:
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:#2d3a4e;border:1px dashed #1e2530;border-radius:12px">
            <div style="font-size:2.5rem;margin-bottom:0.5rem">📊</div>
            <div style="font-family:'Space Mono',monospace;font-size:0.8rem">Analiz için kapalı işlem gerekli</div>
        </div>""", unsafe_allow_html=True)
        return

    try:
        a = compute_analytics(closed)
    except Exception as ex:
        st.error(f"Analytics hesaplama hatası: {ex}")
        return

    if not a:
        st.info("Yeterli veri yok.")
        return

    # ── Setup Performance ─────────────────────────────────────────────────────
    st.markdown('<div class="section-title">⚙️ Setup Performansı</div>', unsafe_allow_html=True)
    setups = a.get("setup_stats", {})
    if setups:
        best_setup = max(setups, key=lambda s: setups[s].get("pnl", 0))
        rows = ""
        for s, v in sorted(setups.items(), key=lambda x: x[1].get("pnl", 0), reverse=True):
            pnl = v.get("pnl", 0)
            wr  = v.get("win_rate", 0)
            cnt = v.get("count", 0)
            pc  = "#10b981" if pnl >= 0 else "#ef4444"
            star = " ⭐" if s == best_setup else ""
            rows += (
                f'<tr style="border-bottom:1px solid #131b28">'
                f'<td style="padding:0.5rem 0.75rem;font-weight:600;color:#e2e8f0">{s}{star}</td>'
                f'<td style="padding:0.5rem 0.75rem;text-align:center;color:#94a3b8">{cnt}</td>'
                f'<td style="padding:0.5rem 0.75rem;text-align:center;color:#94a3b8">{wr:.0f}%</td>'
                f'<td style="padding:0.5rem 0.75rem;text-align:right;color:{pc};font-family:\'Space Mono\',monospace">{format_pnl(pnl)}</td>'
                f'</tr>'
            )
        st.markdown(_table(["Setup", "İşlem", "Win Rate", "PnL"], rows,
                            ["left", "center", "center", "right"]), unsafe_allow_html=True)
    else:
        st.markdown('<div class="card" style="color:#4a5568;font-size:0.8rem">Setup verisi yok.</div>', unsafe_allow_html=True)

    # ── Market Condition ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Piyasa Koşulu</div>', unsafe_allow_html=True)
    conds = a.get("cond_stats", {})
    if conds:
        rows = ""
        for c, v in sorted(conds.items(), key=lambda x: x[1].get("pnl", 0), reverse=True):
            pnl = v.get("pnl", 0)
            wr  = v.get("win_rate", 0)
            tot = v.get("total", 0)
            pc  = "#10b981" if pnl >= 0 else "#ef4444"
            rows += (
                f'<tr style="border-bottom:1px solid #131b28">'
                f'<td style="padding:0.5rem 0.75rem;font-weight:600;color:#e2e8f0">{c}</td>'
                f'<td style="padding:0.5rem 0.75rem;text-align:center;color:#94a3b8">{tot}</td>'
                f'<td style="padding:0.5rem 0.75rem;text-align:center;color:#94a3b8">{wr:.0f}%</td>'
                f'<td style="padding:0.5rem 0.75rem;text-align:right;color:{pc};font-family:\'Space Mono\',monospace">{format_pnl(pnl)}</td>'
                f'</tr>'
            )
        st.markdown(_table(["Koşul", "İşlem", "Win Rate", "PnL"], rows,
                            ["left", "center", "center", "right"]), unsafe_allow_html=True)

    # ── Psychology + Plan ─────────────────────────────────────────────────────
    col_emo, col_plan = st.columns(2)

    with col_emo:
        st.markdown('<div class="section-title">🧠 Psikoloji</div>', unsafe_allow_html=True)
        emos = a.get("emo_stats", {})
        if emos:
            rows = ""
            for e, v in sorted(emos.items(), key=lambda x: x[1].get("pnl", 0), reverse=True):
                pnl = v.get("pnl", 0)
                wr  = v.get("win_rate", 0)
                tot = v.get("total", 0)
                pc  = "#10b981" if pnl >= 0 else "#ef4444"
                rows += (
                    f'<tr style="border-bottom:1px solid #131b28">'
                    f'<td style="padding:0.4rem 0.6rem;font-weight:600;color:#e2e8f0">{e}</td>'
                    f'<td style="padding:0.4rem 0.6rem;text-align:center;color:#94a3b8">{tot}</td>'
                    f'<td style="padding:0.4rem 0.6rem;text-align:center;color:#94a3b8">{wr:.0f}%</td>'
                    f'<td style="padding:0.4rem 0.6rem;text-align:right;color:{pc};font-family:\'Space Mono\',monospace;font-size:0.8rem">{format_pnl(pnl)}</td>'
                    f'</tr>'
                )
            st.markdown(_table(["Duygu", "N", "WR", "PnL"], rows,
                                ["left", "center", "center", "right"]), unsafe_allow_html=True)

    with col_plan:
        st.markdown('<div class="section-title">📋 Plan Uyumu</div>', unsafe_allow_html=True)
        pyw = float(a.get("plan_yes_wr", 0))
        pnw = float(a.get("plan_no_wr", 0))

        yes_bar = _bar_row("✅ Plana Uyuldu", pyw, 100, "#10b981")
        no_bar  = _bar_row("❌ Plan Dışı",    pnw, 100, "#ef4444")
        st.markdown(f'<div class="card">{yes_bar}{no_bar}</div>', unsafe_allow_html=True)

    # ── Mistakes ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">⚠️ Hata Analizi</div>', unsafe_allow_html=True)
    mistakes = a.get("mistake_freq", {})
    if mistakes:
        max_cnt  = max(mistakes.values())
        bars_html = ""
        for m, cnt in sorted(mistakes.items(), key=lambda x: x[1], reverse=True):
            pct = (cnt / max_cnt * 100) if max_cnt > 0 else 0
            bars_html += (
                f'<div style="margin-bottom:0.6rem">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:0.2rem">'
                f'<span style="font-size:0.8rem;color:#e2e8f0">{m}</span>'
                f'<span style="font-family:\'Space Mono\',monospace;font-size:0.75rem;color:#f59e0b">{cnt}×</span>'
                f'</div>'
                f'<div style="background:#1e2530;border-radius:4px;height:7px">'
                f'<div style="background:#f59e0b;width:{pct:.1f}%;height:7px;border-radius:4px"></div>'
                f'</div></div>'
            )
        st.markdown(f'<div class="card">{bars_html}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="card" style="color:#4a5568;font-size:0.8rem;padding:1rem">Hata etiketi girilmemiş.</div>', unsafe_allow_html=True)

    # ── Direction + Execution ─────────────────────────────────────────────────
    dc1, dc2 = st.columns(2)

    with dc1:
        st.markdown('<div class="section-title">📈 LONG vs SHORT</div>', unsafe_allow_html=True)
        lwr = float(a.get("long_win_rate",  0))
        swr = float(a.get("short_win_rate", 0))
        lc  = int(a.get("long_count",  0))
        sc  = int(a.get("short_count", 0))
        long_bar  = _bar_row(f"LONG ({lc})",  lwr, 100, "#10b981")
        short_bar = _bar_row(f"SHORT ({sc})", swr, 100, "#ef4444")
        st.markdown(f'<div class="card">{long_bar}{short_bar}</div>', unsafe_allow_html=True)

    with dc2:
        st.markdown('<div class="section-title">⭐ Execution Skoru</div>', unsafe_allow_html=True)
        exec_a = a.get("exec_analysis", {})
        if exec_a:
            rows = ""
            for bucket in sorted(exec_a.keys()):
                v   = exec_a[bucket]
                ap  = float(v.get("avg_pnl", 0))
                cnt = int(v.get("count", 0))
                pc  = "#10b981" if ap >= 0 else "#ef4444"
                rows += (
                    f'<tr style="border-bottom:1px solid #131b28">'
                    f'<td style="padding:0.4rem 0.6rem;color:#e2e8f0">Skor {bucket}</td>'
                    f'<td style="padding:0.4rem 0.6rem;text-align:center;color:#94a3b8">{cnt}</td>'
                    f'<td style="padding:0.4rem 0.6rem;text-align:right;color:{pc};font-family:\'Space Mono\',monospace;font-size:0.8rem">{format_pnl(ap)}</td>'
                    f'</tr>'
                )
            st.markdown(_table(["Skor", "N", "Avg PnL"], rows,
                                ["left", "center", "right"]), unsafe_allow_html=True)
        else:
            st.markdown('<div class="card" style="color:#4a5568;font-size:0.8rem;padding:1rem">Execution skoru girilmemiş.</div>', unsafe_allow_html=True)

    # ── R Multiple Distribution ───────────────────────────────────────────────
    st.markdown('<div class="section-title">📉 R Multiple Dağılımı</div>', unsafe_allow_html=True)
    r_mults = [r for r in a.get("r_multiples", []) if r is not None]
    if r_mults:
        avg_r    = sum(r_mults) / len(r_mults)
        pos_r    = [r for r in r_mults if r > 0]
        neg_r    = [r for r in r_mults if r <= 0]
        avg_r_c  = "#10b981" if avg_r >= 0 else "#ef4444"
        max_abs  = max(abs(r) for r in r_mults) or 1

        bars_html = ""
        for r in sorted(r_mults, reverse=True):
            c    = "#10b981" if r > 0 else "#ef4444"
            w    = min(abs(r) / max_abs * 100, 100)
            sign = "+" if r >= 0 else ""
            bars_html += (
                f'<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.3rem">'
                f'<div style="width:55px;text-align:right;font-family:\'Space Mono\',monospace;font-size:0.72rem;color:{c}">{sign}{r:.2f}R</div>'
                f'<div style="flex:1;background:#1e2530;border-radius:3px;height:14px">'
                f'<div style="background:{c};width:{w:.1f}%;height:14px;border-radius:3px;opacity:0.9"></div>'
                f'</div></div>'
            )

        st.markdown(f"""
        <div class="card">
          <div style="display:flex;gap:1.5rem;margin-bottom:0.85rem;flex-wrap:wrap">
            <span style="font-size:0.8rem;color:#64748b">Avg R: <b style="color:{avg_r_c};font-family:'Space Mono',monospace">{"+" if avg_r>=0 else ""}{avg_r:.2f}R</b></span>
            <span style="font-size:0.8rem;color:#64748b">Pozitif: <b style="color:#10b981">{len(pos_r)}</b></span>
            <span style="font-size:0.8rem;color:#64748b">Negatif: <b style="color:#ef4444">{len(neg_r)}</b></span>
          </div>
          {bars_html}
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="card" style="color:#4a5568;font-size:0.8rem;padding:1rem">R Multiple verisi yok (kapatırken PnL girmelisiniz).</div>', unsafe_allow_html=True)
