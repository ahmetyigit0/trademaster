import streamlit as st
from analytics.engine import compute_analytics
from utils.calculations import format_pnl


def render_analytics(data: dict):
    closed = data.get("closed_trades", [])
    if len(closed) < 2:
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:#2d3a4e;border:1px dashed #1e2530;border-radius:12px">
            <div style="font-size:2.5rem;margin-bottom:0.5rem">📊</div>
            <div style="font-family:'Space Mono',monospace;font-size:0.8rem">Analiz için en az 2 kapalı işlem gerekli</div>
        </div>""", unsafe_allow_html=True)
        return

    a = compute_analytics(closed)

    # ── Section: Setup Performance ────────────────────────────────────────────
    st.markdown('<div class="section-title">⚙️ Setup Performansı</div>', unsafe_allow_html=True)
    setups = a.get("setup_stats", {})
    if setups:
        best = max(setups, key=lambda s: setups[s]["pnl"])
        rows = ""
        for s, v in sorted(setups.items(), key=lambda x: x[1]["pnl"], reverse=True):
            wr   = v["win_rate"]
            pnl  = v["pnl"]
            cnt  = v["count"]
            pc   = "#10b981" if pnl >= 0 else "#ef4444"
            star = " ⭐" if s == best else ""
            rows += f"""
            <tr>
              <td style="padding:0.5rem 0.75rem;font-weight:600;color:#e2e8f0">{s}{star}</td>
              <td style="padding:0.5rem 0.75rem;text-align:center">{cnt}</td>
              <td style="padding:0.5rem 0.75rem;text-align:center">{wr:.0f}%</td>
              <td style="padding:0.5rem 0.75rem;text-align:right;color:{pc};font-family:'Space Mono',monospace">{format_pnl(pnl)}</td>
            </tr>"""
        st.markdown(f"""
        <div class="card">
          <table style="width:100%;border-collapse:collapse">
            <thead>
              <tr style="border-bottom:1px solid #1e2530;color:#4a5568;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em">
                <th style="padding:0.4rem 0.75rem;text-align:left">Setup</th>
                <th style="padding:0.4rem 0.75rem;text-align:center">İşlem</th>
                <th style="padding:0.4rem 0.75rem;text-align:center">Win Rate</th>
                <th style="padding:0.4rem 0.75rem;text-align:right">PnL</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </div>""", unsafe_allow_html=True)

    # ── Section: Market Condition ─────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Piyasa Koşulu Analizi</div>', unsafe_allow_html=True)
    conds = a.get("cond_stats", {})
    if conds:
        rows = ""
        for c, v in sorted(conds.items(), key=lambda x: x[1]["pnl"], reverse=True):
            pc = "#10b981" if v["pnl"] >= 0 else "#ef4444"
            rows += f"""
            <tr>
              <td style="padding:0.5rem 0.75rem;font-weight:600;color:#e2e8f0">{c}</td>
              <td style="padding:0.5rem 0.75rem;text-align:center">{v['total']}</td>
              <td style="padding:0.5rem 0.75rem;text-align:center">{v['win_rate']:.0f}%</td>
              <td style="padding:0.5rem 0.75rem;text-align:right;color:{pc};font-family:'Space Mono',monospace">{format_pnl(v['pnl'])}</td>
            </tr>"""
        st.markdown(f"""
        <div class="card">
          <table style="width:100%;border-collapse:collapse">
            <thead>
              <tr style="border-bottom:1px solid #1e2530;color:#4a5568;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em">
                <th style="padding:0.4rem 0.75rem;text-align:left">Koşul</th>
                <th style="padding:0.4rem 0.75rem;text-align:center">İşlem</th>
                <th style="padding:0.4rem 0.75rem;text-align:center">Win Rate</th>
                <th style="padding:0.4rem 0.75rem;text-align:right">PnL</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </div>""", unsafe_allow_html=True)

    # ── Section: Psychology & Plan ────────────────────────────────────────────
    acol1, acol2 = st.columns(2)
    with acol1:
        st.markdown('<div class="section-title">🧠 Psikoloji Analizi</div>', unsafe_allow_html=True)
        emos = a.get("emo_stats", {})
        if emos:
            rows = ""
            for e, v in sorted(emos.items(), key=lambda x: x[1]["pnl"], reverse=True):
                pc = "#10b981" if v["pnl"] >= 0 else "#ef4444"
                rows += f"""
                <tr>
                  <td style="padding:0.4rem 0.6rem;font-weight:600;color:#e2e8f0">{e}</td>
                  <td style="padding:0.4rem 0.6rem;text-align:center">{v['total']}</td>
                  <td style="padding:0.4rem 0.6rem;text-align:center">{v['win_rate']:.0f}%</td>
                  <td style="padding:0.4rem 0.6rem;text-align:right;color:{pc};font-family:'Space Mono',monospace;font-size:0.8rem">{format_pnl(v['pnl'])}</td>
                </tr>"""
            st.markdown(f"""
            <div class="card">
              <table style="width:100%;border-collapse:collapse">
                <thead>
                  <tr style="border-bottom:1px solid #1e2530;color:#4a5568;font-size:0.68rem;text-transform:uppercase">
                    <th style="padding:0.3rem 0.6rem;text-align:left">Duygu</th>
                    <th style="padding:0.3rem 0.6rem;text-align:center">N</th>
                    <th style="padding:0.3rem 0.6rem;text-align:center">WR</th>
                    <th style="padding:0.3rem 0.6rem;text-align:right">PnL</th>
                  </tr>
                </thead>
                <tbody>{rows}</tbody>
              </table>
            </div>""", unsafe_allow_html=True)

    with acol2:
        st.markdown('<div class="section-title">📋 Plan Uyumu</div>', unsafe_allow_html=True)
        plan_yes_wr = a.get("plan_yes_wr", 0)
        plan_no_wr  = a.get("plan_no_wr", 0)
        st.markdown(f"""
        <div class="card">
          <div style="display:flex;flex-direction:column;gap:0.75rem">
            <div>
              <div style="display:flex;justify-content:space-between;margin-bottom:0.3rem">
                <span style="color:#10b981;font-size:0.8rem">✅ Plana Uyuldu</span>
                <span style="font-family:'Space Mono',monospace;font-size:0.85rem;color:#10b981">{plan_yes_wr:.1f}%</span>
              </div>
              <div style="background:#1e2530;border-radius:4px;height:8px">
                <div style="background:#10b981;width:{plan_yes_wr}%;height:8px;border-radius:4px;transition:width 0.5s"></div>
              </div>
            </div>
            <div>
              <div style="display:flex;justify-content:space-between;margin-bottom:0.3rem">
                <span style="color:#ef4444;font-size:0.8rem">❌ Plan Dışı</span>
                <span style="font-family:'Space Mono',monospace;font-size:0.85rem;color:#ef4444">{plan_no_wr:.1f}%</span>
              </div>
              <div style="background:#1e2530;border-radius:4px;height:8px">
                <div style="background:#ef4444;width:{plan_no_wr}%;height:8px;border-radius:4px;transition:width 0.5s"></div>
              </div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Section: Mistake Analysis ─────────────────────────────────────────────
    st.markdown('<div class="section-title">⚠️ Hata Analizi</div>', unsafe_allow_html=True)
    mistakes = a.get("mistake_freq", {})
    if mistakes:
        total_m = sum(mistakes.values())
        sorted_m = sorted(mistakes.items(), key=lambda x: x[1], reverse=True)
        bars = ""
        for m, cnt in sorted_m:
            pct = cnt / total_m * 100
            bars += f"""
            <div style="margin-bottom:0.6rem">
              <div style="display:flex;justify-content:space-between;margin-bottom:0.2rem">
                <span style="font-size:0.8rem;color:#e2e8f0">{m}</span>
                <span style="font-family:'Space Mono',monospace;font-size:0.78rem;color:#f59e0b">{cnt}x</span>
              </div>
              <div style="background:#1e2530;border-radius:4px;height:6px">
                <div style="background:#f59e0b;width:{pct}%;height:6px;border-radius:4px"></div>
              </div>
            </div>"""
        st.markdown(f'<div class="card">{bars}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="card" style="color:#4a5568;font-size:0.8rem">Hata etiketi girilmemiş.</div>', unsafe_allow_html=True)

    # ── Section: LONG vs SHORT + Execution ───────────────────────────────────
    bc1, bc2 = st.columns(2)
    with bc1:
        st.markdown('<div class="section-title">📈 Yön Performansı</div>', unsafe_allow_html=True)
        lwr = a.get("long_win_rate", 0)
        swr = a.get("short_win_rate", 0)
        lc  = a.get("long_count", 0)
        sc  = a.get("short_count", 0)
        st.markdown(f"""
        <div class="card">
          <div style="display:flex;justify-content:space-around;text-align:center">
            <div>
              <div style="font-size:0.68rem;color:#4a5568;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.3rem">LONG ({lc})</div>
              <div style="font-size:1.6rem;font-weight:700;color:#10b981;font-family:'Space Mono',monospace">{lwr:.1f}%</div>
              <div style="font-size:0.7rem;color:#64748b">win rate</div>
            </div>
            <div style="width:1px;background:#1e2530"></div>
            <div>
              <div style="font-size:0.68rem;color:#4a5568;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.3rem">SHORT ({sc})</div>
              <div style="font-size:1.6rem;font-weight:700;color:#ef4444;font-family:'Space Mono',monospace">{swr:.1f}%</div>
              <div style="font-size:0.7rem;color:#64748b">win rate</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    with bc2:
        st.markdown('<div class="section-title">⭐ Execution Analizi</div>', unsafe_allow_html=True)
        exec_a = a.get("exec_analysis", {})
        if exec_a:
            rows = ""
            for bucket in sorted(exec_a.keys()):
                v = exec_a[bucket]
                pc = "#10b981" if v["avg_pnl"] >= 0 else "#ef4444"
                rows += f"""
                <tr>
                  <td style="padding:0.4rem 0.6rem;color:#e2e8f0">Skor {bucket}</td>
                  <td style="padding:0.4rem 0.6rem;text-align:center">{v['count']}</td>
                  <td style="padding:0.4rem 0.6rem;text-align:right;color:{pc};font-family:'Space Mono',monospace;font-size:0.8rem">{format_pnl(v['avg_pnl'])}</td>
                </tr>"""
            st.markdown(f"""
            <div class="card">
              <table style="width:100%;border-collapse:collapse">
                <thead>
                  <tr style="border-bottom:1px solid #1e2530;color:#4a5568;font-size:0.68rem;text-transform:uppercase">
                    <th style="padding:0.3rem 0.6rem;text-align:left">Skor</th>
                    <th style="padding:0.3rem 0.6rem;text-align:center">N</th>
                    <th style="padding:0.3rem 0.6rem;text-align:right">Avg PnL</th>
                  </tr>
                </thead>
                <tbody>{rows}</tbody>
              </table>
            </div>""", unsafe_allow_html=True)

    # ── R Distribution ────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📉 R Multiple Dağılımı</div>', unsafe_allow_html=True)
    r_mults = a.get("r_multiples", [])
    if r_mults:
        avg_r = sum(r_mults) / len(r_mults)
        positive_r = [r for r in r_mults if r > 0]
        negative_r = [r for r in r_mults if r <= 0]
        avg_r_c = "#10b981" if avg_r >= 0 else "#ef4444"
        bars_html = ""
        for r in sorted(r_mults, reverse=True):
            c = "#10b981" if r > 0 else "#ef4444"
            w = min(abs(r) * 15, 100)
            sign = "+" if r >= 0 else ""
            bars_html += f"""
            <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.2rem">
              <div style="width:50px;text-align:right;font-family:'Space Mono',monospace;font-size:0.72rem;color:{c}">{sign}{r:.1f}R</div>
              <div style="flex:1;background:#1e2530;border-radius:3px;height:14px">
                <div style="background:{c};width:{w}%;height:14px;border-radius:3px;opacity:0.85"></div>
              </div>
            </div>"""
        st.markdown(f"""
        <div class="card">
          <div style="display:flex;gap:1.5rem;margin-bottom:0.75rem;flex-wrap:wrap">
            <span style="font-size:0.8rem;color:#64748b">Avg R: <b style="color:{avg_r_c};font-family:'Space Mono',monospace">{"+" if avg_r>=0 else ""}{avg_r:.2f}R</b></span>
            <span style="font-size:0.8rem;color:#64748b">Pozitif: <b style="color:#10b981">{len(positive_r)}</b></span>
            <span style="font-size:0.8rem;color:#64748b">Negatif: <b style="color:#ef4444">{len(negative_r)}</b></span>
          </div>
          {bars_html}
        </div>""", unsafe_allow_html=True)
