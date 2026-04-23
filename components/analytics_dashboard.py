import streamlit as st
from analytics.engine import compute_analytics
from utils.calculations import format_pnl

# ── Palette ───────────────────────────────────────────────────────────────────
_G  = "#3fb950"   # green
_R  = "#ff7b72"   # red
_B  = "#58a6ff"   # blue
_Y  = "#e3b341"   # yellow
_DG = "#21262d"   # dark grid
_TX = "#c9d1d9"   # text
_DT = "#8b949e"   # dim text
_BG = "#161b22"   # card bg
_DBG = "#0d1117"  # deep bg


def _c(val):
    """PnL colour."""
    return _G if val >= 0 else _R


def _card(inner, accent=None, extra_style=""):
    border = f"border-top:3px solid {accent};" if accent else ""
    return (f'<div style="background:{_BG};border:1px solid {_DG};'
            f'border-radius:14px;padding:1.1rem 1.2rem;{border}{extra_style}">'
            f'{inner}</div>')


def _mini_label(txt):
    return f'<div style="font-size:0.65rem;color:{_DT};text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.25rem">{txt}</div>'


def _big_val(txt, color=_TX, size="1.4rem"):
    return f'<div style="font-family:\'Space Mono\',monospace;font-size:{size};font-weight:700;color:{color}">{txt}</div>'


def _bar(pct, color, height=7):
    p = max(0, min(100, pct))
    return (f'<div style="background:{_DG};border-radius:4px;height:{height}px;margin-top:0.3rem">'
            f'<div style="background:{color};width:{p:.1f}%;height:{height}px;border-radius:4px;transition:width 0.4s"></div></div>')


def _section(icon, title):
    st.markdown(
        f'<div style="font-family:\'Space Mono\',monospace;font-size:0.7rem;'
        f'letter-spacing:0.18em;text-transform:uppercase;color:#484f58;'
        f'margin:1.8rem 0 0.8rem;display:flex;align-items:center;gap:0.6rem">'
        f'{icon} {title}'
        f'<span style="flex:1;height:1px;background:{_DG};display:block"></span></div>',
        unsafe_allow_html=True,
    )


def render_analytics(data: dict):
    closed = data.get("closed_trades", [])
    if not closed:
        st.markdown(f"""
        <div style="text-align:center;padding:3rem;border:1px dashed {_DG};border-radius:14px">
          <div style="font-size:2.5rem;margin-bottom:0.5rem">📊</div>
          <div style="font-family:'Space Mono',monospace;font-size:0.85rem;color:{_DT}">
            Analiz için kapalı işlem gerekli
          </div>
        </div>""", unsafe_allow_html=True)
        return

    try:
        a = compute_analytics(closed)
    except Exception as ex:
        st.error(f"Analytics hatası: {ex}")
        return
    if not a:
        st.info("Yeterli veri yok.")
        return

    # ══════════════════════════════════════════════════════
    # KPIs — büyük rakamlar üst bar
    # ══════════════════════════════════════════════════════
    total_pnl   = sum(t.get("pnl", 0) for t in closed)
    win_rate    = a["win_rate"] * 100
    pf          = a["profit_factor"]
    ev          = a["ev"]
    n           = a["n"]
    avg_hold    = a.get("avg_hold_hours", 0)
    consistency = a.get("consistency", 0)
    pf_str      = f"{pf:.2f}" if pf != float("inf") else "∞"
    hold_str    = f"{avg_hold:.1f}h" if avg_hold < 48 else f"{avg_hold/24:.1f}g"

    cur_streak_val  = a.get("cur_streak_val", 0)
    cur_streak_type = a.get("cur_streak_type")
    streak_color    = _G if cur_streak_type == "W" else _R
    streak_str      = f"{'🟢' if cur_streak_type=='W' else '🔴'} {cur_streak_val} {'Kazanan' if cur_streak_type=='W' else 'Kaybeden'}"

    # row of KPI cards
    k = st.columns(7)
    kpis = [
        ("Toplam PnL",     format_pnl(total_pnl), _c(total_pnl)),
        ("Win Rate",       f"{win_rate:.1f}%",      _G if win_rate >= 50 else _R),
        ("Profit Factor",  pf_str,                  _G if pf >= 1 else _R),
        ("Expected Value", format_pnl(ev),           _c(ev)),
        ("Toplam İşlem",   str(n),                   _B),
        ("Tutarlılık",     f"{consistency:.0f}%",    _G if consistency >= 60 else _Y),
        ("Ort. Tutma",     hold_str,                  _B),
    ]
    for col, (lbl, val, col_) in zip(k, kpis):
        with col:
            st.markdown(
                f'<div style="background:{_BG};border:1px solid {_DG};border-radius:14px;padding:0.85rem 0.9rem">'
                f'{_mini_label(lbl)}{_big_val(val, col_, "1.1rem")}</div>',
                unsafe_allow_html=True,
            )

    # Streak + max streaks inline
    mws = a.get("max_win_streak", 0)
    mls = a.get("max_loss_streak", 0)
    st.markdown(
        f'<div style="display:flex;gap:1rem;flex-wrap:wrap;margin:0.6rem 0 0">'
        f'<span style="font-size:13px;color:{_DT}">Mevcut seri: <b style="color:{streak_color}">{streak_str}</b></span>'
        f'<span style="font-size:13px;color:{_DT}">En uzun kazanma serisi: <b style="color:{_G}">{mws}</b></span>'
        f'<span style="font-size:13px;color:{_DT}">En uzun kayıp serisi: <b style="color:{_R}">{mls}</b></span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════
    # TOP / WORST trades + Symbol
    # ══════════════════════════════════════════════════════
    col_top, col_bot, col_sym = st.columns(3)

    with col_top:
        _section("🏆", "En Kârlı 5 İşlem")
        rows = ""
        for t in a.get("top_trades", []):
            pnl = t.get("pnl", 0)
            sym = t.get("symbol", "?")
            rid = t.get("id", "?")
            dir_ = t.get("direction", "")
            dir_c = _G if dir_ == "LONG" else _R
            rows += (
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:0.45rem 0;border-bottom:1px solid {_DG}">'
                f'<span style="color:{_TX};font-size:14px">#{rid} '
                f'<b style="color:{dir_c}">{sym}</b></span>'
                f'<span style="font-family:\'Space Mono\',monospace;font-size:13px;color:{_G}">{format_pnl(pnl)}</span>'
                f'</div>'
            )
        st.markdown(_card(rows or f'<span style="color:{_DT}">Veri yok</span>', _G), unsafe_allow_html=True)

    with col_bot:
        _section("💀", "En Zararlı 5 İşlem")
        rows = ""
        for t in a.get("worst_trades", []):
            pnl = t.get("pnl", 0)
            sym = t.get("symbol", "?")
            rid = t.get("id", "?")
            dir_ = t.get("direction", "")
            dir_c = _G if dir_ == "LONG" else _R
            rows += (
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:0.45rem 0;border-bottom:1px solid {_DG}">'
                f'<span style="color:{_TX};font-size:14px">#{rid} '
                f'<b style="color:{dir_c}">{sym}</b></span>'
                f'<span style="font-family:\'Space Mono\',monospace;font-size:13px;color:{_R}">{format_pnl(pnl)}</span>'
                f'</div>'
            )
        st.markdown(_card(rows or f'<span style="color:{_DT}">Veri yok</span>', _R), unsafe_allow_html=True)

    with col_sym:
        _section("🪙", "Sembol İstatistikleri")
        sym_stats = a.get("symbol_stats", {})
        highlights = []
        if a.get("most_traded_sym"):
            s = a["most_traded_sym"]
            highlights.append(("📊 En çok işlem", s, f"{sym_stats[s]['count']} işlem", _B))
        if a.get("best_sym_wr"):
            s = a["best_sym_wr"]
            highlights.append(("🎯 En yüksek WR", s, f"{sym_stats[s]['win_rate']:.0f}%", _G))
        if a.get("best_sym_pnl"):
            s = a["best_sym_pnl"]
            highlights.append(("💰 En kârlı", s, format_pnl(sym_stats[s]["pnl"]), _G))
        if a.get("worst_sym_pnl"):
            s = a["worst_sym_pnl"]
            highlights.append(("🩸 En zararlı", s, format_pnl(sym_stats[s]["pnl"]), _R))

        inner = ""
        for label, sym, val, color in highlights:
            inner += (
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:0.45rem 0;border-bottom:1px solid {_DG}">'
                f'<div><div style="font-size:11px;color:{_DT}">{label}</div>'
                f'<div style="font-size:15px;font-weight:600;color:{_TX}">{sym}</div></div>'
                f'<div style="font-family:\'Space Mono\',monospace;font-size:13px;color:{color}">{val}</div>'
                f'</div>'
            )
        st.markdown(_card(inner or f'<span style="color:{_DT}">Veri yok</span>', _B), unsafe_allow_html=True)

    # ── Tüm semboller tablosu ─────────────────────────────────────────────────
    if sym_stats and len(sym_stats) > 1:
        _section("📋", "Sembol Performans Tablosu")
        rows = ""
        for s, v in sorted(sym_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
            pnl = v["pnl"]; wr = v["win_rate"]; cnt = v["count"]; ap = v["avg_pnl"]
            pc  = _c(pnl); ac = _c(ap)
            rows += (
                f'<tr style="border-bottom:1px solid {_DG}">'
                f'<td style="padding:0.5rem 0.7rem;font-weight:700;color:{_TX}">{s}</td>'
                f'<td style="padding:0.5rem 0.7rem;text-align:center;color:{_DT}">{cnt}</td>'
                f'<td style="padding:0.5rem 0.7rem;text-align:center;color:{_DT}">{wr:.0f}%</td>'
                f'<td style="padding:0.5rem 0.7rem;text-align:right;font-family:\'Space Mono\',monospace;color:{ac};font-size:13px">{format_pnl(ap)}</td>'
                f'<td style="padding:0.5rem 0.7rem;text-align:right;font-family:\'Space Mono\',monospace;color:{pc}">{format_pnl(pnl)}</td>'
                f'</tr>'
            )
        ths = "".join(
            f'<th style="padding:0.4rem 0.7rem;text-align:{a2};color:{_DT};font-size:11px;text-transform:uppercase">{h}</th>'
            for h, a2 in [("Sembol","left"),("İşlem","center"),("WR","center"),("Avg PnL","right"),("Toplam PnL","right")]
        )
        st.markdown(
            f'<div style="background:{_BG};border:1px solid {_DG};border-radius:14px;padding:0">'
            f'<table style="width:100%;border-collapse:collapse">'
            f'<thead><tr style="border-bottom:1px solid {_DG}">{ths}</tr></thead>'
            f'<tbody>{rows}</tbody></table></div>',
            unsafe_allow_html=True,
        )

    # ══════════════════════════════════════════════════════
    # Setup + Market condition
    # ══════════════════════════════════════════════════════
    col_s, col_m = st.columns(2)

    with col_s:
        _section("⚙️", "Setup Performansı")
        setups = a.get("setup_stats", {})
        if setups:
            best_s = max(setups, key=lambda s: setups[s].get("pnl", 0))
            inner  = ""
            for s, v in sorted(setups.items(), key=lambda x: x[1].get("pnl", 0), reverse=True):
                pnl = v["pnl"]; wr = v["win_rate"]; cnt = v["count"]
                star = " ⭐" if s == best_s else ""
                inner += (
                    f'<div style="margin-bottom:0.7rem">'
                    f'<div style="display:flex;justify-content:space-between;margin-bottom:0.15rem">'
                    f'<span style="font-size:14px;font-weight:600;color:{_TX}">{s}{star}</span>'
                    f'<span style="font-family:\'Space Mono\',monospace;font-size:13px;color:{_c(pnl)}">{format_pnl(pnl)}</span>'
                    f'</div>'
                    f'<div style="font-size:12px;color:{_DT};margin-bottom:0.2rem">{cnt} işlem · WR {wr:.0f}%</div>'
                    f'{_bar(wr, _G if pnl >= 0 else _R)}'
                    f'</div>'
                )
            st.markdown(_card(inner, _B), unsafe_allow_html=True)

    with col_m:
        _section("🌊", "Piyasa Koşulu")
        conds = a.get("cond_stats", {})
        if conds:
            inner = ""
            for c, v in sorted(conds.items(), key=lambda x: x[1].get("pnl", 0), reverse=True):
                pnl = v["pnl"]; wr = v["win_rate"]; tot = v["total"]
                inner += (
                    f'<div style="margin-bottom:0.7rem">'
                    f'<div style="display:flex;justify-content:space-between;margin-bottom:0.15rem">'
                    f'<span style="font-size:14px;font-weight:600;color:{_TX}">{c}</span>'
                    f'<span style="font-family:\'Space Mono\',monospace;font-size:13px;color:{_c(pnl)}">{format_pnl(pnl)}</span>'
                    f'</div>'
                    f'<div style="font-size:12px;color:{_DT};margin-bottom:0.2rem">{tot} işlem · WR {wr:.0f}%</div>'
                    f'{_bar(wr, _G if pnl >= 0 else _Y)}'
                    f'</div>'
                )
            st.markdown(_card(inner, _Y), unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════
    # Psychology + Plan + Direction
    # ══════════════════════════════════════════════════════
    col_e, col_p, col_d = st.columns(3)

    with col_e:
        _section("🧠", "Psikoloji")
        emos = a.get("emo_stats", {})
        if emos:
            inner = ""
            for e, v in sorted(emos.items(), key=lambda x: x[1].get("pnl", 0), reverse=True):
                pnl = v["pnl"]; wr = v["win_rate"]; tot = v["total"]
                inner += (
                    f'<div style="display:flex;justify-content:space-between;align-items:center;'
                    f'padding:0.4rem 0;border-bottom:1px solid {_DG}">'
                    f'<div><div style="font-size:14px;font-weight:600;color:{_TX}">{e}</div>'
                    f'<div style="font-size:12px;color:{_DT}">{tot} işlem · {wr:.0f}% WR</div></div>'
                    f'<div style="font-family:\'Space Mono\',monospace;font-size:13px;color:{_c(pnl)}">{format_pnl(pnl)}</div>'
                    f'</div>'
                )
            st.markdown(_card(inner), unsafe_allow_html=True)

    with col_p:
        _section("📋", "Plan Uyumu")
        pyw = float(a.get("plan_yes_wr", 0))
        pnw = float(a.get("plan_no_wr", 0))
        plan_yes = [t for t in closed if t.get("plan_followed") is True]
        plan_no  = [t for t in closed if not t.get("plan_followed")]
        py_pnl   = sum(t.get("pnl", 0) for t in plan_yes)
        pn_pnl   = sum(t.get("pnl", 0) for t in plan_no)
        inner = (
            f'<div style="margin-bottom:0.9rem">'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:0.15rem">'
            f'<span style="font-size:14px;color:{_G}">✅ Plana Uyuldu ({len(plan_yes)})</span>'
            f'<span style="font-family:\'Space Mono\',monospace;font-size:13px;color:{_G}">{pyw:.0f}% WR</span>'
            f'</div>'
            f'<div style="font-size:12px;color:{_DT};margin-bottom:0.2rem">{format_pnl(py_pnl)}</div>'
            f'{_bar(pyw, _G)}</div>'
            f'<div>'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:0.15rem">'
            f'<span style="font-size:14px;color:{_R}">❌ Plan Dışı ({len(plan_no)})</span>'
            f'<span style="font-family:\'Space Mono\',monospace;font-size:13px;color:{_R}">{pnw:.0f}% WR</span>'
            f'</div>'
            f'<div style="font-size:12px;color:{_DT};margin-bottom:0.2rem">{format_pnl(pn_pnl)}</div>'
            f'{_bar(pnw, _R)}</div>'
        )
        st.markdown(_card(inner), unsafe_allow_html=True)

    with col_d:
        _section("📈", "LONG vs SHORT")
        lwr = float(a.get("long_win_rate", 0))
        swr = float(a.get("short_win_rate", 0))
        lc  = int(a.get("long_count", 0))
        sc  = int(a.get("short_count", 0))
        long_t  = [t for t in closed if t.get("direction") == "LONG"]
        short_t = [t for t in closed if t.get("direction") == "SHORT"]
        l_pnl   = sum(t.get("pnl", 0) for t in long_t)
        s_pnl   = sum(t.get("pnl", 0) for t in short_t)
        inner = (
            f'<div style="margin-bottom:0.9rem">'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:0.15rem">'
            f'<span style="font-size:14px;color:{_G}">LONG ({lc})</span>'
            f'<span style="font-family:\'Space Mono\',monospace;font-size:13px;color:{_G}">{lwr:.0f}% WR</span>'
            f'</div>'
            f'<div style="font-size:12px;color:{_DT};margin-bottom:0.2rem">{format_pnl(l_pnl)}</div>'
            f'{_bar(lwr, _G)}</div>'
            f'<div>'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:0.15rem">'
            f'<span style="font-size:14px;color:{_R}">SHORT ({sc})</span>'
            f'<span style="font-family:\'Space Mono\',monospace;font-size:13px;color:{_R}">{swr:.0f}% WR</span>'
            f'</div>'
            f'<div style="font-size:12px;color:{_DT};margin-bottom:0.2rem">{format_pnl(s_pnl)}</div>'
            f'{_bar(swr, _R)}</div>'
        )
        st.markdown(_card(inner), unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════
    # Mistake analysis + Execution + R distribution
    # ══════════════════════════════════════════════════════
    col_mis, col_exec = st.columns(2)

    with col_mis:
        _section("⚠️", "Hata Analizi")
        mistakes = a.get("mistake_freq", {})
        if mistakes:
            max_cnt = max(mistakes.values())
            inner   = ""
            for m, cnt in sorted(mistakes.items(), key=lambda x: x[1], reverse=True):
                pct = (cnt / max_cnt * 100) if max_cnt else 0
                inner += (
                    f'<div style="margin-bottom:0.65rem">'
                    f'<div style="display:flex;justify-content:space-between;margin-bottom:0.15rem">'
                    f'<span style="font-size:14px;color:{_TX}">{m}</span>'
                    f'<span style="font-family:\'Space Mono\',monospace;font-size:13px;color:{_Y}">{cnt}×</span>'
                    f'</div>'
                    f'{_bar(pct, _Y)}</div>'
                )
            st.markdown(_card(inner), unsafe_allow_html=True)
        else:
            st.markdown(_card(f'<span style="color:{_DT}">Hata etiketi girilmemiş.</span>'), unsafe_allow_html=True)

    with col_exec:
        _section("⭐", "Execution Skoru Analizi")
        exec_a = a.get("exec_analysis", {})
        if exec_a:
            inner = ""
            for bucket in sorted(exec_a.keys()):
                v  = exec_a[bucket]
                ap = float(v.get("avg_pnl", 0))
                c2 = int(v.get("count", 0))
                inner += (
                    f'<div style="display:flex;justify-content:space-between;align-items:center;'
                    f'padding:0.4rem 0;border-bottom:1px solid {_DG}">'
                    f'<span style="font-size:14px;color:{_TX}">Skor {bucket}</span>'
                    f'<span style="color:{_DT};font-size:13px">{c2} işlem</span>'
                    f'<span style="font-family:\'Space Mono\',monospace;font-size:13px;color:{_c(ap)}">{format_pnl(ap)}</span>'
                    f'</div>'
                )
            st.markdown(_card(inner), unsafe_allow_html=True)
        else:
            st.markdown(_card(f'<span style="color:{_DT}">Execution skoru girilmemiş.</span>'), unsafe_allow_html=True)

    # ── R Multiple dağılımı ───────────────────────────────────────────────────
    _section("📉", "R Multiple Dağılımı")
    r_mults = [r for r in a.get("r_multiples", []) if r is not None]
    if r_mults:
        avg_r   = sum(r_mults) / len(r_mults)
        pos_r   = [r for r in r_mults if r > 0]
        neg_r   = [r for r in r_mults if r <= 0]
        max_abs = max(abs(r) for r in r_mults) or 1
        bars    = ""
        for r in sorted(r_mults, reverse=True):
            c_ = _G if r > 0 else _R
            w  = min(abs(r) / max_abs * 100, 100)
            s  = "+" if r >= 0 else ""
            bars += (
                f'<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.3rem">'
                f'<div style="width:58px;text-align:right;font-family:\'Space Mono\',monospace;'
                f'font-size:12px;color:{c_}">{s}{r:.2f}R</div>'
                f'<div style="flex:1;background:{_DG};border-radius:3px;height:14px">'
                f'<div style="background:{c_};width:{w:.1f}%;height:14px;border-radius:3px;opacity:0.9"></div>'
                f'</div></div>'
            )
        header = (
            f'<div style="display:flex;gap:1.5rem;margin-bottom:0.85rem;flex-wrap:wrap">'
            f'<span style="font-size:13px;color:{_DT}">Avg R: '
            f'<b style="color:{_c(avg_r)};font-family:\'Space Mono\',monospace">'
            f'{"+" if avg_r>=0 else ""}{avg_r:.2f}R</b></span>'
            f'<span style="font-size:13px;color:{_DT}">Pozitif: <b style="color:{_G}">{len(pos_r)}</b></span>'
            f'<span style="font-size:13px;color:{_DT}">Negatif: <b style="color:{_R}">{len(neg_r)}</b></span>'
            f'</div>'
        )
        st.markdown(_card(header + bars), unsafe_allow_html=True)
