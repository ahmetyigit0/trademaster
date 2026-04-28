"""
Pozisyon Giriş Formu — Kompakt Premium UI
Sade, okunabilir, her şey tek ekranda.
"""
import streamlit as st
from datetime import datetime
from utils.data_manager import save_data
from utils.calculations import (
    calculate_avg_entry, calculate_position_size,
    calculate_rr, calculate_position_heat, rr_color
)

SETUP_TYPES  = ["Liquidity Sweep", "Breakout", "Trend Devamı", "Range", "Haber", "Diğer"]
EMOTIONS     = ["Sakin 😌", "FOMO 😰", "İntikam 😤", "Endişeli 😟", "Güvenli 💪"]
MARKET_CONDS = ["Trend", "Range", "Volatil", "Düşük Hacim", "Haber"]
SL_PRESETS   = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0]
TP_PRESETS   = [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]

_SZ = "pf__sz"; _VER = "pf__ver"
_G="#3fb950"; _R="#ff7b72"; _B="#58a6ff"; _Y="#e3b341"
_DG="#21262d"; _DT="#8b949e"; _TX="#e6edf3"; _DB="#0d1117"
_BG="#161b22"


def render_position_form():   _render_form()
def render_edit_form(pos_id): _render_form(edit_id=pos_id)


# ── UI yardımcıları ────────────────────────────────────────────────────────────

def _divider(title: str, color: str = _B):
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;margin:14px 0 10px'>"
        f"<div style='width:3px;height:16px;background:{color};"
        f"border-radius:2px;flex-shrink:0'></div>"
        f"<span style='font-size:12px;font-weight:700;color:{color};"
        f"text-transform:uppercase;letter-spacing:0.12em'>{title}</span>"
        f"<div style='flex:1;height:1px;background:{_DG}'></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

def _info_box(label, value, color=_TX, bg=_DB):
    st.markdown(
        f"<div style='background:{bg};border:1px solid {_DG};border-radius:8px;"
        f"padding:7px 10px;text-align:center'>"
        f"<div style='font-size:11px;color:{_DT};margin-bottom:2px'>{label}</div>"
        f"<div style='font-size:14px;font-weight:700;color:{color};"
        f"font-family:\"Space Mono\",monospace'>{value}</div></div>",
        unsafe_allow_html=True,
    )

def _section_wrap(content_fn, bg=_BG, border=_DG, radius=12, pad="0.9rem 1rem 0.6rem"):
    st.markdown(
        f"<div style='background:{bg};border:1px solid {border};"
        f"border-radius:{radius}px;padding:{pad};margin-bottom:8px'>",
        unsafe_allow_html=True,
    )
    content_fn()
    st.markdown("</div>", unsafe_allow_html=True)


# ── Ana form ──────────────────────────────────────────────────────────────────

def _render_form(edit_id=None, draft_edit=None):
    # draft_edit modu: taslağı düzenle
    is_draft_edit = draft_edit is not None
    editing       = edit_id is not None
    did           = draft_edit.get("id") if is_draft_edit else None
    px            = f"pf_de_{did}_" if is_draft_edit else (f"pf_{edit_id}_" if editing else "pf_")
    pos           = None

    if is_draft_edit:
        pos     = draft_edit   # taslak veriyi pos olarak kullan
        editing = True         # form editing modunda açılsın (alanlar dolu)
    elif editing:
        pos = next((p for p in st.session_state.data["active_positions"]
                    if p["id"] == edit_id), None)
        if pos is None:
            st.error("Pozisyon bulunamadı."); return

    sz_key  = f"{_SZ}_{px}"
    ver_key = f"{_VER}_{px}"

    # ── Başlık ────────────────────────────────────────────────────────────────
    if is_draft_edit:
        st.markdown(
            f"<div style='background:#0d1a2a;border:1px solid {_B}60;"
            f"border-left:3px solid {_B};border-radius:10px;"
            f"padding:8px 14px;margin-bottom:10px;font-size:14px;color:#f0f6fc'>"
            f"📋 <b>Taslak Düzenle</b> — #{did} {pos.get('symbol','')} "
            f"<span style='font-size:12px;font-weight:400;color:{_DT}'>"
            f"· Tüm alanlar düzenlenebilir · ✅ İşlemi Aç veya 📋 Güncelle</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    elif not editing:
        st.markdown(
            f"<div style='background:#0d1520;border:1px solid {_B}40;"
            f"border-left:3px solid {_B};border-radius:10px;"
            f"padding:8px 14px;margin-bottom:10px;"
            f"font-style:italic;color:#6e7681;font-size:13px;line-height:1.6'>"
            f"\"Herkes teknik analiz bilebilir — çok azı plana sadık kalabilir.\"</div>",
            unsafe_allow_html=True,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # SATIR 1 — Sembol / Yön / Kaldıraç / Sermaye / Risk  (tek satır kompakt)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(
        f"<div style='background:{_BG};border:1px solid {_DG};"
        f"border-radius:12px;padding:10px 12px;margin-bottom:8px'>",
        unsafe_allow_html=True,
    )

    r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns([2.5, 2, 1.5, 2.5, 2])

    with r1c1:
        symbol = st.text_input(
            "Sembol", value=pos["symbol"] if editing else "",
            placeholder="BTC / ETH / SOL...",
            key=f"{px}sym",
        ).upper().strip()

    with r1c2:
        dir_default = pos["direction"] if editing else "LONG"
        if f"{px}dir" not in st.session_state:
            st.session_state[f"{px}dir"] = dir_default
        direction = st.session_state[f"{px}dir"]

        st.markdown(
            f"<div style='font-size:12px;font-weight:600;color:{_DT};"
            f"margin-bottom:5px'>Yön</div>",
            unsafe_allow_html=True,
        )
        d1, d2 = st.columns(2)
        with d1:
            is_long = direction == "LONG"
            long_style = (
                "background:linear-gradient(135deg,#0e3320,#196c2e);"
                "color:#3fb950;border:2px solid #3fb950;font-weight:800;"
                "font-size:15px;letter-spacing:0.05em"
                if is_long else
                "background:#161b22;color:#484f58;border:1.5px solid #21262d;"
                "font-size:14px"
            )
            if st.button(
                "▲  LONG",
                key=f"{px}long",
                use_container_width=True,
                type="primary" if is_long else "secondary",
            ):
                st.session_state[f"{px}dir"] = "LONG"; st.rerun()
        with d2:
            is_short = direction == "SHORT"
            if st.button(
                "▼  SHORT",
                key=f"{px}short",
                use_container_width=True,
                type="primary" if is_short else "secondary",
            ):
                st.session_state[f"{px}dir"] = "SHORT"; st.rerun()

        dc  = _G if direction == "LONG" else _R
        dbg = "#071a0e" if direction == "LONG" else "#1c0505"
        arrow = "▲" if direction == "LONG" else "▼"
        st.markdown(
            f"<div style='text-align:center;background:{dbg};"
            f"border:1.5px solid {dc};border-radius:8px;padding:5px 0;"
            f"font-size:13px;font-weight:800;color:{dc};margin-top:4px;"
            f"letter-spacing:0.08em;font-family:\"Space Mono\",monospace'>"
            f"{arrow} {direction}</div>",
            unsafe_allow_html=True,
        )

    with r1c3:
        lev = st.number_input("Kaldıraç ×",
            min_value=1.0, max_value=500.0,
            value=float(pos.get("leverage",1)) if editing else 1.0,
            step=1.0, key=f"{px}lev")

    with r1c4:
        capital = st.number_input("Sermaye ($)",
            min_value=1.0,
            value=float(pos["capital"]) if editing else 10000.0,
            step=100.0, key=f"{px}cap")
        # sz_key başlangıçta sermaye olsun
        if sz_key not in st.session_state:
            st.session_state[sz_key]  = float(pos.get("position_size", capital)) if editing else float(capital)
            st.session_state[ver_key] = 0

    with r1c5:
        risk_pct = st.number_input("Max Risk %",
            min_value=0.1, max_value=50.0,
            value=float(pos["risk_pct"]) if editing else 2.0,
            step=0.1, key=f"{px}rp")

    st.markdown("</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SATIR 2 — ENTRY + SL/TP  (2/3 + 1/3)
    # ══════════════════════════════════════════════════════════════════════════
    left_col, right_col = st.columns([2, 1], gap="medium")

    # ── SOL: Entry Noktaları ──────────────────────────────────────────────────
    with left_col:
        st.markdown(
            f"<div style='background:{_BG};border:1px solid {_DG};"
            f"border-radius:12px;padding:10px 12px 6px'>",
            unsafe_allow_html=True,
        )
        _divider("📍 Entry Noktaları", _B)

        if f"{px}nentries" not in st.session_state:
            ne_init = len(pos.get("entries",[])) if editing else 1
            st.session_state[f"{px}nentries"] = max(ne_init, 1)

        ne = st.session_state[f"{px}nentries"]
        entries = []
        default_ew = round(100.0 / ne, 1)

        for i in range(ne):
            ep_d = float(pos["entries"][i]["price"])  if editing and i < len(pos.get("entries",[])) else 0.0
            ew_d = float(pos["entries"][i]["weight"]) if editing and i < len(pos.get("entries",[])) else default_ew

            ec1, ec2, ec3, ec4 = st.columns([0.25, 1.3, 0.9, 0.5])
            with ec1:
                st.markdown(
                    f"<div style='padding-top:26px;font-size:12px;"
                    f"color:{_B};font-weight:700;text-align:center'>{i+1}</div>",
                    unsafe_allow_html=True,
                )
            with ec2:
                ep = st.number_input(f"Fiyat {i+1}", min_value=0.0, value=ep_d,
                    format="%.4f", key=f"{px}ep{i}", label_visibility="collapsed")
            with ec3:
                ew = st.number_input(f"% {i+1}", min_value=0.1, max_value=100.0,
                    value=ew_d, step=5.0, key=f"{px}ew{i}",
                    label_visibility="collapsed")
            with ec4:
                usdt_v = st.session_state.get(sz_key, capital) * (ew/100)
                st.markdown(
                    f"<div style='padding-top:26px;font-size:12px;"
                    f"color:{_B};font-family:\"Space Mono\",monospace'>"
                    f"${usdt_v:,.0f}</div>",
                    unsafe_allow_html=True,
                )
            if ep > 0:
                entries.append({"price": ep, "weight": ew})

        # + Entry ekleme butonu
        badd, brem, binfo = st.columns([1, 1, 3])
        with badd:
            if st.button("＋ Entry Ekle", key=f"{px}addent", use_container_width=True):
                st.session_state[f"{px}nentries"] += 1; st.rerun()
        with brem:
            if ne > 1 and st.button("－ Sil", key=f"{px}remdent", use_container_width=True):
                st.session_state[f"{px}nentries"] -= 1; st.rerun()

        # Entry özet
        if entries:
            avg_e = calculate_avg_entry(entries)
            total_w = sum(e["weight"] for e in entries)
            wc = _G if abs(total_w-100)<=1 else _Y
            st.markdown(
                f"<div style='display:flex;gap:12px;margin-top:6px;"
                f"padding:6px 8px;background:{_DB};border-radius:7px;"
                f"font-size:12px;flex-wrap:wrap'>"
                f"<span style='color:{_DT}'>Ort. Entry: "
                f"<b style='color:{_TX};font-family:\"Space Mono\",monospace'>${avg_e:,.4f}</b></span>"
                f"<span style='color:{_DT}'>Ağırlık: <b style='color:{wc}'>{total_w:.0f}%</b></span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            avg_e = 0.0

        st.markdown("</div>", unsafe_allow_html=True)

    # ── SAĞ: SL / TP ─────────────────────────────────────────────────────────
    with right_col:
        st.markdown(
            f"<div style='background:{_BG};border:1px solid {_DG};"
            f"border-radius:12px;padding:10px 12px 6px'>",
            unsafe_allow_html=True,
        )
        _divider("🛑 Stop Loss", _R)

        sl_pct_key = f"{px}sl_pct"
        sl_price_key = f"{px}sl"

        # Slider değişince fiyat güncelle
        def _sl_slider_changed():
            pct = st.session_state.get(sl_pct_key, 1.0)
            ae  = st.session_state.get(f"{px}_avg_e", 0.0)
            if ae > 0:
                d = st.session_state.get(f"{px}dir", "LONG")
                price = ae * (1 - pct/100) if d == "LONG" else ae * (1 + pct/100)
                st.session_state[sl_price_key] = round(price, 6)

        # Fiyat değişince slider güncelle
        def _sl_price_changed():
            ae = st.session_state.get(f"{px}_avg_e", 0.0)
            p  = st.session_state.get(sl_price_key, 0.0)
            if ae > 0 and p > 0:
                pct = abs(ae - p) / ae * 100
                # En yakın preset'e snap
                closest = min(SL_PRESETS, key=lambda x: abs(x - pct))
                st.session_state[sl_pct_key] = closest

        # avg_e'yi session_state'e yaz (callback'ler erişebilsin)
        st.session_state[f"{px}_avg_e"] = avg_e

        if sl_pct_key not in st.session_state:
            # Editing modunda mevcut fiyattan başlangıç yüzdesi hesapla
            if editing and pos.get("stop_loss",0) > 0 and avg_e > 0:
                init_pct = abs(avg_e - pos["stop_loss"]) / avg_e * 100
                closest  = min(SL_PRESETS, key=lambda x: abs(x - init_pct))
                st.session_state[sl_pct_key] = closest
            else:
                st.session_state[sl_pct_key] = 1.0

        sl_pct_v = st.select_slider(
            "SL %", options=SL_PRESETS,
            value=st.session_state[sl_pct_key],
            key=sl_pct_key,
            on_change=_sl_slider_changed,
            label_visibility="visible",
        )

        # Slider'dan hesaplanan fiyat (eğer henüz elle yazılmadıysa)
        sl_default = 0.0
        if editing and sl_price_key not in st.session_state:
            sl_default = float(pos.get("stop_loss", 0.0))
        elif avg_e > 0 and sl_price_key not in st.session_state:
            d = direction
            sl_default = round(avg_e * (1 - sl_pct_v/100) if d=="LONG" else avg_e * (1 + sl_pct_v/100), 6)
        else:
            sl_default = float(st.session_state.get(sl_price_key, 0.0))

        stop_loss = st.number_input("SL Fiyat",
            min_value=0.0,
            value=sl_default,
            format="%.4f", key=sl_price_key,
            on_change=_sl_price_changed,
            label_visibility="collapsed")

        # SL risk bilgisi
        if stop_loss > 0 and avg_e > 0:
            pdiff   = abs(avg_e - stop_loss)
            sl_pct_r = pdiff / avg_e * 100
            sl_loss  = st.session_state.get(sz_key, capital) * pdiff / avg_e
            sl_cap   = sl_loss / capital * 100
            rc = _R if sl_cap > risk_pct * 1.2 else _Y if sl_cap > risk_pct else _G
            st.markdown(
                f"<div style='background:#120808;border:1px solid #2d1010;"
                f"border-radius:7px;padding:5px 8px;font-size:12px;margin-top:4px'>"
                f"<span style='color:{rc};font-weight:700'>-${sl_loss:,.2f}</span>"
                f"<span style='color:{_DT};margin-left:8px'>%{sl_cap:.2f} sermaye</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Take Profits
        _divider("🎯 Take Profit", _G)

        if f"{px}ntp" not in st.session_state:
            ntp_init = len(pos.get("take_profits",[])) if editing else 1
            st.session_state[f"{px}ntp"] = max(ntp_init, 1)

        ntp         = st.session_state[f"{px}ntp"]
        take_profits = []
        default_tw   = round(100.0/ntp, 1)

        for i in range(ntp):
            tp_pd = float(pos["take_profits"][i]["price"])  if editing and i < len(pos.get("take_profits",[])) else 0.0
            tp_wd = float(pos["take_profits"][i]["weight"]) if editing and i < len(pos.get("take_profits",[])) else default_tw
            tp_pct_key  = f"{px}tp_pct_{i}"
            tp_price_key= f"{px}tpp{i}"

            # Slider → fiyat callback
            def _tp_slider_cb(idx=i, ppk=tp_pct_key, tpk=tp_price_key):
                pct = st.session_state.get(ppk, 1.5)
                ae  = st.session_state.get(f"{px}_avg_e", 0.0)
                if ae > 0:
                    d = st.session_state.get(f"{px}dir", "LONG")
                    price = ae * (1 + pct/100) if d == "LONG" else ae * (1 - pct/100)
                    st.session_state[tpk] = round(price, 6)

            # Fiyat → slider callback
            def _tp_price_cb(ppk=tp_pct_key, tpk=tp_price_key):
                ae = st.session_state.get(f"{px}_avg_e", 0.0)
                p  = st.session_state.get(tpk, 0.0)
                if ae > 0 and p > 0:
                    pct     = abs(p - ae) / ae * 100
                    closest = min(TP_PRESETS, key=lambda x: abs(x - pct))
                    st.session_state[ppk] = closest

            if tp_pct_key not in st.session_state:
                if editing and tp_pd > 0 and avg_e > 0:
                    init_pct = abs(avg_e - tp_pd) / avg_e * 100
                    st.session_state[tp_pct_key] = min(TP_PRESETS, key=lambda x: abs(x - init_pct))
                else:
                    st.session_state[tp_pct_key] = TP_PRESETS[min(i*2, len(TP_PRESETS)-1)]

            tp_pct_v = st.select_slider(
                f"TP{i+1} %", options=TP_PRESETS,
                value=st.session_state[tp_pct_key],
                key=tp_pct_key,
                on_change=_tp_slider_cb,
                label_visibility="collapsed",
            )

            # Slider'dan default fiyat
            if editing and tp_price_key not in st.session_state:
                tp_default = tp_pd
            elif avg_e > 0 and tp_price_key not in st.session_state:
                d = direction
                tp_default = round(avg_e * (1 + tp_pct_v/100) if d=="LONG" else avg_e * (1 - tp_pct_v/100), 6)
            else:
                tp_default = float(st.session_state.get(tp_price_key, 0.0))

            tc1, tc2 = st.columns([1.3, 0.7])
            with tc1:
                tp_p = st.number_input(
                    f"TP{i+1} Fiyat", min_value=0.0, value=tp_default,
                    format="%.4f", key=tp_price_key,
                    on_change=_tp_price_cb,
                    label_visibility="collapsed")
            with tc2:
                tp_w = st.number_input(
                    f"TP{i+1} %", min_value=0.1, max_value=100.0,
                    value=tp_wd, step=5.0, key=f"{px}tpw{i}",
                    label_visibility="collapsed")

            # TP kâr + RR hint
            if tp_p > 0 and avg_e > 0:
                move   = abs(tp_p - avg_e) / avg_e
                profit = st.session_state.get(sz_key, capital) * (tp_w/100) * move
                sl_d   = abs(avg_e - stop_loss) if stop_loss > 0 else 0
                rr_v   = round(move / (sl_d/avg_e), 2) if sl_d > 0 else 0
                rrc    = _G if rr_v >= 2 else _Y if rr_v >= 1 else _R
                pct_move = round(move * 100, 2)
                st.markdown(
                    f"<div style='font-size:11px;color:{_DT};margin-bottom:6px'>"
                    f"<b style='color:{_G}'>+${profit:,.2f}</b>  "
                    f"<span style='color:{_DT}'>%{pct_move}</span>  "
                    f"<span style='color:{rrc}'>RR:{rr_v:.1f}</span></div>",
                    unsafe_allow_html=True,
                )
            if tp_p > 0:
                take_profits.append({"price": tp_p, "weight": tp_w})

        # TP ekleme
        ta1, ta2 = st.columns(2)
        with ta1:
            if st.button("＋ TP", key=f"{px}addtp", use_container_width=True):
                st.session_state[f"{px}ntp"] += 1; st.rerun()
        with ta2:
            if ntp > 1 and st.button("－ TP", key=f"{px}remtp", use_container_width=True):
                st.session_state[f"{px}ntp"] -= 1; st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SATIR 3 — Risk / Pozisyon analizi (compact)
    # ══════════════════════════════════════════════════════════════════════════
    calc = {}
    rr   = None
    if entries and stop_loss > 0 and avg_e > 0:
        calc = calculate_position_size(capital, risk_pct, avg_e, stop_loss)
        rr   = calculate_rr(avg_e, stop_loss, take_profits, direction)

    if calc:
        st.markdown(
            f"<div style='background:{_BG};border:1px solid {_DG};"
            f"border-radius:12px;padding:10px 12px;margin-bottom:8px'>",
            unsafe_allow_html=True,
        )

        # Önerilen pozisyon büyüklüğü kontrolü
        if not calc["can_use_full"]:
            rec = calc["recommended_size"]
            ra1, ra2 = st.columns([3, 1])
            with ra1:
                st.warning(
                    f"⚠️ Tam sermaye riski **%{calc['full_capital_risk_pct']:.2f}** "
                    f"(limit: %{risk_pct:.1f}) — Önerilen: **${rec:,.0f}**"
                )
            with ra2:
                if st.button("✓ Uygula", key=f"{px}apply", type="primary",
                             use_container_width=True):
                    st.session_state[sz_key]  = float(rec)
                    st.session_state[ver_key] = st.session_state.get(ver_key,0) + 1
                    st.rerun()
        else:
            if st.session_state.get(sz_key, 0) < 1:
                st.session_state[sz_key] = float(capital)

        # Metrikler satırı
        ver        = st.session_state.get(ver_key, 0)
        widget_key = f"{px}psz_v{ver}"
        def _sync():
            st.session_state[sz_key] = float(st.session_state.get(widget_key, 0.0))

        # position_size session_state'ten oku — widget render'dan önce hesapla
        position_size = float(st.session_state.get(sz_key, capital))

        # Metrik hesaplamaları (position_size kullanarak)
        risk_per_unit = calc.get("risk_per_unit", 0)
        pos_risk_usd  = position_size * risk_per_unit          # yatırılan tutar × risk oranı
        pos_risk_pct  = pos_risk_usd / capital * 100 if capital > 0 else 0
        # Sermaye riski: TÜM sermayeyi yatırırsan sl'e gidersen ne kaybedersin
        cap_risk_usd  = capital * risk_per_unit
        cap_risk_pct  = cap_risk_usd / capital * 100           # = full_capital_risk_pct

        mc0, mc1, mc2, mc3, mc4, mc5 = st.columns(6)
        with mc0:
            position_size = st.number_input(
                "💰 Pozisyon (USDT)", min_value=0.0,
                max_value=float(capital)*200,
                value=position_size,
                step=10.0, key=widget_key, on_change=_sync,
            )
            # Widget değiştikten sonra güncelle
            position_size = float(st.session_state.get(sz_key, position_size))
            pos_risk_usd  = position_size * risk_per_unit
            pos_risk_pct  = pos_risk_usd / capital * 100 if capital > 0 else 0
        with mc1:
            _info_box("Risk $",
                      f"${pos_risk_usd:,.2f}",
                      _Y)
        with mc2:
            # Poz. Riski: yatırılan tutar üzerinden stop'a giderse kayıp / toplam sermaye
            _info_box("Poz. Riski",
                      f"%{pos_risk_pct:.2f}",
                      _R if pos_risk_pct > risk_pct else _Y if pos_risk_pct > risk_pct * 0.8 else _G)
        with mc3:
            # Sermaye Riski: tüm sermaye yatırılsaydı stop'ta kayıp
            _info_box("Sermaye Risk",
                      f"%{cap_risk_pct:.2f}",
                      _R if cap_risk_pct > risk_pct else _G)
        with mc4:
            rr_str = f"1:{rr}" if rr else "—"
            rrc    = rr_color(rr) if rr else _DT
            _info_box("R:R", rr_str, rrc)
        with mc5:
            _info_box("Kaldıraçlı",
                      f"${position_size*lev:,.0f}" if lev > 1 else "—",
                      _B)

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        position_size = float(st.session_state.get(sz_key, capital))

    # ══════════════════════════════════════════════════════════════════════════
    # SATIR 4 — Journal / Bilgi Notu (tek sıra kompakt)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(
        f"<div style='background:{_BG};border:1px solid {_DG};"
        f"border-radius:12px;padding:10px 12px;margin-bottom:10px'>",
        unsafe_allow_html=True,
    )
    _divider("📓 İşlem Notu & Analiz", _Y)

    jc1, jc2, jc3, jc4 = st.columns([2, 1, 1, 1])
    with jc1:
        notes = st.text_area(
            "Notlar",
            value=pos.get("notes","") if editing else "",
            placeholder="Setup gerekçesi, beklenti, risk faktörleri...",
            key=f"{px}notes", height=85,
            label_visibility="collapsed",
        )
    with jc2:
        si    = SETUP_TYPES.index(pos["setup_type"]) if editing and pos.get("setup_type") in SETUP_TYPES else 0
        setup = st.selectbox("Setup", SETUP_TYPES, index=si, key=f"{px}setup")
        mi    = MARKET_CONDS.index(pos["market_condition"]) if editing and pos.get("market_condition") in MARKET_CONDS else 0
        mkt   = st.selectbox("Piyasa", MARKET_CONDS, index=mi, key=f"{px}mkt")
    with jc3:
        ei    = EMOTIONS.index(pos["emotion"]) if editing and pos.get("emotion") in EMOTIONS else 0
        emo   = st.selectbox("Psikoloji", EMOTIONS, index=ei, key=f"{px}emo")
        pf    = st.checkbox("Plana uyuldu", value=pos.get("plan_followed",True) if editing else True,
                            key=f"{px}plan")
    with jc4:
        es    = st.slider("Execution", 0, 10,
                          int(pos.get("execution_score",7)) if editing else 7,
                          key=f"{px}exec")

    # ── Tarih alanları ────────────────────────────────────────────────────────
    st.markdown("")
    da1, da2, da3, da4 = st.columns([1, 1, 1, 2])
    with da1:
        default_open = datetime.now()
        if editing and pos.get("created_at"):
            try: default_open = datetime.fromisoformat(pos["created_at"])
            except: pass
        open_date = st.date_input("📅 Açılış Tarihi", value=default_open.date(),
                                  key=f"{px}open_date")
    with da2:
        open_time = st.time_input("⏰ Açılış Saati", value=default_open.time(),
                                  key=f"{px}open_time")
    with da3:
        pass  # kapanış tarihi kapatma formunda girilir
    with da4:
        pass

    # Kapanış tarihi kapatma formunda girilir
    closed_at_val = pos.get("closed_at", "") if editing else ""

    st.markdown("</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # KAYDET
    # ══════════════════════════════════════════════════════════════════════════
    if is_draft_edit:
        sv1, sv2, sv3, sv4 = st.columns([1.5, 1.5, 1, 2])
        with sv1:
            save_btn  = st.button("✅ İşlemi Aç", type="primary",
                                  use_container_width=True, key=f"{px}save")
        with sv2:
            draft_btn = st.button("📋 Taslak Güncelle",
                                  use_container_width=True, key=f"{px}draft")
        with sv3:
            cancel_btn = st.button("✕ İptal", use_container_width=True,
                                   key=f"{px}cancel")
    else:
        sv1, sv2, sv3, sv4 = st.columns([1.2, 1.2, 1, 2])
        with sv1:
            save_btn = st.button(
                "✅ İşlemi Aç" if not editing else "💾 Güncelle",
                type="primary", use_container_width=True, key=f"{px}save",
            )
        with sv2:
            draft_btn = st.button(
                "📋 Taslak Kaydet",
                use_container_width=True, key=f"{px}draft",
            ) if not editing else False
        with sv3:
            cancel_btn = st.button("✕ İptal", use_container_width=True,
                                   key=f"{px}cancel")

    if cancel_btn:
        if is_draft_edit:
            st.session_state[f"edit_draft_{did}"] = False
        elif editing:
            st.session_state[f"edit_mode_{edit_id}"] = False
        _clear(px, sz_key, ver_key); st.rerun()

    if (not editing or is_draft_edit) and draft_btn:
        if not symbol:
            st.error("Sembol giriniz."); return
        if not entries:
            st.error("En az bir entry fiyatı giriniz."); return
        avg_e2 = calculate_avg_entry(entries)
        calc_s = calculate_position_size(capital, risk_pct, avg_e2, stop_loss) if stop_loss > 0 else {}
        rr_s   = calculate_rr(avg_e2, stop_loss, take_profits, direction) if stop_loss > 0 else None
        data   = st.session_state.data
        open_dt = datetime.combine(open_date, open_time).isoformat()
        draft  = dict(
            symbol=symbol, direction=direction, leverage=lev,
            capital=capital, risk_pct=risk_pct,
            entries=entries, avg_entry=avg_e2, stop_loss=stop_loss,
            take_profits=take_profits, position_size=position_size,
            effective_size=position_size * lev,
            risk_calc=calc_s, rr=rr_s, heat=0,
            setup_type=setup, market_condition=mkt,
            emotion=emo, plan_followed=pf,
            execution_score=es, mistakes=[], notes=notes,
            closed_at=closed_at_val,
            is_draft=True,
            id=data["next_id"],
            created_at=open_dt,
        )
        if "drafts" not in data:
            data["drafts"] = []
        if is_draft_edit:
            # Mevcut taslağı güncelle
            data["drafts"] = [draft if d.get("id")==did else d
                              for d in data["drafts"]]
            st.session_state[f"edit_draft_{did}"] = False
            st.success(f"📋 {symbol} {direction} taslak güncellendi!")
        else:
            data["drafts"].append(draft)
            data["next_id"] += 1
            st.success(f"📋 {symbol} {direction} taslak olarak kaydedildi!")
        save_data(data)
        st.session_state.data = data
        _clear(px, sz_key, ver_key); st.rerun()

    if save_btn:
        if not symbol:
            st.error("Sembol giriniz."); return
        if not entries:
            st.error("En az bir entry fiyatı giriniz."); return
        if stop_loss <= 0:
            st.error("Stop loss zorunludur."); return

        avg_e2 = calculate_avg_entry(entries)
        calc_s = calculate_position_size(capital, risk_pct, avg_e2, stop_loss)
        rr_s   = calculate_rr(avg_e2, stop_loss, take_profits, direction)
        heat_s = calculate_position_heat(
            position_size, capital, calc_s.get("risk_per_unit",0)
        ) if calc_s else 0

        data   = st.session_state.data
        # Açılış tarihi
        open_dt = datetime.combine(open_date, open_time).isoformat()
        record = dict(
            symbol=symbol, direction=direction, leverage=lev,
            capital=capital, risk_pct=risk_pct,
            entries=entries, avg_entry=avg_e2, stop_loss=stop_loss,
            take_profits=take_profits, position_size=position_size,
            effective_size=position_size * lev,
            risk_calc=calc_s, rr=rr_s, heat=heat_s,
            setup_type=setup, market_condition=mkt,
            emotion=emo, plan_followed=pf,
            execution_score=es,
            mistakes=[], notes=notes,
            closed_at=closed_at_val,
        )

        if is_draft_edit:
            # Taslaktan pozisyon aç
            record["id"]         = did
            record["created_at"] = open_dt
            data["active_positions"].append(record)
            data["drafts"] = [d for d in data.get("drafts",[])
                              if d.get("id") != did]
            st.session_state[f"edit_draft_{did}"] = False
            st.success(f"✅ {symbol} {direction} pozisyonu açıldı!")
        elif editing:
            idx = next((i for i,p in enumerate(data["active_positions"])
                        if p["id"]==edit_id), None)
            if idx is not None:
                record["id"]         = edit_id
                record["created_at"] = open_dt
                record["updated_at"] = datetime.now().isoformat()
                data["active_positions"][idx] = record
            st.session_state[f"edit_mode_{edit_id}"] = False
        else:
            record["id"]         = data["next_id"]
            record["created_at"] = open_dt
            data["next_id"]     += 1
            data["active_positions"].append(record)

        save_data(data)
        st.session_state.data = data
        if not is_draft_edit:
            st.success(f"✅ {symbol} {direction} pozisyonu kaydedildi!")
        _clear(px, sz_key, ver_key)
        st.rerun()


def _clear(px, sz_key, ver_key):
    for k in list(st.session_state.keys()):
        if k.startswith(px) or k in (sz_key, ver_key):
            try: del st.session_state[k]
            except: pass


# ══════════════════════════════════════════════════════════════════════════════
# TASLAK DÜZENLEME — Asıl form ile aynı sayfa
# ══════════════════════════════════════════════════════════════════════════════

def _render_draft_edit(draft: dict):
    """
    Taslağı düzenlemek için asıl _render_form'u kullan.
    Taslak verisi editing modundaki pos gibi davranır.
    Kaydet → taslağı günceller. İşlemi Aç → active_positions'a taşır.
    """
    did = draft.get("id")
    edit_draft_key = f"edit_draft_{did}"

    # draft_id'yi session_state'e yaz ki _render_form bilsin
    st.session_state[f"draft_mode_{did}"] = True

    # Taslağı düzenleyip kaydetmek için özel render
    _render_form(edit_id=None, draft_edit=draft)
