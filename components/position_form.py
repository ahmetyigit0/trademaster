"""
Pozisyon Giriş Formu — Premium UI
Resme benzer 3-kolon kart düzeni: Temel Bilgiler | Risk Ayarları | İşlem Özeti
"""
import streamlit as st
from datetime import datetime
from utils.data_manager import save_data
from utils.calculations import (
    calculate_avg_entry, calculate_position_size, calculate_rr,
    calculate_position_heat, rr_color
)

SETUP_TYPES  = ["liquidity", "breakout", "trend", "range", "diğer"]
EMOTIONS     = ["calm", "fomo", "revenge", "anxious", "confident"]
MISTAKES     = ["early exit", "late entry", "no stop", "oversize", "revenge trade", "ignored plan"]
MARKET_CONDS = ["trend", "range", "news", "volatile", "choppy"]

_SZ     = "pf__pos_size"
_SZ_VER = "pf__ps_ver"

_G="#3fb950"; _R="#ff7b72"; _B="#58a6ff"; _Y="#e3b341"
_DG="#21262d"; _DT="#b1bac4"; _TX="#e6edf3"; _DB="#0d1117"


def render_position_form():
    _render_form()

def render_edit_form(pos_id: int):
    _render_form(edit_id=pos_id)


# ── Kart container ─────────────────────────────────────────────────────────
def _card_start(num: str, title: str, color: str = _B):
    st.markdown(
        f"<div style='background:#161b22;border:1px solid {_DG};"
        f"border-radius:14px;padding:1rem 1.1rem 0.5rem'>"
        f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:10px'>"
        f"<span style='background:{color}22;color:{color};border-radius:50%;"
        f"width:26px;height:26px;display:flex;align-items:center;justify-content:center;"
        f"font-size:13px;font-weight:800;flex-shrink:0'>{num}</span>"
        f"<span style='font-size:13px;font-weight:700;color:#f0f6fc;"
        f"text-transform:uppercase;letter-spacing:0.08em'>{title}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

def _card_end():
    st.markdown("</div>", unsafe_allow_html=True)


def _metric_row(label: str, value: str, color: str = _TX):
    st.markdown(
        f"<div style='display:flex;justify-content:space-between;"
        f"padding:5px 0;border-bottom:1px solid {_DG}'>"
        f"<span style='font-size:14px;color:{_DT}'>{label}</span>"
        f"<span style='font-size:14px;font-weight:700;color:{color};"
        f"font-family:\"Space Mono\",monospace'>{value}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_form(edit_id=None):
    editing = edit_id is not None
    px      = f"pf_{edit_id}_" if editing else "pf_"
    pos     = None

    if editing:
        pos = next((p for p in st.session_state.data["active_positions"]
                    if p["id"] == edit_id), None)
        if pos is None:
            st.error("Pozisyon bulunamadı."); return
        st.markdown(
            f"<div style='font-size:1.05rem;font-weight:700;color:#f0f6fc;"
            f"padding-bottom:8px;border-bottom:1px solid {_DG};margin-bottom:12px'>"
            f"✏️ Pozisyonu Düzenle — #{edit_id} {pos['symbol']}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='font-style:italic;color:#6e7681;font-size:13px;"
            f"border-left:2px solid {_B};padding:4px 10px;margin-bottom:14px;line-height:1.6'>"
            f"\"Herkes teknik analiz bilebilir, ancak çok az kişi belirli bir plana sadık kalabilir.\"</div>",
            unsafe_allow_html=True,
        )

    sz_key  = f"{_SZ}_{px}"
    ver_key = f"{_SZ_VER}_{px}"
    if sz_key not in st.session_state:
        st.session_state[sz_key]  = float(pos.get("position_size", 10000)) if editing else 10000.0
        st.session_state[ver_key] = 0

    # ══════════════════════════════════════════════════════════════════════════
    # ÜSTTE 3 KOLON KART: Temel | Risk | İşlem Özeti (canlı)
    # ══════════════════════════════════════════════════════════════════════════
    col1, col2, col3 = st.columns([1, 1, 1], gap="small")

    # ── KART 1: Temel Bilgiler ─────────────────────────────────────────────
    with col1:
        _card_start("1", "Temel Bilgiler", _B)

        symbol = st.text_input(
            "Sembol",
            value=pos["symbol"] if editing else "",
            placeholder="BTC, ETH, SOL...",
            key=f"{px}sym",
        ).upper().strip()

        # LONG / SHORT toggle
        dir_idx = 0 if not editing or pos["direction"] == "LONG" else 1
        dir_val = st.session_state.get(f"{px}dir_val",
                  "LONG" if dir_idx == 0 else "SHORT")
        d1, d2  = st.columns(2)
        with d1:
            long_style = (f"background:linear-gradient(135deg,#196c2e,#238636);"
                          f"color:#fff;border:2px solid #3fb950;border-radius:9px;"
                          f"padding:9px;font-weight:700;font-size:15px;width:100%;cursor:pointer")
            short_style= (f"background:#161b22;color:{_DT};border:1.5px solid {_DG};"
                          f"border-radius:9px;padding:9px;font-weight:600;font-size:15px;"
                          f"width:100%;cursor:pointer")
            if dir_val == "SHORT":
                long_style, short_style = short_style.replace("linear-gradient(135deg,#196c2e,#238636)",_DB).replace("color:#fff","color:"+_DT).replace("border:2px solid #3fb950","border:1.5px solid "+_DG), \
                    (f"background:linear-gradient(135deg,#6e1a1a,#da3633);color:#fff;"
                     f"border:2px solid #ff7b72;border-radius:9px;padding:9px;"
                     f"font-weight:700;font-size:15px;width:100%;cursor:pointer")
            if st.button("📈 LONG", key=f"{px}long_btn", use_container_width=True):
                st.session_state[f"{px}dir_val"] = "LONG"; st.rerun()
        with d2:
            if st.button("📉 SHORT", key=f"{px}short_btn", use_container_width=True):
                st.session_state[f"{px}dir_val"] = "SHORT"; st.rerun()

        direction = st.session_state.get(f"{px}dir_val",
                    "LONG" if dir_idx == 0 else "SHORT")
        dir_color = _G if direction == "LONG" else _R
        st.markdown(
            f"<div style='text-align:center;font-size:13px;font-weight:700;"
            f"color:{dir_color};margin-top:-4px;margin-bottom:4px'>"
            f"{'📈 LONG seçili' if direction=='LONG' else '📉 SHORT seçili'}</div>",
            unsafe_allow_html=True,
        )

        lev_def  = float(pos.get("leverage", 1)) if editing else 1.0
        leverage = st.number_input(
            "Kaldıraç ×",
            min_value=1.0, max_value=500.0,
            value=lev_def, step=1.0,
            key=f"{px}lev",
        )
        _card_end()

    # ── KART 2: Risk Ayarları ──────────────────────────────────────────────
    with col2:
        _card_start("2", "Risk Ayarları", _Y)

        capital = st.number_input(
            "Sermaye (USDT)",
            min_value=1.0,
            value=float(pos["capital"]) if editing else 10000.0,
            step=100.0, key=f"{px}cap",
        )
        risk_pct = st.number_input(
            "Risk %",
            min_value=0.1, max_value=100.0,
            value=float(pos["risk_pct"]) if editing else 2.0,
            step=0.1, key=f"{px}rp",
        )

        # Risk tutarı
        risk_amount = capital * risk_pct / 100
        max_loss    = risk_amount * leverage if leverage > 1 else risk_amount
        st.markdown(
            f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:4px'>"
            f"<div style='background:{_DB};border-radius:8px;padding:7px 10px;text-align:center'>"
            f"<div style='font-size:11px;color:{_DT}'>Risk Tutarı</div>"
            f"<div style='font-size:15px;font-weight:700;color:{_Y};"
            f"font-family:\"Space Mono\",monospace'>${risk_amount:,.2f}</div></div>"
            f"<div style='background:{_DB};border-radius:8px;padding:7px 10px;text-align:center'>"
            f"<div style='font-size:11px;color:{_DT}'>Maks. Zarar</div>"
            f"<div style='font-size:15px;font-weight:700;color:{_R};"
            f"font-family:\"Space Mono\",monospace'>-${risk_amount:,.2f}</div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        _card_end()

    # ── KART 3: İşlem Özeti (canlı) ───────────────────────────────────────
    with col3:
        _card_start("3", "İşlem Özeti", _G)
        # Özet kart — entry/stop sonrası hesaplanır, şimdi placeholder
        st.markdown(
            f"<div style='color:{_DT};font-size:13px;padding:4px 0 8px'>"
            f"Entry ve Stop girince burada canlı hesaplanır.</div>",
            unsafe_allow_html=True,
        )
        _card_end()

    # ══════════════════════════════════════════════════════════════════════════
    # BÖLÜM 4: ENTRY NOKTALARI
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("")
    _card_start("4", "Entry Noktaları", _B)

    ne_def      = len(pos["entries"]) if editing and pos.get("entries") else 1
    num_entries = int(st.number_input(
        "Kaç entry?", min_value=1, max_value=10,
        value=ne_def, step=1, key=f"{px}ne",
    ))

    entries       = []
    default_ew    = round(100.0 / num_entries, 1)
    total_ew      = 0.0

    # Tablo başlığı
    st.markdown(
        f"<div style='display:grid;"
        f"grid-template-columns:30px 1fr 110px 90px 1fr;"
        f"gap:6px;padding:6px 4px;border-bottom:1px solid {_DG};"
        f"font-size:12px;font-weight:700;color:{_DT};"
        f"text-transform:uppercase;letter-spacing:0.07em'>"
        f"<div>#</div><div>Fiyat (USDT)</div>"
        f"<div>Ağırlık (%)</div><div>Tutar</div><div>Açıklama</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    for i in range(num_entries):
        ep_d = float(pos["entries"][i]["price"])  if editing and i < len(pos.get("entries",[])) else 0.0
        ew_d = float(pos["entries"][i]["weight"]) if editing and i < len(pos.get("entries",[])) else default_ew
        desc_key = f"{px}edesc{i}"

        ec1, ec2, ec3, ec4, ec5 = st.columns([0.3, 1.2, 0.9, 0.8, 1.2])
        with ec1:
            st.markdown(
                f"<div style='padding-top:30px;font-size:13px;"
                f"color:{_B};font-weight:700'>{i+1}</div>",
                unsafe_allow_html=True,
            )
        with ec2:
            ep = st.number_input(
                f"Fiyat {i+1}", min_value=0.0, value=ep_d,
                format="%.4f", key=f"{px}ep{i}",
                label_visibility="collapsed",
            )
        with ec3:
            ew = st.number_input(
                f"Ağırlık {i+1}", min_value=0.0, max_value=100.0,
                value=ew_d, step=5.0, key=f"{px}ew{i}",
                label_visibility="collapsed",
            )
        with ec4:
            usdt_v = st.session_state.get(sz_key, capital) * (ew / 100.0)
            st.markdown(
                f"<div style='padding-top:28px;font-size:13px;"
                f"font-weight:700;color:{_B};"
                f"font-family:\"Space Mono\",monospace'>${usdt_v:,.0f}</div>",
                unsafe_allow_html=True,
            )
        with ec5:
            st.text_input(
                f"Açıklama {i+1}", value="",
                placeholder="Opsiyonel...",
                key=desc_key, label_visibility="collapsed",
            )
        total_ew += ew
        if ep > 0:
            entries.append({"price": ep, "weight": ew})

    # Ağırlık toplamı
    tw_c = _G if abs(total_ew - 100) <= 1 else _Y
    if num_entries > 1:
        st.markdown(
            f"<div style='font-size:12px;color:{tw_c};margin-top:4px'>"
            f"Toplam Ağırlık: <b>{total_ew:.0f}%</b>"
            f"{'  ✓' if abs(total_ew-100)<=1 else '  — 100% olmalı'}</div>",
            unsafe_allow_html=True,
        )

    if entries:
        avg_entry = calculate_avg_entry(entries)
        st.markdown(
            f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;"
            f"gap:8px;margin-top:8px'>"
            f"<div style='background:{_DB};border-radius:8px;padding:7px 10px'>"
            f"<div style='font-size:11px;color:{_DT}'>Toplam Ağırlık</div>"
            f"<div style='font-size:14px;font-weight:700;color:{tw_c}'>{total_ew:.0f}%</div></div>"
            f"<div style='background:{_DB};border-radius:8px;padding:7px 10px'>"
            f"<div style='font-size:11px;color:{_DT}'>Toplam Tutar</div>"
            f"<div style='font-size:14px;font-weight:700;color:{_B};"
            f"font-family:\"Space Mono\",monospace'>${st.session_state.get(sz_key,capital):,.0f}</div></div>"
            f"<div style='background:{_DB};border-radius:8px;padding:7px 10px'>"
            f"<div style='font-size:11px;color:{_DT}'>Ortalama Entry</div>"
            f"<div style='font-size:14px;font-weight:700;color:{_TX};"
            f"font-family:\"Space Mono\",monospace'>${avg_entry:,.2f}</div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        avg_entry = 0.0

    _card_end()

    # ══════════════════════════════════════════════════════════════════════════
    # BÖLÜM 5 & 6: Stop Loss + Take Profit yan yana
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("")
    sl_sec, tp_sec = st.columns([1, 1], gap="small")

    with sl_sec:
        _card_start("5", "Stop Loss & Risk", _R)

        sl_def    = float(pos["stop_loss"]) if editing else 0.0
        stop_loss = st.number_input(
            "Stop Loss Fiyatı (Zorunlu)",
            min_value=0.0, value=sl_def,
            format="%.4f", key=f"{px}sl",
        )

        # Risk görseli
        if stop_loss > 0 and avg_entry > 0:
            pdiff    = abs(avg_entry - stop_loss)
            trade_r  = pdiff / avg_entry * 100
            cur_sz   = float(st.session_state.get(sz_key, capital))
            act_loss = cur_sz * (pdiff / avg_entry)
            act_pct  = act_loss / capital * 100
            liq_est  = (stop_loss * 0.97 if direction == "LONG"
                        else stop_loss * 1.03)
            rc = _R if act_pct > risk_pct * 1.2 else _Y if act_pct > risk_pct else _G

            st.markdown(
                f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;"
                f"gap:6px;margin:8px 0'>"
                f"<div style='background:{_DB};border-radius:8px;padding:7px 8px;text-align:center'>"
                f"<div style='font-size:10px;color:{_DT}'>Risk (Entry'e göre)</div>"
                f"<div style='font-size:14px;font-weight:700;color:{rc}'>{act_pct:.2f}%</div></div>"
                f"<div style='background:{_DB};border-radius:8px;padding:7px 8px;text-align:center'>"
                f"<div style='font-size:10px;color:{_DT}'>Risk Tutarı</div>"
                f"<div style='font-size:14px;font-weight:700;color:{rc};"
                f"font-family:\"Space Mono\",monospace'>${act_loss:,.2f}</div></div>"
                f"<div style='background:{_DB};border-radius:8px;padding:7px 8px;text-align:center'>"
                f"<div style='font-size:10px;color:{_DT}'>Likidasyon (~)</div>"
                f"<div style='font-size:14px;font-weight:700;color:{_Y};"
                f"font-family:\"Space Mono\",monospace'>${liq_est:,.2f}</div></div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Risk bar
            bar_pct = min(act_pct / (risk_pct * 2) * 100, 100)
            bar_col = _R if act_pct > risk_pct else _G
            st.markdown(
                f"<div style='font-size:11px;color:{_DT};margin-bottom:3px'>Risk Görseli</div>"
                f"<div style='background:{_DG};border-radius:4px;height:8px;overflow:hidden'>"
                f"<div style='background:{bar_col};width:{bar_pct:.0f}%;height:8px;"
                f"border-radius:4px;transition:width 0.3s'></div></div>"
                f"<div style='display:flex;justify-content:space-between;"
                f"font-size:11px;color:{_DT};margin-top:2px'>"
                f"<span>Entry: ${avg_entry:,.2f}</span>"
                f"<span>SL: ${stop_loss:,.2f}</span></div>",
                unsafe_allow_html=True,
            )
        _card_end()

    with tp_sec:
        _card_start("6", "Take Profit Hedefleri", _G)

        ntp_def = len(pos["take_profits"]) if editing and pos.get("take_profits") else 1
        num_tp  = int(st.number_input(
            "TP sayısı", min_value=1, max_value=5,
            value=ntp_def, step=1, key=f"{px}ntp",
        ))
        default_tpw = round(100.0 / num_tp, 1)
        take_profits = []

        # TP tablo başlığı
        st.markdown(
            f"<div style='display:grid;"
            f"grid-template-columns:50px 1fr 80px 70px;"
            f"gap:4px;padding:4px 0;border-bottom:1px solid {_DG};"
            f"font-size:12px;font-weight:700;color:{_DT};"
            f"text-transform:uppercase'>"
            f"<div>Hedef</div><div>Fiyat</div><div>Çıkış %</div><div>R:R</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        for i in range(num_tp):
            tp_pd = float(pos["take_profits"][i]["price"])  if editing and i < len(pos.get("take_profits",[])) else 0.0
            tp_wd = float(pos["take_profits"][i]["weight"]) if editing and i < len(pos.get("take_profits",[])) else default_tpw
            tc1, tc2, tc3, tc4 = st.columns([0.5, 1.2, 0.8, 0.7])
            with tc1:
                st.markdown(
                    f"<div style='padding-top:28px;font-size:13px;"
                    f"color:{_G};font-weight:700'>TP{i+1}</div>",
                    unsafe_allow_html=True,
                )
            with tc2:
                tp_p = st.number_input(
                    f"TP{i+1} Fiyat", min_value=0.0, value=tp_pd,
                    format="%.4f", key=f"{px}tpp{i}",
                    label_visibility="collapsed",
                )
            with tc3:
                tp_w = st.number_input(
                    f"TP{i+1} %", min_value=0.0, max_value=100.0,
                    value=tp_wd, step=5.0, key=f"{px}tpw{i}",
                    label_visibility="collapsed",
                )
            with tc4:
                if tp_p > 0 and stop_loss > 0 and avg_entry > 0:
                    sl_d = abs(avg_entry - stop_loss)
                    tp_d = abs(tp_p - avg_entry)
                    rr_v = round(tp_d / sl_d, 2) if sl_d > 0 else 0
                    rrc  = _G if rr_v >= 2 else _Y if rr_v >= 1 else _R
                    st.markdown(
                        f"<div style='padding-top:28px;font-size:13px;"
                        f"font-weight:700;color:{rrc}'>{rr_v:.2f}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<div style='padding-top:28px;font-size:13px;color:{_DT}'>—</div>",
                        unsafe_allow_html=True,
                    )
            if tp_p > 0:
                take_profits.append({"price": tp_p, "weight": tp_w})

        # TP Özeti
        if take_profits and avg_entry > 0 and stop_loss > 0:
            sl_d  = abs(avg_entry - stop_loss)
            tp_rrs = []
            for t in take_profits:
                td = abs(t["price"] - avg_entry)
                tp_rrs.append(round(td / sl_d, 2) if sl_d > 0 else 0)
            avg_rr = sum(tp_rrs) / len(tp_rrs) if tp_rrs else 0
            rr_c   = _G if avg_rr >= 2 else _Y if avg_rr >= 1 else _R
            tw_tp  = sum(t["weight"] for t in take_profits)
            st.markdown(
                f"<div style='display:grid;grid-template-columns:1fr 1fr;"
                f"gap:6px;margin-top:8px'>"
                f"<div style='background:{_DB};border-radius:8px;padding:7px 10px'>"
                f"<div style='font-size:11px;color:{_DT}'>Toplam Çıkış</div>"
                f"<div style='font-size:14px;font-weight:700;color:{_G}'>{tw_tp:.0f}%</div></div>"
                f"<div style='background:{_DB};border-radius:8px;padding:7px 10px'>"
                f"<div style='font-size:11px;color:{_DT}'>Ağırlıklı R:R</div>"
                f"<div style='font-size:14px;font-weight:700;color:{rr_c}'>"
                f"1:{avg_rr:.2f}</div></div></div>",
                unsafe_allow_html=True,
            )
        _card_end()

    # ══════════════════════════════════════════════════════════════════════════
    # Risk hesabı + pozisyon büyüklüğü
    # ══════════════════════════════════════════════════════════════════════════
    calc = {}
    if entries and stop_loss > 0:
        calc = calculate_position_size(capital, risk_pct, avg_entry, stop_loss)
        if calc and not calc["can_use_full"]:
            rec = calc["recommended_size"]
            st.markdown("")
            wa, _ = st.columns([2, 4])
            with wa:
                st.warning(
                    f"⚠️ Tam sermaye riski **%{calc['full_capital_risk_pct']:.2f}** "
                    f"— Önerilen: **${rec:,.2f}**"
                )
            wb, _ = st.columns([1, 5])
            with wb:
                if st.button("💡 Öneriyi Uygula", key=f"{px}apply", type="primary"):
                    st.session_state[sz_key]  = float(rec)
                    st.session_state[ver_key] = st.session_state.get(ver_key, 0) + 1
                    st.rerun()
        elif calc and calc["can_use_full"]:
            if st.session_state.get(sz_key, 0) == 0:
                st.session_state[sz_key] = float(capital)

    ver        = st.session_state.get(ver_key, 0)
    widget_key = f"{px}psw_v{ver}"

    def _sync():
        st.session_state[sz_key] = float(st.session_state.get(widget_key, 0.0))

    pa, pb = st.columns([2, 4])
    with pa:
        position_size = st.number_input(
            "💰 Pozisyon Büyüklüğü (USDT)",
            min_value=0.0, max_value=float(capital) * 200,
            value=float(st.session_state.get(sz_key, capital)),
            step=10.0, key=widget_key, on_change=_sync,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # İŞLEM NOTU + ETİKETLER
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("")
    note_col, tag_col = st.columns([2, 1], gap="small")

    with note_col:
        _card_start("✍", "İşlem Notu", _DT)
        notes = st.text_area(
            "Notlar",
            value=pos.get("notes","") if editing else "",
            placeholder="Bu işlemle ilgili düşüncelerini, analizini ve notlarını buraya yaz...",
            key=f"{px}notes", height=90,
            label_visibility="collapsed",
        )
        j1, j2 = st.columns(2)
        with j1:
            si    = SETUP_TYPES.index(pos["setup_type"]) if editing and pos.get("setup_type") in SETUP_TYPES else 0
            setup = st.selectbox("Setup Tipi", SETUP_TYPES, index=si, key=f"{px}setup")
            ei    = EMOTIONS.index(pos["emotion"]) if editing and pos.get("emotion") in EMOTIONS else 0
            emotion = st.selectbox("Psikoloji", EMOTIONS, index=ei, key=f"{px}emo")
        with j2:
            mi    = MARKET_CONDS.index(pos["market_condition"]) if editing and pos.get("market_condition") in MARKET_CONDS else 0
            market = st.selectbox("Piyasa", MARKET_CONDS, index=mi, key=f"{px}market")
            pf_def = pos.get("plan_followed", True) if editing else True
            plan   = st.checkbox("✅ Plana uyuldu", value=pf_def, key=f"{px}plan")
        _card_end()

    with tag_col:
        _card_start("🏷", "Etiketler", _Y)
        mk_def   = pos.get("mistakes", []) if editing else []
        mistakes = st.multiselect("Hata Etiketleri", MISTAKES, default=mk_def, key=f"{px}mist")
        es_def   = int(pos.get("execution_score", 7)) if editing else 7
        exec_s   = st.slider("Execution (0–10)", 0, 10, es_def, key=f"{px}exec")
        _card_end()

    # ══════════════════════════════════════════════════════════════════════════
    # ALT AKSIYON BUTONLARI
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("")
    ba1, ba2, ba3, ba4 = st.columns([1, 1, 1, 2])
    with ba1:
        clear_btn = st.button("🗑 Temizle", key=f"{px}clear", use_container_width=True)
    with ba3:
        save_btn = st.button(
            "✅ İşlemi Kaydet" if not editing else "💾 Güncelle",
            type="primary", use_container_width=True, key=f"{px}save",
        )
    with ba2:
        cancel_btn = st.button("✕ İptal", key=f"{px}cancel", use_container_width=True)

    if clear_btn or cancel_btn:
        if editing:
            st.session_state[f"edit_mode_{edit_id}"] = False
        _clear_form(px, sz_key, ver_key)
        st.rerun()

    if save_btn:
        if not symbol:
            st.error("Sembol giriniz."); return
        if not entries:
            st.error("En az bir entry fiyatı giriniz."); return
        if stop_loss <= 0:
            st.error("Stop loss zorunludur."); return

        avg_e  = calculate_avg_entry(entries)
        calc_s = calculate_position_size(capital, risk_pct, avg_e, stop_loss)
        rr_s   = calculate_rr(avg_e, stop_loss, take_profits, direction)
        heat_s = calculate_position_heat(
            position_size, capital, calc_s.get("risk_per_unit", 0)
        ) if calc_s else 0

        data   = st.session_state.data
        record = dict(
            symbol=symbol, direction=direction, leverage=leverage,
            capital=capital, risk_pct=risk_pct,
            entries=entries, avg_entry=avg_e, stop_loss=stop_loss,
            take_profits=take_profits, position_size=position_size,
            effective_size=position_size * leverage,
            risk_calc=calc_s, rr=rr_s, heat=heat_s,
            setup_type=setup, market_condition=market,
            emotion=emotion, plan_followed=plan,
            execution_score=exec_s, mistakes=mistakes, notes=notes,
        )

        if editing:
            idx = next((i for i,p in enumerate(data["active_positions"])
                        if p["id"] == edit_id), None)
            if idx is not None:
                record["id"]         = edit_id
                record["created_at"] = data["active_positions"][idx].get(
                    "created_at", datetime.now().isoformat())
                record["updated_at"] = datetime.now().isoformat()
                data["active_positions"][idx] = record
            st.session_state[f"edit_mode_{edit_id}"] = False
        else:
            record["id"]         = data["next_id"]
            record["created_at"] = datetime.now().isoformat()
            data["next_id"]     += 1
            data["active_positions"].append(record)

        save_data(data)
        st.session_state.data = data
        _clear_form(px, sz_key, ver_key)
        st.rerun()


def _clear_form(px, sz_key, ver_key):
    for k in list(st.session_state.keys()):
        if k.startswith(px) or k in (sz_key, ver_key):
            try:
                del st.session_state[k]
            except Exception:
                pass
