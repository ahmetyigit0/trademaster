"""Trade Yasaları — kalıcı JSON tabanlı kural listesi."""
import streamlit as st
import json
import os

# ── Renk sabitleri ────────────────────────────────────────────────────────────
_G  = "#3fb950";  _R = "#ff7b72";  _B = "#58a6ff"
_Y  = "#e3b341";  _DT= "#b1bac4";  _TX= "#e6edf3"
_BG = "#161b22";  _DB= "#0d1117";  _DG= "#21262d"

RULES_FILE = "trade_rules.json"

DEFAULT_RULES = [
    {
        "id": 1,
        "category": "Risk Yönetimi",
        "rule": "Her işlemde sermayenin maksimum %2'sini riske at. Bu sınırı asla aşma.",
        "active": True,
    },
    {
        "id": 2,
        "category": "Risk Yönetimi",
        "rule": "Stop loss olmadan pozisyon açma. Stop loss, işlemi açmadan önce belirlenir — açtıktan sonra değil.",
        "active": True,
    },
    {
        "id": 3,
        "category": "Risk Yönetimi",
        "rule": "Aynı anda açık olan pozisyonların toplam riski sermayenin %6'sını geçemez.",
        "active": True,
    },
    {
        "id": 4,
        "category": "Giriş Kuralları",
        "rule": "Minimum 1:2 R:R oranı olmadan işleme girme. Piyasa seni zorlamıyorsa beklemeye devam et.",
        "active": True,
    },
    {
        "id": 5,
        "category": "Giriş Kuralları",
        "rule": "FOMO ile giriş yapma. Kaçırdığın her setup yerine daha iyisi gelecek.",
        "active": True,
    },
    {
        "id": 6,
        "category": "Giriş Kuralları",
        "rule": "Büyük haber/event öncesi pozisyon açma ya da mevcut pozisyon büyüklüğünü azalt.",
        "active": True,
    },
    {
        "id": 7,
        "category": "Çıkış Kuralları",
        "rule": "Kârdayken stop'u maliyete çek (break-even). Kârlı bir işlemi zarara döndürme.",
        "active": True,
    },
    {
        "id": 8,
        "category": "Çıkış Kuralları",
        "rule": "TP seviyelerine ulaşıldığında planı uygula. 'Biraz daha bekleyeyim' tuzağına düşme.",
        "active": True,
    },
    {
        "id": 9,
        "category": "Çıkış Kuralları",
        "rule": "Stop loss'a ulaşıldığında pozisyonu kapat. Stop'u asla uzatma veya kaldırma.",
        "active": True,
    },
    {
        "id": 10,
        "category": "Psikoloji",
        "rule": "3 üst üste zararlı işlemden sonra o gün işlemi bırak. Kayıp serisinde iken boyut artırma.",
        "active": True,
    },
    {
        "id": 11,
        "category": "Psikoloji",
        "rule": "İntikam işlemi (revenge trade) yapma. Zarar sonrası verilen kararlar çoğunlukla hatalıdır.",
        "active": True,
    },
    {
        "id": 12,
        "category": "Psikoloji",
        "rule": "Büyük kâr sonrası aşırı özgüvenle pozisyon büyütme. Sistem her koşulda aynı kalır.",
        "active": True,
    },
    {
        "id": 13,
        "category": "Plan & Analiz",
        "rule": "Her işlemi bu journale kaydet. Kaydetmediğin işlemden öğrenemezsin.",
        "active": True,
    },
    {
        "id": 14,
        "category": "Plan & Analiz",
        "rule": "Haftalık olarak kapalı işlemleri gözden geçir. Tekrarlayan hatalar var mı?",
        "active": True,
    },
    {
        "id": 15,
        "category": "Plan & Analiz",
        "rule": "Setup'ı net göremiyorsan işlem açma. Belirsizlik = pozisyon yok.",
        "active": True,
    },
]

CATEGORIES = ["Risk Yönetimi", "Giriş Kuralları", "Çıkış Kuralları", "Psikoloji", "Plan & Analiz", "Diğer"]

CAT_COLORS = {
    "Risk Yönetimi":  ("#da3633", "#2d0f0f"),
    "Giriş Kuralları":("#1f6feb", "#0d2238"),
    "Çıkış Kuralları":("#e3b341", "#2b1d0a"),
    "Psikoloji":      ("#a371f7", "#1e1030"),
    "Plan & Analiz":  ("#3fb950", "#071a0e"),
    "Diğer":          ("#8b949e", "#161b22"),
}

CAT_ICONS = {
    "Risk Yönetimi":  "🛡️",
    "Giriş Kuralları":"📥",
    "Çıkış Kuralları":"📤",
    "Psikoloji":      "🧠",
    "Plan & Analiz":  "📋",
    "Diğer":          "📌",
}


def _load_rules() -> list[dict]:
    if os.path.exists(RULES_FILE):
        try:
            with open(RULES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # İlk çalıştırmada default kuralları kaydet
    _save_rules(DEFAULT_RULES)
    return DEFAULT_RULES[:]


def _save_rules(rules: list[dict]):
    with open(RULES_FILE, "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)


def _next_id(rules: list[dict]) -> int:
    return max((r["id"] for r in rules), default=0) + 1


def render_trade_rules():
    if "tr_rules" not in st.session_state:
        st.session_state.tr_rules = _load_rules()

    rules = st.session_state.tr_rules

    # ── Header ────────────────────────────────────────────────────────────────
    active_count = sum(1 for r in rules if r.get("active", True))
    st.markdown(
        f"<div style='display:flex;align-items:center;justify-content:space-between;"
        f"margin-bottom:1rem'>"
        f"<div>"
        f"<div style='font-family:\"Space Mono\",monospace;font-size:1rem;"
        f"font-weight:700;color:#f0f6fc'>⚖️ Trade Yasaları</div>"
        f"<div style='font-size:13px;color:#6e7681;margin-top:2px'>"
        f"{active_count} aktif kural · {len(rules)} toplam</div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    # ── Motivasyon banner ─────────────────────────────────────────────────────
    st.markdown(
        "<div style='background:linear-gradient(135deg,#0d2238,#071a0e);"
        "border:1px solid #1f6feb;border-radius:12px;padding:0.9rem 1.1rem;"
        "margin-bottom:1.2rem;font-style:italic;color:#79c0ff;font-size:14px;"
        "line-height:1.7'>"
        "\"Disiplin, motivasyonun bittiği yerde devreye girer. "
        "Kurallar seni kötü günlerde korur — iyi günlerde değil.\""
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Yeni kural ekleme formu ───────────────────────────────────────────────
    with st.expander("➕ Yeni Kural Ekle", expanded=False):
        na1, na2 = st.columns([3, 1])
        with na1:
            new_rule_text = st.text_area(
                "Kural metni",
                placeholder="Kural açıklamasını buraya yaz...",
                key="tr_new_text", height=80,
            )
        with na2:
            new_cat = st.selectbox("Kategori", CATEGORIES, key="tr_new_cat")
        if st.button("✅ Ekle", type="primary", key="tr_add_btn", use_container_width=True):
            if new_rule_text.strip():
                rules.append({
                    "id":       _next_id(rules),
                    "category": new_cat,
                    "rule":     new_rule_text.strip(),
                    "active":   True,
                })
                _save_rules(rules)
                st.session_state.tr_rules = rules
                # clear text
                if "tr_new_text" in st.session_state:
                    del st.session_state["tr_new_text"]
                st.rerun()
            else:
                st.error("Kural metni boş olamaz.")

    st.markdown("")

    # ── Kategori filtresi ─────────────────────────────────────────────────────
    all_cats = sorted(set(r.get("category", "Diğer") for r in rules))
    cf1, cf2 = st.columns([2, 2])
    with cf1:
        cat_filter = st.selectbox("Kategori filtresi", ["Tümü"] + all_cats, key="tr_cat_filter")
    with cf2:
        show_inactive = st.checkbox("Devre dışıları da göster", value=False, key="tr_show_inactive")

    filtered = [
        r for r in rules
        if (cat_filter == "Tümü" or r.get("category") == cat_filter)
        and (show_inactive or r.get("active", True))
    ]

    # ── Kuralları kategoriye göre grupla ──────────────────────────────────────
    from collections import defaultdict
    grouped: dict[str, list] = defaultdict(list)
    for r in filtered:
        grouped[r.get("category", "Diğer")].append(r)

    for cat, cat_rules in grouped.items():
        fg, bg = CAT_COLORS.get(cat, ("#8b949e", "#161b22"))
        icon   = CAT_ICONS.get(cat, "📌")

        # Kategori başlığı — şerit yok, sadece başlık + sayı badge
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:0.7rem;"
            f"margin:1.4rem 0 0.6rem'>"
            f"<div style='background:{bg};border:1px solid {fg}33;"
            f"color:{fg};padding:5px 14px;border-radius:8px;"
            f"font-size:14px;font-weight:700;letter-spacing:0.03em'>"
            f"{icon} {cat}</div>"
            f"<div style='background:#21262d;color:#8b949e;padding:3px 10px;"
            f"border-radius:20px;font-size:13px;font-weight:600'>"
            f"{len(cat_rules)} kural</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        for r in cat_rules:
            rid      = r["id"]
            active   = r.get("active", True)
            opacity  = "1" if active else "0.45"
            edit_key = f"tr_edit_{rid}"
            st.session_state.setdefault(edit_key, False)
            border_c = fg if active else "#30363d"
            card_bg  = "#161b22" if active else "#0f1520"
            num      = cat_rules.index(r) + 1

            if st.session_state.get(edit_key):
                # ── Düzenleme modu ────────────────────────────────────────
                with st.container():
                    st.markdown(
                        f"<div style='background:{card_bg};border:1px solid {_DG};"
                        f"border-left:3px solid {fg};border-radius:10px;"
                        f"padding:0.7rem 1rem 0.4rem;margin-bottom:0.45rem'>",
                        unsafe_allow_html=True,
                    )
                    edited_text = st.text_area(
                        "Kuralı düzenle", value=r["rule"],
                        key=f"tr_edit_text_{rid}", height=80,
                    )
                    edited_cat = st.selectbox(
                        "Kategori", CATEGORIES,
                        index=CATEGORIES.index(r.get("category","Diğer"))
                              if r.get("category") in CATEGORIES else 0,
                        key=f"tr_edit_cat_{rid}",
                    )
                    ec1, ec2, _ = st.columns([1, 1, 4])
                    with ec1:
                        if st.button("💾 Kaydet", key=f"tr_save_{rid}",
                                     use_container_width=True, type="primary"):
                            idx = next((i for i,x in enumerate(rules) if x["id"]==rid), None)
                            if idx is not None:
                                rules[idx]["rule"]     = edited_text.strip()
                                rules[idx]["category"] = edited_cat
                            _save_rules(rules)
                            st.session_state.tr_rules = rules
                            st.session_state[edit_key] = False
                            st.rerun()
                    with ec2:
                        if st.button("İptal", key=f"tr_cancel_{rid}",
                                     use_container_width=True):
                            st.session_state[edit_key] = False
                            st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                # ── Accordion görünüm modu ────────────────────────────────
                open_key = f"tr_open_{rid}"
                st.session_state.setdefault(open_key, False)
                is_open  = st.session_state[open_key]

                dot_c  = fg if active else "#30363d"
                hdr_bg = card_bg
                brd_c  = fg if is_open else (border_c if active else "#30363d")
                brad   = "10px 10px 0 0" if is_open else "10px"

                # Header
                h_col, btn_col = st.columns([14, 1])
                with h_col:
                    st.markdown(
                        f"<div style='background:{hdr_bg};border:1px solid {brd_c};"
                        f"border-left:3px solid {dot_c};border-radius:{brad};"
                        f"padding:0.65rem 1rem;opacity:{opacity};cursor:pointer;"
                        f"display:flex;align-items:center;gap:0.75rem'>"
                        f"<span style='color:{fg};font-family:\"Space Mono\",monospace;"
                        f"font-size:12px;min-width:22px;font-weight:700;"
                        f"flex-shrink:0'>{num:02d}</span>"
                        f"<span style='color:#e6edf3;font-size:14px;line-height:1.6;"
                        f"font-weight:500;flex:1'>{r['rule']}</span>"
                        f"<span style='font-size:11px;color:{_DT};flex-shrink:0'>"
                        f"{'▲' if is_open else '▼'}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with btn_col:
                    if st.button("▼" if not is_open else "▲",
                                 key=f"tr_acc_{rid}",
                                 use_container_width=True,
                                 label_visibility="collapsed"):
                        st.session_state[open_key] = not is_open
                        st.rerun()

                # Açık gövde: butonlar + kategori
                if is_open:
                    st.markdown(
                        f"<div style='background:{_BG3};border:1px solid {fg}50;"
                        f"border-top:none;border-radius:0 0 10px 10px;"
                        f"padding:10px 14px;margin-bottom:2px'>",
                        unsafe_allow_html=True,
                    )
                    # Kategori badge
                    cat_name = r.get("category","")
                    if cat_name:
                        st.markdown(
                            f"<span style='background:{fg}18;color:{fg};"
                            f"border:1px solid {fg}40;border-radius:20px;"
                            f"padding:2px 10px;font-size:11px;font-weight:600'>"
                            f"{cat_name}</span>",
                            unsafe_allow_html=True,
                        )
                        st.markdown("<div style='height:8px'></div>",
                                    unsafe_allow_html=True)

                    # Butonlar
                    ba1, ba2, ba3, _ = st.columns([1, 1, 1, 6])
                    with ba1:
                        lbl = "⏸ Durdur" if active else "▶ Aç"
                        if st.button(lbl, key=f"tr_toggle_{rid}",
                                     use_container_width=True):
                            idx = next((i for i,x in enumerate(rules)
                                        if x["id"]==rid), None)
                            if idx is not None:
                                rules[idx]["active"] = not active
                            _save_rules(rules)
                            st.session_state.tr_rules = rules
                            st.rerun()
                    with ba2:
                        if st.button("✏️ Düzenle", key=f"tr_edit_btn_{rid}",
                                     use_container_width=True):
                            st.session_state[edit_key] = True
                            st.rerun()
                    with ba3:
                        if st.button("🗑️ Sil", key=f"tr_del_{rid}",
                                     use_container_width=True):
                            rules = [x for x in rules if x["id"] != rid]
                            _save_rules(rules)
                            st.session_state.tr_rules = rules
                            st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)


    if not filtered:
        st.markdown(
            f"<div style='text-align:center;padding:2.5rem;color:#30363d;"
            f"border:1px dashed #21262d;border-radius:10px'>"
            f"<div style='font-size:2rem;margin-bottom:0.5rem'>📜</div>"
            f"<div style='font-size:14px;color:#484f58'>Bu kategoride kural yok</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
