"""
AR-GE LAB — Trader Gelişim ve Araştırma Merkezi
================================================
8 modül:
1. Günlük Gelişim Paneli (puan kartı)
2. Trade Challenge Sistemi (gamification)
3. Strateji Laboratuvarı
4. Edge Keşif Paneli (journal verisi analizi)
5. Trader Not Defteri
6. Market Gözlem
7. Koç Paneli
8. XP / Seviye Sistemi
"""

import streamlit as st
import json
import os
from datetime import datetime, timedelta, date

# ── Tema ─────────────────────────────────────────────────────────────────────
_BG  = "#0d1117"; _BG2 = "#161b22"; _BG3 = "#1c2128"
_DG  = "#21262d"; _DG2 = "#30363d"
_TX  = "#e6edf3"; _DT  = "#8b949e"; _DT2 = "#6e7681"
_G   = "#3fb950"; _R   = "#ff7b72"; _B   = "#58a6ff"
_Y   = "#e3b341"; _P   = "#a371f7"; _C   = "#39d353"

ARGE_FILE = "arge_data.json"

# ── Seviye sistemi ────────────────────────────────────────────────────────────
LEVELS = [
    (0,    "Rookie Trader",      "🌱", _DT),
    (100,  "Disciplined Trader", "⚔️", _B),
    (300,  "Sniper Trader",      "🎯", _G),
    (600,  "Elite Trader",       "💎", _P),
    (1000, "Master Trader",      "👑", _Y),
]

CHALLENGES = [
    {"id":1,  "title":"7 Gün Revenge Trade Yok",     "desc":"7 gün boyunca hiç revenge trade açma",        "target":7,  "xp":80,  "icon":"🧘"},
    {"id":2,  "title":"5 İşlem Plana Sadık",          "desc":"5 işlem üst üste plana uyu",                  "target":5,  "xp":50,  "icon":"📋"},
    {"id":3,  "title":"3 Gün Overtrade Yok",          "desc":"3 gün günde max 2 işlem",                     "target":3,  "xp":40,  "icon":"🛑"},
    {"id":4,  "title":"10 İşlem Stop Taşıma Yok",     "desc":"Stop loss taşımadan 10 işlem tamamla",        "target":10, "xp":100, "icon":"🔒"},
    {"id":5,  "title":"5 Gün Günlük Review",          "desc":"5 gün üst üste günlük review yap",            "target":5,  "xp":60,  "icon":"📊"},
    {"id":6,  "title":"RR 1:2 Altına Düşme",         "desc":"10 işlemde RR 1:2 üstünü koru",               "target":10, "xp":120, "icon":"⚖️"},
    {"id":7,  "title":"30 Gün Not Al",                "desc":"30 gün boyunca en az 1 not gir",              "target":30, "xp":150, "icon":"✍️"},
    {"id":8,  "title":"Haftalık Strateji Testi",      "desc":"Bir stratejiyi 2 hafta boyunca kaydet",       "target":14, "xp":90,  "icon":"🔬"},
]

NOTE_CATS = ["Psikoloji 🧠", "Teknik Analiz 📈", "Risk Yönetimi ⚖️",
             "Öğrenilen Dersler 📚", "Video Notları 🎥", "Coin Fikirleri 💡"]

STRATEGY_STATUSES = ["💡 Fikir", "🧪 Testte", "🔧 Geliştiriliyor", "✅ Canlıya Hazır", "❌ İptal"]
RISK_LEVELS = ["Düşük 🟢", "Orta 🟡", "Yüksek 🔴"]
MARKETS = ["BTC", "ETH", "Altcoin", "Tümü", "Majors"]

COACH_TIPS = [
    ("🔴", "Son kayıplardan sonra lot küçült — hesabın iyileşmesine izin ver"),
    ("🔴", "Üst üste 2 zarar varsa bugünkü işlemlere son ver"),
    ("🟡", "Plansız trade açma — setup yoksa bekle"),
    ("🟡", "RR 1:2 altına düşme — kazanç/kayıp oranını koru"),
    ("🟢", "Güçlü setup bekle — fırsat her gün gelmez"),
    ("🟢", "Günlük max işlem sayını belirle ve ona uyu"),
    ("🔵", "Stop taşıma hem en yaygın hem en pahalı hata"),
    ("🔵", "Kayıp serisinde lot küçült, kazanç serisinde de temkinli ol"),
    ("🟣", "Her işlem öncesi: setup var mı? RR uygun mu? Plan var mı?"),
    ("🟣", "Piyasa çok volatilse kenardan izle — trade yapmak zorunda değilsin"),
]

DAILY_MOTIVATIONS = [
    "Disiplin, motivasyondan çok daha güçlüdür. 💪",
    "En iyi setup'ı beklemek de bir stratejidir. 🎯",
    "Kayıptan değil, kayıptan ne öğrendiğinden kork. 📚",
    "Sabır, trader'ın en değerli silahıdır. ⏳",
    "Sermayeni koru — fırsat her zaman tekrar gelir. 🛡️",
    "Günde 2 kaliteli trade, 10 kötü trade'den iyidir. ✂️",
    "Plan yoksa trade yok. Bu kadar basit. 📋",
]


# ── Veri yönetimi ─────────────────────────────────────────────────────────────

def _load() -> dict:
    if os.path.exists(ARGE_FILE):
        try:
            with open(ARGE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return _default()

def _save(data: dict):
    try:
        with open(ARGE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _default() -> dict:
    return {
        "xp": 0,
        "daily_scores": {},     # {date_str: {disiplin, sabir, risk, psikoloji, dogru, gelisim}}
        "challenges": {},       # {challenge_id: {progress, completed, completed_at}}
        "strategies": [],
        "notes": [],
        "market_obs": {},       # {date_str: {type, notes}}
        "weekly_focus": "",
    }

def _get_arge() -> dict:
    if "arge" not in st.session_state:
        st.session_state.arge = _load()
    return st.session_state.arge

def _save_arge():
    _save(st.session_state.arge)

def _add_xp(amount: int, reason: str = ""):
    d = _get_arge()
    d["xp"] = d.get("xp", 0) + amount
    _save_arge()
    st.toast(f"✨ +{amount} XP — {reason}", icon="🏆")

def _get_level(xp: int):
    lvl = LEVELS[0]
    for threshold, name, icon, color in LEVELS:
        if xp >= threshold:
            lvl = (threshold, name, icon, color)
    return lvl

def _today() -> str:
    return date.today().isoformat()


# ── UI Yardımcıları ───────────────────────────────────────────────────────────

def _card(content_fn, border_color=_DG, bg=_BG2, glow=False):
    glow_css = f"box-shadow:0 0 20px {border_color}25;" if glow else ""
    st.markdown(
        f"<div style='background:{bg};border:1px solid {border_color};"
        f"border-radius:14px;padding:0.9rem 1rem 0.6rem;margin-bottom:8px;{glow_css}'>",
        unsafe_allow_html=True,
    )
    content_fn()
    st.markdown("</div>", unsafe_allow_html=True)

def _section_title(icon: str, title: str, color: str = _B, subtitle: str = ""):
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:12px'>"
        f"<span style='font-size:1.4rem'>{icon}</span>"
        f"<div>"
        f"<div style='font-size:1rem;font-weight:700;color:{_TX};"
        f"letter-spacing:-0.01em'>{title}</div>"
        f"{'<div style=\"font-size:12px;color:'+_DT+'\">'+subtitle+'</div>' if subtitle else ''}"
        f"</div>"
        f"<div style='flex:1;height:1px;background:linear-gradient({color}40,transparent)'></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

def _progress_bar(val: int, max_val: int = 100, color: str = _B, height: int = 6):
    pct = min(int(val / max_val * 100), 100)
    st.markdown(
        f"<div style='background:{_DG};border-radius:{height}px;height:{height}px;"
        f"overflow:hidden;margin:3px 0'>"
        f"<div style='width:{pct}%;height:{height}px;background:linear-gradient(90deg,{color},{color}cc);"
        f"border-radius:{height}px;transition:width 0.4s'></div></div>",
        unsafe_allow_html=True,
    )

def _badge(text: str, color: str = _B, bg: str = ""):
    bg = bg or f"{color}18"
    st.markdown(
        f"<span style='background:{bg};color:{color};border:1px solid {color}40;"
        f"border-radius:20px;padding:3px 10px;font-size:12px;font-weight:700;"
        f"display:inline-block;margin:2px'>{text}</span>",
        unsafe_allow_html=True,
    )

def _metric_card(label: str, value: str, color: str = _TX, sub: str = ""):
    st.markdown(
        f"<div style='background:{_BG3};border:1px solid {_DG};border-radius:10px;"
        f"padding:10px 14px;text-align:center'>"
        f"<div style='font-size:11px;color:{_DT};text-transform:uppercase;"
        f"letter-spacing:0.1em;margin-bottom:3px'>{label}</div>"
        f"<div style='font-size:1.3rem;font-weight:800;color:{color};"
        f"font-family:\"Space Mono\",monospace'>{value}</div>"
        f"{'<div style=\"font-size:11px;color:'+_DT2+';margin-top:2px\">'+sub+'</div>' if sub else ''}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. GÜNLÜK GELİŞİM PANELİ
# ══════════════════════════════════════════════════════════════════════════════

def _render_daily(d: dict):
    _section_title("📅", "Günlük Gelişim Paneli",
                   _B, f"Bugün: {date.today().strftime('%d %B %Y, %A')}")

    today = _today()
    scores = d.get("daily_scores", {}).get(today, {})

    # Motivasyon
    import hashlib
    mot_idx = int(hashlib.md5(today.encode()).hexdigest(), 16) % len(DAILY_MOTIVATIONS)
    st.markdown(
        f"<div style='background:linear-gradient(135deg,#0d1a2a,#0a2018);"
        f"border:1px solid {_B}30;border-left:3px solid {_B};"
        f"border-radius:10px;padding:10px 14px;margin-bottom:10px;"
        f"font-style:italic;font-size:14px;color:{_DT}'>"
        f"💬 {DAILY_MOTIVATIONS[mot_idx]}</div>",
        unsafe_allow_html=True,
    )

    # Skor kartları
    sc1, sc2, sc3, sc4 = st.columns(4)
    score_fields = [
        (sc1, "disiplin",  "🎯 Disiplin",   _B),
        (sc2, "sabir",     "⏳ Sabır",       _G),
        (sc3, "risk",      "⚖️ Risk Yön.",  _Y),
        (sc4, "psikoloji", "🧠 Psikoloji",   _P),
    ]
    score_vals = {}
    for col, key, label, color in score_fields:
        with col:
            val = st.slider(label, 0, 100,
                            scores.get(key, 70),
                            key=f"arge_score_{key}_{today}",
                            help=f"Bugünkü {label} puanın")
            score_vals[key] = val
            _progress_bar(val, 100, color, 5)
            avg_c = _G if val >= 70 else _Y if val >= 40 else _R
            st.markdown(
                f"<div style='text-align:center;font-size:13px;font-weight:700;"
                f"color:{avg_c};font-family:\"Space Mono\",monospace'>{val}/100</div>",
                unsafe_allow_html=True,
            )

    # Genel rating
    overall = int(sum(score_vals.values()) / 4)
    oc = _G if overall >= 70 else _Y if overall >= 50 else _R
    st.markdown(
        f"<div style='background:{_BG3};border:1px solid {oc}40;"
        f"border-radius:10px;padding:10px 14px;margin:8px 0;"
        f"display:flex;align-items:center;justify-content:space-between'>"
        f"<span style='font-size:14px;color:{_DT};font-weight:600'>Genel Trader Rating</span>"
        f"<div style='display:flex;align-items:center;gap:12px'>"
        f"<div style='width:120px'>",
        unsafe_allow_html=True,
    )
    _progress_bar(overall, 100, oc, 8)
    st.markdown(
        f"</div>"
        f"<span style='font-family:\"Space Mono\",monospace;font-size:1.4rem;"
        f"font-weight:800;color:{oc}'>{overall}</span>"
        f"<span style='font-size:12px;color:{_DT}'>/100</span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    # Doğru yapılanlar / gelişim
    dc1, dc2 = st.columns(2)
    with dc1:
        dogru = st.text_area(
            "✅ Bugün yapılan doğru şeyler",
            value=scores.get("dogru", ""),
            placeholder="• Plana sadık kaldım\n• RR 1:2 üstünü korudum...",
            key=f"arge_dogru_{today}", height=80,
        )
    with dc2:
        gelisim = st.text_area(
            "🔧 Geliştireceğim alanlar",
            value=scores.get("gelisim", ""),
            placeholder="• Erken exit yaptım\n• Stop taşıdım...",
            key=f"arge_gelisim_{today}", height=80,
        )

    sa1, _ = st.columns([1, 4])
    with sa1:
        if st.button("💾 Günü Kaydet", key="arge_save_daily", type="primary",
                     use_container_width=True):
            if "daily_scores" not in d:
                d["daily_scores"] = {}
            d["daily_scores"][today] = {
                **score_vals, "dogru": dogru, "gelisim": gelisim,
                "saved_at": datetime.now().isoformat(),
            }
            _save_arge()
            _add_xp(10, "Günlük review")
            st.success("✅ Günlük gelişim kaydedildi! +10 XP")
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# 2. CHALLENGE SİSTEMİ
# ══════════════════════════════════════════════════════════════════════════════

def _render_challenges(d: dict):
    _section_title("🏆", "Trade Challenge Sistemi",
                   _Y, "Görevleri tamamla, rozet kazan, seviye atla")

    ch_data = d.get("challenges", {})
    active_tab, completed_tab = st.tabs(["⚡ Aktif Görevler", "🎖️ Tamamlananlar"])

    with active_tab:
        for ch in CHALLENGES:
            cid = str(ch["id"])
            cdata = ch_data.get(cid, {"progress": 0, "completed": False})
            if cdata.get("completed"):
                continue

            progress = cdata.get("progress", 0)
            pct      = min(int(progress / ch["target"] * 100), 100)
            pc       = _G if pct >= 70 else _Y if pct >= 30 else _B

            row_col, btn_col = st.columns([10, 2])
            with row_col:
                st.markdown(
                    f"<div style='background:{_BG2};border:1px solid {_DG};"
                    f"border-radius:12px;padding:10px 14px;margin-bottom:4px'>"
                    f"<div style='display:flex;justify-content:space-between;"
                    f"align-items:center;margin-bottom:6px'>"
                    f"<div style='display:flex;align-items:center;gap:8px'>"
                    f"<span style='font-size:1.3rem'>{ch['icon']}</span>"
                    f"<div>"
                    f"<div style='font-size:14px;font-weight:700;color:{_TX}'>"
                    f"{ch['title']}</div>"
                    f"<div style='font-size:12px;color:{_DT}'>{ch['desc']}</div>"
                    f"</div></div>"
                    f"<div style='text-align:right'>"
                    f"<div style='font-size:11px;color:{_DT}'>İlerleme</div>"
                    f"<div style='font-family:\"Space Mono\",monospace;font-size:14px;"
                    f"font-weight:700;color:{pc}'>{progress}/{ch['target']}</div>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
                _progress_bar(progress, ch["target"], pc, 6)
                st.markdown(
                    f"<div style='font-size:11px;color:{_Y};margin-top:4px'>"
                    f"🏅 Ödül: +{ch['xp']} XP</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with btn_col:
                if st.button("＋1", key=f"ch_inc_{cid}", use_container_width=True,
                             help="İlerleme ekle"):
                    new_p = progress + 1
                    if "challenges" not in d:
                        d["challenges"] = {}
                    if new_p >= ch["target"]:
                        d["challenges"][cid] = {
                            "progress": new_p, "completed": True,
                            "completed_at": datetime.now().isoformat(),
                        }
                        _save_arge()
                        _add_xp(ch["xp"], ch["title"])
                        st.balloons()
                        st.success(f"🎖️ '{ch['title']}' tamamlandı! +{ch['xp']} XP")
                    else:
                        d["challenges"][cid] = {
                            "progress": new_p, "completed": False,
                        }
                        _save_arge()
                    st.rerun()

    with completed_tab:
        completed = [ch for ch in CHALLENGES
                     if ch_data.get(str(ch["id"]), {}).get("completed")]
        if not completed:
            st.markdown(
                f"<div style='text-align:center;padding:2rem;color:{_DT}'>"
                f"Henüz tamamlanan görev yok. Devam et! 💪</div>",
                unsafe_allow_html=True,
            )
        for ch in completed:
            cid   = str(ch["id"])
            cat   = ch_data[cid].get("completed_at", "")
            cat_s = ""
            if cat:
                try: cat_s = datetime.fromisoformat(cat).strftime("%d %b %Y")
                except: pass
            st.markdown(
                f"<div style='background:linear-gradient(135deg,#071a0e,#0a2018);"
                f"border:1px solid {_G}40;border-radius:12px;"
                f"padding:10px 14px;margin-bottom:6px;"
                f"display:flex;align-items:center;justify-content:space-between'>"
                f"<div style='display:flex;align-items:center;gap:10px'>"
                f"<span style='font-size:1.5rem'>{ch['icon']}</span>"
                f"<div>"
                f"<div style='font-size:14px;font-weight:700;color:{_G}'>{ch['title']}</div>"
                f"<div style='font-size:12px;color:{_DT}'>{cat_s}</div>"
                f"</div></div>"
                f"<div style='background:{_G}20;color:{_G};border:1px solid {_G}40;"
                f"border-radius:20px;padding:4px 12px;font-size:12px;font-weight:700'>"
                f"🏅 +{ch['xp']} XP</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# 3. STRATEJİ LABORATUVARI
# ══════════════════════════════════════════════════════════════════════════════

def _render_strategies(d: dict):
    _section_title("🔬", "Strateji Laboratuvarı",
                   _P, "Fikirleri test et, stratejileri geliştir")

    strategies = d.get("strategies", [])

    # Yeni strateji ekle
    with st.expander("＋ Yeni Strateji Ekle", expanded=not strategies):
        sc1, sc2 = st.columns(2)
        with sc1:
            s_name   = st.text_input("Strateji Adı", key="arge_s_name",
                                     placeholder="Örn: SMC Liquidity Grab")
            s_dir    = st.selectbox("Yön", ["Long", "Short", "Her İkisi"],
                                    key="arge_s_dir")
            s_market = st.selectbox("Market", MARKETS, key="arge_s_market")
            s_risk   = st.selectbox("Risk Seviyesi", RISK_LEVELS, key="arge_s_risk")
        with sc2:
            s_status = st.selectbox("Durum", STRATEGY_STATUSES, key="arge_s_status")
            s_desc   = st.text_area("Açıklama", key="arge_s_desc", height=80,
                                    placeholder="Strateji mantığını anlat...")
            s_notes  = st.text_area("Notlar", key="arge_s_notes", height=60,
                                    placeholder="Ek notlar, backtesting sonuçları...")
        if st.button("💾 Strateji Kaydet", key="arge_save_strat",
                     type="primary", use_container_width=False):
            if s_name.strip():
                strategies.append({
                    "id": len(strategies) + 1,
                    "name": s_name.strip(),
                    "direction": s_dir,
                    "market": s_market,
                    "risk": s_risk,
                    "status": s_status,
                    "desc": s_desc,
                    "notes": s_notes,
                    "created_at": datetime.now().isoformat(),
                })
                d["strategies"] = strategies
                _save_arge()
                _add_xp(15, "Strateji eklendi")
                st.success("✅ Strateji kaydedildi! +15 XP")
                st.rerun()

    # Strateji kartları
    if not strategies:
        st.markdown(
            f"<div style='text-align:center;padding:2rem;color:{_DT}'>"
            f"Henüz strateji yok. İlk stratejini ekle! 🔬</div>",
            unsafe_allow_html=True,
        )
        return

    status_colors = {
        "💡 Fikir": _DT, "🧪 Testte": _B, "🔧 Geliştiriliyor": _Y,
        "✅ Canlıya Hazır": _G, "❌ İptal": _R,
    }
    cols = st.columns(3)
    for i, strat in enumerate(strategies):
        with cols[i % 3]:
            sc = status_colors.get(strat["status"], _DT)
            st.markdown(
                f"<div style='background:{_BG2};border:1px solid {_DG};"
                f"border-top:3px solid {sc};border-radius:12px;"
                f"padding:12px 14px;margin-bottom:8px;min-height:140px'>"
                f"<div style='font-size:14px;font-weight:700;color:{_TX};"
                f"margin-bottom:4px'>{strat['name']}</div>"
                f"<div style='font-size:12px;color:{_DT};margin-bottom:6px'>"
                f"{strat.get('desc','')[:80]}{'...' if len(strat.get('desc',''))>80 else ''}</div>"
                f"<div style='display:flex;flex-wrap:wrap;gap:4px;margin-bottom:6px'>"
                f"<span style='background:{sc}20;color:{sc};border:1px solid {sc}40;"
                f"border-radius:5px;padding:2px 7px;font-size:11px;font-weight:700'>"
                f"{strat['status']}</span>"
                f"<span style='background:{_BG3};color:{_DT};border-radius:5px;"
                f"padding:2px 7px;font-size:11px'>{strat['direction']}</span>"
                f"<span style='background:{_BG3};color:{_DT};border-radius:5px;"
                f"padding:2px 7px;font-size:11px'>{strat['market']}</span>"
                f"<span style='background:{_BG3};color:{_DT};border-radius:5px;"
                f"padding:2px 7px;font-size:11px'>{strat['risk']}</span>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            btn_a, btn_b = st.columns(2)
            with btn_a:
                new_status = st.selectbox(
                    "", STRATEGY_STATUSES, key=f"strat_status_{i}",
                    index=STRATEGY_STATUSES.index(strat["status"])
                          if strat["status"] in STRATEGY_STATUSES else 0,
                    label_visibility="collapsed",
                )
            with btn_b:
                if st.button("🗑️", key=f"del_strat_{i}", use_container_width=True):
                    d["strategies"] = [s for j, s in enumerate(strategies) if j != i]
                    _save_arge(); st.rerun()
            if new_status != strat["status"]:
                strategies[i]["status"] = new_status
                d["strategies"] = strategies
                _save_arge(); st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# 4. EDGE KEŞİF PANELİ
# ══════════════════════════════════════════════════════════════════════════════

def _render_edge(trade_data: dict):
    _section_title("🔭", "Edge Keşif Paneli",
                   _C, "Journal verilerinden kendi edge'ini keşfet")

    trades = trade_data.get("closed_trades", [])

    if len(trades) < 3:
        st.info("📊 En az 3 kapalı işlem gerekiyor. Daha fazla işlem kaydet!")
        return

    # Hesaplamalar
    wins   = [t for t in trades if t.get("pnl", 0) > 0]
    losses = [t for t in trades if t.get("pnl", 0) <= 0]
    longs  = [t for t in trades if t.get("direction") == "LONG"]
    shorts = [t for t in trades if t.get("direction") == "SHORT"]
    long_wr  = len([t for t in longs  if t.get("pnl",0)>0]) / max(len(longs), 1) * 100
    short_wr = len([t for t in shorts if t.get("pnl",0)>0]) / max(len(shorts), 1) * 100

    # Saat analizi
    hour_wins = {}
    hour_total= {}
    for t in trades:
        try:
            h = datetime.fromisoformat(t.get("created_at","")).hour
            hour_total[h] = hour_total.get(h, 0) + 1
            if t.get("pnl", 0) > 0:
                hour_wins[h] = hour_wins.get(h, 0) + 1
        except Exception:
            pass
    best_hour = None
    best_wr   = 0
    for h, total in hour_total.items():
        if total >= 2:
            wr = hour_wins.get(h, 0) / total * 100
            if wr > best_wr:
                best_wr, best_hour = wr, h

    # Setup analizi
    setup_wins  = {}
    setup_total = {}
    for t in trades:
        s = t.get("setup_type", "")
        if s:
            setup_total[s] = setup_total.get(s, 0) + 1
            if t.get("pnl", 0) > 0:
                setup_wins[s] = setup_wins.get(s, 0) + 1
    best_setup = None
    best_swr   = 0
    for s, total in setup_total.items():
        if total >= 2:
            swr = setup_wins.get(s, 0) / total * 100
            if swr > best_swr:
                best_swr, best_setup = swr, s

    # En kötü hata
    all_mistakes = []
    for t in trades:
        all_mistakes.extend(t.get("mistakes", []))
    worst_mistake = max(set(all_mistakes), key=all_mistakes.count) if all_mistakes else "—"

    # Ortalama kazançlı işlem süresi
    win_durations = []
    for t in wins:
        try:
            opened = datetime.fromisoformat(t.get("created_at",""))
            closed = datetime.fromisoformat(t.get("closed_at",""))
            win_durations.append((closed - opened).total_seconds() / 3600)
        except Exception:
            pass
    avg_win_dur = sum(win_durations) / len(win_durations) if win_durations else 0

    # Gün analizi
    day_wins  = {}
    day_total = {}
    day_names = {0:"Pzt",1:"Sal",2:"Çar",3:"Per",4:"Cum",5:"Cmt",6:"Paz"}
    for t in trades:
        try:
            wd = datetime.fromisoformat(t.get("created_at","")).weekday()
            day_total[wd] = day_total.get(wd, 0) + 1
            if t.get("pnl", 0) > 0:
                day_wins[wd] = day_wins.get(wd, 0) + 1
        except Exception:
            pass
    best_day = None
    best_dwr = 0
    for wd, total in day_total.items():
        if total >= 2:
            dwr = day_wins.get(wd, 0) / total * 100
            if dwr > best_dwr:
                best_dwr, best_day = dwr, wd

    # Metrikler
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        dc = _G if long_wr > short_wr else _DT
        sc = _G if short_wr > long_wr  else _DT
        _metric_card("Long WR", f"%{long_wr:.0f}", dc, f"{len(longs)} işlem")
        _metric_card("Short WR", f"%{short_wr:.0f}", sc, f"{len(shorts)} işlem")
    with m2:
        _metric_card("En İyi Saat",
                     f"{best_hour:02d}:00" if best_hour is not None else "—",
                     _B, f"%{best_wr:.0f} WR" if best_hour else "")
        _metric_card("En İyi Gün",
                     day_names.get(best_day,"—") if best_day is not None else "—",
                     _G, f"%{best_dwr:.0f} WR" if best_day else "")
    with m3:
        _metric_card("En İyi Setup", best_setup or "—", _P,
                     f"%{best_swr:.0f} WR" if best_setup else "")
        _metric_card("En Kötü Hata", worst_mistake, _R)
    with m4:
        wr_total = len(wins) / len(trades) * 100
        _metric_card("Genel WR", f"%{wr_total:.0f}", _G if wr_total >= 50 else _R,
                     f"{len(wins)}W / {len(losses)}L")
        _metric_card("Ort. Win Süresi",
                     f"{avg_win_dur:.1f}h" if avg_win_dur else "—",
                     _B)

    # Long vs Short özet
    better = "LONG" if long_wr >= short_wr else "SHORT"
    bc     = _G if better == "LONG" else _R
    st.markdown(
        f"<div style='background:{bc}15;border:1px solid {bc}40;"
        f"border-radius:10px;padding:10px 16px;margin-top:8px;"
        f"font-size:14px;color:{_TX}'>"
        f"💡 <b>Edge'in:</b> "
        f"<span style='color:{bc};font-weight:700'>{better}</span> tarafında "
        f"daha başarılısın "
        f"({long_wr:.0f}% vs {short_wr:.0f}% WR). "
        f"{'Long setuplara odaklan.' if better=='LONG' else 'Short setuplara odaklan.'}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRADER NOT DEFTERİ
# ══════════════════════════════════════════════════════════════════════════════

def _render_notes(d: dict):
    _section_title("📓", "Trader Not Defteri",
                   _Y, "Kategori bazlı bilgi birikimi")

    notes = d.get("notes", [])

    # Yeni not
    with st.expander("＋ Yeni Not Ekle"):
        nc1, nc2 = st.columns([2, 1])
        with nc1:
            n_title = st.text_input("Başlık", key="arge_n_title",
                                    placeholder="Öğrendiğin şeyi özetle...")
            n_body  = st.text_area("İçerik", key="arge_n_body",
                                   height=100, placeholder="Detaylar...")
        with nc2:
            n_cat   = st.selectbox("Kategori", NOTE_CATS, key="arge_n_cat")
            n_tags  = st.text_input("Etiketler", key="arge_n_tags",
                                    placeholder="rr, psikoloji, stop...")
        if st.button("💾 Not Kaydet", key="arge_save_note",
                     type="primary"):
            if n_title.strip():
                notes.insert(0, {
                    "id": len(notes) + 1,
                    "title": n_title.strip(),
                    "body": n_body,
                    "category": n_cat,
                    "tags": [t.strip() for t in n_tags.split(",") if t.strip()],
                    "created_at": datetime.now().isoformat(),
                })
                d["notes"] = notes
                _save_arge()
                _add_xp(5, "Not eklendi")
                st.success("✅ Not kaydedildi! +5 XP")
                st.rerun()

    if not notes:
        st.markdown(
            f"<div style='text-align:center;padding:2rem;color:{_DT}'>"
            f"Henüz not yok. İlk notunu ekle! ✍️</div>",
            unsafe_allow_html=True,
        )
        return

    # Arama + filtre
    sf1, sf2, sf3 = st.columns([3, 2, 1])
    with sf1:
        search = st.text_input("🔍 Ara", key="arge_note_search",
                               placeholder="Başlık veya içerik...",
                               label_visibility="collapsed")
    with sf2:
        cat_f = st.selectbox("Kategori", ["Tümü"] + NOTE_CATS,
                             key="arge_note_cat_f", label_visibility="collapsed")
    with sf3:
        st.markdown(
            f"<div style='padding-top:8px;font-size:13px;color:{_DT}'>"
            f"{len(notes)} not</div>",
            unsafe_allow_html=True,
        )

    filtered = notes
    if search:
        filtered = [n for n in filtered
                    if search.lower() in n.get("title","").lower()
                    or search.lower() in n.get("body","").lower()]
    if cat_f != "Tümü":
        filtered = [n for n in filtered if n.get("category") == cat_f]

    cat_colors = {
        "Psikoloji 🧠": _P, "Teknik Analiz 📈": _B,
        "Risk Yönetimi ⚖️": _Y, "Öğrenilen Dersler 📚": _G,
        "Video Notları 🎥": _R, "Coin Fikirleri 💡": _C,
    }
    for note in filtered[:20]:
        cc = cat_colors.get(note.get("category",""), _DT)
        try: created = datetime.fromisoformat(note["created_at"]).strftime("%d %b")
        except: created = ""
        note_c, del_c = st.columns([14, 1])
        with note_c:
            with st.expander(f"{note['title']}   —   {note.get('category','')}   {created}"):
                st.markdown(
                    f"<div style='color:{_TX};font-size:14px;line-height:1.7'>"
                    f"{note.get('body','').replace(chr(10),'<br>')}</div>",
                    unsafe_allow_html=True,
                )
                if note.get("tags"):
                    for tag in note["tags"]:
                        _badge(f"#{tag}", cc)
        with del_c:
            if st.button("🗑️", key=f"del_note_{note['id']}",
                         use_container_width=True):
                d["notes"] = [n for n in notes if n["id"] != note["id"]]
                _save_arge(); st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# 6. MARKET GÖZLEM
# ══════════════════════════════════════════════════════════════════════════════

def _render_market_obs(d: dict):
    _section_title("🌍", "Market Gözlem & AR-GE",
                   _C, "Bugünün piyasa yapısını kaydet")

    today    = _today()
    obs_data = d.get("market_obs", {})
    today_obs= obs_data.get(today, {})

    MARKET_TYPES = ["Trend 📈", "Range ↔️", "Fake Breakout ⚡",
                    "News Driven 📰", "Chop 🌊", "Volatile 💥"]

    mc1, mc2 = st.columns([1, 2])
    with mc1:
        m_type = st.selectbox(
            "Bugün piyasa tipi",
            MARKET_TYPES,
            index=MARKET_TYPES.index(today_obs.get("type", MARKET_TYPES[0]))
                  if today_obs.get("type") in MARKET_TYPES else 0,
            key=f"mobs_type_{today}",
        )
        st.markdown("")
        if st.button("💾 Gözlemi Kaydet", key="save_mobs",
                     type="primary", use_container_width=True):
            obs_data[today] = {
                "type": m_type,
                "notes": st.session_state.get(f"mobs_notes_{today}", ""),
                "saved_at": datetime.now().isoformat(),
            }
            d["market_obs"] = obs_data
            _save_arge()
            st.success("✅ Gözlem kaydedildi!")
            st.rerun()

        # Son 7 günün özeti
        st.markdown(
            f"<div style='font-size:12px;font-weight:600;color:{_DT};"
            f"margin-top:12px;margin-bottom:6px'>Son 7 Gün</div>",
            unsafe_allow_html=True,
        )
        for i in range(7):
            d_str = (date.today() - timedelta(days=i)).isoformat()
            od    = obs_data.get(d_str)
            if od:
                dt_label = "Bugün" if i == 0 else f"{i}g önce"
                st.markdown(
                    f"<div style='font-size:12px;color:{_DT};padding:2px 0'>"
                    f"<span style='color:{_TX}'>{dt_label}:</span> "
                    f"{od.get('type','—')}</div>",
                    unsafe_allow_html=True,
                )

    with mc2:
        m_notes = st.text_area(
            "Bugünkü gözlemler",
            value=today_obs.get("notes", ""),
            key=f"mobs_notes_{today}",
            height=200,
            placeholder=(
                "• BTC 78K üstünde tutunuyor, direnç güçlü\n"
                "• ETH zayıf görünüyor\n"
                "• Funding rate yükseliyor — dikkatli ol\n"
                "• Setup: ---\n"
                "• Plan: ---"
            ),
        )


# ══════════════════════════════════════════════════════════════════════════════
# 7. KOÇ PANELİ
# ══════════════════════════════════════════════════════════════════════════════

def _render_coach(trade_data: dict):
    _section_title("🎓", "Manuel Koç Paneli",
                   _P, "Disiplin rehberi ve günlük uyarılar")

    trades = trade_data.get("closed_trades", [])
    recent = sorted(trades, key=lambda x: x.get("created_at",""), reverse=True)[:10]

    # Otomatik uyarılar
    warnings = []
    if len(recent) >= 2:
        last2_pnl = [t.get("pnl", 0) for t in recent[:2]]
        if all(p < 0 for p in last2_pnl):
            warnings.append(("🔴", "Son 2 işlem zararda — bugün mola ver veya lot küçült!"))
    if len(recent) >= 3:
        last3 = recent[:3]
        revenge = [t for t in last3 if "revenge trade" in t.get("mistakes", [])]
        if revenge:
            warnings.append(("🔴", "Son 3 işlemde revenge trade var — dur, nefes al!"))
    recent_today = [t for t in recent
                    if t.get("created_at","")[:10] == _today()]
    if len(recent_today) >= 2:
        warnings.append(("🟡", f"Bugün {len(recent_today)} işlem açtın — günlük limitine dikkat!"))

    if warnings:
        st.markdown(
            f"<div style='background:#1c0a0a;border:1px solid {_R}40;"
            f"border-radius:12px;padding:10px 14px;margin-bottom:12px'>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='font-size:13px;font-weight:700;color:{_R};"
            f"margin-bottom:8px'>⚠️ Otomatik Uyarılar</div>",
            unsafe_allow_html=True,
        )
        for icon, msg in warnings:
            st.markdown(
                f"<div style='font-size:13px;color:{_TX};padding:3px 0'>"
                f"{icon} {msg}</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # Koç kartları
    st.markdown(
        f"<div style='font-size:13px;font-weight:600;color:{_DT};"
        f"margin-bottom:8px'>📌 Disiplin Rehberi</div>",
        unsafe_allow_html=True,
    )
    tip_colors = {"🔴": _R, "🟡": _Y, "🟢": _G, "🔵": _B, "🟣": _P}

    cols = st.columns(2)
    for i, (icon, tip) in enumerate(COACH_TIPS):
        with cols[i % 2]:
            tc = tip_colors.get(icon, _DT)
            st.markdown(
                f"<div style='background:{_BG2};border:1px solid {_DG};"
                f"border-left:3px solid {tc};border-radius:10px;"
                f"padding:9px 12px;margin-bottom:6px'>"
                f"<div style='font-size:13px;color:{_TX}'>"
                f"{icon} {tip}</div></div>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# 8. XP / SEVİYE SİSTEMİ
# ══════════════════════════════════════════════════════════════════════════════

def _render_xp(d: dict):
    xp  = d.get("xp", 0)
    lvl = _get_level(xp)
    threshold, name, icon, color = lvl

    next_lvl = None
    for t, n, i, c in LEVELS:
        if t > xp:
            next_lvl = (t, n, i, c)
            break

    st.markdown(
        f"<div style='background:linear-gradient(135deg,{_BG2},{_BG3});"
        f"border:1px solid {color}40;border-radius:16px;"
        f"padding:1.2rem 1.4rem;margin-bottom:12px'>"
        f"<div style='display:flex;align-items:center;gap:16px'>"
        f"<div style='font-size:3rem'>{icon}</div>"
        f"<div style='flex:1'>"
        f"<div style='font-size:11px;color:{_DT};text-transform:uppercase;"
        f"letter-spacing:0.12em;margin-bottom:2px'>Seviye</div>"
        f"<div style='font-family:\"Space Mono\",monospace;font-size:1.3rem;"
        f"font-weight:800;color:{color}'>{name}</div>"
        f"<div style='font-size:13px;color:{_DT};margin-top:2px'>"
        f"{xp} XP biriktirdin</div>"
        f"</div>"
        f"<div style='text-align:right'>"
        f"<div style='font-family:\"Space Mono\",monospace;font-size:2rem;"
        f"font-weight:800;color:{color}'>{xp}</div>"
        f"<div style='font-size:12px;color:{_DT}'>toplam XP</div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    if next_lvl:
        nt, nn, ni, nc = next_lvl
        needed = nt - xp
        prog   = (xp - threshold) / (nt - threshold) * 100 if nt > threshold else 100
        st.markdown(
            f"<div style='margin-top:8px'>"
            f"<div style='display:flex;justify-content:space-between;"
            f"font-size:12px;color:{_DT};margin-bottom:4px'>"
            f"<span>Sonraki: {ni} {nn}</span>"
            f"<span>{needed} XP kaldı</span></div>",
            unsafe_allow_html=True,
        )
        _progress_bar(int(prog), 100, nc, 8)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # XP Kazanma Rehberi
    st.markdown(
        f"<div style='font-size:13px;font-weight:600;color:{_DT};"
        f"margin-bottom:6px'>💡 XP Nasıl Kazanılır?</div>",
        unsafe_allow_html=True,
    )
    xp_guide = [
        ("📅 Günlük review yap",    "+10 XP"),
        ("✍️ Not ekle",              "+5 XP"),
        ("🔬 Strateji ekle",         "+15 XP"),
        ("🏆 Challenge tamamla",     "+40-150 XP"),
        ("📋 İşlem kaydet",         "+2 XP"),
    ]
    gc = st.columns(len(xp_guide))
    for col, (action, xp_val) in zip(gc, xp_guide):
        with col:
            st.markdown(
                f"<div style='background:{_BG2};border:1px solid {_DG};"
                f"border-radius:8px;padding:7px;text-align:center'>"
                f"<div style='font-size:12px;color:{_DT};margin-bottom:3px'>{action}</div>"
                f"<div style='font-size:13px;font-weight:700;color:{_Y}'>{xp_val}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# ANA RENDER
# ══════════════════════════════════════════════════════════════════════════════

def render_arge_lab(trade_data: dict):
    d = _get_arge()

    # ── Header ────────────────────────────────────────────────────────────────
    xp   = d.get("xp", 0)
    lvl  = _get_level(xp)
    _, lname, licon, lcolor = lvl

    st.markdown(
        f"<div style='background:linear-gradient(135deg,#0d1117,#0d1a2a);"
        f"border:1px solid {_B}20;border-radius:16px;"
        f"padding:1rem 1.4rem;margin-bottom:16px;"
        f"display:flex;align-items:center;justify-content:space-between'>"
        f"<div>"
        f"<div style='font-family:\"Space Mono\",monospace;font-size:1.1rem;"
        f"font-weight:700;color:{_TX};letter-spacing:0.05em'>"
        f"🧪 AR-GE LAB</div>"
        f"<div style='font-size:13px;color:{_DT};margin-top:2px'>"
        f"Trader Gelişim & Araştırma Merkezi</div>"
        f"</div>"
        f"<div style='display:flex;align-items:center;gap:12px'>"
        f"<div style='text-align:center'>"
        f"<div style='font-size:1.5rem'>{licon}</div>"
        f"<div style='font-size:12px;font-weight:700;color:{lcolor}'>{lname}</div>"
        f"</div>"
        f"<div style='background:{lcolor}20;border:1.5px solid {lcolor}50;"
        f"border-radius:10px;padding:6px 14px;text-align:center'>"
        f"<div style='font-family:\"Space Mono\",monospace;font-size:1.2rem;"
        f"font-weight:800;color:{lcolor}'>{xp}</div>"
        f"<div style='font-size:10px;color:{_DT};text-transform:uppercase;"
        f"letter-spacing:0.1em'>XP</div>"
        f"</div></div></div>",
        unsafe_allow_html=True,
    )

    # ── Sekmeler ──────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📅 Günlük",
        "🏆 Challenge",
        "🔬 Strateji Lab",
        "🔭 Edge Keşif",
        "📓 Notlar",
        "🌍 Market",
        "🎓 Koç",
        "⭐ XP & Seviye",
    ])

    with tabs[0]: _render_daily(d)
    with tabs[1]: _render_challenges(d)
    with tabs[2]: _render_strategies(d)
    with tabs[3]: _render_edge(trade_data)
    with tabs[4]: _render_notes(d)
    with tabs[5]: _render_market_obs(d)
    with tabs[6]: _render_coach(trade_data)
    with tabs[7]: _render_xp(d)
