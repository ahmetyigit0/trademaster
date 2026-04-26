"""
TradeBot UI — Streamlit sekmesi
================================
Strateji ayarları, başlat/durdur, canlı durum, log, işlem tablosu.
"""

import streamlit as st
import time
from datetime import datetime

from bot.engine import get_state, start_bot, stop_bot, HAS_BINANCE

# ── Tema ─────────────────────────────────────────────────────────────────────
_G  = "#3fb950";  _R = "#ff7b72";  _B = "#58a6ff"
_Y  = "#e3b341";  _TX= "#e6edf3";  _DT= "#b1bac4"
_BG = "#161b22";  _DB= "#0d1117";  _DG= "#21262d"

STRATEGIES = {
    "EMA_CROSS":       "EMA Crossover + RSI filtresi",
    "RSI_MEAN_REVERT": "RSI Aşırı Al/Sat",
    "BREAKOUT":        "Breakout (yüksek/düşük kırılım)",
}

TIMEFRAMES = ["1m","3m","5m","15m","30m","1h","2h","4h"]

# ── Coin kataloğu ──────────────────────────────────────────────────────────
# (Binance Futures USDT-M perp çiftleri, kategorilendirilmiş)
COINS = {
    "🥇 Major":       ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT"],
    "🔵 Layer 1":     ["AVAXUSDT","ADAUSDT","DOTUSDT","NEARUSDT","ATOMUSDT",
                       "APTUSDT","SUIUSDT","TIAUSDT","INJUSDT","SEIUSDT"],
    "🟣 Layer 2":     ["MATICUSDT","OPUSDT","ARBUSDT","STRKUSDT","MANTAUSDT"],
    "🟡 DeFi":        ["UNIUSDT","AAVEUSDT","MKRUSDT","CRVUSDT","SNXUSDT",
                       "COMPUSDT","1INCHUSDT","RUNEUSDT"],
    "🟢 AI / Data":   ["FETUSDT","AGIXUSDT","RENDERUSDT","WLDUSDT","ARKMUSDT"],
    "🔴 Meme":        ["DOGEUSDT","SHIBUSDT","PEPEUSDT","WIFUSDT","BONKUSDT",
                       "FLOKIUSDT"],
    "⚪ Diğer":       ["LINKUSDT","LTCUSDT","FILUSDT","SANDUSDT","MANAUSDT",
                       "AXSUSDT","GALAUSDT","APEUSDT","GMXUSDT","JUPUSDT"],
}

# Düz liste (arama için)
_ALL_COINS: list[str] = [c for cats in COINS.values() for c in cats]

LOG_COLORS = {
    "INFO":  _DT,
    "LONG":  _G,
    "SHORT": _R,
    "SELL":  _Y,
    "BUY":   _G,
    "WARN":  _Y,
    "ERROR": _R,
}


def render_tradebot():
    st.markdown(
        f"<div style='font-size:1.15rem;font-weight:700;color:#f0f6fc;"
        f"margin-bottom:4px'>🤖 TradeBot</div>"
        f"<div style='font-size:13px;color:{_DT};margin-bottom:1rem'>"
        f"Binance Futures otomatik alım-satım botu. "
        f"WebSocket tabanlı — rate limit / ban riski yok.</div>",
        unsafe_allow_html=True,
    )

    if not HAS_BINANCE:
        st.error(
            "**python-binance** yüklü değil.\n\n"
            "```\npip install python-binance\n```\n\n"
            "Sonra uygulamayı yeniden başlatın."
        )
        return

    snap = get_state().snapshot()

    # ── 2 kolon: sol=ayarlar, sağ=canlı durum ────────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    # ════════════════════════════════════════
    # SOL — Ayarlar
    # ════════════════════════════════════════
    with left:
        _section("⚙️", "Bot Ayarları")

        # API Keys (session_state'te sakla — JSON'a yazma)
        with st.expander("🔑 Binance API Anahtarları", expanded=not snap["running"]):
            api_key = st.text_input(
                "API Key",
                value=st.session_state.get("bot_api_key", ""),
                type="password",
                key="bot_api_key_input",
                placeholder="Binance API Key",
            )
            api_secret = st.text_input(
                "API Secret",
                value=st.session_state.get("bot_api_secret", ""),
                type="password",
                key="bot_api_secret_input",
                placeholder="Binance API Secret",
            )
            use_testnet = st.checkbox(
                "Testnet kullan (önerilen — gerçek para riski yok)",
                value=st.session_state.get("bot_testnet", True),
                key="bot_testnet_chk",
            )
            if st.button("💾 Kaydet", key="bot_save_keys", use_container_width=True):
                st.session_state["bot_api_key"]    = api_key
                st.session_state["bot_api_secret"] = api_secret
                st.session_state["bot_testnet"]    = use_testnet
                st.success("API bilgileri kaydedildi (sadece oturumda tutulur).")

        _section("📐", "Strateji")

        strategy = st.selectbox(
            "Strateji",
            list(STRATEGIES.keys()),
            format_func=lambda k: f"{k} — {STRATEGIES[k]}",
            key="bot_strategy",
            disabled=snap["running"],
        )

        sym_col, tf_col = st.columns(2)
        with sym_col:
            symbol = _render_coin_selector(snap["running"])
        with tf_col:
            timeframe = st.selectbox(
                "Zaman Dilimi",
                TIMEFRAMES,
                index=TIMEFRAMES.index("5m"),
                key="bot_tf",
                disabled=snap["running"],
            )

        # Strateji parametreleri
        with st.expander("🔧 Strateji Parametreleri", expanded=False):
            if strategy == "EMA_CROSS":
                c1, c2 = st.columns(2)
                with c1:
                    ema_fast = st.number_input("EMA Hızlı", 2, 50, 9,  key="bot_ema_fast",  disabled=snap["running"])
                    rsi_lo   = st.number_input("RSI Aşırı Sat", 10, 45, 30, key="bot_rsi_lo", disabled=snap["running"])
                with c2:
                    ema_slow = st.number_input("EMA Yavaş",  5, 200, 21, key="bot_ema_slow",  disabled=snap["running"])
                    rsi_hi   = st.number_input("RSI Aşırı Al", 55, 90, 70, key="bot_rsi_hi", disabled=snap["running"])
            elif strategy == "RSI_MEAN_REVERT":
                c1, c2 = st.columns(2)
                with c1:
                    rsi_lo   = st.number_input("RSI Aşırı Sat", 10, 45, 30, key="bot_rsi_lo2", disabled=snap["running"])
                    rsi_period = st.number_input("RSI Periyot", 5, 30, 14, key="bot_rsi_p", disabled=snap["running"])
                with c2:
                    rsi_hi   = st.number_input("RSI Aşırı Al", 55, 90, 70, key="bot_rsi_hi2", disabled=snap["running"])
            elif strategy == "BREAKOUT":
                bp       = st.number_input("Kırılım Periyodu", 5, 100, 20, key="bot_bp", disabled=snap["running"])

        _section("💰", "Risk Yönetimi")

        r1, r2 = st.columns(2)
        with r1:
            risk_pct  = st.number_input("Risk % (sermaye başına)", 0.1, 10.0, 1.0,
                                         step=0.1, key="bot_risk", disabled=snap["running"])
            tp_pct    = st.number_input("Take Profit %",           0.1, 20.0, 1.5,
                                         step=0.1, key="bot_tp",   disabled=snap["running"])
        with r2:
            leverage  = st.number_input("Kaldıraç (×)",  1.0, 20.0, 1.0,
                                         step=1.0, key="bot_lev",  disabled=snap["running"])
            sl_pct    = st.number_input("Stop Loss %",              0.1, 20.0, 1.0,
                                         step=0.1, key="bot_sl",   disabled=snap["running"])

        cooldown  = st.number_input(
            "Trade cooldown (saniye) — aynı yönde art arda işlem engeli",
            min_value=10, max_value=3600, value=60, step=10,
            key="bot_cooldown", disabled=snap["running"],
        )

        dry_run = st.checkbox(
            "🧪 Dry Run modu (gerçek order göndermez, simüle eder)",
            value=st.session_state.get("bot_dry_run", True),
            key="bot_dry_run_chk",
            disabled=snap["running"],
        )
        if not dry_run:
            st.warning(
                "⚠️ **Dry Run kapalı** — gerçek Binance siparişleri gönderilecek! "
                "Testnet kullanıyorsanız gerçek para riski yoktur."
            )

        # ── Başlat / Durdur ──────────────────────────────────────────────────
        st.markdown("")
        if not snap["running"]:
            if st.button("▶️ Botu Başlat", type="primary",
                         use_container_width=True, key="bot_start"):
                saved_key    = st.session_state.get("bot_api_key", "").strip()
                saved_secret = st.session_state.get("bot_api_secret", "").strip()
                if not saved_key or not saved_secret:
                    st.error("API Key ve Secret giriniz.")
                else:
                    cfg = _build_cfg(strategy, symbol, timeframe,
                                     risk_pct, tp_pct, sl_pct, leverage, cooldown,
                                     dry_run, use_testnet)
                    ok  = start_bot(saved_key, saved_secret, cfg)
                    if ok:
                        st.success("Bot başlatıldı!")
                        time.sleep(0.8)
                        st.rerun()
                    else:
                        st.error("Bot başlatılamadı.")
        else:
            if st.button("⏹️ Botu Durdur", use_container_width=True,
                         key="bot_stop"):
                stop_bot()
                time.sleep(0.5)
                st.rerun()

    # ════════════════════════════════════════
    # SAĞ — Canlı Durum
    # ════════════════════════════════════════
    with right:
        _section("📡", "Canlı Durum")
        _render_status_bar(snap)

        _section("📊", "Aktif Pozisyon")
        _render_open_trade(snap)

        _section("📋", "Günlük Log")
        _render_log(snap)

    # ── Alt: İşlem tablosu ────────────────────────────────────────────────────
    st.markdown("")
    _section("📁", "Bot İşlem Geçmişi")
    _render_trade_table(snap)

    # ── Auto-refresh: çalışıyorsa 3s'de bir yenile ───────────────────────────
    if snap["running"]:
        time.sleep(3)
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SUB-RENDERERS
# ══════════════════════════════════════════════════════════════════════════════

def _render_status_bar(snap: dict):
    running   = snap["running"]
    status_c  = _G if running else _DT
    dot       = "🟢" if running else "⚪"
    total_pnl = snap["total_pnl"]
    pnl_c     = _G if total_pnl >= 0 else _R
    wins      = snap["win_count"]
    losses    = snap["loss_count"]
    total_t   = wins + losses
    wr        = wins / total_t * 100 if total_t else 0

    st.markdown(f"""
    <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:0.5rem;margin-bottom:0.5rem'>
      <div style='background:{_DB};border:1px solid {_DG};border-radius:10px;padding:0.7rem;text-align:center'>
        <div style='font-size:12px;color:{_DT};margin-bottom:3px'>DURUM</div>
        <div style='font-size:15px;font-weight:700;color:{status_c}'>{dot} {snap["status"]}</div>
      </div>
      <div style='background:{_DB};border:1px solid {_DG};border-radius:10px;padding:0.7rem;text-align:center'>
        <div style='font-size:12px;color:{_DT};margin-bottom:3px'>TOPLAM PnL</div>
        <div style='font-family:"Space Mono",monospace;font-size:16px;font-weight:700;color:{pnl_c}'>
          {'+'if total_pnl>=0 else ''}{total_pnl:.4f}$</div>
      </div>
      <div style='background:{_DB};border:1px solid {_DG};border-radius:10px;padding:0.7rem;text-align:center'>
        <div style='font-size:12px;color:{_DT};margin-bottom:3px'>WIN RATE</div>
        <div style='font-size:16px;font-weight:700;color:{_G if wr>=50 else _R}'>
          {wr:.1f}% <span style='color:{_DT};font-size:13px'>({wins}W/{losses}L)</span></div>
      </div>
    </div>
    <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:0.5rem'>
      <div style='background:{_DB};border:1px solid {_DG};border-radius:10px;padding:0.7rem;text-align:center'>
        <div style='font-size:12px;color:{_DT};margin-bottom:3px'>SEMBOL</div>
        <div style='font-size:15px;font-weight:700;color:{_B}'>{snap["symbol"] or "—"}</div>
      </div>
      <div style='background:{_DB};border:1px solid {_DG};border-radius:10px;padding:0.7rem;text-align:center'>
        <div style='font-size:12px;color:{_DT};margin-bottom:3px'>SON FİYAT</div>
        <div style='font-family:"Space Mono",monospace;font-size:15px;font-weight:700;color:{_TX}'>
          ${snap["last_price"]:,.4f}</div>
      </div>
      <div style='background:{_DB};border:1px solid {_DG};border-radius:10px;padding:0.7rem;text-align:center'>
        <div style='font-size:12px;color:{_DT};margin-bottom:3px'>UNREALIZED</div>
        <div style='font-family:"Space Mono",monospace;font-size:15px;font-weight:700;
             color:{_G if snap["current_pnl"]>=0 else _R}'>
          {'+'if snap["current_pnl"]>=0 else ''}{snap["current_pnl"]:.4f}$</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if snap.get("error"):
        st.error(f"Hata: {snap['error']}")


def _render_open_trade(snap: dict):
    ot = snap.get("open_trade")
    if not ot:
        st.markdown(
            f"<div style='text-align:center;padding:1.5rem;color:{_DT};"
            f"border:1px dashed {_DG};border-radius:10px;font-size:14px'>"
            f"Açık pozisyon yok</div>",
            unsafe_allow_html=True,
        )
        return

    side_c  = _G if ot["side"] == "LONG" else _R
    side_bg = "#0a2e1a" if ot["side"] == "LONG" else "#2d0f0f"
    upnl    = snap["current_pnl"]
    upnl_c  = _G if upnl >= 0 else _R

    opened  = ot.get("opened_at", "")
    try:
        opened_str = datetime.fromisoformat(opened).strftime("%H:%M:%S")
    except Exception:
        opened_str = opened

    dry_badge = (
        f"<span style='background:#21262d;color:{_Y};padding:2px 7px;"
        f"border-radius:5px;font-size:11px;margin-left:6px'>DRY</span>"
        if ot.get("dry_run") else ""
    )

    st.markdown(f"""
    <div style='background:{_DB};border:1.5px solid {side_c}40;
                border-left:3px solid {side_c};border-radius:10px;padding:0.9rem'>
      <div style='display:flex;align-items:center;gap:0.6rem;margin-bottom:0.8rem'>
        <span style='background:{side_bg};color:{side_c};padding:4px 12px;
              border-radius:7px;font-size:14px;font-weight:700'>{ot["side"]}</span>
        <span style='font-size:13px;color:{_DT}'>{snap["symbol"]}</span>
        {dry_badge}
        <span style='margin-left:auto;font-size:13px;color:{_DT}'>{opened_str}</span>
      </div>
      <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:0.4rem'>
        <div class='detail-item'>
          <div class='detail-label'>Entry</div>
          <div class='detail-value'>${ot["entry"]:,.4f}</div>
        </div>
        <div class='detail-item'>
          <div class='detail-label'>TP</div>
          <div class='detail-value' style='color:{_G}'>${ot["tp"]:,.4f}</div>
        </div>
        <div class='detail-item'>
          <div class='detail-label'>SL</div>
          <div class='detail-value' style='color:{_R}'>${ot["sl"]:,.4f}</div>
        </div>
        <div class='detail-item'>
          <div class='detail-label'>Unr. PnL</div>
          <div class='detail-value' style='color:{upnl_c}'>
            {'+'if upnl>=0 else ''}{upnl:.4f}$</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def _render_log(snap: dict):
    logs = snap.get("log", [])
    if not logs:
        st.markdown(
            f"<div style='color:{_DT};font-size:13px;padding:0.5rem 0'>Log boş.</div>",
            unsafe_allow_html=True,
        )
        return

    rows = ""
    for entry in logs[:30]:
        lv  = entry["level"]
        clr = LOG_COLORS.get(lv, _DT)
        rows += (
            f"<div style='display:flex;gap:0.6rem;padding:3px 0;"
            f"border-bottom:1px solid {_DG}20;font-size:13px'>"
            f"<span style='color:{_DT};min-width:55px;font-family:\"Space Mono\",monospace'>"
            f"{entry['time']}</span>"
            f"<span style='color:{clr};min-width:42px;font-weight:700'>{lv}</span>"
            f"<span style='color:{_TX}'>{entry['msg']}</span>"
            f"</div>"
        )

    st.markdown(
        f"<div style='background:{_DB};border:1px solid {_DG};border-radius:10px;"
        f"padding:0.75rem 0.9rem;max-height:260px;overflow-y:auto'>{rows}</div>",
        unsafe_allow_html=True,
    )


def _render_trade_table(snap: dict):
    trades = snap.get("trades", [])
    if not trades:
        st.markdown(
            f"<div style='text-align:center;padding:2rem;color:{_DT};"
            f"border:1px dashed {_DG};border-radius:10px;font-size:15px'>"
            f"Henüz tamamlanan bot işlemi yok</div>",
            unsafe_allow_html=True,
        )
        return

    # Tablo başlığı
    st.markdown(
        f"<div style='display:grid;"
        f"grid-template-columns:60px 70px 110px 110px 110px 80px 70px 100px 130px;"
        f"padding:7px 12px;background:{_DB};border:1.5px solid {_DG};"
        f"border-radius:12px 12px 0 0;font-size:12px;font-weight:700;"
        f"color:{_DT};text-transform:uppercase;letter-spacing:0.07em'>"
        f"<div>#</div><div>Yön</div><div>Entry</div><div>Exit</div>"
        f"<div>PnL</div><div>Sebep</div><div>DRY</div><div>Açılış</div><div>Kapanış</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    for i, t in enumerate(trades[:50]):
        pnl    = t.get("pnl", 0)
        side   = t.get("side", "?")
        pnl_c  = _G if pnl >= 0 else _R
        dir_c  = _G if side == "LONG" else _R
        dir_bg = "#0a2e1a" if side == "LONG" else "#2d0f0f"
        reason = t.get("reason", "—")
        dry    = "✅" if t.get("dry_run") else "❌"

        try:
            opened = datetime.fromisoformat(t.get("opened_at","")).strftime("%m/%d %H:%M")
        except Exception:
            opened = "—"
        try:
            closed = datetime.fromisoformat(t.get("closed_at","")).strftime("%m/%d %H:%M")
        except Exception:
            closed = "—"

        row_bg = _BG if i % 2 == 0 else "#1a1f29"

        st.markdown(
            f"<div style='display:grid;"
            f"grid-template-columns:60px 70px 110px 110px 110px 80px 70px 100px 130px;"
            f"padding:9px 12px;background:{row_bg};border:1.5px solid {_DG};"
            f"border-top:none;align-items:center;font-size:14px'>"
            f"<div style='color:{_DT};font-family:\"Space Mono\",monospace'>{i+1}</div>"
            f"<div><span style='background:{dir_bg};color:{dir_c};"
            f"padding:3px 9px;border-radius:6px;font-size:12px;font-weight:700'>"
            f"{side}</span></div>"
            f"<div style='font-family:\"Space Mono\",monospace;color:{_TX}'>"
            f"${t.get('entry',0):,.4f}</div>"
            f"<div style='font-family:\"Space Mono\",monospace;color:{_TX}'>"
            f"${t.get('exit',0):,.4f}</div>"
            f"<div style='font-family:\"Space Mono\",monospace;font-weight:700;color:{pnl_c}'>"
            f"{'+'if pnl>=0 else ''}{pnl:.4f}$</div>"
            f"<div style='color:{_DT};font-size:13px'>{reason}</div>"
            f"<div style='font-size:13px'>{dry}</div>"
            f"<div style='color:{_DT};font-size:12px'>{opened}</div>"
            f"<div style='color:{_DT};font-size:12px'>{closed}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _render_coin_selector(disabled: bool) -> str:
    """
    Kategori dropdown + arama filtresi + Manuel giriş.
    Returns: seçilen USDT çift adı (örn. 'BTCUSDT')
    """
    # Mod seçimi: listeden seç | elle yaz
    mode = st.radio(
        "Coin seçim modu",
        ["📋 Listeden seç", "⌨️ Elle gir"],
        horizontal=True,
        key="bot_sym_mode",
        disabled=disabled,
        label_visibility="collapsed",
    )

    if mode == "⌨️ Elle gir":
        manual = st.text_input(
            "Sembol (USDT çifti)",
            value=st.session_state.get("bot_symbol_manual", "BTCUSDT"),
            key="bot_symbol_manual",
            disabled=disabled,
            placeholder="örn. BTCUSDT",
        ).upper().strip()
        symbol = manual if manual else "BTCUSDT"
        st.session_state["bot_symbol_final"] = symbol
        return symbol

    # ── Listeden seç ─────────────────────────────────────────────────────
    # 1) Arama kutusu
    search = st.text_input(
        "🔍 Coin ara",
        value="",
        key="bot_sym_search",
        disabled=disabled,
        placeholder="BTC, ETH, SOL...",
        label_visibility="collapsed",
    ).upper().strip()

    # 2) Kategori seçici
    cat_names = list(COINS.keys())
    sel_cat   = st.selectbox(
        "Kategori",
        ["🔎 Tümünde ara"] + cat_names,
        key="bot_sym_cat",
        disabled=disabled,
        label_visibility="collapsed",
    )

    # 3) Filtrelenmiş liste
    if sel_cat == "🔎 Tümünde ara":
        pool = _ALL_COINS
    else:
        pool = COINS[sel_cat]

    if search:
        pool = [c for c in pool if search in c]

    if not pool:
        st.caption("Eşleşen coin bulunamadı.")
        pool = _ALL_COINS[:5]

    # 4) Coin seçici
    prev_sym = st.session_state.get("bot_symbol_final", "BTCUSDT")
    default_idx = pool.index(prev_sym) if prev_sym in pool else 0

    selected = st.selectbox(
        "Coin",
        pool,
        index=default_idx,
        key=f"bot_sym_pick_{'_'.join(pool[:3])}",   # key arama değişince resetlensin
        disabled=disabled,
        label_visibility="collapsed",
    )
    st.session_state["bot_symbol_final"] = selected

    # Seçilen coini bilgi kutusunda göster
    base = selected.replace("USDT","")
    st.markdown(
        f"<div style='background:#0a1220;border:1px solid #1f6feb40;"
        f"border-radius:8px;padding:6px 10px;font-size:13px;"
        f"display:flex;align-items:center;gap:8px;margin-top:2px'>"
        f"<span style='font-family:\"Space Mono\",monospace;font-weight:700;"
        f"font-size:15px;color:#58a6ff'>{base}</span>"
        f"<span style='color:#6e7681'>/</span>"
        f"<span style='color:#6e7681'>USDT</span>"
        f"<span style='margin-left:auto;background:#21262d;color:#8b949e;"
        f"padding:2px 8px;border-radius:5px;font-size:12px'>FUTURES PERP</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    return selected


def _section(icon: str, title: str):
    st.markdown(
        f"<div style='font-family:\"Space Mono\",monospace;font-size:0.7rem;"
        f"letter-spacing:0.18em;text-transform:uppercase;color:#484f58;"
        f"margin:1.2rem 0 0.6rem;display:flex;align-items:center;gap:0.6rem'>"
        f"{icon} {title}"
        f"<span style='flex:1;height:1px;background:{_DG};display:block'></span></div>",
        unsafe_allow_html=True,
    )


def _build_cfg(strategy, symbol, timeframe, risk_pct, tp_pct,
               sl_pct, leverage, cooldown, dry_run, testnet) -> dict:
    cfg: dict = {
        "strategy":        strategy,
        "symbol":          symbol,
        "timeframe":       timeframe,
        "risk_pct":        risk_pct,
        "tp_pct":          tp_pct,
        "sl_pct":          sl_pct,
        "leverage":        int(leverage),
        "trade_cooldown_s": cooldown,
        "dry_run":         dry_run,
        "testnet":         testnet,
    }
    # Strateji özel parametreler
    if strategy == "EMA_CROSS":
        cfg["ema_fast"]       = st.session_state.get("bot_ema_fast", 9)
        cfg["ema_slow"]       = st.session_state.get("bot_ema_slow", 21)
        cfg["rsi_oversold"]   = st.session_state.get("bot_rsi_lo",   30)
        cfg["rsi_overbought"] = st.session_state.get("bot_rsi_hi",   70)
    elif strategy == "RSI_MEAN_REVERT":
        cfg["rsi_period"]     = st.session_state.get("bot_rsi_p",   14)
        cfg["rsi_oversold"]   = st.session_state.get("bot_rsi_lo2", 30)
        cfg["rsi_overbought"] = st.session_state.get("bot_rsi_hi2", 70)
    elif strategy == "BREAKOUT":
        cfg["breakout_period"] = st.session_state.get("bot_bp", 20)
    return cfg
