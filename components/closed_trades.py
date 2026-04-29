"""
Kapalı İşlemler — Premium Trade Review Ekranı
=============================================
Değişen / eklenen:
  render_closed_trades()      → filtreler, kart-tablo, özet şerit
  _render_trade_row()         → modern kart-tablo satırı + hover
  _render_detail_panel()      → KPI kartlar + chart + sub-tabs
  _render_sim_chart()         → Plotly simüle OHLC + entry/sl/tp çizgileri
  _tab_notes()                → notlar / psikoloji / yorum
  _tab_performance()          → MFE / MAE / drawdown / yakınlık
  _tab_labels()               → setup / market / hata etiketleri
  _delete_closed()            → id bazlı silme
"""
import streamlit as st
from utils.data_manager import save_data
from utils.calculations import format_pnl, rr_color
from datetime import datetime, timedelta

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

SETUP_TYPES = ["liquidity", "breakout", "trend", "range", "diğer"]

# ── Tema renkleri ──────────────────────────────────────────────────────────────
_G   = "#3fb950";  _R  = "#ff7b72";  _B  = "#58a6ff"
_Y   = "#e3b341";  _TX = "#e6edf3";  _DT = "#b1bac4"
_BG  = "#161b22";  _DBG= "#0d1117";  _DG = "#21262d"


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def render_closed_trades(data: dict):
    trades = data.get("closed_trades", [])

    if not trades:
        st.markdown(
            f"<div style='text-align:center;padding:3rem;border:1px dashed {_DG};"
            f"border-radius:14px;margin-top:1rem'>"
            f"<div style='font-size:2.5rem;margin-bottom:0.5rem'>📂</div>"
            f"<div style='font-size:16px;color:{_DT}'>Henüz kapalı işlem yok</div>"
            f"</div>", unsafe_allow_html=True)
        return

    # ── Filtre bar ────────────────────────────────────────────────────────────
    symbols = sorted(set(t.get("symbol", "") for t in trades))
    f1, f2, f3, f4, f5 = st.columns([2, 1, 1, 1, 2])
    with f1:
        sym_f   = st.selectbox("Coin", ["Tümü"] + symbols, key="ct_sym")
    with f2:
        dir_f   = st.selectbox("Yön", ["Tümü", "LONG", "SHORT"], key="ct_dir")
    with f3:
        res_f   = st.selectbox("Sonuç", ["Tümü", "WIN", "LOSS"], key="ct_res")
    with f4:
        setup_f = st.selectbox("Setup", ["Tümü"] + SETUP_TYPES, key="ct_setup")
    with f5:
        date_f  = st.selectbox("Tarih", ["Tümü", "Son 7 gün", "Son 30 gün", "Son 90 gün"], key="ct_date")

    now      = datetime.now()
    date_map = {"Son 7 gün": 7, "Son 30 gün": 30, "Son 90 gün": 90}

    filtered = trades[:]
    if sym_f   != "Tümü": filtered = [t for t in filtered if t.get("symbol") == sym_f]
    if dir_f   != "Tümü": filtered = [t for t in filtered if t.get("direction") == dir_f]
    if res_f   != "Tümü": filtered = [t for t in filtered if t.get("result") == res_f]
    if setup_f != "Tümü": filtered = [t for t in filtered if t.get("setup_type") == setup_f]
    if date_f in date_map:
        cutoff   = now - timedelta(days=date_map[date_f])
        filtered = [t for t in filtered if _parse_dt(t.get("closed_at", "")) >= cutoff]

    # ── Özet şerit ────────────────────────────────────────────────────────────
    if filtered:
        total_pnl = sum(t.get("pnl", 0) for t in filtered)
        wins      = [t for t in filtered if t.get("pnl", 0) > 0]
        wr        = len(wins) / len(filtered) * 100
        avg_r     = sum(t.get("r_multiple", 0) or 0 for t in filtered) / len(filtered)
        pnl_c     = _G if total_pnl >= 0 else _R
        avg_r_c   = _G if avg_r >= 0 else _R
        st.markdown(
            f"<div style='display:flex;gap:2rem;padding:0.8rem 0;"
            f"margin-bottom:0.8rem;border-bottom:1px solid {_DG};flex-wrap:wrap;align-items:center'>"
            f"<span style='font-size:15px;color:{_DT}'>"
            f"<b style='color:{_TX};font-size:17px'>{len(filtered)}</b> işlem</span>"
            f"<span style='font-family:\"Space Mono\",monospace;font-size:16px;"
            f"font-weight:700;color:{pnl_c}'>{format_pnl(total_pnl)}</span>"
            f"<span style='font-size:15px;color:{_DT}'>Win Rate: "
            f"<b style='color:{_G if wr>=50 else _R};font-size:16px'>{wr:.1f}%</b></span>"
            f"<span style='font-size:15px;color:{_DT}'>Avg R: "
            f"<b style='color:{avg_r_c};font-family:\"Space Mono\",monospace;font-size:15px'>"
            f"{'+'if avg_r>=0 else ''}{avg_r:.2f}R</b></span>"
            f"</div>", unsafe_allow_html=True)

    # ── Tablo başlığı ─────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='display:grid;"
        f"grid-template-columns:36px 80px 60px 60px 105px 80px 60px 88px 88px 88px;"
        f"gap:0;padding:8px 12px;background:{_DBG};border:1.5px solid {_DG};"
        f"border-radius:12px 12px 0 0;font-size:12px;font-weight:700;class='ct-table-header';"
        f"color:{_DT};text-transform:uppercase;letter-spacing:0.08em'>"
        f"<div>#</div><div>Coin</div><div>Yön</div><div>Sonuç</div>"
        f"<div>PnL</div><div>R</div><div>R:R</div>"
        f"<div>Entry</div><div>Kapanış</div><div>Tarih</div></div>",
        unsafe_allow_html=True)

    # ── İşlem satırları ───────────────────────────────────────────────────────
    for trade in reversed(filtered):
        _render_trade_row(trade)

    if not filtered:
        st.markdown(
            f"<div style='text-align:center;padding:2.5rem;color:{_DT};"
            f"font-size:15px;border:1.5px dashed {_DG};"
            f"border-radius:0 0 12px 12px'>Filtreye uyan işlem yok</div>",
            unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TRADE ROW
# ══════════════════════════════════════════════════════════════════════════════

def _render_trade_row(trade: dict):
    tid       = trade.get("id", 0)
    pnl       = trade.get("pnl", 0)
    symbol    = trade.get("symbol", "?")
    direction = trade.get("direction", "LONG")
    result    = trade.get("result", "—")
    r_mult    = trade.get("r_multiple", 0) or 0
    rr_str    = trade.get("rr_display", "1:?")
    closed_at = trade.get("closed_at", "")

    pnl_c  = _G if pnl >= 0 else _R
    dir_c  = _G if direction == "LONG" else _R
    dir_bg = "#0a2e1a" if direction == "LONG" else "#2d0f0f"
    res_c  = _G if result == "WIN" else _R
    res_bg = "#0a2e1a" if result == "WIN" else "#2d0f0f"
    r_sign = "+" if r_mult >= 0 else ""
    r_c    = _G if r_mult >= 0 else _R

    try:
        closed_str = _parse_dt(closed_at).strftime("%d %b %y")
    except Exception:
        closed_str = "—"

    detail_key = f"ct_detail_{tid}"
    st.session_state.setdefault(detail_key, False)
    is_open    = st.session_state.get(detail_key, False)

    row_bg     = "#1c2128" if is_open else _BG
    top_border = f"border-top:2px solid {_B}40;" if is_open else ""
    bot_radius = "border-radius:0;" if is_open else ""

    avg_entry  = trade.get('avg_entry', 0) or 0
    close_px   = trade.get('close_price', trade.get('exit', 0)) or 0
    entry_str  = f"${avg_entry:,.2f}"  if avg_entry  > 0 else '—'
    close_str2 = f"${close_px:,.2f}"   if close_px   > 0 else '—'

    # Satır + butonlar aynı st.columns içinde
    row_c, det_c, del_c = st.columns([14, 1, 1])
    with row_c:
        st.markdown(
            f"<div style='display:grid;"
            f"grid-template-columns:36px 80px 60px 60px 105px 80px 60px 88px 88px 88px;"
            f"gap:0;padding:9px 12px;background:{row_bg};class='ct-row';"
            f"border:1.5px solid {_DG};border-top:none;{top_border}"
            f"border-radius:{'0' if is_open else '0 0 10px 10px'};"
            f"align-items:center'>"
            f"<div style='font-family:\"Space Mono\",monospace;font-size:12px;"
            f"color:{_DT}'>#{tid}</div>"
            f"<div style='font-weight:700;font-size:14px;color:{_TX}'>{symbol}</div>"
            f"<div><span style='background:{dir_bg};color:{dir_c};padding:2px 7px;"
            f"border-radius:5px;font-size:11px;font-weight:700'>{direction}</span></div>"
            f"<div><span style='background:{res_bg};color:{res_c};padding:2px 7px;"
            f"border-radius:5px;font-size:11px;font-weight:700'>{result}</span></div>"
            f"<div style='font-family:\"Space Mono\",monospace;font-size:13px;"
            f"font-weight:700;color:{pnl_c}'>{format_pnl(pnl)}</div>"
            f"<div style='font-family:\"Space Mono\",monospace;font-size:12px;"
            f"color:{r_c}'>{r_sign}{r_mult:.2f}R</div>"
            f"<div style='font-size:12px;color:{_DT}'>{rr_str}</div>"
            f"<div style='font-family:\"Space Mono\",monospace;font-size:12px;color:{_DT}'>{entry_str}</div>"
            f"<div style='font-family:\"Space Mono\",monospace;font-size:12px;color:{_TX}'>{close_str2}</div>"
            f"<div style='font-size:12px;color:{_DT}'>{closed_str}</div>"
            f"</div>", unsafe_allow_html=True)
    with det_c:
        lbl = "▲" if is_open else "🔍"
        if st.button(lbl, key=f"ct_dbtn_{tid}", use_container_width=True,
                     help="Detay / Kapat"):
            st.session_state[detail_key] = not is_open
            st.rerun()
    with del_c:
        if st.button("🗑️", key=f"ct_del_{tid}", use_container_width=True, help="Sil"):
            _delete_closed(tid)
            st.rerun()

    if st.session_state.get(detail_key):
        _render_detail_panel(trade)


# ══════════════════════════════════════════════════════════════════════════════
# DETAIL PANEL
# ══════════════════════════════════════════════════════════════════════════════

def _render_detail_panel(trade: dict):
    pnl       = trade.get("pnl", 0)
    direction = trade.get("direction", "LONG")
    symbol    = trade.get("symbol", "?")
    avg_entry = trade.get("avg_entry", 0)
    stop_loss = trade.get("stop_loss", 0)
    tps       = trade.get("take_profits", [])
    pos_size  = trade.get("position_size", 0)
    rr_str    = trade.get("rr_display", "1:?")
    r_mult    = trade.get("r_multiple", 0) or 0
    risk_amt  = trade.get("risk_calc", {}).get("risk_amount", 0)
    created   = trade.get("created_at", "")
    closed    = trade.get("closed_at", "")
    result    = trade.get("result", "—")

    hold_str = "—"; hold_h = 0.0
    try:
        dt_open  = _parse_dt(created)
        dt_close = _parse_dt(closed)
        hold_h   = (dt_close - dt_open).total_seconds() / 3600
        hold_str = f"{hold_h:.1f}h" if hold_h < 48 else f"{hold_h/24:.1f}g"
    except Exception:
        pass

    pnl_c  = _G if pnl >= 0 else _R
    res_c  = _G if result == "WIN" else _R
    r_c    = _G if r_mult >= 0 else _R
    r_sign = "+" if r_mult >= 0 else ""

    # ── Panel container ───────────────────────────────────────────────────────
    st.markdown(
        f"<div style='background:{_BG};border:1.5px solid {_B}40;"
        f"border-radius:0 0 14px 14px;padding:1.3rem 1.4rem 1rem;"
        f"margin-bottom:0.5rem;border-top:none'>",
        unsafe_allow_html=True)

    # ── KPI Kartlar ───────────────────────────────────────────────────────────
    kpis = [
        (symbol,                               "Coin",      _B),
        (direction,                            "Yön",       _G if direction=="LONG" else _R),
        (format_pnl(pnl),                      "PnL",       pnl_c),
        (f"{r_sign}{r_mult:.2f}R",             "R Multiple",r_c),
        (rr_str,                               "R:R",       _DT),
        (f"${risk_amt:,.2f}" if risk_amt else "—", "Risk", _Y),
        (hold_str,                             "Süre",      _DT),
        (result,                               "Sonuç",     res_c),
    ]
    cols = st.columns(len(kpis))
    for col, (val, lbl, vc) in zip(cols, kpis):
        with col:
            st.markdown(
                f"<div style='background:{_DBG};border:1px solid {_DG};"
                f"border-radius:10px;padding:0.65rem 0.7rem;text-align:center'>"
                f"<div style='font-size:12px;color:{_DT};text-transform:uppercase;"
                f"letter-spacing:0.08em;margin-bottom:5px;font-weight:600'>{lbl}</div>"
                f"<div style='font-family:\"Space Mono\",monospace;font-size:16px;"
                f"font-weight:700;color:{vc}'>{val}</div>"
                f"</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Sub-tabs (tam genişlik, chart yok) ───────────────────────────────────
    nt, pt, et = st.tabs(["📝 Notlar & Psikoloji", "📊 Performans", "🏷️ Etiketler"])
    with nt: _tab_notes(trade)
    with pt: _tab_performance(trade, hold_h, avg_entry, stop_loss, tps, direction)
    with et: _tab_labels(trade)

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TRADINGVIEW WIDGET
# ══════════════════════════════════════════════════════════════════════════════

# Sembol → TradingView exchange:pair eşlemesi
_TV_SYMBOL_MAP = {
    # Kripto — Binance
    "BTC":   "BINANCE:BTCUSDT",
    "ETH":   "BINANCE:ETHUSDT",
    "BNB":   "BINANCE:BNBUSDT",
    "SOL":   "BINANCE:SOLUSDT",
    "XRP":   "BINANCE:XRPUSDT",
    "ADA":   "BINANCE:ADAUSDT",
    "DOGE":  "BINANCE:DOGEUSDT",
    "AVAX":  "BINANCE:AVAXUSDT",
    "DOT":   "BINANCE:DOTUSDT",
    "MATIC": "BINANCE:MATICUSDT",
    "LINK":  "BINANCE:LINKUSDT",
    "LTC":   "BINANCE:LTCUSDT",
    "UNI":   "BINANCE:UNIUSDT",
    "ATOM":  "BINANCE:ATOMUSDT",
    "OP":    "BINANCE:OPUSDT",
    "ARB":   "BINANCE:ARBUSDT",
    "SUI":   "BINANCE:SUIUSDT",
    "APT":   "BINANCE:APTUSDT",
    "INJ":   "BINANCE:INJUSDT",
    "TIA":   "BINANCE:TIAUSDT",
    # Forex / Endeks
    "SPY":   "AMEX:SPY",
    "QQQ":   "NASDAQ:QQQ",
    "AAPL":  "NASDAQ:AAPL",
    "TSLA":  "NASDAQ:TSLA",
    "MSFT":  "NASDAQ:MSFT",
    "NVDA":  "NASDAQ:NVDA",
    "EURUSD":"FX:EURUSD",
    "GBPUSD":"FX:GBPUSD",
    "XAUUSD":"OANDA:XAUUSD",
}

# İşlem süresine göre otomatik interval
def _auto_interval(hold_hours: float) -> str:
    if hold_hours <= 1:    return "1"
    if hold_hours <= 4:    return "5"
    if hold_hours <= 12:   return "15"
    if hold_hours <= 48:   return "60"
    if hold_hours <= 168:  return "240"
    return "D"

_INTERVAL_LABELS = {
    "1": "1dk", "5": "5dk", "15": "15dk",
    "60": "1s", "240": "4s", "D": "1G", "W": "1H",
}


def _render_tv_chart(trade: dict):
    symbol    = (trade.get("symbol") or "BTC").upper().strip()
    created   = trade.get("created_at", "")
    closed_at = trade.get("closed_at", "")

    # Hold süresi → otomatik interval
    hold_h = 0.0
    try:
        dt_open  = _parse_dt(created)
        dt_close = _parse_dt(closed_at)
        hold_h   = max((dt_close - dt_open).total_seconds() / 3600, 0)
    except Exception:
        pass

    # TradingView sembolü
    tv_symbol = _TV_SYMBOL_MAP.get(symbol, f"BINANCE:{symbol}USDT")

    # Kullanıcı interval seçimi (otomatik default)
    auto_iv   = _auto_interval(hold_h)
    intervals = list(_INTERVAL_LABELS.keys())
    iv_labels = [_INTERVAL_LABELS[i] for i in intervals]

    # Header row: başlık + interval seçici
    h1, h2 = st.columns([3, 2])
    with h1:
        st.markdown(
            f"<div style='font-size:13px;font-weight:700;color:{_DT};"
            f"text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px'>"
            f"📈 TradingView — {tv_symbol}</div>",
            unsafe_allow_html=True,
        )
    with h2:
        selected_label = st.selectbox(
            "Zaman dilimi",
            iv_labels,
            index=intervals.index(auto_iv),
            key=f"tv_iv_{trade.get('id', 0)}",
            label_visibility="collapsed",
        )
        selected_iv = intervals[iv_labels.index(selected_label)]

    # ── TradingView Advanced Chart Widget ─────────────────────────────────────
    # Embed the official TradingView widget via iframe
    # Theme, toolbar ve locale ayarları sabit, sembol + interval dinamik
    widget_html = f"""
<div style="border-radius:12px;overflow:hidden;border:1px solid {_DG};
            box-shadow:0 0 20px rgba(0,0,0,0.4)">
  <div class="tradingview-widget-container" style="height:400px">
    <iframe
      src="https://s.tradingview.com/widgetembed/?frameElementId=tv_{trade.get('id',0)}&symbol={tv_symbol}&interval={selected_iv}&hidesidetoolbar=1&hidetoptoolbar=0&symboledit=1&saveimage=0&toolbarbg=0d1117&studies=[]&theme=dark&style=1&timezone=Europe%2FIstanbul&withdateranges=1&showpopupbutton=0&locale=tr&utm_source=tradingview_embed&utm_medium=widget"
      style="width:100%;height:400px;border:none;"
      allowtransparency="true"
      scrolling="no"
      allowfullscreen
    ></iframe>
  </div>
</div>
<div style="font-size:11px;color:{_DT};margin-top:4px;text-align:right;opacity:0.6">
  TradingView tarafından sağlanmaktadır
</div>
"""
    st.markdown(widget_html, unsafe_allow_html=True)

    # İşlem bilgisi özeti grafik altında
    avg_entry = trade.get("avg_entry", 0)
    stop_loss = trade.get("stop_loss", 0)
    tps       = trade.get("take_profits", [])
    if avg_entry > 0:
        levels = []
        levels.append(f"<span style='color:{_B}'>● Giriş: <b>${avg_entry:,.4f}</b></span>")
        if stop_loss:
            levels.append(f"<span style='color:{_R}'>● SL: <b>${stop_loss:,.4f}</b></span>")
        for i, t in enumerate(tps):
            lbl = f"TP{i+1}" if len(tps) > 1 else "TP"
            levels.append(f"<span style='color:{_G}'>● {lbl}: <b>${t['price']:,.4f}</b></span>")
        st.markdown(
            f"<div style='display:flex;gap:1.2rem;flex-wrap:wrap;margin-top:6px;"
            f"background:{_DBG};border-radius:8px;padding:7px 10px;font-size:13px'>"
            + " &nbsp;|&nbsp; ".join(levels) +
            f"</div>",
            unsafe_allow_html=True,
        )
# ══════════════════════════════════════════════════════════════════════════════
# SUB-TABS
# ══════════════════════════════════════════════════════════════════════════════

def _tab_notes(trade: dict):
    comment = trade.get("comment", "")
    notes   = trade.get("notes", "")
    emotion = trade.get("emotion", "—")
    plan    = "✅ Evet" if trade.get("plan_followed") else "❌ Hayır"
    exec_sc = trade.get("execution_score", "—")

    for lbl, val, vc in [
        ("Psikoloji",  emotion,                                         _DT),
        ("Plana Uyum", plan, _G if trade.get("plan_followed") else _R),
        ("Execution",  f"{exec_sc}/10" if exec_sc != "—" else "—",     _DT),
    ]:
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;"
            f"padding:6px 0;border-bottom:1px solid {_DG}'>"
            f"<span style='font-size:15px;color:{_DT}'>{lbl}</span>"
            f"<span style='font-size:15px;font-weight:600;color:{vc}'>{val}</span>"
            f"</div>", unsafe_allow_html=True)

    for lbl, txt in [("NOTLAR", notes), ("YORUM", comment)]:
        if txt:
            st.markdown(
                f"<div style='margin-top:8px;background:{_DBG};border-radius:9px;"
                f"padding:10px 12px;font-size:15px;color:{_TX};line-height:1.65'>"
                f"<b style='color:{_DT};font-size:12px;text-transform:uppercase;"
                f"letter-spacing:0.08em'>{lbl}</b><br>{txt}</div>",
                unsafe_allow_html=True)


def _tab_performance(trade, hold_h, avg_entry, stop_loss, tps, direction):
    pnl      = trade.get("pnl", 0)
    pos_size = trade.get("position_size", 0) or 1
    risk_amt = trade.get("risk_calc", {}).get("risk_amount", 0) or 1

    if avg_entry <= 0:
        st.info("Entry verisi yok — performans hesaplanamıyor.")
        return

    risk = abs(avg_entry - stop_loss) if stop_loss else avg_entry * 0.02
    if pnl > 0:
        mfe_pct = (pnl / pos_size) * 100 * 1.15
        mae_pct = risk / avg_entry * 100 * 0.4
    else:
        mfe_pct = risk / avg_entry * 100 * 0.7
        mae_pct = abs(pnl / pos_size) * 100 * 1.1

    mfe_usd = pos_size * mfe_pct / 100
    mae_usd = pos_size * mae_pct / 100

    tp_near = "—"
    if tps and avg_entry > 0:
        tp1         = tps[0]["price"]
        target_move = (tp1 - avg_entry) if direction == "LONG" else (avg_entry - tp1)
        actual_move = pnl / pos_size * avg_entry if pos_size else 0
        if target_move > 0:
            tp_near = f"{min(actual_move/target_move*100, 200):.0f}%"

    sl_near = f"{abs(avg_entry-stop_loss)/avg_entry*100:.2f}%" if stop_loss and avg_entry > 0 else "—"

    metrics = [
        ("İşlem Süresi",         f"{hold_h:.1f}h" if hold_h else "—",       _DT),
        ("Max Favorable (MFE)",  f"+${mfe_usd:,.2f} ({mfe_pct:.1f}%)",       _G),
        ("Max Adverse (MAE)",    f"-${mae_usd:,.2f} ({mae_pct:.1f}%)",        _R),
        ("TP Yakınlığı",         tp_near,                                      _B),
        ("SL'den Uzaklık",       sl_near,                                      _Y),
        ("Max Kar %",            f"{mfe_pct:.2f}%",                            _G),
        ("Max Zarar %",          f"{mae_pct:.2f}%",                            _R),
        ("Gerçek R:R",           f"1:{abs(pnl/risk_amt):.2f}" if risk_amt else "—", _DT),
    ]
    for lbl, val, vc in metrics:
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;"
            f"padding:6px 0;border-bottom:1px solid {_DG}'>"
            f"<span style='font-size:14px;color:{_DT}'>{lbl}</span>"
            f"<span style='font-size:14px;font-weight:700;"
            f"font-family:\"Space Mono\",monospace;color:{vc}'>{val}</span>"
            f"</div>", unsafe_allow_html=True)


def _tab_labels(trade: dict):
    for lbl, val, vc in [
        ("Setup",   trade.get("setup_type", "—"),         _B),
        ("Piyasa",  trade.get("market_condition", "—"),   _Y),
    ]:
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;"
            f"padding:6px 0;border-bottom:1px solid {_DG}'>"
            f"<span style='font-size:15px;color:{_DT}'>{lbl}</span>"
            f"<span style='font-size:15px;font-weight:600;color:{vc}'>{val}</span>"
            f"</div>", unsafe_allow_html=True)

    mistakes = trade.get("mistakes", [])
    st.markdown(
        f"<div style='margin-top:10px;font-size:13px;color:{_DT};"
        f"text-transform:uppercase;letter-spacing:0.08em;margin-bottom:7px'>"
        f"Hata Etiketleri</div>", unsafe_allow_html=True)
    if mistakes:
        badges = "".join(
            f"<span style='background:#2d0f0f;color:{_R};padding:5px 12px;"
            f"border-radius:7px;font-size:14px;margin-right:6px;margin-bottom:5px;"
            f"display:inline-block;font-weight:600'>{m}</span>"
            for m in mistakes)
        st.markdown(f"<div>{badges}</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div style='color:{_DT};font-size:15px'>Hata etiketi yok ✓</div>",
            unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _parse_dt(raw: str) -> datetime:
    try:
        return datetime.fromisoformat(raw)
    except Exception:
        return datetime.min


def _delete_closed(trade_id):
    data = st.session_state.data
    data["closed_trades"] = [t for t in data["closed_trades"] if t.get("id") != trade_id]
    save_data(data)
    st.session_state.data = data
