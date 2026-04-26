"""
Binance OHLC Chart — Gerçek piyasa verisi + Entry/SL/TP/Exit işaretleri
=======================================================================
Binance public API'den (auth gerektirmez) gerçek kline verisi çeker.
İşlem tarih aralığını ± padding ile kapsayacak şekilde mum sayısını ayarlar.
Üzerine Plotly ile Entry, Stop Loss, Take Profit ve Exit çizgileri çizer.

Entegrasyon:
    from components.chart_binance import render_binance_chart
    render_binance_chart(trade)

Bağımlılık:
    pip install requests plotly   (requirements.txt'e eklenmiştir)
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import Optional

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ── Tema ─────────────────────────────────────────────────────────────────────
_G   = "#3fb950";  _R  = "#ff7b72";  _B  = "#58a6ff"
_Y   = "#e3b341";  _TX = "#e6edf3";  _DT = "#b1bac4"
_BG  = "#161b22";  _DBG= "#0d1117";  _DG = "#21262d"

# ── Binance sembol eşlemesi ───────────────────────────────────────────────────
_BINANCE_PAIR = {
    "BTC":"BTCUSDT","ETH":"ETHUSDT","BNB":"BNBUSDT","SOL":"SOLUSDT",
    "XRP":"XRPUSDT","ADA":"ADAUSDT","DOGE":"DOGEUSDT","AVAX":"AVAXUSDT",
    "DOT":"DOTUSDT","MATIC":"MATICUSDT","LINK":"LINKUSDT","LTC":"LTCUSDT",
    "UNI":"UNIUSDT","ATOM":"ATOMUSDT","OP":"OPUSDT","ARB":"ARBUSDT",
    "SUI":"SUIUSDT","APT":"APTUSDT","INJ":"INJUSDT","TIA":"TIAUSDT",
    "NEAR":"NEARUSDT","FIL":"FILUSDT","AAVE":"AAVEUSDT","MKR":"MKRUSDT",
    "SNX":"SNXUSDT","CRV":"CRVUSDT","PEPE":"PEPEUSDT","WIF":"WIFUSDT",
    "JUP":"JUPUSDT","STRK":"STRKUSDT",
}

# ── İşlem süresine göre otomatik interval ────────────────────────────────────
def _best_interval(hold_hours: float) -> tuple[str, int]:
    """(binance_interval_str, minutes_per_candle)"""
    if hold_hours <= 0.5:  return "1m",   1
    if hold_hours <= 2:    return "3m",   3
    if hold_hours <= 6:    return "5m",   5
    if hold_hours <= 16:   return "15m",  15
    if hold_hours <= 48:   return "1h",   60
    if hold_hours <= 200:  return "4h",   240
    return "1d", 1440

_INTERVAL_OPTIONS = {
    "1m":"1dk","3m":"3dk","5m":"5dk","15m":"15dk",
    "30m":"30dk","1h":"1s","2h":"2s","4h":"4s","1d":"1G",
}
_INTERVAL_MINUTES = {
    "1m":1,"3m":3,"5m":5,"15m":15,"30m":30,
    "1h":60,"2h":120,"4h":240,"1d":1440,
}


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC
# ══════════════════════════════════════════════════════════════════════════════

def render_binance_chart(trade: dict):
    """Ana entry point. closed_trades detay panelinden çağrılır."""
    if not HAS_PLOTLY:
        st.warning("Plotly yüklü değil: `pip install plotly`")
        return
    if not HAS_REQUESTS:
        st.warning("Requests yüklü değil: `pip install requests`")
        return

    symbol    = (trade.get("symbol") or "BTC").upper().strip()
    created   = trade.get("created_at", "")
    closed_at = trade.get("closed_at", "")
    avg_entry = float(trade.get("avg_entry") or 0)
    stop_loss = float(trade.get("stop_loss") or 0)
    tps       = trade.get("take_profits") or []
    pnl       = float(trade.get("pnl") or 0)
    pos_size  = float(trade.get("position_size") or 1) or 1
    direction = trade.get("direction", "LONG")
    tid       = trade.get("id", 0)

    # Tarihleri parse et
    dt_open  = _parse_dt(created)
    dt_close = _parse_dt(closed_at)
    if dt_open == datetime.min:
        st.info("İşlem tarihi girilmemiş — grafik gösterilemiyor.")
        return

    hold_h = max((dt_close - dt_open).total_seconds() / 3600, 0.25)

    # ── Interval seçici ──────────────────────────────────────────────────────
    auto_iv, _  = _best_interval(hold_h)
    iv_keys     = list(_INTERVAL_OPTIONS.keys())
    iv_labels   = [_INTERVAL_OPTIONS[k] for k in iv_keys]
    auto_idx    = iv_keys.index(auto_iv) if auto_iv in iv_keys else 2

    h1, h2 = st.columns([4, 1])
    with h1:
        binance_pair = _BINANCE_PAIR.get(symbol, f"{symbol}USDT")
        st.markdown(
            f"<div style='font-size:13px;font-weight:700;color:{_DT};"
            f"text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px'>"
            f"📈 {binance_pair} — Gerçek Piyasa Verisi</div>",
            unsafe_allow_html=True,
        )
    with h2:
        sel_label = st.selectbox(
            "interval", iv_labels, index=auto_idx,
            key=f"bc_iv_{tid}", label_visibility="collapsed",
        )
        selected_iv = iv_keys[iv_labels.index(sel_label)]

    iv_min = _INTERVAL_MINUTES[selected_iv]

    # ── Veri çek ─────────────────────────────────────────────────────────────
    with st.spinner("Piyasa verisi yükleniyor..."):
        candles = _fetch_klines(symbol, selected_iv, dt_open, dt_close, iv_min)

    if candles is None:
        # Hata mesajı zaten gösterildi
        return
    if len(candles) < 3:
        st.warning(
            f"⚠️ Bu tarih aralığında yeterli veri bulunamadı "
            f"({binance_pair} · {selected_iv}). "
            f"İşlem tarihi çok eski veya sembol o dönemde mevcut olmayabilir."
        )
        _render_level_summary(avg_entry, stop_loss, tps, direction)
        return

    # ── Grafik çiz ────────────────────────────────────────────────────────────
    _render_chart(
        candles, dt_open, dt_close,
        avg_entry, stop_loss, tps, pnl, pos_size, direction, tid,
    )

    # ── Seviye özeti (grafik altı) ────────────────────────────────────────────
    _render_level_summary(avg_entry, stop_loss, tps, direction)


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHER
# ══════════════════════════════════════════════════════════════════════════════

_BINANCE_ENDPOINTS = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://data-api.binance.vision",
]


def _fetch_klines(
    symbol: str,
    interval: str,
    dt_open: datetime,
    dt_close: datetime,
    iv_min: int,
) -> Optional[list]:
    """
    Binance /api/v3/klines endpoint'inden gerçek OHLC verisi çeker.
    İşlem öncesi 30 mum + işlem süresi + işlem sonrası 30 mum = tam görüntü.
    Birden fazla endpoint dener (failover).
    """
    pair      = _BINANCE_PAIR.get(symbol, f"{symbol}USDT")
    padding   = timedelta(minutes=iv_min * 30)
    start_ms  = int((dt_open  - padding).timestamp() * 1000)
    end_ms    = int((dt_close + padding).timestamp() * 1000)
    limit     = min(int((end_ms - start_ms) / (iv_min * 60000)) + 10, 1000)

    params = {
        "symbol":    pair,
        "interval":  interval,
        "startTime": start_ms,
        "endTime":   end_ms,
        "limit":     limit,
    }

    last_error = ""
    for base in _BINANCE_ENDPOINTS:
        try:
            resp = requests.get(
                f"{base}/api/v3/klines",
                params=params,
                timeout=8,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 Chrome/120.0"
                    )
                },
            )
            if resp.status_code == 200:
                return resp.json()
            last_error = f"HTTP {resp.status_code}"
        except requests.exceptions.Timeout:
            last_error = "Bağlantı zaman aşımı"
        except requests.exceptions.ConnectionError:
            last_error = "Bağlantı hatası"
        except Exception as e:
            last_error = str(e)

    st.error(
        f"❌ Binance verisi alınamadı: {last_error}\n\n"
        f"İnternet bağlantısını kontrol edin veya "
        f"sembol/tarih bilgilerinin doğru olduğundan emin olun."
    )
    return None


# ══════════════════════════════════════════════════════════════════════════════
# CHART RENDERER
# ══════════════════════════════════════════════════════════════════════════════

def _render_chart(
    candles, dt_open, dt_close,
    avg_entry, stop_loss, tps, pnl, pos_size, direction, tid,
):
    dates  = [datetime.fromtimestamp(c[0] / 1000) for c in candles]
    opens  = [float(c[1]) for c in candles]
    highs  = [float(c[2]) for c in candles]
    lows   = [float(c[3]) for c in candles]
    closes = [float(c[4]) for c in candles]

    fig = go.Figure()

    # ── RR shading ────────────────────────────────────────────────────────────
    tp1_price = tps[0]["price"] if tps else None
    if avg_entry > 0 and stop_loss > 0 and tp1_price:
        if direction == "LONG":
            y_low, y_high      = stop_loss, tp1_price
            shade_color        = "rgba(63,185,80,0.09)"
            shade_line         = "rgba(63,185,80,0.3)"
        else:
            y_low, y_high      = tp1_price, stop_loss
            shade_color        = "rgba(255,123,114,0.09)"
            shade_line         = "rgba(255,123,114,0.3)"

        if y_low != y_high:
            fig.add_hrect(
                y0=min(y_low, y_high), y1=max(y_low, y_high),
                fillcolor=shade_color,
                line_color=shade_line, line_width=1,
            )

    # ── Mumlar ────────────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=dates,
        open=opens, high=highs, low=lows, close=closes,
        increasing_line_color="#3fb950", decreasing_line_color="#ff7b72",
        increasing_fillcolor="#26a641",  decreasing_fillcolor="#da3633",
        line_width=1,
        name="Fiyat",
        hoverinfo="x+y",
    ))

    # ── Yatay seviye çizgileri ────────────────────────────────────────────────
    def _hline(y, color, dash, label, width=1.8):
        fig.add_hline(
            y=y,
            line_dash=dash,
            line_color=color,
            line_width=width,
            annotation_text=f"  {label}",
            annotation_position="right",
            annotation_font=dict(size=12, color=color, family="DM Sans"),
        )

    if avg_entry > 0:
        _hline(avg_entry, _B, "solid", f"ENTRY  ${avg_entry:,.2f}", width=2)
    if stop_loss > 0:
        _hline(stop_loss, _R, "dot",   f"SL  ${stop_loss:,.2f}")
    for i, t in enumerate(tps):
        lbl = f"TP{i+1}  ${t['price']:,.2f}" if len(tps) > 1 else f"TP  ${t['price']:,.2f}"
        _hline(t["price"], _G, "dash", lbl)

    # ── Exit fiyatını PnL'den hesapla ─────────────────────────────────────────
    exit_price = None
    if avg_entry > 0 and pos_size > 0:
        price_move = (pnl / pos_size) * avg_entry
        exit_price = (avg_entry + price_move
                      if direction == "LONG"
                      else avg_entry - price_move)
        exit_color = _G if pnl >= 0 else _R
        _hline(exit_price, exit_color, "dashdot",
               f"EXIT  ${exit_price:,.2f}", width=1.5)

    # ── Dikey giriş / çıkış bantları ─────────────────────────────────────────
    # Giriş: mavi dikey çizgi + ENTRY etiket
    fig.add_vline(
        x=dt_open,
        line_dash="dot", line_color=_B, line_width=2,
        annotation_text="  ENTRY",
        annotation_position="top left",
        annotation_font=dict(size=12, color=_B, family="DM Sans"),
    )
    # Çıkış: yeşil/kırmızı dikey çizgi + EXIT etiket
    exit_vc = _G if pnl >= 0 else _R
    fig.add_vline(
        x=dt_close,
        line_dash="dot", line_color=exit_vc, line_width=2,
        annotation_text="  EXIT",
        annotation_position="top right",
        annotation_font=dict(size=12, color=exit_vc, family="DM Sans"),
    )

    # ── İşaret noktaları (scatter) ───────────────────────────────────────────
    # Entry marker — mavi daire
    fig.add_trace(go.Scatter(
        x=[dt_open], y=[avg_entry],
        mode="markers+text",
        marker=dict(size=12, color=_B, symbol="circle",
                    line=dict(color="#0d1117", width=2)),
        text=["  ENTRY"], textposition="middle right",
        textfont=dict(size=12, color=_B, family="DM Sans"),
        name="Entry", showlegend=False,
        hovertemplate=f"Entry: ${avg_entry:,.4f}<extra></extra>",
    ))

    # Exit marker — yeşil/kırmızı daire
    if exit_price:
        fig.add_trace(go.Scatter(
            x=[dt_close], y=[exit_price],
            mode="markers+text",
            marker=dict(size=12, color=exit_vc, symbol="circle",
                        line=dict(color="#0d1117", width=2)),
            text=["  EXIT"], textposition="middle right",
            textfont=dict(size=12, color=exit_vc, family="DM Sans"),
            name="Exit", showlegend=False,
            hovertemplate=f"Exit: ${exit_price:,.4f}<br>PnL: ${pnl:+,.2f}<extra></extra>",
        ))

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        height=420,
        paper_bgcolor=_DBG,
        plot_bgcolor=_DBG,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=100, t=20, b=10),
        font=dict(family="DM Sans, sans-serif", size=13, color=_DT),
        showlegend=False,
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=_BG, bordercolor=_DG,
            font=dict(size=13, color=_TX, family="DM Sans"),
        ),
    )
    fig.update_xaxes(
        gridcolor=_DG, showgrid=True, zeroline=False,
        tickfont=dict(size=11, color=_DT),
        tickformat="%d %b\n%H:%M",
        rangeslider_visible=False,
    )
    fig.update_yaxes(
        gridcolor=_DG, showgrid=True, zeroline=False,
        tickfont=dict(size=11, color=_DT),
        tickprefix="$",
        side="right",
    )

    st.plotly_chart(fig, use_container_width=True,
                    config={
                        "displayModeBar": True,
                        "modeBarButtonsToRemove": [
                            "select2d","lasso2d","autoScale2d"
                        ],
                        "displaylogo": False,
                    })


# ══════════════════════════════════════════════════════════════════════════════
# LEVEL SUMMARY (grafik altı bilgi şeridi)
# ══════════════════════════════════════════════════════════════════════════════

def _render_level_summary(avg_entry, stop_loss, tps, direction):
    if avg_entry <= 0:
        return
    parts = []
    parts.append(
        f"<span style='color:{_B}'>● <b>GİRİŞ</b>&nbsp;${avg_entry:,.4f}</span>"
    )
    if stop_loss:
        sl_dist = abs(avg_entry - stop_loss) / avg_entry * 100
        parts.append(
            f"<span style='color:{_R}'>● <b>SL</b>&nbsp;${stop_loss:,.4f}"
            f"&nbsp;<small>(-{sl_dist:.2f}%)</small></span>"
        )
    for i, t in enumerate(tps):
        lbl      = f"TP{i+1}" if len(tps) > 1 else "TP"
        tp_dist  = abs(t["price"] - avg_entry) / avg_entry * 100
        sign     = "+" if (
            (direction == "LONG" and t["price"] > avg_entry) or
            (direction == "SHORT" and t["price"] < avg_entry)
        ) else "-"
        parts.append(
            f"<span style='color:{_G}'>● <b>{lbl}</b>&nbsp;${t['price']:,.4f}"
            f"&nbsp;<small>({sign}{tp_dist:.2f}%)</small></span>"
        )

    st.markdown(
        f"<div style='display:flex;gap:1.4rem;flex-wrap:wrap;"
        f"background:{_DBG};border:1px solid {_DG};"
        f"border-radius:9px;padding:8px 12px;font-size:13px;margin-top:4px'>"
        + "&nbsp;&nbsp;".join(parts)
        + "</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _parse_dt(raw: str) -> datetime:
    try:
        return datetime.fromisoformat(raw)
    except Exception:
        return datetime.min
