"""
Pusu Terminali
==============
Binance'den EMA/ATR çek → seviyeler gir → plan kur → ateşle
"""

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import json, os, math
from datetime import datetime

# ── Tema ─────────────────────────────────────────────────────────────────────
_BG  = "#0a0a0a"; _BG2 = "#0f0f0f"; _BG3 = "#141414"
_DG  = "#1e1e1e"; _DG2 = "#252525"; _DG3 = "#2e2e2e"
_TX  = "#e6edf3"; _DT  = "#8b949e"; _DT2 = "#6e7681"
_G   = "#00ff88"; _R   = "#ff4444"; _Y   = "#f0b429"
_B   = "#58a6ff"; _O   = "#ff8c00"; _P   = "#c084fc"
_W   = "#ffffff"; _GR  = "#3fb950"

INTERVALS = {"15dk":"15m","1 Saat":"1h","4 Saat":"4h","1 Gün":"1d","1 Hafta":"1w"}
PUSU_FILE = "pusu_data.json"

# ── Veri yönetimi ─────────────────────────────────────────────────────────────

def _load_pusu() -> list:
    if os.path.exists(PUSU_FILE):
        try:
            return json.load(open(PUSU_FILE, encoding="utf-8"))
        except Exception:
            pass
    return []

def _save_pusu(pusular: list):
    with open(PUSU_FILE, "w", encoding="utf-8") as f:
        json.dump(pusular, f, ensure_ascii=False, indent=2)

# ── Binance API ───────────────────────────────────────────────────────────────

def _render_js_fetcher(symbol: str, interval: str):
    """Tarayıcı JS ile Binance'den veri çek, query_params üzerinden Streamlit'e gönder."""
    import streamlit.components.v1 as components

    # Binance interval formatı
    html = f"""
<div id="fetch-status" style="
    background:#0f0f0f;border:1px solid #f0b42940;border-radius:8px;
    padding:10px 14px;font-family:monospace;font-size:12px;color:#f0b429;
    margin:4px 0">
  ⏳ Tarayıcıdan {symbol} {interval} verisi çekiliyor...
</div>
<script>
(async function() {{
  const sym      = "{symbol.upper()}";
  const interval = "{interval}";
  const status   = document.getElementById("fetch-status");

  function calcEMA(prices, n) {{
    const k = 2 / (n + 1);
    let e = prices[0];
    for (let i = 1; i < prices.length; i++) e = prices[i] * k + e * (1 - k);
    return Math.round(e * 10000) / 10000;
  }}

  function calcATR(highs, lows, closes, period=14) {{
    const trs = [];
    for (let i = 1; i < closes.length; i++) {{
      const hl = highs[i] - lows[i];
      const hc = Math.abs(highs[i] - closes[i-1]);
      const lc = Math.abs(lows[i] - closes[i-1]);
      trs.push(Math.max(hl, hc, lc));
    }}
    const last = trs.slice(-period);
    return Math.round(last.reduce((a,b) => a+b, 0) / last.length * 10000) / 10000;
  }}

  async function tryFetch(url) {{
    const r = await fetch(url, {{signal: AbortSignal.timeout(8000)}});
    if (!r.ok) throw new Error("HTTP " + r.status);
    return r.json();
  }}

  const sources = [
    [`https://api.binance.com/api/v3/klines?symbol=${{sym}}&interval=${{interval}}&limit=210`, "Binance"],
    [`https://fapi.binance.com/fapi/v1/klines?symbol=${{sym}}&interval=${{interval}}&limit=210`, "BinanceFut"],
    [`https://api.bybit.com/v5/market/kline?symbol=${{sym}}&interval=${{interval.replace("h","").replace("4h","240").replace("1h","60").replace("15m","15").replace("1d","D").replace("1w","W")}}&limit=210&category=linear`, "Bybit"],
  ];

  let candles = null;
  let srcName = "";

  for (const [url, name] of sources) {{
    try {{
      status.textContent = `⏳ ${{name}} deneniyor...`;
      const data = await tryFetch(url);
      const arr = Array.isArray(data) ? data : (data?.result?.list ? data.result.list.reverse() : null);
      if (arr && arr.length >= 50) {{
        candles = arr;
        srcName = name;
        break;
      }}
    }} catch(e) {{ /* continue */ }}
  }}

  if (!candles) {{
    status.innerHTML = `<span style="color:#ff4444">❌ Tüm kaynaklar başarısız. Değerleri manuel gir.</span>`;
    return;
  }}

  const closes = candles.map(c => parseFloat(c[4]));
  const highs  = candles.map(c => parseFloat(c[2]));
  const lows   = candles.map(c => parseFloat(c[3]));

  const price  = closes[closes.length - 1];
  const ema20  = calcEMA(closes, 20);
  const ema50  = calcEMA(closes, 50);
  const ema200 = calcEMA(closes, 200);
  const atr    = calcATR(highs, lows, closes);

  status.innerHTML = `<span style="color:#00ff88">✅ ${{srcName}} · Veri alındı — price=${{price.toFixed(2)}} ema20=${{ema20.toFixed(2)}}</span>`;

  // Streamlit query params üzerinden gönder
  const params = new URLSearchParams(window.location.search);
  params.set("price",  price.toFixed(4));
  params.set("ema20",  ema20.toFixed(4));
  params.set("ema50",  ema50.toFixed(4));
  params.set("ema200", ema200.toFixed(4));
  params.set("atr",    atr.toFixed(4));
  params.set("src",    srcName);
  window.history.replaceState(null, "", "?" + params.toString());

  // Streamlit'i tetikle
  setTimeout(() => window.location.reload(), 800);
}})();
</script>
"""
    components.html(html, height=60)


def _calc_ema(prices, n):
    k = 2 / (n + 1)
    e = prices[0]
    for p in prices[1:]:
        e = p * k + e * (1 - k)
    return round(e, 4)

def _calc_atr(highs, lows, closes, period=14):
    trs = []
    for i in range(1, len(closes)):
        h, l, pc = highs[i], lows[i], closes[i-1]
        trs.append(max(h-l, abs(h-pc), abs(l-pc)))
    vals = trs[-period:]
    return round(sum(vals)/len(vals), 4) if vals else 0

def _parse_candles(candles):
    """Klines listesini [open,high,low,close,vol,...] formatından parse et."""
    closes = [float(c[4]) for c in candles]
    highs  = [float(c[2]) for c in candles]
    lows   = [float(c[3]) for c in candles]
    return closes, highs, lows

def _build_result(closes, highs, lows, source):
    return {
        "price":      closes[-1],
        "ema20":      _calc_ema(closes, 20),
        "ema50":      _calc_ema(closes, 50),
        "ema200":     _calc_ema(closes, 200),
        "atr":        _calc_atr(highs, lows, closes),
        "source":     source,
        "fetched_at": datetime.now().strftime("%H:%M:%S"),
    }

def _fetch_binance(symbol: str, interval: str) -> dict | None:
    """Binance → OKX → Bybit sırasıyla dener. Başarısızsa None döner."""
    import requests

    # Interval map'leri
    okx_map   = {"1m":"1m","3m":"3m","5m":"5m","15m":"15m","30m":"30m",
                 "1h":"1H","2h":"2H","4h":"4H","6h":"6H","12h":"12H",
                 "1d":"1D","1w":"1W","1M":"1M"}
    bybit_map = {"1m":"1","3m":"3","5m":"5","15m":"15","30m":"30",
                 "1h":"60","2h":"120","4h":"240","6h":"360","12h":"720",
                 "1d":"D","1w":"W","1M":"M"}

    sym = symbol.upper()

    # ── 1. Binance Spot ───────────────────────────────────────────────────
    try:
        r = requests.get("https://api.binance.com/api/v3/klines",
            params={"symbol": sym, "interval": interval, "limit": 210},
            timeout=7)
        if r.status_code == 200:
            candles = r.json()
            if isinstance(candles, list) and len(candles) >= 50:
                closes, highs, lows = _parse_candles(candles)
                return _build_result(closes, highs, lows, "Binance")
    except Exception:
        pass

    # ── 2. Binance Futures ────────────────────────────────────────────────
    try:
        r = requests.get("https://fapi.binance.com/fapi/v1/klines",
            params={"symbol": sym, "interval": interval, "limit": 210},
            timeout=7)
        if r.status_code == 200:
            candles = r.json()
            if isinstance(candles, list) and len(candles) >= 50:
                closes, highs, lows = _parse_candles(candles)
                return _build_result(closes, highs, lows, "Binance Futures")
    except Exception:
        pass

    # ── 3. OKX ────────────────────────────────────────────────────────────
    try:
        okx_sym = sym.replace("USDT","-USDT").replace("BTC","BTC") if "-" not in sym else sym
        okx_bar = okx_map.get(interval, "4H")
        r = requests.get("https://www.okx.com/api/v5/market/candles",
            params={"instId": okx_sym, "bar": okx_bar, "limit": "210"},
            timeout=7)
        if r.status_code == 200:
            data = r.json()
            candles = data.get("data", [])
            if len(candles) >= 50:
                # OKX format: [ts, o, h, l, c, vol, volCcy, ...]
                closes = [float(c[4]) for c in candles][::-1]
                highs  = [float(c[2]) for c in candles][::-1]
                lows   = [float(c[3]) for c in candles][::-1]
                return _build_result(closes, highs, lows, "OKX")
    except Exception:
        pass

    # ── 4. Bybit ──────────────────────────────────────────────────────────
    try:
        bybit_iv = bybit_map.get(interval, "240")
        r = requests.get("https://api.bybit.com/v5/market/kline",
            params={"symbol": sym, "interval": bybit_iv,
                    "limit": "210", "category": "linear"},
            timeout=7)
        if r.status_code == 200:
            data = r.json()
            candles = data.get("result",{}).get("list",[])
            if len(candles) >= 50:
                # Bybit: [startTime, open, high, low, close, volume, turnover]
                closes = [float(c[4]) for c in candles][::-1]
                highs  = [float(c[2]) for c in candles][::-1]
                lows   = [float(c[3]) for c in candles][::-1]
                return _build_result(closes, highs, lows, "Bybit")
    except Exception:
        pass

    return None

# ── Yardımcılar ───────────────────────────────────────────────────────────────

def _parse_levels(raw: str) -> list[float]:
    out = []
    for p in raw.replace(",", " ").split():
        try:
            v = float(p.strip())
            if v > 0: out.append(v)
        except ValueError: pass
    return sorted(out)

def _parse_zone(raw: str) -> tuple[float,float] | None:
    raw = raw.strip()
    if "-" in raw:
        parts = raw.split("-")
        try:
            a, b = float(parts[0].strip()), float(parts[1].strip())
            return (min(a,b), max(a,b))
        except: return None
    return None

def _parse_zones(raw: str) -> list[tuple[float,float]]:
    zones = []
    for line in raw.replace(",","\n").splitlines():
        z = _parse_zone(line.strip())
        if z: zones.append(z)
    return zones

def _avg(lst): return sum(lst)/len(lst) if lst else 0

def _fmt(v):
    if v == 0: return "—"
    if v >= 1000: return f"{v:,.2f}"
    return f"{v:.4f}"

def _calc_rr(entries: list, sl: float, tps: list, direction: str) -> dict:
    if not entries or not tps or sl == 0: return {}
    avg_e = _avg(entries)
    risk  = abs(avg_e - sl)
    if risk == 0: return {}
    results = {}
    for i, tp in enumerate(tps):
        reward = abs(tp - avg_e)
        results[f"tp{i+1}"] = round(reward / risk, 2)
    avg_rr = round(_avg(list(results.values())), 2)
    results["avg"] = avg_rr
    return results

# ── Grafik ────────────────────────────────────────────────────────────────────

def _build_chart(symbol, price, direction,
                 sup_zones, res_zones, liq_levels,
                 ema20, ema50, ema200,
                 show_ema20, show_ema50, show_ema200,
                 entries, sl, tps):

    all_prices = [price]
    for lo,hi in sup_zones + res_zones: all_prices += [lo, hi]
    all_prices += liq_levels
    if entries: all_prices += entries
    if sl > 0:  all_prices.append(sl)
    if tps:     all_prices += tps
    for v in [ema20, ema50, ema200]:
        if v > 0: all_prices.append(v)

    p_min = min(all_prices) * 0.990
    p_max = max(all_prices) * 1.010

    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        margin=dict(l=8, r=130, t=8, b=8),
        height=500,
        xaxis=dict(visible=False, range=[0,100], fixedrange=True),
        yaxis=dict(
            gridcolor="#191919", gridwidth=0.5, zeroline=False,
            tickfont=dict(color=_DT, size=10, family="monospace"),
            tickformat=",.0f", range=[p_min, p_max],
            side="right", fixedrange=True,
        ),
        showlegend=False,
        dragmode=False,
        hovermode=False,
        font=dict(family="monospace", color=_TX),
    )

    # ── Destek bölgeleri ─────────────────────────────────────────────────────
    for lo, hi in sup_zones:
        mid = (lo+hi)/2
        fig.add_shape(type="rect", x0=0, x1=100, y0=lo, y1=hi,
            fillcolor="rgba(0,255,136,0.07)",
            line=dict(color="rgba(0,255,136,0.45)", width=1))
        fig.add_annotation(x=1, y=mid,
            text=f"SUP  {_fmt(lo)} – {_fmt(hi)}",
            showarrow=False, font=dict(color="#00ff88", size=10, family="monospace"),
            xanchor="left", bgcolor="rgba(0,0,0,0.55)", borderpad=3)

    # ── Direnç bölgeleri ─────────────────────────────────────────────────────
    for lo, hi in res_zones:
        mid = (lo+hi)/2
        fig.add_shape(type="rect", x0=0, x1=100, y0=lo, y1=hi,
            fillcolor="rgba(255,68,68,0.07)",
            line=dict(color="rgba(255,68,68,0.45)", width=1))
        fig.add_annotation(x=1, y=mid,
            text=f"RES  {_fmt(lo)} – {_fmt(hi)}",
            showarrow=False, font=dict(color="#ff4444", size=10, family="monospace"),
            xanchor="left", bgcolor="rgba(0,0,0,0.55)", borderpad=3)

    # ── Likidasyon çizgileri ─────────────────────────────────────────────────
    for lv in liq_levels:
        fig.add_shape(type="line", x0=0, x1=100, y0=lv, y1=lv,
            line=dict(color=_O, width=2, dash="dot"))
        fig.add_annotation(x=1, y=lv,
            text=f"LIQ  {_fmt(lv)}",
            showarrow=False, font=dict(color=_O, size=10, family="monospace"),
            xanchor="left", bgcolor="rgba(0,0,0,0.55)", borderpad=3)

    # ── EMA çizgileri (yazı ortada) ──────────────────────────────────────────
    ema_defs = [
        (ema20,  show_ema20,  _W,  "EMA20"),
        (ema50,  show_ema50,  _Y,  "EMA50"),
        (ema200, show_ema200, _R,  "EMA200"),
    ]
    for val, show, color, label in ema_defs:
        if show and val > 0:
            fig.add_shape(type="line", x0=0, x1=100, y0=val, y1=val,
                line=dict(color=color, width=1.5, dash="solid"))
            fig.add_annotation(x=50, y=val,
                text=f"{label}  {_fmt(val)}",
                showarrow=False, font=dict(color=color, size=10, family="monospace"),
                xanchor="center", yanchor="bottom",
                bgcolor="rgba(0,0,0,0.65)", borderpad=3)

    # ── Güncel fiyat ─────────────────────────────────────────────────────────
    fig.add_shape(type="line", x0=0, x1=100, y0=price, y1=price,
        line=dict(color=_B, width=1.5, dash="dash"))
    fig.add_annotation(x=99, y=price,
        text=f"◀ {_fmt(price)}",
        showarrow=False, font=dict(color=_B, size=12, family="monospace"),
        xanchor="right", bgcolor="rgba(88,166,255,0.18)",
        bordercolor=_B, borderwidth=1, borderpad=4)

    is_long = direction == "LONG"
    e_color = _G if is_long else _R

    # ── Entry çizgileri ──────────────────────────────────────────────────────
    for i, en in enumerate(entries):
        fig.add_shape(type="line", x0=0, x1=100, y0=en, y1=en,
            line=dict(color=e_color, width=1.5, dash="dashdot"))
        fig.add_annotation(x=99, y=en,
            text=f"ENTRY {i+1}  {_fmt(en)}",
            showarrow=False, font=dict(color=e_color, size=10, family="monospace"),
            xanchor="right", bgcolor="rgba(0,0,0,0.7)",
            bordercolor=e_color, borderwidth=1, borderpad=3)

    # ── Stop Loss ─────────────────────────────────────────────────────────────
    if sl > 0:
        fig.add_shape(type="line", x0=0, x1=100, y0=sl, y1=sl,
            line=dict(color=_R, width=2, dash="dash"))
        fig.add_annotation(x=99, y=sl,
            text=f"SL  {_fmt(sl)}",
            showarrow=False, font=dict(color=_R, size=11, family="monospace"),
            xanchor="right", bgcolor="rgba(255,68,68,0.18)",
            bordercolor=_R, borderwidth=1, borderpad=4)

    # ── Take Profit çizgileri ────────────────────────────────────────────────
    tp_colors = [_GR, _G, "#00ffcc", "#7fff00"]
    for i, tp in enumerate(tps):
        tc = tp_colors[i % len(tp_colors)]
        fig.add_shape(type="line", x0=0, x1=100, y0=tp, y1=tp,
            line=dict(color=tc, width=1.5, dash="dash"))
        rr_vals = _calc_rr(entries, sl, tps, direction)
        rr_str  = f"  RR {rr_vals.get(f'tp{i+1}','')}" if rr_vals else ""
        fig.add_annotation(x=99, y=tp,
            text=f"TP{i+1}  {_fmt(tp)}{rr_str}",
            showarrow=False, font=dict(color=tc, size=10, family="monospace"),
            xanchor="right", bgcolor="rgba(0,0,0,0.7)",
            bordercolor=tc, borderwidth=1, borderpad=3)

    return fig

# ── R:R özet HTML ──────────────────────────────────────────────────────────────

def _rr_summary_html(entries, sl, tps, direction):
    if not entries or sl == 0 or not tps:
        return ""
    rr = _calc_rr(entries, sl, tps, direction)
    avg_e = _avg(entries)
    risk_pct = abs(avg_e - sl) / avg_e * 100

    parts = []
    for i, tp in enumerate(tps):
        v = rr.get(f"tp{i+1}", "—")
        c = _G if isinstance(v, float) and v >= 2 else _Y if isinstance(v, float) and v >= 1 else _R
        parts.append(f"<span style='color:{c}'>TP{i+1}: {v}</span>")

    avg_c = _G if rr.get("avg",0) >= 2 else _Y if rr.get("avg",0) >= 1 else _R
    html = (
        f"<div style='display:flex;align-items:center;gap:14px;flex-wrap:wrap;"
        f"background:{_BG3};border:1px solid {_DG3};border-radius:8px;"
        f"padding:7px 12px;font-family:monospace;font-size:12px;margin:4px 0'>"
        f"<span style='color:{_DT2}'>Avg Entry: <b style='color:{_TX}'>{_fmt(avg_e)}</b></span>"
        f"<span style='color:{_DT2}'>Risk: <b style='color:{_R}'>{risk_pct:.2f}%</b></span>"
        + "".join(f"<span style='color:{_DT2}'>TP{i+1}: <b style='color:{_GR}'>{_fmt(tp)}</b></span>" for i,tp in enumerate(tps))
        + "  " + "  ".join(parts)
        + f"  <b style='color:{avg_c}'>Avg RR: {rr.get('avg','—')}</b>"
        f"</div>"
    )
    return html

# ── Pusu kart ──────────────────────────────────────────────────────────────────

def _render_pusu_card(p: dict, idx: int, pusular: list):
    pid       = p.get("id", idx)
    symbol    = p.get("symbol","?")
    direction = p.get("direction","LONG")
    entries   = p.get("entries",[])
    sl        = p.get("sl", 0)
    tps       = p.get("tps",[])
    interval  = p.get("interval","4h")
    created   = p.get("created_at","")
    fired     = p.get("fired", False)
    is_long   = direction == "LONG"
    dir_c     = _G if is_long else _R
    dir_bg    = "#051a0e" if is_long else "#1a0505"
    avg_e     = _avg(entries)
    rr        = _calc_rr(entries, sl, tps, direction)

    open_key  = f"pusu_open_{pid}"
    st.session_state.setdefault(open_key, False)
    is_open   = st.session_state[open_key]

    status_c  = "#f0b429" if not fired else "#6e7681"
    status    = "🟡 Bekleniyor" if not fired else "✅ Ateşlendi"

    # Ana kart
    card_col, btn_col = st.columns([14, 1])
    with card_col:
        st.markdown(
            f"<div style='background:{_BG2};border:1px solid {_DG3};"
            f"border-left:3px solid {dir_c};"
            f"border-radius:{'10px 10px 0 0' if is_open else '10px'};"
            f"padding:12px 16px;cursor:pointer'>"
            # Üst
            f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:6px'>"
            f"<span style='font-family:\"Space Mono\",monospace;font-size:16px;"
            f"font-weight:700;color:{_TX}'>{symbol}</span>"
            f"<span style='background:{dir_bg};color:{dir_c};padding:2px 9px;"
            f"border-radius:4px;font-size:11px;font-weight:700'>{direction}</span>"
            f"<span style='font-size:11px;color:{_DT}'>{interval.upper()}</span>"
            f"<span style='flex:1'></span>"
            f"<span style='font-size:11px;color:{status_c};font-weight:600'>{status}</span>"
            f"</div>"
            # Pusu kuruldu
            f"<div style='font-family:\"Space Mono\",monospace;font-size:13px;"
            f"font-weight:700;color:{_Y};letter-spacing:0.08em;"
            f"margin-bottom:6px'>🎯 Pusu Kuruldu... Bekleniyor</div>"
            # Alt bilgi
            f"<div style='display:flex;gap:14px;flex-wrap:wrap;font-family:monospace;"
            f"font-size:12px;color:{_DT}'>"
            f"<span>Avg E: <b style='color:{dir_c}'>{_fmt(avg_e)}</b></span>"
            f"<span>SL: <b style='color:{_R}'>{_fmt(sl)}</b></span>"
            + "".join(f"<span>TP{i+1}: <b style='color:{_GR}'>{_fmt(tp)}</b></span>"
                      for i,tp in enumerate(tps))
            + (f"<span>Avg RR: <b style='color:{_G}'>{rr.get('avg','—')}</b></span>"
               if rr else "")
            + (f"<span style='color:{_DT2};margin-left:auto'>{created}</span>" if created else "")
            + f"</div></div>",
            unsafe_allow_html=True)

    with btn_col:
        if st.button("▼" if not is_open else "▲",
                     key=f"pusu_tog_{pid}", use_container_width=True):
            st.session_state[open_key] = not is_open
            st.rerun()

    # Açık panel: Ateş / Düzenle / Sil
    if is_open:
        st.markdown(
            f"<div style='background:{_BG3};border:1px solid {_DG3};"
            f"border-top:none;border-radius:0 0 10px 10px;"
            f"padding:10px 14px;margin-bottom:6px'>",
            unsafe_allow_html=True)

        rr_html = _rr_summary_html(entries, sl, tps, direction)
        if rr_html:
            st.markdown(rr_html, unsafe_allow_html=True)

        ba1, ba2, ba3, _ = st.columns([1.2, 1, 1, 5])
        with ba1:
            if st.button("🔥 Ateş", key=f"pusu_fire_{pid}",
                         type="primary", use_container_width=True):
                _fire_pusu(p)
                pusular[idx]["fired"] = True
                _save_pusu(pusular)
                st.session_state.pusu_pusular = pusular
                st.success("🔥 Pozisyon forma aktarıldı!")
                st.rerun()
        with ba2:
            if st.button("✏️ Düzenle", key=f"pusu_edit_{pid}",
                         use_container_width=True):
                st.session_state["pusu_edit_idx"] = idx
                st.session_state["pusu_tab"] = "edit"
                st.rerun()
        with ba3:
            if st.button("🗑️ Sil", key=f"pusu_del_{pid}",
                         use_container_width=True):
                pusular.pop(idx)
                _save_pusu(pusular)
                st.session_state.pusu_pusular = pusular
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


def _fire_pusu(p: dict):
    """Pusu verilerini session_state'e yaz → position form okuyacak."""
    entries = p.get("entries", [])
    tps     = p.get("tps", [])
    avg_e   = _avg(entries)
    sl      = p.get("sl", 0)
    tp1     = tps[0] if tps else 0

    st.session_state["pusu_fire"] = {
        "symbol":    p.get("symbol",""),
        "direction": p.get("direction","LONG"),
        "avg_entry": avg_e,
        "entries":   entries,
        "sl":        sl,
        "tps":       tps,
        "tp1":       tp1,
    }
    st.session_state["active_tab"] = 0  # Aktif Pozisyonlar sekmesine geç


# ── Ana render ────────────────────────────────────────────────────────────────

def render_pusu():

    # CSS
    st.markdown("""<style>
    .pusu-title {
        font-family: 'Space Mono', monospace;
        font-size: 1.05rem; font-weight: 700;
        color: #f0b429; letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    </style>""", unsafe_allow_html=True)

    # Session state init
    if "pusu_pusular" not in st.session_state:
        st.session_state.pusu_pusular = _load_pusu()
    if "pusu_tab" not in st.session_state:
        st.session_state.pusu_tab = "new"

    pusular = st.session_state.pusu_pusular

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='display:flex;align-items:center;justify-content:space-between;"
        f"margin-bottom:14px'>"
        f"<div>"
        f"<div class='pusu-title'>🎯 Pusu Terminali</div>"
        f"<div style='font-size:12px;color:{_DT}'>Seviye analizi · Plan kur · Ateşle</div>"
        f"</div>"
        f"<div style='background:#1a1200;border:1px solid {_Y}40;border-radius:8px;"
        f"padding:4px 14px;font-size:11px;color:{_Y};letter-spacing:0.1em'>"
        f"{len([p for p in pusular if not p.get('fired')])} AKTİF PUSU</div>"
        f"</div>",
        unsafe_allow_html=True)

    # ── İç sekmeler ───────────────────────────────────────────────────────────
    inner_tabs = st.tabs(["⚡ Yeni Pusu Kur", "🎯 Kurulan Pusular"])

    # ══════════════════════════════════════════════════════════════════════════
    # SEKME 1 — YENİ PUSU KUR
    # ══════════════════════════════════════════════════════════════════════════
    with inner_tabs[0]:

        # Edit mode kontrolü
        edit_idx = st.session_state.pop("pusu_edit_idx", None)
        edit_data = pusular[edit_idx] if edit_idx is not None and edit_idx < len(pusular) else None

        # ── Veri çekme paneli ─────────────────────────────────────────────
        st.markdown(
            f"<div style='font-size:11px;font-weight:600;color:{_DT};text-transform:uppercase;"
            f"letter-spacing:0.08em;margin-bottom:8px'>📡 Binance Verisi</div>",
            unsafe_allow_html=True)

        fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 1.5])
        with fc1:
            symbol = st.text_input("Sembol", key="pus_symbol",
                value=edit_data.get("symbol","BTCUSDT") if edit_data else
                      st.session_state.get("pus_symbol_val","BTCUSDT"),
                label_visibility="collapsed",
                placeholder="BTCUSDT").upper().strip()
            st.session_state["pus_symbol_val"] = symbol
        with fc2:
            iv_label = st.selectbox("Interval", list(INTERVALS.keys()),
                                    index=2, key="pus_interval",
                                    label_visibility="collapsed")
            interval = INTERVALS[iv_label]
        with fc3:
            direction = st.radio("Yön", ["LONG","SHORT"], horizontal=True,
                                 key="pus_dir", label_visibility="collapsed")
        with fc4:
            fetch_btn = st.button("🔄 Veri Çek", key="pus_fetch",
                                  use_container_width=True)

        # Veri çek
        fetched_key = f"pus_fetched_{symbol}_{interval}"
        if fetch_btn:
            with st.spinner(f"{symbol} {iv_label} verisi çekiliyor..."):
                result = _fetch_binance(symbol, interval)
            if result:
                st.session_state[fetched_key] = result
                src_lbl = result.get("source","?")
                st.success(f"✅ {src_lbl} · {result['fetched_at']} — Veri alındı")
            else:
                # Tarayıcı JS fallback
                st.session_state["pus_use_js"] = True
                st.warning("⚠️ Sunucu erişimi yok — Tarayıcı üzerinden deneniyor...")
                st.rerun()

        # JS Fetch fallback
        if st.session_state.get("pus_use_js"):
            _render_js_fetcher(symbol, interval)

        fetched = st.session_state.get(fetched_key, {})

        # JS'den gelen veri kontrolü
        qp = st.query_params
        if "ema20" in qp and f"pus_fetched_{symbol}_{interval}" not in st.session_state:
            try:
                result = {
                    "price":  float(qp.get("price",0)),
                    "ema20":  float(qp.get("ema20",0)),
                    "ema50":  float(qp.get("ema50",0)),
                    "ema200": float(qp.get("ema200",0)),
                    "atr":    float(qp.get("atr",0)),
                    "source": qp.get("src","Browser"),
                    "fetched_at": datetime.now().strftime("%H:%M:%S"),
                }
                st.session_state[fetched_key] = result
                st.session_state["pus_use_js"] = False
                # Query param'ları temizle
                st.query_params.clear()
                st.rerun()
            except Exception:
                pass

        fetched = st.session_state.get(fetched_key, {})

        # ── Fiyat + EMA + ATR ─────────────────────────────────────────────
        st.markdown(
            f"<div style='font-size:11px;font-weight:600;color:{_DT};text-transform:uppercase;"
            f"letter-spacing:0.08em;margin:10px 0 6px'>📊 Fiyat & Göstergeler</div>",
            unsafe_allow_html=True)

        mv1, mv2, mv3, mv4, mv5 = st.columns(5)
        def _num(col, label, key, fetched_val, edit_val=None, fmt="%.2f", step=10.0):
            init = edit_val if edit_val is not None else fetched_val if fetched_val else 0.0
            with col:
                st.markdown(f"<div style='font-size:10px;color:{_DT2};margin-bottom:2px'>{label}</div>",
                            unsafe_allow_html=True)
                return st.number_input(label, value=float(init), format=fmt,
                                       step=step, key=key, label_visibility="collapsed")

        price  = _num(mv1, "Güncel Fiyat", "pus_price",  fetched.get("price",0),  step=10.0)
        ema20  = _num(mv2, "EMA 20",       "pus_ema20",  fetched.get("ema20",0),   step=10.0)
        ema50  = _num(mv3, "EMA 50",       "pus_ema50",  fetched.get("ema50",0),   step=10.0)
        ema200 = _num(mv4, "EMA 200",      "pus_ema200", fetched.get("ema200",0),  step=10.0)
        atr    = _num(mv5, "ATR",          "pus_atr",    fetched.get("atr",0),     step=1.0, fmt="%.4f")

        if fetched:
            src_lbl = fetched.get("source","?")
            src_c   = {"Binance":_G,"Binance Futures":_G,"OKX":_B,"Bybit":_P}.get(src_lbl,_DT)
            st.markdown(
                f"<div style='font-size:11px;color:{_DT2};margin-bottom:2px'>"
                f"<span style='color:{src_c};font-weight:600'>{src_lbl}</span>"
                f" · {fetched.get('fetched_at','—')} &nbsp;·&nbsp; "
                f"Manuel değiştirebilirsin</div>",
                unsafe_allow_html=True)

        ec1, ec2, ec3 = st.columns([1,1,1])
        with ec1: show_ema20  = st.checkbox("EMA20 Göster",  True, key="pus_se20")
        with ec2: show_ema50  = st.checkbox("EMA50 Göster",  True, key="pus_se50")
        with ec3: show_ema200 = st.checkbox("EMA200 Göster", True, key="pus_se200")

        # ── Bölgeler ──────────────────────────────────────────────────────
        bc1, bc2 = st.columns(2)
        with bc1:
            st.markdown(f"<div style='font-size:11px;color:{_G};font-weight:600;"
                        f"text-transform:uppercase;letter-spacing:0.08em;"
                        f"margin-bottom:4px'>Destek Bölgeleri</div>", unsafe_allow_html=True)
            sup_raw = st.text_area("Destek", height=75, key="pus_sup",
                label_visibility="collapsed",
                value=edit_data.get("sup_raw","") if edit_data else "",
                placeholder="73000-74000\n71000-71500")
        with bc2:
            st.markdown(f"<div style='font-size:11px;color:{_R};font-weight:600;"
                        f"text-transform:uppercase;letter-spacing:0.08em;"
                        f"margin-bottom:4px'>Direnç Bölgeleri</div>", unsafe_allow_html=True)
            res_raw = st.text_area("Direnç", height=75, key="pus_res",
                label_visibility="collapsed",
                value=edit_data.get("res_raw","") if edit_data else "",
                placeholder="79000-81250\n83000-84000")

        st.markdown(f"<div style='font-size:11px;color:{_O};font-weight:600;"
                    f"text-transform:uppercase;letter-spacing:0.08em;"
                    f"margin-bottom:4px'>Likidasyon Seviyeleri (boşlukla ayır)</div>",
                    unsafe_allow_html=True)
        liq_raw = st.text_input("Likidasyon", key="pus_liq",
            label_visibility="collapsed",
            value=edit_data.get("liq_raw","") if edit_data else "",
            placeholder="78500 80200 82500")

        # ── Plan: Entry / SL / TP ─────────────────────────────────────────
        st.markdown(
            f"<div style='font-size:11px;font-weight:600;color:{_DT};"
            f"text-transform:uppercase;letter-spacing:0.08em;margin:10px 0 6px'>"
            f"📋 Trade Planı (Manuel Gir)</div>",
            unsafe_allow_html=True)

        # Entry sayısı
        n_entries = st.slider("Entry sayısı", 1, 4, 1, key="pus_nentry",
                              label_visibility="collapsed")
        entry_cols = st.columns(n_entries)
        entries = []
        for i, col in enumerate(entry_cols):
            with col:
                st.markdown(f"<div style='font-size:10px;color:{_G};margin-bottom:2px'>"
                            f"Entry {i+1}</div>", unsafe_allow_html=True)
                default_e = float(edit_data["entries"][i]) if (edit_data and i < len(edit_data.get("entries",[]))) else float(price or 0)
                v = st.number_input(f"e{i}", value=default_e, format="%.2f",
                                    step=10.0, key=f"pus_entry_{i}",
                                    label_visibility="collapsed")
                entries.append(v)

        sl_col, _ = st.columns([2, 5])
        with sl_col:
            st.markdown(f"<div style='font-size:10px;color:{_R};margin-bottom:2px'>"
                        f"Stop Loss</div>", unsafe_allow_html=True)
            default_sl = float(edit_data.get("sl",0)) if edit_data else 0.0
            sl = st.number_input("SL", value=default_sl, format="%.2f",
                                 step=10.0, key="pus_sl",
                                 label_visibility="collapsed")

        n_tps = st.slider("TP sayısı", 1, 4, 2, key="pus_ntp",
                          label_visibility="collapsed")
        tp_cols = st.columns(n_tps)
        tps = []
        for i, col in enumerate(tp_cols):
            with col:
                st.markdown(f"<div style='font-size:10px;color:{_GR};margin-bottom:2px'>"
                            f"TP {i+1}</div>", unsafe_allow_html=True)
                default_tp = float(edit_data["tps"][i]) if (edit_data and i < len(edit_data.get("tps",[]))) else 0.0
                v = st.number_input(f"tp{i}", value=default_tp, format="%.2f",
                                    step=10.0, key=f"pus_tp_{i}",
                                    label_visibility="collapsed")
                tps.append(v)

        # R:R canlı göster
        valid_entries = [e for e in entries if e > 0]
        valid_tps     = [t for t in tps if t > 0]
        if valid_entries and sl > 0 and valid_tps:
            rr_html = _rr_summary_html(valid_entries, sl, valid_tps, direction)
            st.markdown(rr_html, unsafe_allow_html=True)

        # ── Grafik + Plan butonları ───────────────────────────────────────
        g1, g2 = st.columns([1, 1])
        with g1:
            grafik_btn = st.button("📈 Grafik Oluştur", key="pus_grafik",
                                   use_container_width=True)
        with g2:
            plan_btn = st.button("🎯 Pusu Kur", key="pus_plan",
                                 type="primary", use_container_width=True)

        sup_zones = _parse_zones(sup_raw)
        res_zones = _parse_zones(res_raw)
        liq_levels = _parse_levels(liq_raw)

        # Grafik render
        if grafik_btn or st.session_state.get("pus_show_chart"):
            st.session_state["pus_show_chart"] = True
            if price > 0:
                fig = _build_chart(
                    symbol, price, direction,
                    sup_zones, res_zones, liq_levels,
                    ema20, ema50, ema200,
                    show_ema20, show_ema50, show_ema200,
                    valid_entries, sl, valid_tps)
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False,
                                        "scrollZoom": False,
                                        "staticPlot": True})

                # Özet tablo: entry/sl/tp değerleri
                if valid_entries or sl > 0 or valid_tps:
                    st.markdown(
                        f"<div style='background:{_BG3};border:1px solid {_DG3};"
                        f"border-radius:8px;padding:10px 14px;"
                        f"font-family:monospace;font-size:12px;color:{_DT};"
                        f"display:flex;flex-wrap:wrap;gap:14px;margin-top:4px'>",
                        unsafe_allow_html=True)
                    parts = []
                    for i,e in enumerate(valid_entries):
                        dc = _G if direction=="LONG" else _R
                        parts.append(f"<span>Entry{i+1}: <b style='color:{dc}'>{_fmt(e)}</b></span>")
                    if sl > 0:
                        parts.append(f"<span>SL: <b style='color:{_R}'>{_fmt(sl)}</b></span>")
                    for i,tp in enumerate(valid_tps):
                        parts.append(f"<span>TP{i+1}: <b style='color:{_GR}'>{_fmt(tp)}</b></span>")
                    st.markdown(
                        f"<div style='display:flex;flex-wrap:wrap;gap:14px;"
                        f"font-family:monospace;font-size:12px;color:{_DT}'>"
                        + "".join(parts) + "</div>",
                        unsafe_allow_html=True)
            else:
                st.warning("Güncel fiyatı gir.")

        # Pusu kur
        if plan_btn:
            if not valid_entries:
                st.error("En az 1 entry giriniz.")
            elif sl <= 0:
                st.error("Stop Loss giriniz.")
            elif not valid_tps:
                st.error("En az 1 TP giriniz.")
            else:
                new_pusu = {
                    "id":        len(pusular) + 1,
                    "symbol":    symbol,
                    "direction": direction,
                    "interval":  interval,
                    "price":     price,
                    "ema20":     ema20,
                    "ema50":     ema50,
                    "ema200":    ema200,
                    "atr":       atr,
                    "entries":   valid_entries,
                    "sl":        sl,
                    "tps":       valid_tps,
                    "sup_raw":   sup_raw,
                    "res_raw":   res_raw,
                    "liq_raw":   liq_raw,
                    "fired":     False,
                    "created_at": datetime.now().strftime("%d %b %Y %H:%M"),
                }
                if edit_idx is not None:
                    new_pusu["id"] = pusular[edit_idx].get("id", edit_idx+1)
                    pusular[edit_idx] = new_pusu
                else:
                    pusular.append(new_pusu)

                _save_pusu(pusular)
                st.session_state.pusu_pusular = pusular
                st.session_state["pus_show_chart"] = False
                st.balloons()
                st.success("🎯 Pusu kuruldu! Kurulan Pusular sekmesine bak.")
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # SEKME 2 — KURULAN PUSULAR
    # ══════════════════════════════════════════════════════════════════════════
    with inner_tabs[1]:
        pusular = st.session_state.pusu_pusular
        active  = [p for p in pusular if not p.get("fired")]
        fired   = [p for p in pusular if p.get("fired")]

        if not pusular:
            st.markdown(
                f"<div style='text-align:center;padding:3rem;color:{_DT};"
                f"border:1.5px dashed {_DG};border-radius:12px'>"
                f"<div style='font-size:2rem;margin-bottom:8px'>🎯</div>"
                f"<div>Henüz pusu kurulmadı. Yeni Pusu Kur sekmesinden başla.</div>"
                f"</div>",
                unsafe_allow_html=True)
            return

        if active:
            st.markdown(
                f"<div style='font-size:12px;font-weight:600;color:{_Y};"
                f"margin-bottom:8px'>⚡ Aktif Pusular ({len(active)})</div>",
                unsafe_allow_html=True)
            for idx, p in enumerate(pusular):
                if not p.get("fired"):
                    _render_pusu_card(p, idx, pusular)

        if fired:
            st.markdown(
                f"<div style='font-size:12px;font-weight:600;color:{_DT};"
                f"margin:14px 0 8px'>✅ Ateşlenenler ({len(fired)})</div>",
                unsafe_allow_html=True)
            for idx, p in enumerate(pusular):
                if p.get("fired"):
                    _render_pusu_card(p, idx, pusular)
