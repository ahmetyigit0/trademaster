"""
TradeBot Engine — ccxt + Binance USDM Futures Testnet
======================================================
Streamlit Cloud üzerinde çalışır:
  - ccxt.binanceusdm + set_sandbox_mode(True)
  - Tüm REST çağrıları testnet.binancefuture.com/fapi/ → TR kısıtı yok
  - Streamlit Cloud ABD/EU IP kullandığı için Binance.com'a da erişir
  - Thread tabanlı: WebSocket veya polling ile canlı veri

Rate limit koruması:
  - Kline: 5s polling (WebSocket ekstra bağımlılık gerektiriyor)
  - Order: sinyal başına 1 adet + 60s cooldown
  - fetchOHLCV: limit=50, 5s arayla → ~10 req/dk (limit: 1200/dk)
"""

import threading
import time
from datetime import datetime, timezone
from collections import deque
from typing import Optional

try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False

HAS_BINANCE = HAS_CCXT
MAX_LOG_ROWS = 200


# ══════════════════════════════════════════════════════════════════════════════
# SHARED STATE
# ══════════════════════════════════════════════════════════════════════════════

class BotState:
    def __init__(self):
        self._lock       = threading.Lock()
        self.running     = False
        self.status      = "Durduruldu"
        self.symbol      = ""
        self.strategy    = ""
        self.trades      = []
        self.open_trade  = None
        self.log         = deque(maxlen=MAX_LOG_ROWS)
        self.equity      = 0.0
        self.current_pnl = 0.0
        self.total_pnl   = 0.0
        self.win_count   = 0
        self.loss_count  = 0
        self.last_price  = 0.0
        self.candles     = deque(maxlen=500)
        self.error       = ""
        self.thought     = "Bot henüz başlatılmadı."   # canlı düşünce
        self.indicators  = {}                           # son indikatör değerleri

    def set(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def add_log(self, level: str, msg: str):
        with self._lock:
            self.log.appendleft({
                "time":  datetime.now(timezone.utc).strftime("%H:%M:%S"),
                "level": level,
                "msg":   msg,
            })

    def add_trade(self, trade: dict):
        with self._lock:
            self.trades.insert(0, trade)
            if len(self.trades) > 200:
                self.trades.pop()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "running":     self.running,
                "status":      self.status,
                "symbol":      self.symbol,
                "strategy":    self.strategy,
                "trades":      list(self.trades),
                "open_trade":  self.open_trade,
                "log":         list(self.log),
                "equity":      self.equity,
                "current_pnl": self.current_pnl,
                "total_pnl":   self.total_pnl,
                "win_count":   self.win_count,
                "loss_count":  self.loss_count,
                "last_price":  self.last_price,
                "error":       self.error,
                "thought":     self.thought,
                "indicators":  dict(self.indicators),
            }


_bot_state = BotState()

def get_state() -> BotState:
    return _bot_state


# ══════════════════════════════════════════════════════════════════════════════
# CCXT EXCHANGE FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def _make_exchange(api_key: str, api_secret: str) -> "ccxt.binanceusdm":
    """
    ccxt.binanceusdm + set_sandbox_mode(True)
    → Tüm istek testnet.binancefuture.com/fapi/ adresine gider.
    TR IP kısıtı bu URL'de yoktur. Streamlit Cloud'dan da çalışır.
    """
    ex = ccxt.binanceusdm({
        "apiKey":  api_key,
        "secret":  api_secret,
        "options": {
            "defaultType":       "future",
            "adjustForTimeDifference": True,   # timestamp auto-sync
        },
        "timeout": 15000,
        "enableRateLimit": True,               # ccxt built-in rate limiter
    })
    ex.set_sandbox_mode(True)   # ← tüm URL'leri testnet'e yönlendirir
    return ex


# ══════════════════════════════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def _ema(prices: list, period: int) -> Optional[float]:
    if len(prices) < period:
        return None
    k, ema = 2 / (period + 1), prices[0]
    for p in prices[1:]:
        ema = p * k + ema * (1 - k)
    return ema


def _rsi(prices: list, period: int = 14) -> Optional[float]:
    if len(prices) < period + 1:
        return None
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains  = [max(d, 0)   for d in deltas[-period:]]
    losses = [abs(min(d, 0)) for d in deltas[-period:]]
    avg_g  = sum(gains)  / period
    avg_l  = sum(losses) / period
    if avg_l == 0:
        return 100.0
    return round(100 - 100 / (1 + avg_g / avg_l), 2)


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_signal(candles_list: list, cfg: dict) -> str:
    """Backward compat wrapper."""
    result = evaluate_signal_with_thoughts(candles_list, cfg)
    return result["signal"]


def evaluate_signal_with_thoughts(candles_list: list, cfg: dict) -> dict:
    """
    Returns:
        signal     : LONG | SHORT | CLOSE_LONG | CLOSE_SHORT | HOLD
        thought    : Türkçe açıklama (neden bu karar?)
        indicators : dict of current indicator values
    """
    NOT_ENOUGH = {
        "signal": "HOLD",
        "thought": f"⏳ Yeterli mum yok ({len(candles_list)}/30). "
                   f"Geçmiş veri bekleniyor...",
        "indicators": {},
    }
    if len(candles_list) < 30:
        return NOT_ENOUGH

    strategy = cfg.get("strategy", "EMA_CROSS")
    closes   = [c["c"] for c in candles_list]
    price    = closes[-1]

    # ── EMA_CROSS ─────────────────────────────────────────────────────────────
    if strategy == "EMA_CROSS":
        fast, slow = cfg.get("ema_fast", 9), cfg.get("ema_slow", 21)
        rsi_hi = cfg.get("rsi_overbought", 70)
        rsi_lo = cfg.get("rsi_oversold",   30)

        ema_f = _ema(closes, fast)
        ema_s = _ema(closes, slow)
        rsi   = _rsi(closes, 14)

        if None in (ema_f, ema_s, rsi):
            return NOT_ENOUGH

        prev    = closes[:-1]
        p_ema_f = _ema(prev, fast)
        p_ema_s = _ema(prev, slow)
        if None in (p_ema_f, p_ema_s):
            return NOT_ENOUGH

        cross_up   = p_ema_f <= p_ema_s and ema_f > ema_s
        cross_down = p_ema_f >= p_ema_s and ema_f < ema_s
        ema_spread = round(((ema_f - ema_s) / price) * 100, 3)
        ema_trend  = "📈 EMA hızlı üstte" if ema_f > ema_s else "📉 EMA hızlı altta"

        inds = {
            f"EMA{fast}": round(ema_f, 4),
            f"EMA{slow}": round(ema_s, 4),
            "RSI":        round(rsi, 1),
            "EMA Spread": f"{ema_spread:+.3f}%",
            "Trend":      ema_trend,
        }

        # Sinyal kararı + düşünce
        if cross_up and rsi < rsi_hi:
            return {
                "signal": "LONG",
                "thought": (
                    f"🟢 LONG SİNYAL!\n"
                    f"EMA{fast} ({ema_f:.2f}) az önce EMA{slow} ({ema_s:.2f}) "
                    f"üzerine çıktı (Golden Cross). "
                    f"RSI {rsi:.1f} — aşırı alım bölgesinde değil ({rsi_hi} altı). "
                    f"Yukarı momentum güçlü → LONG açılıyor."
                ),
                "indicators": inds,
            }
        if cross_down and rsi > rsi_lo:
            return {
                "signal": "SHORT",
                "thought": (
                    f"🔴 SHORT SİNYAL!\n"
                    f"EMA{fast} ({ema_f:.2f}) EMA{slow} ({ema_s:.2f}) "
                    f"altına geçti (Death Cross). "
                    f"RSI {rsi:.1f} — aşırı satım bölgesinde değil ({rsi_lo} üstü). "
                    f"Aşağı momentum güçlü → SHORT açılıyor."
                ),
                "indicators": inds,
            }
        if ema_f < ema_s and rsi > 75:
            return {
                "signal": "CLOSE_LONG",
                "thought": (
                    f"⚠️ LONG KAPAT\n"
                    f"EMA{fast} EMA{slow} altında ve RSI {rsi:.1f} > 75 "
                    f"(aşırı alım). Trend zayıflıyor → LONG kapatılıyor."
                ),
                "indicators": inds,
            }
        if ema_f > ema_s and rsi < 25:
            return {
                "signal": "CLOSE_SHORT",
                "thought": (
                    f"⚠️ SHORT KAPAT\n"
                    f"EMA{fast} EMA{slow} üstünde ve RSI {rsi:.1f} < 25 "
                    f"(aşırı satım). Dip yakın → SHORT kapatılıyor."
                ),
                "indicators": inds,
            }

        # HOLD — neden?
        parts = []
        if not cross_up and not cross_down:
            parts.append(f"{ema_trend} (spread {ema_spread:+.3f}%)")
        if rsi > rsi_hi:
            parts.append(f"RSI {rsi:.1f} aşırı alım bölgesinde → LONG bekleniyor")
        elif rsi < rsi_lo:
            parts.append(f"RSI {rsi:.1f} aşırı satım bölgesinde → SHORT bekleniyor")
        else:
            parts.append(f"RSI {rsi:.1f} nötr bölgede")
        if ema_f > ema_s:
            dist = abs(ema_f - p_ema_f) / price * 100
            parts.append(
                f"Cross yok: EMA{fast} üstte ama kesişme henüz olmadı "
                f"(yaklaşma hızı: {dist:.4f}%/mum)"
            )
        return {
            "signal": "HOLD",
            "thought": "⏸ BEKLE\n" + " · ".join(parts),
            "indicators": inds,
        }

    # ── RSI_MEAN_REVERT ───────────────────────────────────────────────────────
    elif strategy == "RSI_MEAN_REVERT":
        period = cfg.get("rsi_period", 14)
        rsi    = _rsi(closes, period)
        rsi_hi = cfg.get("rsi_overbought", 70)
        rsi_lo = cfg.get("rsi_oversold",   30)

        if rsi is None:
            return NOT_ENOUGH

        inds = {"RSI": round(rsi, 1), "RSI Aşırı Al": rsi_hi, "RSI Aşırı Sat": rsi_lo}

        if rsi < rsi_lo:
            return {"signal": "LONG",
                    "thought": f"🟢 LONG — RSI {rsi:.1f} < {rsi_lo} (aşırı satım). Dip fırsatı.",
                    "indicators": inds}
        if rsi > rsi_hi:
            return {"signal": "SHORT",
                    "thought": f"🔴 SHORT — RSI {rsi:.1f} > {rsi_hi} (aşırı alım). Zirve fırsatı.",
                    "indicators": inds}
        if rsi > 55:
            return {"signal": "CLOSE_LONG",
                    "thought": f"⚠️ LONG KAPAT — RSI {rsi:.1f} > 55, güç kaybediyor.",
                    "indicators": inds}
        if rsi < 45:
            return {"signal": "CLOSE_SHORT",
                    "thought": f"⚠️ SHORT KAPAT — RSI {rsi:.1f} < 45, güç kaybediyor.",
                    "indicators": inds}

        bar = "▓" * int(rsi / 10) + "░" * (10 - int(rsi / 10))
        return {
            "signal": "HOLD",
            "thought": (f"⏸ BEKLE — RSI {rsi:.1f} [{bar}] nötr bölgede "
                        f"({rsi_lo}–{rsi_hi}). Kenar bölge bekleniyor."),
            "indicators": inds,
        }

    # ── BREAKOUT ──────────────────────────────────────────────────────────────
    elif strategy == "BREAKOUT":
        period  = cfg.get("breakout_period", 20)
        if len(closes) < period:
            return NOT_ENOUGH
        recent  = closes[-period:]
        highest = max(recent[:-1])
        lowest  = min(recent[:-1])
        cur     = closes[-1]
        dist_hi = round((cur / highest - 1) * 100, 3)
        dist_lo = round((cur / lowest  - 1) * 100, 3)

        inds = {
            f"{period}p High": round(highest, 4),
            f"{period}p Low":  round(lowest,  4),
            "Fiyat/High":      f"{dist_hi:+.3f}%",
            "Fiyat/Low":       f"{dist_lo:+.3f}%",
        }

        if cur > highest:
            return {"signal": "LONG",
                    "thought": (f"🟢 LONG — Fiyat ({cur:.2f}) {period} mumun "
                                f"zirvesini ({highest:.2f}) kırdı! Kırılım onaylandı."),
                    "indicators": inds}
        if cur < lowest:
            return {"signal": "SHORT",
                    "thought": (f"🔴 SHORT — Fiyat ({cur:.2f}) {period} mumun "
                                f"tabanını ({lowest:.2f}) kırdı! Kırılım onaylandı."),
                    "indicators": inds}
        return {
            "signal": "HOLD",
            "thought": (f"⏸ BEKLE — Fiyat range içinde. "
                        f"Zirveye {dist_hi:+.3f}% · Tabana {dist_lo:+.3f}%. "
                        f"Kırılım bekleniyor."),
            "indicators": inds,
        }

    return {"signal": "HOLD", "thought": "⏸ Bilinmeyen strateji.",
            "indicators": {}}


# ══════════════════════════════════════════════════════════════════════════════
# BOT WORKER
# ══════════════════════════════════════════════════════════════════════════════

# ccxt timeframe → saniye
_TF_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900,
    "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400,
}


class BotWorker:
    def __init__(self, api_key: str, api_secret: str, cfg: dict):
        self.api_key    = api_key
        self.api_secret = api_secret
        self.cfg        = cfg
        self.state      = _bot_state
        self._stop_evt  = threading.Event()
        self._ex: Optional["ccxt.binanceusdm"] = None
        self._last_trade_ts  = 0.0
        self._last_candle_ts = 0      # son okunan kapanmış mum zaman damgası (ms)

    def stop(self):
        self._stop_evt.set()
        self.state.set(running=False, status="Durduruldu")
        self.state.add_log("INFO", "⏹ Bot durduruldu.")

    # ── Ana döngü ─────────────────────────────────────────────────────────────
    def run(self):
        state = self.state
        cfg   = self.cfg
        sym   = cfg["symbol"].upper().replace("/", "")
        # ccxt sembol formatı: BTC/USDT
        ccxt_sym = sym[:-4] + "/USDT" if sym.endswith("USDT") else sym
        tf      = cfg.get("timeframe", "5m")
        dry     = cfg.get("dry_run", True)
        tf_sec  = _TF_SECONDS.get(tf, 300)
        # Polling: mum kapanma periyodunun 1/6'sı kadar bekle ama min 5s
        poll_s  = max(tf_sec // 6, 5)

        state.set(running=True, status="Başlatılıyor...",
                  symbol=sym, strategy=cfg.get("strategy", "EMA_CROSS"), error="")
        state.add_log("INFO",
            f"[TESTNET·ccxt] {'[DRY]' if dry else '[CANLI]'} "
            f"{ccxt_sym} · {tf} · poll:{poll_s}s")

        # ── 1. Exchange bağlantısı ────────────────────────────────────────────
        if not HAS_CCXT:
            state.set(running=False, status="Hata",
                      error="ccxt yüklü değil: pip install ccxt")
            state.add_log("ERROR", "ccxt yüklü değil.")
            return

        try:
            self._ex = _make_exchange(self.api_key, self.api_secret)
        except Exception as e:
            state.set(running=False, status="Hata", error=str(e))
            state.add_log("ERROR", f"Exchange oluşturulamadı: {e}")
            return

        # ── 2. Bağlantı testi + bakiye ────────────────────────────────────────
        try:
            balance = self._ex.fetch_balance()
            equity  = float(balance.get("USDT", {}).get("free", 0))
            state.set(equity=equity)
            state.add_log("INFO",
                f"✅ ccxt testnet bağlandı. Bakiye: ${equity:,.2f} USDT")
            state.add_log("INFO",
                f"Sandbox URL: {self._ex.urls.get('api',{}).get('fapiPublic','?')}")
            # Bakiye uyarısı
            if equity < 10:
                state.add_log("WARN",
                    f"⚠️ Bakiye çok düşük (${equity:.2f}). "
                    f"testnet.binancefuture.com → Assets → "
                    f"'Transfer' ile USDT ekle veya yeni hesap oluştur.")
        except ccxt.AuthenticationError as e:
            hint = ("API Key veya Secret hatalı. "
                    "testnet.binancefuture.com → Profil → API Key → "
                    "Generate HMAC_SHA256 Key ile yeni key oluştur.")
            state.set(running=False, status="Hata",
                      error=f"AuthError: {e} → {hint}")
            state.add_log("ERROR", f"Kimlik doğrulama hatası: {e}")
            state.add_log("WARN",  hint)
            return
        except ccxt.NetworkError as e:
            hint = ("Ağ hatası. İnternet bağlantısını veya "
                    "testnet.binancefuture.com erişimini kontrol et.")
            state.set(running=False, status="Hata", error=f"NetworkError: {e}")
            state.add_log("ERROR", f"Ağ hatası: {e}")
            state.add_log("WARN",  hint)
            return
        except Exception as e:
            state.set(running=False, status="Hata", error=str(e))
            state.add_log("ERROR", f"Bakiye hatası: {e}")
            return

        # ── Market bilgisi: min qty, step size ───────────────────────────────
        self._min_qty  = 0.001
        self._step_dec = 3
        try:
            markets = self._ex.load_markets()
            mkt = markets.get(ccxt_sym, {})
            limits = mkt.get("limits", {}).get("amount", {})
            prec   = mkt.get("precision", {})
            self._min_qty  = float(limits.get("min", 0.001))
            # ccxt precision: integer = decimal places
            p = prec.get("amount", 3)
            self._step_dec = int(p) if isinstance(p, (int, float)) else 3
            state.add_log("INFO",
                f"Market bilgisi: min_qty={self._min_qty}, "
                f"decimal={self._step_dec}")
        except Exception as e:
            state.add_log("WARN", f"Market bilgisi alınamadı, default: {e}")

        # ── 3. Position mode + Kaldıraç ─────────────────────────────────────
        lev = int(cfg.get("leverage", 1))

        # Önce One-Way mode'a al (hedge mode kapalı → positionSide=BOTH çalışır)
        try:
            self._ex.fapiPrivatePostPositionSideDual(params={"dualSidePosition": "false"})
            state.add_log("INFO", "Pozisyon modu: One-Way ✅")
        except Exception as e:
            # Zaten one-way moddaysa hata verir — yoksay
            if "No need to change position side" in str(e) or "-4059" in str(e):
                state.add_log("INFO", "Pozisyon modu: One-Way (zaten ayarlı) ✅")
            else:
                state.add_log("WARN", f"Position mode ayarlanamadı: {e}")

        # Kaldıraç
        try:
            self._ex.set_leverage(lev, ccxt_sym, params={"marginType": "CROSSED"})
            state.add_log("INFO", f"Kaldıraç: {lev}× ✅")
        except Exception as e:
            err_str = str(e)
            if "No need to change" in err_str or "-4046" in err_str or "-4047" in err_str:
                state.add_log("INFO", f"Kaldıraç: {lev}× (zaten ayarlı) ✅")
            else:
                state.add_log("WARN", f"Kaldıraç ayarlanamadı: {e} — devam ediliyor")

        # ── 4. Geçmiş kline (1 kez) ───────────────────────────────────────────
        try:
            ohlcv = self._ex.fetch_ohlcv(ccxt_sym, tf, limit=300)
            for row in ohlcv:
                state.candles.append({
                    "t": row[0], "o": row[1], "h": row[2],
                    "l": row[3], "c": row[4], "v": row[5],
                })
            if ohlcv:
                self._last_candle_ts = ohlcv[-1][0]
                state.last_price     = float(ohlcv[-1][4])
            state.add_log("INFO", f"{len(ohlcv)} geçmiş mum yüklendi.")
        except Exception as e:
            state.add_log("WARN", f"Geçmiş kline alınamadı: {e}")

        state.set(status=f"✅ Çalışıyor [TESTNET·ccxt] — {ccxt_sym} · {tf}")
        state.add_log("INFO",
            f"Polling modu: {poll_s}s arayla veri çekiliyor. "
            f"Mum kapandığında strateji çalıştırılır.")

        # ── 5. Ana polling döngüsü ────────────────────────────────────────────
        while not self._stop_evt.is_set():
            try:
                self._tick(ccxt_sym, tf, cfg, dry)
            except ccxt.RateLimitExceeded:
                state.add_log("WARN", "Rate limit — 30s bekleniyor")
                time.sleep(30)
            except ccxt.NetworkError as e:
                state.add_log("WARN", f"Ağ hatası, 10s sonra tekrar: {e}")
                time.sleep(10)
            except Exception as e:
                state.add_log("ERROR", f"Tick hatası: {e}")

            # Stop event'i kontrol ederek bekle
            self._stop_evt.wait(timeout=poll_s)

    # ── Tek tick ──────────────────────────────────────────────────────────────
    def _tick(self, ccxt_sym: str, tf: str, cfg: dict, dry: bool):
        state = self.state

        # En son 2 mumu çek (yeni kapanmış mumu yakala)
        ohlcv = self._ex.fetch_ohlcv(ccxt_sym, tf, limit=3)
        if not ohlcv:
            return

        last_closed = ohlcv[-2]   # son kapanmış mum (index -1 hâlâ açık)
        live_candle = ohlcv[-1]   # canlı mum (fiyat bilgisi için)

        state.last_price = float(live_candle[4])

        # Unrealized PnL güncelle
        ot = state.open_trade
        if ot:
            cur = state.last_price
            if ot["side"] == "LONG":
                upnl = (cur - ot["entry"]) / ot["entry"] * ot["notional"]
            else:
                upnl = (ot["entry"] - cur) / ot["entry"] * ot["notional"]
            state.set(current_pnl=round(upnl, 4))

        # Yeni kapanmış mum var mı?
        new_ts = last_closed[0]
        if new_ts <= self._last_candle_ts:
            return   # henüz yeni mum yok

        self._last_candle_ts = new_ts
        state.candles.append({
            "t": last_closed[0], "o": last_closed[1], "h": last_closed[2],
            "l": last_closed[3], "c": last_closed[4], "v": last_closed[5],
        })
        candles_list = list(state.candles)

        # Cooldown
        now_ts   = time.time()
        cooldown = cfg.get("trade_cooldown_s", 60)
        cooldown_remaining = cooldown - (now_ts - self._last_trade_ts)

        # ── Strateji değerlendirmesi + düşünce ───────────────────────────────
        result     = evaluate_signal_with_thoughts(candles_list, cfg)
        signal     = result["signal"]
        thought    = result["thought"]
        indicators = result["indicators"]

        # Cooldown aktifse düşünceye not ekle
        if cooldown_remaining > 0 and state.open_trade is None:
            thought += (f"\n⏱ Cooldown: {cooldown_remaining:.0f}s daha beklenecek "
                        f"(tekrar işlem için)")

        state.set(thought=thought, indicators=indicators)

        # Cooldown aktifse işlem yapma ama düşünceyi yazmaya devam et
        if cooldown_remaining > 0:
            return

        if state.open_trade is None:
            if signal in ("LONG", "SHORT"):
                self._open_pos(ccxt_sym, signal, state.last_price, cfg, dry)
                self._last_trade_ts = now_ts
        else:
            ot     = state.open_trade
            close  = False
            reason = ""
            if (ot["side"] == "LONG"  and signal == "CLOSE_LONG") or \
               (ot["side"] == "SHORT" and signal == "CLOSE_SHORT"):
                close = True; reason = "Sinyal"
            if ot["side"] == "LONG":
                if   state.last_price >= ot["tp"]: close = True; reason = "TP"
                elif state.last_price <= ot["sl"]: close = True; reason = "SL"
            else:
                if   state.last_price <= ot["tp"]: close = True; reason = "TP"
                elif state.last_price >= ot["sl"]: close = True; reason = "SL"
            if close:
                self._close_pos(ccxt_sym, state.last_price, reason, cfg, dry)
                self._last_trade_ts = now_ts

    # ── Pozisyon aç ───────────────────────────────────────────────────────────
    def _open_pos(self, ccxt_sym: str, side: str, price: float,
                  cfg: dict, dry: bool):
        state    = self.state
        eq       = state.equity or 0.0
        rp       = cfg.get("risk_pct",  1.0)
        lev      = float(cfg.get("leverage", 1))
        tp_pct   = cfg.get("tp_pct",    1.5)
        sl_pct   = cfg.get("sl_pct",    1.0)
        min_qty  = getattr(self, "_min_qty",  0.001)
        step_dec = getattr(self, "_step_dec", 3)

        # Notional ve qty hesapla
        notional = eq * (rp / 100) * lev
        raw_qty  = notional / price if price > 0 else 0
        qty      = max(round(raw_qty, step_dec), min_qty)

        # Bakiye + minimum miktar kontrolü
        if eq < 10:
            state.add_log("ERROR",
                f"Bakiye çok düşük: ${eq:.2f} USDT. "
                f"testnet.binancefuture.com adresinden bakiye ekle.")
            return

        min_notional = 5.0   # Binance Futures min 5 USDT notional
        if qty * price < min_notional:
            qty = max(round(min_notional / price, step_dec), min_qty)
            notional = qty * price
            state.add_log("WARN",
                f"Qty minimum notional için artırıldı: {qty} "
                f"(${notional:.2f})")

        tp = round(price * (1 + tp_pct/100) if side == "LONG"
                   else price * (1 - tp_pct/100), 6)
        sl = round(price * (1 - sl_pct/100) if side == "LONG"
                   else price * (1 + sl_pct/100), 6)

        if not dry:
            try:
                b_side = "buy" if side == "LONG" else "sell"
                # One-Way mode: positionSide=BOTH, reduceOnly=False
                self._ex.create_market_order(
                    ccxt_sym, b_side, qty,
                    params={"positionSide": "BOTH"},
                )
            except ccxt.InsufficientFunds as e:
                state.add_log("ERROR", f"Yetersiz bakiye: {e}"); return
            except ccxt.InvalidOrder as e:
                state.add_log("ERROR", f"Geçersiz order: {e}"); return
            except Exception as e:
                state.add_log("ERROR", f"Order hatası: {e}"); return

        trade = {
            "id":        int(time.time() * 1000),
            "side":      side,
            "entry":     price,
            "qty":       qty,
            "notional":  round(notional, 4),
            "tp":        tp,
            "sl":        sl,
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "dry_run":   dry,
        }
        state.set(open_trade=trade)
        emoji = "🟢" if side == "LONG" else "🔴"
        state.add_log(side,
            f"{'[DRY] ' if dry else ''}{emoji} {side} @ ${price:,.4f} | "
            f"qty={qty} | TP:{tp:,.4f} | SL:{sl:,.4f} | "
            f"Notional:${notional:,.2f}")

    # ── Pozisyon kapat ────────────────────────────────────────────────────────
    def _close_pos(self, ccxt_sym: str, price: float, reason: str,
                   cfg: dict, dry: bool):
        state = self.state
        ot    = state.open_trade
        if not ot:
            return

        pnl    = ((price - ot["entry"]) / ot["entry"] * ot["notional"]
                  if ot["side"] == "LONG"
                  else (ot["entry"] - price) / ot["entry"] * ot["notional"])
        pnl    = round(pnl, 4)
        # Kapatmak için ters taraf
        b_side = "sell" if ot["side"] == "LONG" else "buy"

        if not dry:
            try:
                # One-Way mode kapatma:
                # positionSide=BOTH + reduceOnly=True
                self._ex.create_market_order(
                    ccxt_sym, b_side, ot["qty"],
                    params={
                        "positionSide": "BOTH",
                        "reduceOnly":   True,
                    }
                )
            except ccxt.InvalidOrder as e:
                # reduceOnly reddedilirse parametresiz dene
                state.add_log("WARN", f"reduceOnly reddedildi, tekrar: {e}")
                try:
                    self._ex.create_market_order(
                        ccxt_sym, b_side, ot["qty"],
                        params={"positionSide": "BOTH"},
                    )
                except Exception as e2:
                    state.add_log("ERROR", f"Kapat hatası: {e2}")
            except Exception as e:
                state.add_log("ERROR", f"Kapat hatası: {e}")

        state.add_trade({**ot,
            "exit":      price,
            "pnl":       pnl,
            "reason":    reason,
            "closed_at": datetime.now(timezone.utc).isoformat(),
        })
        state.set(
            open_trade=None,
            current_pnl=0.0,
            total_pnl  =round(state.total_pnl + pnl, 4),
            win_count  =state.win_count  + (1 if pnl > 0 else 0),
            loss_count =state.loss_count + (1 if pnl <= 0 else 0),
        )
        emoji = "✅" if pnl > 0 else "❌"
        state.add_log("SELL",
            f"{'[DRY] ' if dry else ''}{emoji} KAPANDI ({reason}) "
            f"@ ${price:,.4f} | PnL:{'+'if pnl>=0 else ''}{pnl:.4f}$")


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

_worker_thread:   Optional[threading.Thread] = None
_worker_instance: Optional[BotWorker]        = None


def start_bot(api_key: str, api_secret: str, cfg: dict) -> bool:
    global _worker_thread, _worker_instance
    if not HAS_CCXT:
        _bot_state.add_log("ERROR", "ccxt yüklü değil: pip install ccxt")
        return False
    if _bot_state.running:
        return False
    # State sıfırla
    _bot_state.set(
        trades=[], open_trade=None, equity=0.0,
        total_pnl=0.0, current_pnl=0.0,
        win_count=0, loss_count=0,
        last_price=0.0, error="",
        thought="Bot başlatılıyor...",
        indicators={},
    )
    _bot_state.log.clear()
    _worker_instance = BotWorker(api_key, api_secret, cfg)
    _worker_thread   = threading.Thread(
        target=_worker_instance.run, daemon=True, name="TradeBot"
    )
    _worker_thread.start()
    return True


def stop_bot():
    global _worker_instance
    if _worker_instance:
        _worker_instance.stop()
    _bot_state.set(running=False, status="Durduruldu", open_trade=None)
