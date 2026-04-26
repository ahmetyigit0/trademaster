"""
TradeBot Engine — ccxt + OKX Demo Trading
==========================================
OKX Demo Trading: gerçek fiyatlar, sanal para.
TR'den erişilebilir. ccxt tam destekli.

Demo Trading URL: https://www.okx.com (sandbox flag ile demo endpoint'e gider)
API key: OKX hesabından demo trading key alınır.
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
        self.thought     = "Bot henüz başlatılmadı."
        self.indicators  = {}

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
# CCXT OKX FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def _make_exchange(api_key: str, api_secret: str,
                   passphrase: str = "") -> "ccxt.okx":
    """
    OKX Demo Trading.
    set_sandbox_mode(True) → demo endpoint'e yönlendirir.

    API key nasıl alınır:
    1. okx.com hesabına giriş → API Management
    2. Create API → Demo Trading seç
    3. Key + Secret + Passphrase kopyala
    """
    ex = ccxt.okx({
        "apiKey":   api_key,
        "secret":   api_secret,
        "password": passphrase,   # OKX passphrase zorunlu
        "options":  {
            "defaultType": "swap",   # USDT-M Perpetual Futures
        },
        "enableRateLimit": True,
        "timeout":  15000,
    })
    ex.set_sandbox_mode(True)   # demo-aws.okx.com endpoint'ine yönlendirir
    return ex


# ── OKX sembol formatı ────────────────────────────────────────────────────────
def _to_okx_sym(symbol: str) -> str:
    """BTCUSDT → BTC/USDT:USDT (ccxt OKX swap format)"""
    symbol = symbol.upper().replace("/", "")
    if symbol.endswith("USDT"):
        base = symbol[:-4]
        return f"{base}/USDT:USDT"
    return symbol


# ccxt timeframe → saniye
_TF_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900,
    "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400,
}


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
    gains  = [max(d, 0)      for d in deltas[-period:]]
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
    return evaluate_signal_with_thoughts(candles_list, cfg)["signal"]


def evaluate_signal_with_thoughts(candles_list: list, cfg: dict) -> dict:
    NOT_ENOUGH = {
        "signal": "HOLD",
        "thought": f"⏳ Yeterli mum yok ({len(candles_list)}/30). Geçmiş veri bekleniyor...",
        "indicators": {},
    }
    if len(candles_list) < 30:
        return NOT_ENOUGH

    strategy = cfg.get("strategy", "EMA_CROSS")
    closes   = [c["c"] for c in candles_list]
    price    = closes[-1]

    if strategy == "EMA_CROSS":
        fast, slow = cfg.get("ema_fast", 9), cfg.get("ema_slow", 21)
        rsi_hi = cfg.get("rsi_overbought", 70)
        rsi_lo = cfg.get("rsi_oversold",   30)
        ema_f  = _ema(closes, fast)
        ema_s  = _ema(closes, slow)
        rsi    = _rsi(closes, 14)
        if None in (ema_f, ema_s, rsi):
            return NOT_ENOUGH
        prev    = closes[:-1]
        p_ema_f = _ema(prev, fast)
        p_ema_s = _ema(prev, slow)
        if None in (p_ema_f, p_ema_s):
            return NOT_ENOUGH

        cross_up   = p_ema_f <= p_ema_s and ema_f > ema_s
        cross_down = p_ema_f >= p_ema_s and ema_f < ema_s
        spread     = round(((ema_f - ema_s) / price) * 100, 3)
        trend      = "📈 EMA hızlı üstte" if ema_f > ema_s else "📉 EMA hızlı altta"
        inds = {f"EMA{fast}": round(ema_f, 4), f"EMA{slow}": round(ema_s, 4),
                "RSI": round(rsi, 1), "Spread": f"{spread:+.3f}%", "Trend": trend}

        if cross_up and rsi < rsi_hi:
            return {"signal": "LONG", "indicators": inds,
                    "thought": (f"🟢 LONG SİNYAL!\nEMA{fast} ({ema_f:.2f}) EMA{slow} ({ema_s:.2f}) "
                                f"üzerine çıktı (Golden Cross). RSI {rsi:.1f} < {rsi_hi} → LONG açılıyor.")}
        if cross_down and rsi > rsi_lo:
            return {"signal": "SHORT", "indicators": inds,
                    "thought": (f"🔴 SHORT SİNYAL!\nEMA{fast} ({ema_f:.2f}) EMA{slow} ({ema_s:.2f}) "
                                f"altına geçti (Death Cross). RSI {rsi:.1f} > {rsi_lo} → SHORT açılıyor.")}
        if ema_f < ema_s and rsi > 75:
            return {"signal": "CLOSE_LONG", "indicators": inds,
                    "thought": f"⚠️ LONG KAPAT\nEMA{fast} altta ve RSI {rsi:.1f} > 75."}
        if ema_f > ema_s and rsi < 25:
            return {"signal": "CLOSE_SHORT", "indicators": inds,
                    "thought": f"⚠️ SHORT KAPAT\nEMA{fast} üstte ve RSI {rsi:.1f} < 25."}

        parts = [trend, f"RSI {rsi:.1f} nötr ({rsi_lo}–{rsi_hi})"]
        if ema_f > ema_s:
            parts.append(f"Cross henüz yok (spread {spread:+.3f}%)")
        return {"signal": "HOLD", "indicators": inds,
                "thought": "⏸ BEKLE\n" + " · ".join(parts)}

    elif strategy == "RSI_MEAN_REVERT":
        rsi    = _rsi(closes, cfg.get("rsi_period", 14))
        rsi_hi = cfg.get("rsi_overbought", 70)
        rsi_lo = cfg.get("rsi_oversold",   30)
        if rsi is None:
            return NOT_ENOUGH
        inds = {"RSI": round(rsi, 1)}
        if rsi < rsi_lo:
            return {"signal": "LONG",  "indicators": inds,
                    "thought": f"🟢 LONG — RSI {rsi:.1f} < {rsi_lo} (aşırı satım)."}
        if rsi > rsi_hi:
            return {"signal": "SHORT", "indicators": inds,
                    "thought": f"🔴 SHORT — RSI {rsi:.1f} > {rsi_hi} (aşırı alım)."}
        if rsi > 55:
            return {"signal": "CLOSE_LONG",  "indicators": inds,
                    "thought": f"⚠️ LONG KAPAT — RSI {rsi:.1f} > 55."}
        if rsi < 45:
            return {"signal": "CLOSE_SHORT", "indicators": inds,
                    "thought": f"⚠️ SHORT KAPAT — RSI {rsi:.1f} < 45."}
        bar = "▓" * int(rsi/10) + "░" * (10-int(rsi/10))
        return {"signal": "HOLD", "indicators": inds,
                "thought": f"⏸ BEKLE — RSI {rsi:.1f} [{bar}] nötr bölge."}

    elif strategy == "BREAKOUT":
        period  = cfg.get("breakout_period", 20)
        if len(closes) < period:
            return NOT_ENOUGH
        recent  = closes[-period:]
        highest = max(recent[:-1])
        lowest  = min(recent[:-1])
        cur     = closes[-1]
        inds = {f"{period}p High": round(highest, 4), f"{period}p Low": round(lowest, 4),
                "Fiyat/High": f"{(cur/highest-1)*100:+.3f}%",
                "Fiyat/Low":  f"{(cur/lowest -1)*100:+.3f}%"}
        if cur > highest:
            return {"signal": "LONG",  "indicators": inds,
                    "thought": f"🟢 LONG — {period} mumun zirvesi ({highest:.2f}) kırıldı!"}
        if cur < lowest:
            return {"signal": "SHORT", "indicators": inds,
                    "thought": f"🔴 SHORT — {period} mumun tabanı ({lowest:.2f}) kırıldı!"}
        return {"signal": "HOLD", "indicators": inds,
                "thought": (f"⏸ BEKLE — Range içinde. "
                            f"Zirveye {(cur/highest-1)*100:+.3f}% · "
                            f"Tabana {(cur/lowest-1)*100:+.3f}%")}

    return {"signal": "HOLD", "thought": "⏸ Bilinmeyen strateji.", "indicators": {}}


# ══════════════════════════════════════════════════════════════════════════════
# BOT WORKER
# ══════════════════════════════════════════════════════════════════════════════

class BotWorker:
    def __init__(self, api_key: str, api_secret: str,
                 passphrase: str, cfg: dict):
        self.api_key     = api_key
        self.api_secret  = api_secret
        self.passphrase  = passphrase
        self.cfg         = cfg
        self.state       = _bot_state
        self._stop_evt   = threading.Event()
        self._ex: Optional["ccxt.okx"] = None
        self._last_trade_ts  = 0.0
        self._last_candle_ts = 0
        self._min_qty        = 1.0
        self._step_dec       = 0

    def stop(self):
        self._stop_evt.set()
        self.state.set(running=False, status="Durduruldu")
        self.state.add_log("INFO", "⏹ Bot durduruldu.")

    def run(self):
        state    = self.state
        cfg      = self.cfg
        sym_raw  = cfg["symbol"].upper().replace("/", "")
        okx_sym  = _to_okx_sym(sym_raw)
        tf       = cfg.get("timeframe", "5m")
        dry      = cfg.get("dry_run", True)
        tf_sec   = _TF_SECONDS.get(tf, 300)
        poll_s   = max(tf_sec // 6, 5)

        state.set(running=True, status="Başlatılıyor...",
                  symbol=sym_raw, strategy=cfg.get("strategy", "EMA_CROSS"),
                  error="", thought="Bağlanıyor...", indicators={})
        state.add_log("INFO",
            f"[OKX DEMO·ccxt] {'[DRY]' if dry else '[CANLI]'} "
            f"{okx_sym} · {tf} · poll:{poll_s}s")

        if not HAS_CCXT:
            state.set(running=False, status="Hata", error="ccxt yüklü değil.")
            return

        # ── Bağlantı ─────────────────────────────────────────────────────────
        try:
            self._ex = _make_exchange(self.api_key, self.api_secret, self.passphrase)
        except Exception as e:
            state.set(running=False, status="Hata", error=str(e))
            state.add_log("ERROR", f"Exchange hatası: {e}")
            return

        # Bakiye
        try:
            balance = self._ex.fetch_balance()
            usdt    = (balance.get("USDT") or
                       balance.get("usdt") or {})
            equity  = float(usdt.get("free") or usdt.get("total") or 0)
            state.set(equity=equity)
            state.add_log("INFO", f"✅ OKX Demo bağlandı. Bakiye: ${equity:,.2f} USDT")
            demo_url = self._ex.urls.get("api", {})
            if isinstance(demo_url, dict):
                state.add_log("INFO",
                    f"URL: {list(demo_url.values())[0] if demo_url else '?'}")
            if equity < 1:
                state.add_log("WARN",
                    "⚠️ Bakiye çok düşük. OKX demo hesabına giriş yap → "
                    "Varlıklar → Fonları aktar.")
        except ccxt.AuthenticationError as e:
            hint = ("API Key/Secret/Passphrase hatalı. "
                    "OKX → API Management → Demo Trading key oluştur.")
            state.set(running=False, status="Hata", error=hint)
            state.add_log("ERROR", f"Auth hatası: {e}")
            state.add_log("WARN", hint)
            return
        except ccxt.NetworkError as e:
            state.set(running=False, status="Hata", error=f"Ağ hatası: {e}")
            state.add_log("ERROR", f"Ağ hatası: {e}")
            return
        except Exception as e:
            state.set(running=False, status="Hata", error=str(e))
            state.add_log("ERROR", f"Bakiye hatası: {e}")
            return

        # Kaldıraç
        try:
            lev = int(cfg.get("leverage", 1))
            self._ex.set_leverage(lev, okx_sym)
            state.add_log("INFO", f"Kaldıraç: {lev}× ✅")
        except Exception as e:
            state.add_log("WARN", f"Kaldıraç ayarlanamadı: {e}")

        # Market bilgisi
        try:
            markets = self._ex.load_markets()
            mkt     = markets.get(okx_sym, {})
            lim     = mkt.get("limits", {}).get("amount", {})
            prec    = mkt.get("precision", {})
            self._min_qty  = float(lim.get("min", 1))
            p = prec.get("amount", 0)
            self._step_dec = int(p) if isinstance(p, (int, float)) else 0
            state.add_log("INFO",
                f"Market: {okx_sym} min={self._min_qty} dec={self._step_dec}")
        except Exception as e:
            state.add_log("WARN", f"Market bilgisi alınamadı: {e}")

        # Geçmiş kline
        try:
            ohlcv = self._ex.fetch_ohlcv(okx_sym, tf, limit=300)
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
            state.add_log("WARN", f"Geçmiş kline: {e}")

        state.set(status=f"✅ Çalışıyor [OKX DEMO] — {okx_sym} · {tf}")

        # ── Ana polling döngüsü ───────────────────────────────────────────────
        while not self._stop_evt.is_set():
            try:
                self._tick(okx_sym, tf, cfg, dry)
            except ccxt.RateLimitExceeded:
                state.add_log("WARN", "Rate limit — 30s bekleniyor")
                time.sleep(30)
            except ccxt.NetworkError as e:
                state.add_log("WARN", f"Ağ hatası, 10s sonra: {e}")
                time.sleep(10)
            except Exception as e:
                state.add_log("ERROR", f"Tick hatası: {e}")
            self._stop_evt.wait(timeout=poll_s)

    def _tick(self, okx_sym: str, tf: str, cfg: dict, dry: bool):
        state = self.state
        ohlcv = self._ex.fetch_ohlcv(okx_sym, tf, limit=3)
        if not ohlcv:
            return

        last_closed = ohlcv[-2]
        live_candle = ohlcv[-1]
        state.last_price = float(live_candle[4])

        # Unrealized PnL
        ot = state.open_trade
        if ot:
            cur = state.last_price
            upnl = ((cur - ot["entry"]) / ot["entry"] * ot["notional"]
                    if ot["side"] == "LONG"
                    else (ot["entry"] - cur) / ot["entry"] * ot["notional"])
            state.set(current_pnl=round(upnl, 4))

        new_ts = last_closed[0]
        if new_ts <= self._last_candle_ts:
            return

        self._last_candle_ts = new_ts
        state.candles.append({
            "t": last_closed[0], "o": last_closed[1], "h": last_closed[2],
            "l": last_closed[3], "c": last_closed[4], "v": last_closed[5],
        })
        candles_list = list(state.candles)

        now_ts   = time.time()
        cooldown = cfg.get("trade_cooldown_s", 60)
        cd_rem   = cooldown - (now_ts - self._last_trade_ts)

        result     = evaluate_signal_with_thoughts(candles_list, cfg)
        signal     = result["signal"]
        thought    = result["thought"]
        indicators = result["indicators"]

        if cd_rem > 0 and state.open_trade is None:
            thought += f"\n⏱ Cooldown: {cd_rem:.0f}s kaldı"

        state.set(thought=thought, indicators=indicators)

        if cd_rem > 0:
            return

        if state.open_trade is None:
            if signal in ("LONG", "SHORT"):
                self._open_pos(okx_sym, signal, state.last_price, cfg, dry)
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
                self._close_pos(okx_sym, state.last_price, reason, cfg, dry)
                self._last_trade_ts = now_ts

    def _open_pos(self, okx_sym: str, side: str, price: float,
                  cfg: dict, dry: bool):
        state    = self.state
        eq       = state.equity or 100.0
        rp       = cfg.get("risk_pct",  1.0)
        lev      = float(cfg.get("leverage", 1))
        tp_pct   = cfg.get("tp_pct",    1.5)
        sl_pct   = cfg.get("sl_pct",    1.0)

        notional = eq * (rp / 100) * lev
        raw_qty  = notional / price
        qty      = max(round(raw_qty, self._step_dec), self._min_qty)

        # OKX kontrat büyüklüğü — çoğu pair 0.01 BTC / kontrat
        # ccxt bunu otomatik handle eder ama min_qty=1 kontrat olabilir
        if qty < self._min_qty:
            qty = self._min_qty

        tp = round(price * (1 + tp_pct/100) if side == "LONG"
                   else price * (1 - tp_pct/100), 6)
        sl = round(price * (1 - sl_pct/100) if side == "LONG"
                   else price * (1 + sl_pct/100), 6)

        if not dry:
            try:
                b_side = "buy" if side == "LONG" else "sell"
                self._ex.create_market_order(
                    okx_sym, b_side, qty,
                    params={"tdMode": "cross"},   # OKX cross margin
                )
            except ccxt.InsufficientFunds as e:
                state.add_log("ERROR", f"Yetersiz bakiye: {e}"); return
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
            f"qty={qty} | TP:{tp:,.4f} | SL:{sl:,.4f} | ${notional:,.2f}")

    def _close_pos(self, okx_sym: str, price: float, reason: str,
                   cfg: dict, dry: bool):
        state = self.state
        ot    = state.open_trade
        if not ot:
            return

        pnl    = ((price - ot["entry"]) / ot["entry"] * ot["notional"]
                  if ot["side"] == "LONG"
                  else (ot["entry"] - price) / ot["entry"] * ot["notional"])
        pnl    = round(pnl, 4)
        b_side = "sell" if ot["side"] == "LONG" else "buy"

        if not dry:
            try:
                self._ex.create_market_order(
                    okx_sym, b_side, ot["qty"],
                    params={"tdMode": "cross", "reduceOnly": True},
                )
            except Exception as e:
                state.add_log("ERROR", f"Kapat hatası: {e}")

        state.add_trade({**ot, "exit": price, "pnl": pnl,
                         "reason": reason,
                         "closed_at": datetime.now(timezone.utc).isoformat()})
        state.set(
            open_trade=None, current_pnl=0.0,
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
    _bot_state.set(
        trades=[], open_trade=None, equity=0.0,
        total_pnl=0.0, current_pnl=0.0,
        win_count=0, loss_count=0,
        last_price=0.0, error="",
        thought="Bot başlatılıyor...", indicators={},
    )
    _bot_state.log.clear()
    passphrase = cfg.get("passphrase", "")
    _worker_instance = BotWorker(api_key, api_secret, passphrase, cfg)
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
