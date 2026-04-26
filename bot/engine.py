"""
TradeBot Engine — Binance Futures Testnet
==========================================
python-binance kütüphanesi KULLANILMIYOR.
Tüm API çağrıları doğrudan requests ile testnet endpoint'lerine gider.

Testnet REST  : https://testnet.binancefuture.com/fapi/v1/...
Testnet WS    : wss://stream.binancefuture.com/ws/  (public, auth gereksiz)

Rate Limit Koruması:
- Kline/fiyat verisi: WebSocket (polling yok)
- Order: sadece sinyal gelince + 60s cooldown
- Geçmiş veri: sadece başlangıçta 1 REST çağrısı
"""

import threading
import time
import hmac
import hashlib
import json
import ssl
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from collections import deque
from typing import Optional

try:
    import websocket          # pip install websocket-client
    HAS_WS = True
except ImportError:
    HAS_WS = False

try:
    import requests as _req
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

HAS_BINANCE = HAS_REQUESTS   # artık python-binance gerekmez

MAX_LOG_ROWS = 200

# ── Testnet endpoint'leri ─────────────────────────────────────────────────────
_BASE_REST = "https://testnet.binancefuture.com"
_BASE_WS   = "wss://stream.binancefuture.com/ws"


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
        self.candles     = deque(maxlen=300)
        self.error       = ""

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
            if len(self.trades) > 100:
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
            }


_bot_state = BotState()

def get_state() -> BotState:
    return _bot_state


# ══════════════════════════════════════════════════════════════════════════════
# TESTNET HTTP CLIENT  (python-binance kullanmaz)
# ══════════════════════════════════════════════════════════════════════════════

class TestnetClient:
    """
    Doğrudan requests ile testnet.binancefuture.com'a bağlanır.
    Hiçbir koşulda Binance.com ana URL'ine gitme riski yok.
    """

    def __init__(self, api_key: str, api_secret: str):
        self.key    = api_key
        self.secret = api_secret.encode()

    # ── İmza ─────────────────────────────────────────────────────────────────
    def _sign(self, params: dict) -> str:
        qs  = urllib.parse.urlencode(params)
        sig = hmac.new(self.secret, qs.encode(), hashlib.sha256).hexdigest()
        return sig

    def _ts(self) -> int:
        return int(time.time() * 1000)

    # ── HTTP yardımcıları ─────────────────────────────────────────────────────
    def _get(self, path: str, params: dict = None, signed: bool = False) -> dict:
        params = dict(params or {})
        if signed:
            params["timestamp"] = self._ts()
            params["signature"] = self._sign(params)
        url = f"{_BASE_REST}{path}"
        headers = {"X-MBX-APIKEY": self.key}
        r = _req.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, params: dict = None) -> dict:
        params = dict(params or {})
        params["timestamp"] = self._ts()
        params["signature"] = self._sign(params)
        url = f"{_BASE_REST}{path}"
        headers = {"X-MBX-APIKEY": self.key}
        r = _req.post(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        return r.json()

    # ── Testnet API metotları ─────────────────────────────────────────────────
    def ping(self) -> bool:
        try:
            self._get("/fapi/v1/ping")
            return True
        except Exception:
            return False

    def server_time(self) -> int:
        data = self._get("/fapi/v1/time")
        return data["serverTime"]

    def account_balance(self) -> list:
        return self._get("/fapi/v2/balance", signed=True)

    def exchange_info(self, symbol: str) -> dict:
        data = self._get("/fapi/v1/exchangeInfo", {"symbol": symbol})
        return data

    def klines(self, symbol: str, interval: str, limit: int = 300) -> list:
        return self._get("/fapi/v1/klines", {
            "symbol": symbol, "interval": interval, "limit": limit
        })

    def change_leverage(self, symbol: str, leverage: int) -> dict:
        return self._post("/fapi/v1/leverage", {
            "symbol": symbol, "leverage": leverage
        })

    def new_order(self, symbol: str, side: str, quantity: float,
                  position_side: str = "BOTH") -> dict:
        return self._post("/fapi/v1/order", {
            "symbol":       symbol,
            "side":         side,
            "type":         "MARKET",
            "quantity":     str(quantity),
            "positionSide": position_side,
        })

    def close_order(self, symbol: str, side: str, quantity: float) -> dict:
        return self._post("/fapi/v1/order", {
            "symbol":       symbol,
            "side":         side,
            "type":         "MARKET",
            "quantity":     str(quantity),
            "positionSide": "BOTH",
            "reduceOnly":   "true",
        })


# ══════════════════════════════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def _ema(prices: list, period: int) -> Optional[float]:
    if len(prices) < period:
        return None
    k   = 2 / (period + 1)
    ema = prices[0]
    for p in prices[1:]:
        ema = p * k + ema * (1 - k)
    return ema


def _rsi(prices: list, period: int = 14) -> Optional[float]:
    if len(prices) < period + 1:
        return None
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains  = [max(d, 0) for d in deltas[-period:]]
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
    """Returns: LONG | SHORT | CLOSE_LONG | CLOSE_SHORT | HOLD"""
    strategy = cfg.get("strategy", "EMA_CROSS")
    closes   = [c["c"] for c in candles_list]

    if strategy == "EMA_CROSS":
        fast   = cfg.get("ema_fast", 9)
        slow   = cfg.get("ema_slow", 21)
        rsi_hi = cfg.get("rsi_overbought", 70)
        rsi_lo = cfg.get("rsi_oversold",   30)
        ema_f  = _ema(closes, fast)
        ema_s  = _ema(closes, slow)
        rsi    = _rsi(closes, 14)
        if ema_f is None or ema_s is None or rsi is None:
            return "HOLD"
        prev    = [c["c"] for c in candles_list[:-1]]
        p_ema_f = _ema(prev, fast)
        p_ema_s = _ema(prev, slow)
        if p_ema_f is None or p_ema_s is None:
            return "HOLD"
        if p_ema_f <= p_ema_s and ema_f > ema_s and rsi < rsi_hi: return "LONG"
        if p_ema_f >= p_ema_s and ema_f < ema_s and rsi > rsi_lo: return "SHORT"
        if ema_f < ema_s and rsi > 70: return "CLOSE_LONG"
        if ema_f > ema_s and rsi < 30: return "CLOSE_SHORT"
        return "HOLD"

    elif strategy == "RSI_MEAN_REVERT":
        rsi    = _rsi(closes, cfg.get("rsi_period", 14))
        rsi_hi = cfg.get("rsi_overbought", 70)
        rsi_lo = cfg.get("rsi_oversold",   30)
        if rsi is None: return "HOLD"
        if rsi < rsi_lo: return "LONG"
        if rsi > rsi_hi: return "SHORT"
        if rsi > 55:     return "CLOSE_LONG"
        if rsi < 45:     return "CLOSE_SHORT"
        return "HOLD"

    elif strategy == "BREAKOUT":
        period = cfg.get("breakout_period", 20)
        if len(closes) < period: return "HOLD"
        recent = closes[-period:]
        if closes[-1] > max(recent[:-1]): return "LONG"
        if closes[-1] < min(recent[:-1]): return "SHORT"
        return "HOLD"

    return "HOLD"


# ══════════════════════════════════════════════════════════════════════════════
# BOT WORKER
# ══════════════════════════════════════════════════════════════════════════════

class BotWorker:
    def __init__(self, api_key: str, api_secret: str, cfg: dict):
        self.api_key    = api_key
        self.api_secret = api_secret
        self.cfg        = cfg
        self.state      = _bot_state
        self._stop_evt  = threading.Event()
        self._ws        = None
        self._client: Optional[TestnetClient] = None
        self._last_trade_time = 0.0
        self._step_size = "0.001"

    def stop(self):
        self._stop_evt.set()
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        self.state.set(running=False, status="Durduruldu")
        self.state.add_log("INFO", "Bot durduruldu.")

    def run(self):
        state = self.state
        cfg   = self.cfg
        sym   = cfg["symbol"].upper().replace("/", "")
        tf    = cfg.get("timeframe", "5m")
        dry   = cfg.get("dry_run", True)

        state.set(running=True, status="Başlatılıyor...",
                  symbol=sym, strategy=cfg.get("strategy", "EMA_CROSS"), error="")
        state.add_log("INFO",
            f"Bot başlatıldı [TESTNET] {'[DRY]' if dry else '[CANLI]'}: {sym} · {tf}")

        # ── 1. Testnet REST bağlantısı ────────────────────────────────────────
        self._client = TestnetClient(self.api_key, self.api_secret)

        # Ping
        if not self._client.ping():
            err = "testnet.binancefuture.com'a ulaşılamıyor. İnternet bağlantınızı kontrol edin."
            state.set(running=False, status="Hata", error=err)
            state.add_log("ERROR", err)
            return
        state.add_log("INFO", "✅ Testnet ping OK")

        # Sunucu saati farkını kontrol et (±1000ms)
        try:
            srv_ts = self._client.server_time()
            diff   = abs(srv_ts - int(time.time() * 1000))
            if diff > 5000:
                state.add_log("WARN",
                    f"Sistem saati farkı: {diff}ms — timestamp hatası alabilirsin. "
                    f"Bilgisayar saatini senkronize et.")
        except Exception:
            pass

        # Bakiye
        try:
            balances = self._client.account_balance()
            equity   = next(
                (float(b["availableBalance"]) for b in balances if b["asset"] == "USDT"),
                0.0,
            )
            state.set(equity=equity)
            state.add_log("INFO", f"✅ Bakiye: ${equity:,.2f} USDT (Testnet)")
        except Exception as e:
            err = str(e)
            if "signature" in err.lower() or "-1022" in err:
                hint = "Timestamp/imza hatası — sistem saatini kontrol et."
            elif "-2014" in err or "api-key" in err.lower():
                hint = "API Key geçersiz. testnet.binancefuture.com'dan yeni key oluştur."
            elif "-2015" in err:
                hint = "API Key/Secret hatalı eşleşiyor."
            else:
                hint = "testnet.binancefuture.com → Profil → API Key → Generate HMAC_SHA256 Key"
            state.set(running=False, status="Hata",
                      error=f"Bakiye alınamadı: {err} | {hint}")
            state.add_log("ERROR", f"Bakiye hatası: {err}")
            state.add_log("WARN",  f"İpucu: {hint}")
            return

        # Lot size
        try:
            info = self._client.exchange_info(sym)
            for s in info.get("symbols", []):
                if s["symbol"] == sym:
                    for f in s.get("filters", []):
                        if f["filterType"] == "LOT_SIZE":
                            self._step_size = f["stepSize"]
            state.add_log("INFO", f"Lot step size: {self._step_size}")
        except Exception as e:
            state.add_log("WARN", f"Step size alınamadı, default kullanılıyor: {e}")

        # Kaldıraç
        try:
            lev = int(cfg.get("leverage", 1))
            self._client.change_leverage(sym, lev)
            state.add_log("INFO", f"Kaldıraç: {lev}×")
        except Exception as e:
            state.add_log("WARN", f"Kaldıraç ayarlanamadı: {e}")

        # Geçmiş kline (1 kez REST)
        try:
            klines = self._client.klines(sym, tf, 300)
            for k in klines:
                state.candles.append({
                    "t": k[0], "o": float(k[1]), "h": float(k[2]),
                    "l": float(k[3]), "c": float(k[4]), "v": float(k[5]),
                })
            state.add_log("INFO", f"{len(klines)} geçmiş mum yüklendi.")
        except Exception as e:
            state.add_log("WARN", f"Geçmiş veri alınamadı: {e}")

        # ── 2. WebSocket stream — Testnet public stream ───────────────────────
        # Public stream: auth gerektirmez, sadece fiyat/kline verisi alır
        ws_url = f"{_BASE_WS}/{sym.lower()}@kline_{tf}"
        state.set(status=f"✅ Çalışıyor [TESTNET] — {sym} izleniyor")
        state.add_log("INFO", f"WebSocket bağlandı: {ws_url}")

        self._run_websocket(ws_url, sym, cfg)

    def _run_websocket(self, ws_url: str, sym: str, cfg: dict):
        """WebSocket thread — bağlantı kopunca otomatik yeniden bağlanır."""
        state = self.state

        def _on_message(ws, raw):
            if self._stop_evt.is_set():
                ws.close()
                return
            try:
                msg    = json.loads(raw)
                k      = msg.get("k", {})
                if not k:
                    return
                closed = k.get("x", False)
                candle = {
                    "t": k["t"], "o": float(k["o"]), "h": float(k["h"]),
                    "l": float(k["l"]), "c": float(k["c"]), "v": float(k["v"]),
                }
                state.last_price = candle["c"]

                # Unrealized PnL
                ot = state.open_trade
                if ot:
                    if ot["side"] == "LONG":
                        upnl = (candle["c"] - ot["entry"]) / ot["entry"] * ot["notional"]
                    else:
                        upnl = (ot["entry"] - candle["c"]) / ot["entry"] * ot["notional"]
                    state.set(current_pnl=round(upnl, 4))

                if not closed:
                    return

                state.candles.append(candle)
                candles_list = list(state.candles)

                # Cooldown kontrolü
                now_ts   = time.time()
                cooldown = cfg.get("trade_cooldown_s", 60)
                if now_ts - self._last_trade_time < cooldown:
                    return

                signal = evaluate_signal(candles_list, cfg)

                if state.open_trade is None:
                    if signal in ("LONG", "SHORT"):
                        self._open_pos(sym, signal, candle["c"], cfg)
                        self._last_trade_time = now_ts
                else:
                    ot     = state.open_trade
                    close  = False
                    reason = ""
                    if (ot["side"] == "LONG"  and signal == "CLOSE_LONG") or \
                       (ot["side"] == "SHORT" and signal == "CLOSE_SHORT"):
                        close = True; reason = "Sinyal"
                    if ot["side"] == "LONG":
                        if   candle["c"] >= ot["tp"]: close = True; reason = "TP"
                        elif candle["c"] <= ot["sl"]: close = True; reason = "SL"
                    else:
                        if   candle["c"] <= ot["tp"]: close = True; reason = "TP"
                        elif candle["c"] >= ot["sl"]: close = True; reason = "SL"
                    if close:
                        self._close_pos(sym, candle["c"], reason, cfg)
                        self._last_trade_time = now_ts

            except Exception as ex:
                state.add_log("ERROR", f"WS mesaj hatası: {ex}")

        def _on_error(ws, err):
            if not self._stop_evt.is_set():
                state.add_log("WARN", f"WS hata: {err}")

        def _on_close(ws, code, msg):
            if not self._stop_evt.is_set():
                state.add_log("WARN", "WS bağlantısı kesildi, yeniden bağlanılıyor...")
                time.sleep(3)
                if not self._stop_evt.is_set():
                    self._run_websocket(ws_url, sym, cfg)   # reconnect

        def _on_open(ws):
            state.add_log("INFO", "WS bağlandı ✅")

        if not HAS_WS:
            # websocket-client yoksa polling fallback (15s'de bir REST)
            state.add_log("WARN",
                "websocket-client yüklü değil. REST polling moduna geçildi. "
                "pip install websocket-client ile WebSocket aktif edilebilir.")
            self._polling_loop(sym, cfg)
            return

        self._ws = websocket.WebSocketApp(
            ws_url,
            on_message=_on_message,
            on_error=_on_error,
            on_close=_on_close,
            on_open=_on_open,
        )
        self._ws.run_forever(
            sslopt={"cert_reqs": ssl.CERT_NONE},
            ping_interval=20,
            ping_timeout=10,
        )

    def _polling_loop(self, sym: str, cfg: dict):
        """Fallback: WebSocket yoksa 15s'de bir REST kline çeker."""
        state = self.state
        while not self._stop_evt.is_set():
            try:
                klines = self._client.klines(sym, cfg.get("timeframe","5m"), 50)
                if klines:
                    last = klines[-1]
                    candle = {
                        "t": last[0], "o": float(last[1]), "h": float(last[2]),
                        "l": float(last[3]), "c": float(last[4]), "v": float(last[5]),
                    }
                    state.last_price = candle["c"]
                    state.candles.append(candle)
            except Exception as e:
                state.add_log("WARN", f"Polling hata: {e}")
            time.sleep(15)

    # ── Pozisyon aç ────────────────────────────────────────────────────────────
    def _open_pos(self, sym: str, side: str, price: float, cfg: dict):
        state   = self.state
        eq      = state.equity or 1000.0
        rp      = cfg.get("risk_pct", 1.0)
        lev     = cfg.get("leverage", 1)
        tp_pct  = cfg.get("tp_pct",   1.5)
        sl_pct  = cfg.get("sl_pct",   1.0)
        dry_run = cfg.get("dry_run",  True)

        notional = eq * (rp / 100) * lev
        raw_qty  = notional / price
        step     = self._step_size
        dec      = len(step.rstrip("0").split(".")[-1]) if "." in step else 0
        qty      = max(round(raw_qty, dec), float(step))

        tp = price * (1 + tp_pct/100) if side == "LONG" else price * (1 - tp_pct/100)
        sl = price * (1 - sl_pct/100) if side == "LONG" else price * (1 + sl_pct/100)
        tp = round(tp, 6)
        sl = round(sl, 6)

        if not dry_run:
            try:
                b_side = "BUY" if side == "LONG" else "SELL"
                self._client.new_order(sym, b_side, qty)
            except Exception as e:
                state.add_log("ERROR", f"Order hatası: {e}")
                return

        trade = {
            "id":        int(time.time() * 1000),
            "side":      side,
            "entry":     price,
            "qty":       qty,
            "notional":  round(notional, 4),
            "tp":        tp,
            "sl":        sl,
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "dry_run":   dry_run,
        }
        state.set(open_trade=trade)
        emoji  = "🟢" if side == "LONG" else "🔴"
        dr     = "[DRY] " if dry_run else ""
        state.add_log(side,
            f"{dr}{emoji} {side} @ ${price:,.4f} | qty={qty} | "
            f"TP:{tp:,.4f} | SL:{sl:,.4f}")

    # ── Pozisyon kapat ──────────────────────────────────────────────────────────
    def _close_pos(self, sym: str, price: float, reason: str, cfg: dict):
        state   = self.state
        ot      = state.open_trade
        if not ot:
            return
        dry_run = cfg.get("dry_run", True)

        if ot["side"] == "LONG":
            pnl    = (price - ot["entry"]) / ot["entry"] * ot["notional"]
            b_side = "SELL"
        else:
            pnl    = (ot["entry"] - price) / ot["entry"] * ot["notional"]
            b_side = "BUY"
        pnl = round(pnl, 4)

        if not dry_run:
            try:
                self._client.close_order(sym, b_side, ot["qty"])
            except Exception as e:
                state.add_log("ERROR", f"Kapat hatası: {e}")

        state.add_trade({**ot, "exit": price, "pnl": pnl,
                         "reason": reason,
                         "closed_at": datetime.now(timezone.utc).isoformat()})
        state.set(
            open_trade=None, current_pnl=0.0,
            total_pnl=round(state.total_pnl + pnl, 4),
            win_count =state.win_count  + (1 if pnl > 0 else 0),
            loss_count=state.loss_count + (1 if pnl <= 0 else 0),
        )
        emoji = "✅" if pnl > 0 else "❌"
        dr    = "[DRY] " if dry_run else ""
        state.add_log("SELL",
            f"{dr}{emoji} KAPANDI ({reason}) @ ${price:,.4f} | "
            f"PnL:{'+'if pnl>=0 else ''}{pnl:.4f} USDT")


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

_worker_thread:   Optional[threading.Thread] = None
_worker_instance: Optional[BotWorker]        = None


def start_bot(api_key: str, api_secret: str, cfg: dict) -> bool:
    global _worker_thread, _worker_instance
    if not HAS_REQUESTS:
        _bot_state.add_log("ERROR", "requests yüklü değil: pip install requests")
        return False
    if _bot_state.running:
        return False
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
