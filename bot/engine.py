"""
TradeBot Engine
==============
Ayrı thread'de çalışır. Streamlit UI'yi bloklamaz.
WebSocket ile canlı veri alır (REST polling YOK → ban riski sıfır).

Rate Limit Koruması:
- Kline verisi: WebSocket stream (REST çağrısı değil)
- Order gönderme: sadece sinyal gelince (throttle korumalı)
- Minimum 60s trade cooldown
- dry_run modu ile gerçek order göndermeden test
"""

import threading
import time
from datetime import datetime, timezone
from collections import deque
from typing import Optional

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
    from binance import ThreadedWebsocketManager
    HAS_BINANCE = True
except ImportError:
    HAS_BINANCE = False

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

        ema_f   = _ema(closes, fast)
        ema_s   = _ema(closes, slow)
        rsi     = _rsi(closes, 14)
        if ema_f is None or ema_s is None or rsi is None:
            return "HOLD"

        prev    = [c["c"] for c in candles_list[:-1]]
        p_ema_f = _ema(prev, fast)
        p_ema_s = _ema(prev, slow)
        if p_ema_f is None or p_ema_s is None:
            return "HOLD"

        if p_ema_f <= p_ema_s and ema_f > ema_s and rsi < rsi_hi:
            return "LONG"
        if p_ema_f >= p_ema_s and ema_f < ema_s and rsi > rsi_lo:
            return "SHORT"
        if ema_f < ema_s and rsi > 70:
            return "CLOSE_LONG"
        if ema_f > ema_s and rsi < 30:
            return "CLOSE_SHORT"
        return "HOLD"

    elif strategy == "RSI_MEAN_REVERT":
        rsi    = _rsi(closes, cfg.get("rsi_period", 14))
        rsi_hi = cfg.get("rsi_overbought", 70)
        rsi_lo = cfg.get("rsi_oversold",   30)
        if rsi is None:
            return "HOLD"
        if rsi < rsi_lo: return "LONG"
        if rsi > rsi_hi: return "SHORT"
        if rsi > 55:     return "CLOSE_LONG"
        if rsi < 45:     return "CLOSE_SHORT"
        return "HOLD"

    elif strategy == "BREAKOUT":
        period = cfg.get("breakout_period", 20)
        if len(closes) < period:
            return "HOLD"
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
        self._twm       = None
        self._client    = None
        self._last_trade_time = 0.0

    def stop(self):
        self._stop_evt.set()
        if self._twm:
            try:
                self._twm.stop()
            except Exception:
                pass
        self.state.set(running=False, status="Durduruldu")
        self.state.add_log("INFO", "Bot durduruldu.")

    def run(self):
        state   = self.state
        cfg     = self.cfg
        sym     = cfg["symbol"].upper().replace("/", "")
        tf      = cfg.get("timeframe", "5m")
        testnet = cfg.get("testnet", True)
        dry_run = cfg.get("dry_run", True)

        state.set(running=True, status="Başlatılıyor...",
                  symbol=sym, strategy=cfg.get("strategy","EMA_CROSS"), error="")
        mode_lbl = "[TESTNET]" if testnet else "[GERÇEK]"
        dry_lbl  = "[DRY RUN]" if dry_run else "[CANLI ORDER]"
        state.add_log("INFO", f"Bot başlatıldı {mode_lbl} {dry_lbl}: {sym} · {tf}")

        # ── Binance bağlantısı ─────────────────────────────────────────────
        try:
            self._client = Client(
                self.api_key, self.api_secret,
                testnet=testnet,
                requests_params={"timeout": 10},
            )
            eq_raw = self._client.futures_account_balance()
            equity = next(
                (float(b["availableBalance"]) for b in eq_raw if b["asset"] == "USDT"),
                0.0,
            )
            state.set(equity=equity)
            state.add_log("INFO", f"Bağlantı OK. Bakiye: ${equity:,.2f} USDT")
        except Exception as e:
            state.set(running=False, status="Hata", error=str(e))
            state.add_log("ERROR", f"Bağlantı hatası: {e}")
            return

        # Lot step size
        step_size = "0.001"
        try:
            info = self._client.futures_exchange_info()
            for s in info["symbols"]:
                if s["symbol"] == sym:
                    for f in s["filters"]:
                        if f["filterType"] == "LOT_SIZE":
                            step_size = f["stepSize"]
                    break
        except Exception:
            pass

        # Leverage
        try:
            lev = int(cfg.get("leverage", 1))
            self._client.futures_change_leverage(symbol=sym, leverage=lev)
            state.add_log("INFO", f"Kaldıraç: {lev}×")
        except Exception as e:
            state.add_log("WARN", f"Kaldıraç ayarlanamadı: {e}")

        # Geçmiş mum verisi (1 kez, REST — WebSocket'ten önce)
        try:
            klines = self._client.futures_klines(symbol=sym, interval=tf, limit=300)
            for k in klines:
                state.candles.append({
                    "t": k[0], "o": float(k[1]), "h": float(k[2]),
                    "l": float(k[3]), "c": float(k[4]), "v": float(k[5]),
                })
            state.add_log("INFO", f"{len(klines)} geçmiş mum yüklendi.")
        except Exception as e:
            state.add_log("WARN", f"Geçmiş veri: {e}")

        # ── WebSocket stream ───────────────────────────────────────────────
        self._twm = ThreadedWebsocketManager(
            api_key=self.api_key, api_secret=self.api_secret,
            testnet=testnet,
        )
        self._twm.start()
        state.set(status=f"Çalışıyor — {sym} izleniyor")

        def _on_kline(msg):
            if self._stop_evt.is_set():
                return
            try:
                k      = msg["data"]["k"]
                closed = k["x"]
                candle = {
                    "t": k["t"], "o": float(k["o"]), "h": float(k["h"]),
                    "l": float(k["l"]), "c": float(k["c"]), "v": float(k["v"]),
                }
                state.last_price = candle["c"]

                if closed:
                    state.candles.append(candle)
                    candles_list = list(state.candles)

                    # Unrealized PnL güncelle
                    ot = state.open_trade
                    if ot:
                        if ot["side"] == "LONG":
                            upnl = (candle["c"] - ot["entry"]) / ot["entry"] * ot["notional"]
                        else:
                            upnl = (ot["entry"] - candle["c"]) / ot["entry"] * ot["notional"]
                        state.set(current_pnl=round(upnl, 4))

                    # Cooldown
                    now_ts   = time.time()
                    cooldown = cfg.get("trade_cooldown_s", 60)
                    if now_ts - self._last_trade_time < cooldown:
                        return

                    signal = evaluate_signal(candles_list, cfg)

                    if state.open_trade is None:
                        if signal in ("LONG", "SHORT"):
                            self._open_pos(sym, signal, candle["c"], step_size, cfg)
                            self._last_trade_time = now_ts
                    else:
                        ot     = state.open_trade
                        close  = False
                        reason = ""

                        if (ot["side"] == "LONG"  and signal == "CLOSE_LONG") or \
                           (ot["side"] == "SHORT" and signal == "CLOSE_SHORT"):
                            close  = True
                            reason = "Sinyal"

                        if ot["side"] == "LONG":
                            if candle["c"] >= ot["tp"]: close = True; reason = "TP"
                            elif candle["c"] <= ot["sl"]: close = True; reason = "SL"
                        else:
                            if candle["c"] <= ot["tp"]: close = True; reason = "TP"
                            elif candle["c"] >= ot["sl"]: close = True; reason = "SL"

                        if close:
                            self._close_pos(sym, candle["c"], reason, cfg)
                            self._last_trade_time = now_ts

            except Exception as ex:
                state.add_log("ERROR", f"WS hata: {ex}")

        self._twm.start_kline_futures_socket(
            callback=_on_kline,
            symbol=sym,
            interval=tf,
        )

        while not self._stop_evt.is_set():
            time.sleep(0.5)

        self._twm.stop()

    def _open_pos(self, sym, side, price, step_size, cfg):
        state    = self.state
        eq       = state.equity or 1000.0
        rp       = cfg.get("risk_pct", 1.0)
        lev      = cfg.get("leverage", 1)
        tp_pct   = cfg.get("tp_pct",   1.5)
        sl_pct   = cfg.get("sl_pct",   1.0)
        dry_run  = cfg.get("dry_run",  True)

        # Qty hesapla
        notional = eq * (rp / 100) * lev
        raw_qty  = notional / price
        dec      = len(step_size.rstrip("0").split(".")[-1]) if "." in step_size else 0
        qty      = max(round(raw_qty, dec), float(step_size))

        tp = price * (1 + tp_pct/100) if side == "LONG" else price * (1 - tp_pct/100)
        sl = price * (1 - sl_pct/100) if side == "LONG" else price * (1 + sl_pct/100)
        tp = round(tp, 6)
        sl = round(sl, 6)

        if not dry_run:
            try:
                b_side = "BUY" if side == "LONG" else "SELL"
                self._client.futures_create_order(
                    symbol=sym, side=b_side, type="MARKET",
                    quantity=qty, positionSide="BOTH",
                )
            except BinanceAPIException as e:
                state.add_log("ERROR", f"Order hatası: {e.message}"); return
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
            "dry_run":   dry_run,
        }
        state.set(open_trade=trade)
        emoji  = "🟢" if side == "LONG" else "🔴"
        dr_lbl = "[DRY] " if dry_run else ""
        state.add_log(side,
            f"{dr_lbl}{emoji} {side} @ ${price:,.4f} | qty={qty} | "
            f"TP:{tp:,.4f} | SL:{sl:,.4f}"
        )

    def _close_pos(self, sym, price, reason, cfg):
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
                self._client.futures_create_order(
                    symbol=sym, side=b_side, type="MARKET",
                    quantity=ot["qty"], positionSide="BOTH", reduceOnly=True,
                )
            except Exception as e:
                state.add_log("ERROR", f"Kapat hatası: {e}")

        closed = {**ot, "exit": price, "pnl": pnl,
                  "reason": reason,
                  "closed_at": datetime.now(timezone.utc).isoformat()}
        state.add_trade(closed)
        state.set(
            open_trade=None, current_pnl=0.0,
            total_pnl=round(state.total_pnl + pnl, 4),
            win_count =state.win_count  + (1 if pnl > 0 else 0),
            loss_count=state.loss_count + (1 if pnl <= 0 else 0),
        )
        emoji  = "✅" if pnl > 0 else "❌"
        dr_lbl = "[DRY] " if dry_run else ""
        state.add_log("SELL",
            f"{dr_lbl}{emoji} KAPANDI ({reason}) @ ${price:,.4f} | "
            f"PnL:{'+'if pnl>=0 else ''}{pnl:.4f} USDT"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

_worker_thread: Optional[threading.Thread]   = None
_worker_instance: Optional[BotWorker] = None


def start_bot(api_key: str, api_secret: str, cfg: dict) -> bool:
    global _worker_thread, _worker_instance
    if not HAS_BINANCE:
        _bot_state.add_log("ERROR", "python-binance yüklü değil: pip install python-binance")
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
