"""Analytics engine — pure Python, no Streamlit dependencies."""
from collections import defaultdict
from typing import Any


def _avg(lst: list) -> float:
    return sum(lst) / len(lst) if lst else 0.0


def compute_analytics(closed_trades: list[dict]) -> dict[str, Any]:
    if not closed_trades:
        return {}

    wins   = [t for t in closed_trades if t.get("pnl", 0) > 0]
    losses = [t for t in closed_trades if t.get("pnl", 0) <= 0]
    n      = len(closed_trades)

    win_rate   = len(wins) / n
    avg_win    = _avg([t["pnl"] for t in wins])
    avg_loss   = _avg([t["pnl"] for t in losses])
    gross_win  = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    pf         = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    ev         = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))

    # R multiples — filter out None / 0 from trades without risk_calc
    r_multiples = [t.get("r_multiple", 0) for t in closed_trades]

    # ── Setup stats ───────────────────────────────────────────────────────────
    _setup: dict[str, dict] = {}
    for t in closed_trades:
        s = (t.get("setup_type") or "unknown").strip()
        if s not in _setup:
            _setup[s] = {"wins": 0, "losses": 0, "pnl": 0.0, "count": 0}
        _setup[s]["count"] += 1
        _setup[s]["pnl"]   += t.get("pnl", 0)
        if t.get("pnl", 0) > 0:
            _setup[s]["wins"] += 1
        else:
            _setup[s]["losses"] += 1
    for s in _setup:
        c = _setup[s]["count"]
        _setup[s]["win_rate"] = (_setup[s]["wins"] / c * 100) if c else 0.0

    # ── Execution buckets ─────────────────────────────────────────────────────
    _exec: dict[str, dict] = {}
    for t in closed_trades:
        score = t.get("execution_score")
        if score is None:
            continue
        try:
            score = int(score)
        except (TypeError, ValueError):
            continue
        lo     = (score // 2) * 2
        hi     = lo + 1
        bucket = f"{lo}-{hi}"
        if bucket not in _exec:
            _exec[bucket] = {"pnl_list": [], "count": 0}
        _exec[bucket]["pnl_list"].append(t.get("pnl", 0))
        _exec[bucket]["count"] += 1
    exec_analysis = {
        k: {"avg_pnl": _avg(v["pnl_list"]), "count": v["count"]}
        for k, v in _exec.items()
    }

    # ── Mistake frequency ─────────────────────────────────────────────────────
    mistake_freq: dict[str, int] = {}
    for t in closed_trades:
        for m in (t.get("mistakes") or []):
            mistake_freq[m] = mistake_freq.get(m, 0) + 1

    # ── Market condition ──────────────────────────────────────────────────────
    _cond: dict[str, dict] = {}
    for t in closed_trades:
        c = (t.get("market_condition") or "unknown").strip()
        if c not in _cond:
            _cond[c] = {"wins": 0, "total": 0, "pnl": 0.0}
        _cond[c]["total"] += 1
        _cond[c]["pnl"]   += t.get("pnl", 0)
        if t.get("pnl", 0) > 0:
            _cond[c]["wins"] += 1
    for c in _cond:
        tot = _cond[c]["total"]
        _cond[c]["win_rate"] = (_cond[c]["wins"] / tot * 100) if tot else 0.0

    # ── Emotion ───────────────────────────────────────────────────────────────
    _emo: dict[str, dict] = {}
    for t in closed_trades:
        e = (t.get("emotion") or "unknown").strip()
        if e not in _emo:
            _emo[e] = {"wins": 0, "total": 0, "pnl": 0.0}
        _emo[e]["total"] += 1
        _emo[e]["pnl"]   += t.get("pnl", 0)
        if t.get("pnl", 0) > 0:
            _emo[e]["wins"] += 1
    for e in _emo:
        tot = _emo[e]["total"]
        _emo[e]["win_rate"] = (_emo[e]["wins"] / tot * 100) if tot else 0.0

    # ── Direction ─────────────────────────────────────────────────────────────
    long_t  = [t for t in closed_trades if t.get("direction") == "LONG"]
    short_t = [t for t in closed_trades if t.get("direction") == "SHORT"]
    long_wr  = (len([t for t in long_t  if t.get("pnl", 0) > 0]) / len(long_t)  * 100) if long_t  else 0.0
    short_wr = (len([t for t in short_t if t.get("pnl", 0) > 0]) / len(short_t) * 100) if short_t else 0.0

    # ── Plan adherence ────────────────────────────────────────────────────────
    plan_yes = [t for t in closed_trades if t.get("plan_followed") is True]
    plan_no  = [t for t in closed_trades if not t.get("plan_followed")]
    plan_yes_wr = (len([t for t in plan_yes if t.get("pnl", 0) > 0]) / len(plan_yes) * 100) if plan_yes else 0.0
    plan_no_wr  = (len([t for t in plan_no  if t.get("pnl", 0) > 0]) / len(plan_no)  * 100) if plan_no  else 0.0

    return {
        "n":             n,
        "win_rate":      win_rate,
        "avg_win":       avg_win,
        "avg_loss":      avg_loss,
        "profit_factor": pf,
        "ev":            ev,
        "r_multiples":   r_multiples,
        "setup_stats":   _setup,
        "exec_analysis": exec_analysis,
        "mistake_freq":  mistake_freq,
        "cond_stats":    _cond,
        "emo_stats":     _emo,
        "long_win_rate": long_wr,
        "short_win_rate":short_wr,
        "long_count":    len(long_t),
        "short_count":   len(short_t),
        "plan_yes_wr":   plan_yes_wr,
        "plan_no_wr":    plan_no_wr,
    }
