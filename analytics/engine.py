"""Analytics engine — pure Python, no Streamlit dependencies."""
from typing import Any
from datetime import datetime


def _avg(lst):
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
    r_multiples = [t.get("r_multiple", 0) or 0 for t in closed_trades]

    # ── Top / Bottom trades ───────────────────────────────────────────────────
    sorted_by_pnl = sorted(closed_trades, key=lambda t: t.get("pnl", 0), reverse=True)
    top_trades    = sorted_by_pnl[:5]
    worst_trades  = sorted_by_pnl[-5:][::-1]

    # ── Symbol stats ──────────────────────────────────────────────────────────
    _sym: dict[str, dict] = {}
    for t in closed_trades:
        s = (t.get("symbol") or "?").upper().strip()
        if s not in _sym:
            _sym[s] = {"wins": 0, "losses": 0, "pnl": 0.0, "count": 0, "pnl_list": []}
        _sym[s]["count"]   += 1
        _sym[s]["pnl"]     += t.get("pnl", 0)
        _sym[s]["pnl_list"].append(t.get("pnl", 0))
        if t.get("pnl", 0) > 0:
            _sym[s]["wins"] += 1
        else:
            _sym[s]["losses"] += 1
    for s in _sym:
        c = _sym[s]["count"]
        _sym[s]["win_rate"] = (_sym[s]["wins"] / c * 100) if c else 0.0
        _sym[s]["avg_pnl"]  = _avg(_sym[s]["pnl_list"])

    most_traded_sym  = max(_sym, key=lambda s: _sym[s]["count"])       if _sym else None
    best_sym_wr      = max(_sym, key=lambda s: _sym[s]["win_rate"])    if _sym else None
    best_sym_pnl     = max(_sym, key=lambda s: _sym[s]["pnl"])         if _sym else None
    worst_sym_pnl    = min(_sym, key=lambda s: _sym[s]["pnl"])         if _sym else None

    # ── Streak ────────────────────────────────────────────────────────────────
    def _parse_dt(t):
        raw = t.get("closed_at") or t.get("created_at") or ""
        try:
            return datetime.fromisoformat(raw)
        except Exception:
            return datetime.min

    sorted_t = sorted(closed_trades, key=_parse_dt)
    results  = ["W" if t.get("pnl", 0) > 0 else "L" for t in sorted_t]
    cur_streak = max_win_streak = max_loss_streak = 0
    cur_type   = None
    for r in results:
        if r == cur_type:
            cur_streak += 1
        else:
            cur_streak = 1
            cur_type   = r
        if r == "W":
            max_win_streak  = max(max_win_streak,  cur_streak)
        else:
            max_loss_streak = max(max_loss_streak, cur_streak)

    # Current streak
    cur_streak_val  = 0
    cur_streak_type = None
    for r in reversed(results):
        if cur_streak_type is None:
            cur_streak_type = r
            cur_streak_val  = 1
        elif r == cur_streak_type:
            cur_streak_val += 1
        else:
            break

    # ── Consistency (% of months profitable) ─────────────────────────────────
    monthly: dict[str, list] = {}
    for t in sorted_t:
        dt = _parse_dt(t)
        key = dt.strftime("%Y-%m") if dt != datetime.min else "unknown"
        monthly.setdefault(key, []).append(t.get("pnl", 0))
    profitable_months = sum(1 for v in monthly.values() if sum(v) > 0)
    consistency = (profitable_months / len(monthly) * 100) if monthly else 0

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
        lo = (score // 2) * 2
        bucket = f"{lo}-{lo+1}"
        if bucket not in _exec:
            _exec[bucket] = {"pnl_list": [], "count": 0}
        _exec[bucket]["pnl_list"].append(t.get("pnl", 0))
        _exec[bucket]["count"] += 1
    exec_analysis = {k: {"avg_pnl": _avg(v["pnl_list"]), "count": v["count"]} for k, v in _exec.items()}

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

    # ── Hold time ─────────────────────────────────────────────────────────────
    hold_times = []
    for t in closed_trades:
        try:
            opened = datetime.fromisoformat(t.get("created_at", ""))
            closed_dt = datetime.fromisoformat(t.get("closed_at", ""))
            hold_times.append((closed_dt - opened).total_seconds() / 3600)
        except Exception:
            pass
    avg_hold_h = _avg(hold_times)

    return {
        "n": n, "win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss,
        "profit_factor": pf, "ev": ev, "r_multiples": r_multiples,
        "setup_stats": _setup, "exec_analysis": exec_analysis,
        "mistake_freq": mistake_freq, "cond_stats": _cond, "emo_stats": _emo,
        "long_win_rate": long_wr, "short_win_rate": short_wr,
        "long_count": len(long_t), "short_count": len(short_t),
        "plan_yes_wr": plan_yes_wr, "plan_no_wr": plan_no_wr,
        # New
        "top_trades": top_trades, "worst_trades": worst_trades,
        "symbol_stats": _sym,
        "most_traded_sym": most_traded_sym, "best_sym_wr": best_sym_wr,
        "best_sym_pnl": best_sym_pnl, "worst_sym_pnl": worst_sym_pnl,
        "max_win_streak": max_win_streak, "max_loss_streak": max_loss_streak,
        "cur_streak_val": cur_streak_val, "cur_streak_type": cur_streak_type,
        "consistency": consistency, "monthly": monthly,
        "avg_hold_hours": avg_hold_h,
    }
