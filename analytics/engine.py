"""Analytics engine — computes all performance metrics from trade data."""
from collections import defaultdict
from typing import Any


def _safe_avg(lst):
    return sum(lst) / len(lst) if lst else 0.0


def compute_analytics(closed_trades: list[dict]) -> dict[str, Any]:
    if not closed_trades:
        return {}

    wins   = [t for t in closed_trades if t.get("pnl", 0) > 0]
    losses = [t for t in closed_trades if t.get("pnl", 0) <= 0]
    n = len(closed_trades)

    # ── Base metrics ──────────────────────────────────────────────────────────
    win_rate    = len(wins) / n
    avg_win     = _safe_avg([t["pnl"] for t in wins])
    avg_loss    = _safe_avg([t["pnl"] for t in losses])
    gross_win   = sum(t["pnl"] for t in wins)
    gross_loss  = abs(sum(t["pnl"] for t in losses))
    pf          = gross_win / gross_loss if gross_loss > 0 else float("inf")
    ev          = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))

    # R multiples
    r_multiples = [t.get("r_multiple", 0) for t in closed_trades]

    # ── Setup performance ─────────────────────────────────────────────────────
    setup_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0, "count": 0})
    for t in closed_trades:
        s = t.get("setup_type", "unknown") or "unknown"
        setup_stats[s]["count"] += 1
        setup_stats[s]["pnl"]   += t.get("pnl", 0)
        if t.get("pnl", 0) > 0:
            setup_stats[s]["wins"] += 1
        else:
            setup_stats[s]["losses"] += 1

    for s in setup_stats:
        c = setup_stats[s]["count"]
        setup_stats[s]["win_rate"] = setup_stats[s]["wins"] / c * 100 if c else 0

    # ── Execution score analysis ──────────────────────────────────────────────
    exec_buckets = defaultdict(list)
    for t in closed_trades:
        score = t.get("execution_score")
        if score is not None:
            bucket = f"{(int(score) // 2) * 2}-{(int(score) // 2) * 2 + 1}"
            exec_buckets[bucket].append(t.get("pnl", 0))

    exec_analysis = {k: {"avg_pnl": _safe_avg(v), "count": len(v)} for k, v in exec_buckets.items()}

    # ── Mistake frequency ─────────────────────────────────────────────────────
    mistake_freq = defaultdict(int)
    for t in closed_trades:
        for m in t.get("mistakes", []):
            mistake_freq[m] += 1

    # ── Market condition analysis ─────────────────────────────────────────────
    cond_stats = defaultdict(lambda: {"wins": 0, "total": 0, "pnl": 0.0})
    for t in closed_trades:
        c = t.get("market_condition", "unknown") or "unknown"
        cond_stats[c]["total"] += 1
        cond_stats[c]["pnl"]   += t.get("pnl", 0)
        if t.get("pnl", 0) > 0:
            cond_stats[c]["wins"] += 1

    for c in cond_stats:
        tot = cond_stats[c]["total"]
        cond_stats[c]["win_rate"] = cond_stats[c]["wins"] / tot * 100 if tot else 0

    # ── Emotion / psychology analysis ────────────────────────────────────────
    emo_stats = defaultdict(lambda: {"wins": 0, "total": 0, "pnl": 0.0})
    for t in closed_trades:
        e = t.get("emotion", "unknown") or "unknown"
        emo_stats[e]["total"] += 1
        emo_stats[e]["pnl"]   += t.get("pnl", 0)
        if t.get("pnl", 0) > 0:
            emo_stats[e]["wins"] += 1
    for e in emo_stats:
        tot = emo_stats[e]["total"]
        emo_stats[e]["win_rate"] = emo_stats[e]["wins"] / tot * 100 if tot else 0

    # ── Direction bias ────────────────────────────────────────────────────────
    long_trades  = [t for t in closed_trades if t.get("direction") == "LONG"]
    short_trades = [t for t in closed_trades if t.get("direction") == "SHORT"]
    long_wr  = len([t for t in long_trades  if t.get("pnl", 0) > 0]) / len(long_trades)  * 100 if long_trades  else 0
    short_wr = len([t for t in short_trades if t.get("pnl", 0) > 0]) / len(short_trades) * 100 if short_trades else 0

    # ── Plan adherence ────────────────────────────────────────────────────────
    plan_yes = [t for t in closed_trades if t.get("plan_followed")]
    plan_no  = [t for t in closed_trades if not t.get("plan_followed")]
    plan_yes_wr = len([t for t in plan_yes if t.get("pnl", 0) > 0]) / len(plan_yes) * 100 if plan_yes else 0
    plan_no_wr  = len([t for t in plan_no  if t.get("pnl", 0) > 0]) / len(plan_no)  * 100 if plan_no  else 0

    return {
        "n": n,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": pf,
        "ev": ev,
        "r_multiples": r_multiples,
        "setup_stats": dict(setup_stats),
        "exec_analysis": exec_analysis,
        "mistake_freq": dict(mistake_freq),
        "cond_stats": dict(cond_stats),
        "emo_stats": dict(emo_stats),
        "long_win_rate": long_wr,
        "short_win_rate": short_wr,
        "plan_yes_wr": plan_yes_wr,
        "plan_no_wr": plan_no_wr,
        "long_count": len(long_trades),
        "short_count": len(short_trades),
    }
