from __future__ import annotations
import pandas as pd
import numpy as np

def rr_ok(entry: float, stop: float, target: float, min_rr: float) -> bool:
    risk = abs(entry - stop)
    reward = abs(target - entry)
    return reward >= min_rr * risk

def simulate(df: pd.DataFrame, entries: pd.DataFrame, min_rr: float = 1.5) -> pd.DataFrame:
    results = []
    for ts, row in entries.iterrows():
        if not (np.isfinite(row.entry_price) and np.isfinite(row.stop) and np.isfinite(row.target)):
            continue
        if not rr_ok(row.entry_price, row.stop, row.target, min_rr):
            continue
        future = df.loc[df.index >= ts]
        hit = None
        for ts2, r2 in future.iloc[1:].iterrows():
            hi, lo = r2["High"], r2["Low"]
            if row.side == "long":
                if lo <= row.stop: hit = ("stop", ts2); break
                if hi >= row.target: hit = ("target", ts2); break
            else:
                if hi >= row.stop: hit = ("stop", ts2); break
                if lo <= row.target: hit = ("target", ts2); break
        if hit is None:
            hit = ("open", future.index[-1])
        pnl = (row.target - row.entry_price) if (hit[0]=="target" and row.side=="long") else               (row.entry_price - row.target) if (hit[0]=="target" and row.side=="short") else               (row.stop - row.entry_price)   if (hit[0]=="stop"   and row.side=="long") else               (row.entry_price - row.stop)   if (hit[0]=="stop"   and row.side=="short") else 0.0
        results.append({
            "time": ts, "side": row.side, "entry": row.entry_price, "stop": row.stop, "target": row.target,
            "exit_type": hit[0], "exit_time": hit[1], "pnl": pnl
        })
    return pd.DataFrame(results).set_index("time") if results else pd.DataFrame(columns=["side","entry","stop","target","exit_type","exit_time","pnl"])
