from __future__ import annotations
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

def pivot_zones(df: pd.DataFrame, order: int = 5, band: float = 0.002) -> pd.DataFrame:
    h, l = df["High"].values, df["Low"].values
    idx_high = argrelextrema(h, np.greater, order=order)[0]
    idx_low  = argrelextrema(l, np.less, order=order)[0]

    levels = []
    for i in idx_high:
        levels.append(("R", df.index[i], h[i]))
    for i in idx_low:
        levels.append(("S", df.index[i], l[i]))
    if not levels:
        return pd.DataFrame(columns=["kind","start","end","lo","hi"])

    levels = sorted(levels, key=lambda x: x[2])
    merged = []
    cur_kind, cur_vals = None, []

    def flush(vals):
        arr = np.array([v[2] for v in vals])
        lo, hi = arr.min(), arr.max()
        return vals[0][0], min(v[1] for v in vals), max(v[1] for v in vals), lo, hi

    for k, t, val in levels:
        if not cur_vals:
            cur_kind, cur_vals = k, [(k,t,val)]
            continue
        ref = np.median([v[2] for v in cur_vals]) or val
        if abs(val - ref) / max(ref, 1e-9) <= band:
            cur_vals.append((k,t,val))
        else:
            merged.append(flush(cur_vals))
            cur_kind, cur_vals = k, [(k,t,val)]
    if cur_vals:
        merged.append(flush(cur_vals))

    zones = pd.DataFrame(merged, columns=["kind","start","end","lo","hi"])
    return zones

def tag_price_context(df: pd.DataFrame, zones: pd.DataFrame, tol: float = 0.002) -> pd.Series:
    close = df["Close"]
    tags = pd.Series(index=df.index, dtype="object")
    for ts in df.index:
        px = close.loc[ts]
        found = None
        for _, row in zones.iterrows():
            if row.lo*(1-tol) <= px <= row.hi*(1+tol):
                found = f"near_{'res' if row.kind=='R' else 'sup'}"
                break
        tags.loc[ts] = found or "mid"
    return tags

def acceptance_rejection(df: pd.DataFrame, zones: pd.DataFrame, lookback: int = 3) -> pd.Series:
    close, high, low, open_ = df["Close"], df["High"], df["Low"], df["Open"]
    sig = pd.Series("none", index=df.index, dtype="object")

    for ts in df.index:
        px_o, px_h, px_l, px_c = open_.loc[ts], high.loc[ts], low.loc[ts], close.loc[ts]
        for _, z in zones.iterrows():
            if z.kind == "R":
                in_band = (z.lo <= px_h) and (px_l <= z.hi)
                if in_band and (px_c < z.lo) and (px_h > z.hi):
                    sig.loc[ts] = "reject_res"
                idx = df.index.get_loc(ts)
                if px_c > z.hi and idx+lookback < len(df):
                    window = close.iloc[idx: idx+lookback]
                    if (window > z.hi).all():
                        sig.loc[ts] = "accept_res_break"
            else:
                in_band = (z.lo <= px_h) and (px_l <= z.hi)
                if in_band and (px_c > z.hi) and (px_l < z.lo):
                    sig.loc[ts] = "reject_sup"
                idx = df.index.get_loc(ts)
                if px_c < z.lo and idx+lookback < len(df):
                    window = close.iloc[idx: idx+lookback]
                    if (window < z.lo).all():
                        sig.loc[ts] = "accept_sup_break"
    return sig
