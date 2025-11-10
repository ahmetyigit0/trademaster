from __future__ import annotations
import pandas as pd
import numpy as np

def wick_stats(df: pd.DataFrame) -> pd.DataFrame:
    o,h,l,c = df["Open"], df["High"], df["Low"], df["Close"]
    upper = h - np.maximum(o,c)
    lower = np.minimum(o,c) - l
    body  = (c - o).abs()
    rng   = (h - l).replace(0, np.nan)
    out = pd.DataFrame({
        "upper_wick": upper,
        "lower_wick": lower,
        "body": body,
        "range": rng,
        "body_pct": (body/rng).clip(0,1),
        "up_wick_pct": (upper/rng).clip(0,1),
        "lo_wick_pct": (lower/rng).clip(0,1)
    }, index=df.index)
    return out.fillna(0.0)

def volume_spikes(df: pd.DataFrame, win: int = 20, z: float = 2.0) -> pd.Series:
    v = df["Volume"].astype(float)
    ma = v.rolling(win).mean()
    std = v.rolling(win).std().replace(0, np.nan)
    spikes = (v > (ma + z*std)).fillna(False)
    return spikes

def classify_spike_bias(df: pd.DataFrame) -> pd.Series:
    w = wick_stats(df)
    cls = pd.Series("neutral", index=df.index, dtype="object")
    up, lo = w["upper_wick"], w["lower_wick"]
    cond_bull = lo > (up * 1.2)
    cond_bear = up > (lo * 1.2)
    cls[cond_bull] = "bullish"
    cls[cond_bear] = "bearish"
    return cls

def price_volume_assessment(df: pd.DataFrame, win: int = 20, z: float = 2.0) -> pd.Series:
    spikes = volume_spikes(df, win, z)
    bias = classify_spike_bias(df)
    out = pd.Series("none", index=df.index, dtype="object")
    out[spikes & (bias=="bullish")] = "vol_bull"
    out[spikes & (bias=="bearish")] = "vol_bear"
    out[spikes & (bias=="neutral")] = "vol_neutral"
    return out
