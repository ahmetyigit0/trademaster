from __future__ import annotations
import pandas as pd
import numpy as np

def _body(o,c): return (c - o).abs()
def _range(h,l): return (h - l).abs()
def _upper_wick(h, o, c): return h - np.maximum(o, c)
def _lower_wick(l, o, c): return np.minimum(o, c) - l

def bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    o,c = df["Open"], df["Close"]
    prev_o, prev_c = o.shift(1), c.shift(1)
    cond1 = (c > o) & (prev_c < prev_o)
    cond2 = (c >= prev_o) & (o <= prev_c)
    return (cond1 & cond2).fillna(False)

def bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    o,c = df["Open"], df["Close"]
    prev_o, prev_c = o.shift(1), c.shift(1)
    cond1 = (c < o) & (prev_c > prev_o)
    cond2 = (o >= prev_c) & (c <= prev_o)
    return (cond1 & cond2).fillna(False)

def hammer(df: pd.DataFrame, factor: float=2.0) -> pd.Series:
    o,h,l,c = df["Open"], df["High"], df["Low"], df["Close"]
    body = _body(o,c)
    rng = _range(h,l)
    lw = _lower_wick(l,o,c)
    uw = _upper_wick(h,o,c)
    return (lw >= factor*body) & (uw <= body) & (c >= o - 1e-9)

def hanging_man(df: pd.DataFrame, factor: float=2.0) -> pd.Series:
    o,h,l,c = df["Open"], df["High"], df["Low"], df["Close"]
    body = _body(o,c)
    lw = _lower_wick(l,o,c)
    uw = _upper_wick(h,o,c)
    return (lw >= factor*body) & (uw <= body) & (c <= o + 1e-9)

def doji(df: pd.DataFrame, thresh: float=0.1) -> pd.Series:
    o,h,l,c = df["Open"], df["High"], df["Low"], df["Close"]
    rng = _range(h,l).replace(0, np.nan)
    return ((_body(o,c)/rng) <= thresh).fillna(False)

def pin_bar_bearish(df: pd.DataFrame, factor: float=2.0) -> pd.Series:
    o,h,l,c = df["Open"], df["High"], df["Low"], df["Close"]
    body = _body(o,c)
    uw = _upper_wick(h,o,c)
    lw = _lower_wick(l,o,c)
    return (uw >= factor*body) & (lw <= body) & (c < o)

def pin_bar_bullish(df: pd.DataFrame, factor: float=2.0) -> pd.Series:
    o,h,l,c = df["Open"], df["High"], df["Low"], df["Close"]
    body = _body(o,c)
    lw = _lower_wick(l,o,c)
    uw = _upper_wick(h,o,c)
    return (lw >= factor*body) & (uw <= body) & (c > o)

def harami_bull(df: pd.DataFrame) -> pd.Series:
    o,c = df["Open"], df["Close"]
    po, pc = o.shift(1), c.shift(1)
    cond = (pc < po) & (c > o) & (c <= po) & (o >= pc)
    return cond.fillna(False)

def harami_bear(df: pd.DataFrame) -> pd.Series:
    o,c = df["Open"], df["Close"]
    po, pc = o.shift(1), c.shift(1)
    cond = (pc > po) & (c < o) & (o <= pc) & (c >= po)
    return cond.fillna(False)

def morning_star(df: pd.DataFrame) -> pd.Series:
    o,c = df["Open"], df["Close"]
    po, pc = o.shift(1), c.shift(1)
    p2o, p2c = o.shift(2), c.shift(2)
    return ((p2c < p2o) & (abs(pc-po) < abs(p2o-p2c)*0.5) & (c > o) & (c > (p2o + p2c)/2)).fillna(False)

def evening_star(df: pd.DataFrame) -> pd.Series:
    o,c = df["Open"], df["Close"]
    po, pc = o.shift(1), c.shift(1)
    p2o, p2c = o.shift(2), c.shift(2)
    return ((p2c > p2o) & (abs(pc-po) < abs(p2o-p2c)*0.5) & (c < o) & (c < (p2o + p2c)/2)).fillna(False)

def annotate_patterns(df: pd.DataFrame) -> pd.DataFrame:
    pats = pd.DataFrame(index=df.index)
    pats["bull_engulf"] = bullish_engulfing(df)
    pats["bear_engulf"] = bearish_engulfing(df)
    pats["hammer"] = hammer(df)
    pats["hangman"] = hanging_man(df)
    pats["doji"] = doji(df)
    pats["pin_bull"] = pin_bar_bullish(df)
    pats["pin_bear"] = pin_bar_bearish(df)
    pats["harami_bull"] = harami_bull(df)
    pats["harami_bear"] = harami_bear(df)
    pats["morning_star"] = morning_star(df)
    pats["evening_star"] = evening_star(df)
    return pats
