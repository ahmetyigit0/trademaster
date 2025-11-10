from __future__ import annotations
import pandas as pd
import yfinance as yf

REQUIRED_PRICE_COLS = ["Open","High","Low","Close"]
OPTIONAL_COLS = ["Adj Close","Volume"]

def _flatten_one_symbol(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Handle MultiIndex columns (e.g., when yfinance returns per-ticker columns)."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    # Prefer exact symbol match on lvl0
    if symbol in df.columns.get_level_values(0):
        return df.xs(symbol, axis=1, level=0)
    # Else take the first top-level as fallback
    first = df.columns.get_level_values(0)[0]
    return df.xs(first, axis=1, level=0)

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make column names canonical: Open, High, Low, Close, Adj Close, Volume"""
    norm = {}
    for c in df.columns:
        key = str(c).strip().lower().replace(" ", "")
        norm[key] = c
    out = pd.DataFrame(index=df.index)
    # Required price columns must exist in some form
    for want in ["open","high","low","close"]:
        if want in norm:
            out[want.title()] = df[norm[want]]
        else:
            raise KeyError(f"Missing required column: {want.title()}")
    # Optional
    if "adjclose" in norm:
        out["Adj Close"] = df[norm["adjclose"]]
    # Volume may be missing for some tickers (indices/crypto on certain intervals). Fill zeros.
    if "volume" in norm:
        out["Volume"] = df[norm["volume"]].fillna(0)
    else:
        out["Volume"] = 0
    return out

def load_ohlcv(symbol: str, interval: str = "1d", period: str = "3y") -> pd.DataFrame:
    df = yf.download(symbol, interval=interval, period=period, auto_adjust=False, progress=False, group_by="column")
    if df is None or df.empty:
        raise ValueError(f"No data returned for {symbol} (interval={interval}, period={period}).")
    df = _flatten_one_symbol(df, symbol)
    df = _normalize_columns(df)
    # Drop rows with NaN in essential price cols
    df = df.dropna(subset=["Open","High","Low","Close"])
    df.index.name = "Date"
    return df
