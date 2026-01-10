import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from pycoingecko import CoinGeckoAPI
import matplotlib.pyplot as plt
import json
from datetime import datetime

st.set_page_config(page_title="Net Worth Dashboard", layout="wide")

@st.cache_data(ttl=60)
def load_portfolio(path="portfolio.csv"):
    df = pd.read_csv(path)
    # normalize blanks
    for col in ["avg_cost_usd","avg_cost_try","manual_price_try","manual_price_usd","quantity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["currency"] = df.get("currency", "USD").fillna("USD")
    df["group"] = df.get("group", "Other").fillna("Other")
    df["name"] = df.get("name", df["symbol"]).fillna(df["symbol"])
    return df

@st.cache_data(ttl=300)
def load_notes(path="notes.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"goals": [], "expectations": {}, "rules": []}

@st.cache_data(ttl=60)
def fetch_fx_usdtry():
    # USDTRY from Yahoo
    try:
        t = yf.Ticker("USDTRY=X")
        px = t.history(period="2d")["Close"].dropna()
        return float(px.iloc[-1]) if len(px) else np.nan
    except Exception:
        return np.nan

@st.cache_data(ttl=60)
def fetch_yf_last_close(tickers):
    if not tickers:
        return {}
    data = yf.download(tickers=tickers, period="5d", interval="1d", group_by="ticker", auto_adjust=True, progress=False)
    out = {}
    # yf returns different shapes for 1 ticker vs many
    if isinstance(tickers, str) or len(tickers) == 1:
        # single
        try:
            close = data["Close"].dropna()
            out[tickers if isinstance(tickers, str) else tickers[0]] = float(close.iloc[-1])
        except Exception:
            pass
        return out

    for tkr in tickers:
        try:
            close = data[tkr]["Close"].dropna()
            out[tkr] = float(close.iloc[-1]) if len(close) else np.nan
        except Exception:
            out[tkr] = np.nan
    return out

@st.cache_data(ttl=60)
def fetch_coingecko_prices(ids, vs="usd"):
    if not ids:
        return {}
    cg = CoinGeckoAPI()
    prices = cg.get_price(ids=ids, vs_currencies=vs)
    # returns dict: {id: {vs: price}}
    out = {}
    for _id in ids:
        try:
            out[_id] = float(prices[_id][vs])
        except Exception:
            out[_id] = np.nan
    return out

def to_try(price, currency, usdtry):
    if currency.upper() == "TRY":
        return price
    if currency.upper() == "USD":
        return price * usdtry if np.isfinite(usdtry) else np.nan
    return np.nan

st.title("📌 Net Worth Dashboard (Altın / Hisse / USD / Kripto)")

# Sidebar
st.sidebar.header("Ayarlar")
portfolio_path = st.sidebar.text_input("portfolio.csv yolu", "portfolio.csv")
notes_path = st.sidebar.text_input("notes.json yolu", "notes.json")
st.sidebar.divider()

df = load_portfolio(portfolio_path)
notes = load_notes(notes_path)

usdtry = fetch_fx_usdtry()
st.sidebar.metric("USD/TRY", f"{usdtry:,.2f}" if np.isfinite(usdtry) else "N/A")

# Collect tickers/ids
crypto_ids = df.loc[df["asset_type"].str.lower()=="crypto", "symbol"].str.lower().tolist()
yf_tickers = df.loc[df["asset_type"].str.lower().isin(["equity","fx","commodity"]), "symbol"].tolist()

# fetch prices
crypto_px_usd = fetch_coingecko_prices(sorted(set(crypto_ids)), vs="usd")
yf_px = fetch_yf_last_close(sorted(set(yf_tickers)))

# build valuation
rows = []
for _, r in df.iterrows():
    atype = str(r["asset_type"]).lower()
    symbol = str(r["symbol"])
    qty = float(r["quantity"])
    avg_cost_usd = float(r["avg_cost_usd"])
    avg_cost_try = float(r["avg_cost_try"])
    manual_try = float(r.get("manual_price_try", 0.0))
    manual_usd = float(r.get("manual_price_usd", 0.0))
    currency = str(r.get("currency", "USD"))

    live_usd = np.nan
    live_try = np.nan

    if atype == "crypto":
        live_usd = crypto_px_usd.get(symbol.lower(), np.nan)
        live_try = to_try(live_usd, "USD", usdtry)
    elif atype in ["equity","fx","commodity"]:
        live_price = yf_px.get(symbol, np.nan)
        # For yf tickers, assume currency is provided by you (USD/TRY)
        live_try = to_try(live_price, currency, usdtry)
        live_usd = live_price if currency.upper()=="USD" else (live_try/usdtry if np.isfinite(usdtry) else np.nan)
    elif atype == "manual":
        # Use manual price if provided
        if manual_try > 0:
            live_try = manual_try
            live_usd = manual_try/usdtry if np.isfinite(usdtry) else np.nan
        elif manual_usd > 0:
            live_usd = manual_usd
            live_try = to_try(manual_usd, "USD", usdtry)

    value_try = live_try * qty if np.isfinite(live_try) else np.nan
    value_usd = live_usd * qty if np.isfinite(live_usd) else np.nan

    # cost basis (best-effort)
    cost_try = (avg_cost_try * qty) if avg_cost_try > 0 else (avg_cost_usd * qty * usdtry if avg_cost_usd > 0 and np.isfinite(usdtry) else np.nan)
    pl_try = (value_try - cost_try) if np.isfinite(value_try) and np.isfinite(cost_try) else np.nan
    pl_pct = (pl_try / cost_try * 100.0) if np.isfinite(pl_try) and np.isfinite(cost_try) and cost_try != 0 else np.nan

    rows.append({
        "Group": r.get("group","Other"),
        "Type": atype,
        "Symbol": symbol,
        "Name": r.get("name", symbol),
        "Qty": qty,
        "Live (USD)": live_usd,
        "Live (TRY)": live_try,
        "Value (TRY)": value_try,
        "Cost (TRY)": cost_try,
        "P/L (TRY)": pl_try,
        "P/L %": pl_pct
    })

val = pd.DataFrame(rows)

# Summary
total_try = val["Value (TRY)"].sum(skipna=True)
total_pl_try = val["P/L (TRY)"].sum(skipna=True)

c1, c2, c3 = st.columns(3)
c1.metric("Toplam Net Worth (TRY)", f"{total_try:,.0f}")
c2.metric("Toplam P/L (TRY)", f"{total_pl_try:,.0f}")
c3.metric("Güncelleme", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

st.divider()

# Main table
st.subheader("📦 Portföy Detayı")
st.dataframe(
    val.sort_values(["Group","Type","Symbol"]),
    use_container_width=True,
    height=420
)

# Allocation chart
st.subheader("📊 Dağılım")
grp = val.groupby("Group", dropna=False)["Value (TRY)"].sum().sort_values(ascending=False)

fig = plt.figure()
ax = plt.gca()
ax.pie(grp.values, labels=grp.index, autopct="%1.1f%%")
ax.set_title("Varlık Dağılımı (TRY)")
st.pyplot(fig)

st.divider()

# Goals / expectations / rules
st.subheader("🎯 Hedefler & Beklentiler")
colA, colB = st.columns([1,1])

with colA:
    st.markdown("### Hedefler")
    goals = notes.get("goals", [])
    if goals:
        st.dataframe(pd.DataFrame(goals), use_container_width=True)
    else:
        st.info("notes.json içine hedef ekleyebilirsin.")

with colB:
    st.markdown("### Kural Seti")
    rules = notes.get("rules", [])
    if rules:
        for r in rules:
            st.write(f"• {r}")
    else:
        st.info("notes.json içine kural seti ekleyebilirsin.")

st.markdown("### Beklentiler / Senaryolar")
exp = notes.get("expectations", {})
if exp:
    for k, v in exp.items():
        st.write(f"**{k.upper()}**: {v}")
else:
    st.info("notes.json içine beklenti/senaryo notlarını ekleyebilirsin.")