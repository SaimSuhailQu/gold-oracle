"""Free, keyless market data for the gold factor model (via yfinance).

Symbols (all verified working Oct 2026):
  GC=F   COMEX gold front-month future  (prediction target)
  UUP    Invesco DB USD Index ETF       (dollar proxy; Yahoo delisted DX-Y.N)
  ^TNX   10Y Treasury nominal yield
  TIP    iShares TIPS ETF               (real-yield proxy: price falls as real yields rise)
  ^IRX   13-week T-bill yield           (short-rate / Fed stance proxy)
  ^VIX   volatility / fear gauge
  ^GSPC  S&P 500                        (risk appetite)
  BTC-USD bitcoin                       (hard-asset competition)
  GLD    gold ETF price + volume        (investment flow proxy)
  TLT    20Y Treasury ETF               (duration bid)

Data is cached to data/cache.csv so backtests don't re-download.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

TICKERS = {
    "gold": "GC=F",
    "dollar": "UUP",
    "tnx": "^TNX",
    "tip": "TIP",
    "irx": "^IRX",
    "vix": "^VIX",
    "spx": "^GSPC",
    "btc": "BTC-USD",
    "gld": "GLD",
    "tlt": "TLT",
}

CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache.csv")


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def download(period: str = "15y", use_cache: bool = True) -> pd.DataFrame:
    """Download OHLCV for all factor tickers; returns wide Close-price frame
    (plus GLD volume). Index is tz-naive dates. Retries once on failure."""
    if use_cache and os.path.exists(CACHE_PATH):
        mtime = datetime.fromtimestamp(os.path.getmtime(CACHE_PATH), tz=timezone.utc)
        if (datetime.now(timezone.utc) - mtime).days < 1:
            df = pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)
            if not df.empty and "gold" in df.columns and len(df) >= 500:
                return df
            print("[data] cache unusable, re-downloading")

    last_err: Exception | None = None
    for attempt in (1, 2):
        try:
            return _fetch(period)
        except Exception as exc:
            last_err = exc
            print(f"[data] download attempt {attempt} failed: {exc}")
            import time
            time.sleep(5)
    raise RuntimeError(f"download failed twice: {last_err}")


def _fetch(period: str) -> pd.DataFrame:
    raw = yf.download(
        list(TICKERS.values()), period=period, progress=False,
        auto_adjust=True, group_by="ticker",
    )
    closes = {}
    for name, sym in TICKERS.items():
        try:
            sub = _flatten(raw[sym])
            closes[name] = sub["Close"]
        except Exception as exc:  # keep going; model tolerates a missing leg
            print(f"[data] {sym} failed: {exc}")
    df = pd.DataFrame(closes).dropna(how="all")
    # Gold is the anchor of everything: drop weekend/holiday rows where gold
    # itself didn't trade (crypto trades 24/7 and would otherwise keep
    # NaN-gold rows alive, poisoning gold_price and the technicals).
    if "gold" in df.columns:
        df = df.dropna(subset=["gold"])
    # real daily high/low for gold (technicals: ATR/ADX/swings need real range)
    try:
        gsub = _flatten(raw[TICKERS["gold"]])
        df["gold_high"] = gsub["High"].reindex(df.index)
        df["gold_low"] = gsub["Low"].reindex(df.index)
    except Exception:
        pass
    if df.empty or len(df) < 500 or "gold" not in df.columns:
        raise RuntimeError(
            f"unusable download: {len(df)} rows, columns={list(df.columns)[:5]}")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    # GLD volume for the flow proxy
    try:
        vol = _flatten(raw[TICKERS["gld"]])["Volume"]
        df["gld_volume"] = vol.reindex(df.index)
    except Exception:
        pass
    df = df.sort_index()
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    df.to_csv(CACHE_PATH)
    print(f"[data] {len(df)} rows, {df.index[0].date()} -> {df.index[-1].date()}")
    return df


def latest() -> pd.DataFrame:
    """Fresh pull bypassing cache (for the live signal)."""
    return download(use_cache=False)
