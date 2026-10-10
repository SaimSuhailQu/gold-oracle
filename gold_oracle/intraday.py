"""
Real intraday gold data for the microstructure layer (free & keyless).

Provider chain — first working source wins:
1. Dukascopy XAU/USD tick feed -> aggregated to 1-minute bars (real FX gold).
   https://datafeed.dukascopy.com/datafeed/XAU/USD/YYYY/MM/DD/HHh_ticks.bi5
   .bi5 = LZMA-compressed 20-byte ticks: >i millis, >i ask_pts, >i bid_pts,
   >f ask_vol, >f bid_vol. Prices are integer points; the divisor is
   sanity-checked against the plausible gold range (100-20000).
2. Kraken PAXG/USD 1m OHLC — gold-pegged token (1 PAXG = 1 troy oz),
   very reliable fallback. Absolute level can deviate slightly from spot;
   microstructure uses it purely relatively (VWAP/z-scores/profiles).

Hourly .bi5 files are cached under data/intraday_cache/ so reruns are free.

Public API:
    bars, provider = get_1m_bars(days=3)   # DataFrame[time,open,high,low,close,volume]
    micro = microstructure_from_bars(bars)  # real VWAP, z-score, POC/VAH/VAL
"""

from __future__ import annotations

import json
import lzma
import os
import struct
import time
import urllib.request
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE, "data", "intraday_cache")
UA = {"User-Agent": "gold-oracle/1.0"}


def _get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# ---------------- Dukascopy ----------------

def _dukascopy_hour(dt: datetime) -> pd.DataFrame | None:
    """One hour of XAU/USD ticks -> DataFrame, cached. None if unavailable."""
    key = dt.strftime("%Y/%m/%d/%H")
    cache_path = os.path.join(CACHE_DIR, "dukascopy", dt.strftime("%Y-%m-%d"), f"{dt.hour:02d}.bi5")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            raw = fh.read()
    else:
        url = (f"https://datafeed.dukascopy.com/datafeed/XAU/USD/{dt.year}/"
               f"{dt.month:02d}/{dt.day:02d}/{dt.hour:02d}h_ticks.bi5")
        try:
            raw = _get(url, timeout=12)
        except Exception:
            return None
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as fh:
            fh.write(raw)
    try:
        try:
            dec = lzma.decompress(raw)
        except Exception:
            dec = lzma.decompress(raw, format=lzma.FORMAT_ALONE)
    except Exception:
        return None
    n = len(dec) // 20
    if n == 0:
        return None
    rows = []
    base_ms = int(dt.replace(minute=0, second=0, microsecond=0).timestamp() * 1000)
    for i in range(n):
        ms, ask_p, bid_p, av, bv = struct.unpack(">3i2f", dec[i * 20:(i + 1) * 20])
        rows.append((base_ms + ms, ask_p, bid_p, av + bv))
    df = pd.DataFrame(rows, columns=["ms", "ask_p", "bid_p", "vol"])
    # divisor sanity check against plausible gold range
    price = None
    for div in (1000.0, 100.0, 10000.0):
        px = float(df["ask_p"].iloc[len(df) // 2]) / div
        if 100.0 < px < 20000.0:
            price = div
            break
    if price is None:
        return None
    df["mid"] = (df["ask_p"] + df["bid_p"]) / 2.0 / price
    df["time"] = pd.to_datetime(df["ms"], unit="ms", utc=True)
    return df[["time", "mid", "vol"]]


def _dukascopy_bars(days: int) -> pd.DataFrame | None:
    # cap the sweep: 30 most recent hours is plenty for session microstructure,
    # and keeps signal generation fast when the feed is unreachable
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    # fast probe: if the latest hour is unreachable, don't sweep 30 failing hours
    try:
        probe = _dukascopy_hour(now - timedelta(hours=1))
    except Exception:
        probe = None
    if probe is None or not len(probe):
        # one more try two hours back (latest file may simply not exist yet)
        try:
            probe = _dukascopy_hour(now - timedelta(hours=2))
        except Exception:
            probe = None
        if probe is None or not len(probe):
            return None
    frames = [probe]
    for h in range(3, min(days * 24, 30)):
        dt = now - timedelta(hours=h)
        if dt.weekday() >= 5:  # no weekend ticks
            continue
        try:
            df = _dukascopy_hour(dt)
        except Exception:
            df = None
        if df is not None and len(df):
            frames.append(df)
    if not frames:
        return None
    ticks = pd.concat(frames).sort_values("time")
    ticks["bar"] = ticks["time"].dt.floor("min")
    g = ticks.groupby("bar")
    bars = pd.DataFrame({
        "open": g["mid"].first(), "high": g["mid"].max(),
        "low": g["mid"].min(), "close": g["mid"].last(),
        "volume": g["vol"].sum(),
    }).reset_index().rename(columns={"bar": "time"})
    return bars


# ---------------- Kraken PAXG ----------------

def _kraken_bars(days: int) -> pd.DataFrame | None:
    """PAXG/USD 1m OHLC, paginated (720 bars/call)."""
    need = days * 24 * 60
    since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
    all_bars = []
    for _ in range(12):
        url = f"https://api.kraken.com/0/public/OHLC?pair=PAXGUSD&interval=1&since={since}"
        try:
            d = json.loads(_get(url, timeout=20).decode())
        except Exception:
            break
        res = d.get("result", {})
        key = next((k for k in res if k != "last"), None)
        bars = res.get(key, []) if key else []
        if not bars:
            break
        all_bars.extend(bars)
        since = int(res.get("last", since)) + 1
        if len(all_bars) >= need:
            break
        time.sleep(0.3)
    if len(all_bars) < 120:
        return None
    df = pd.DataFrame(all_bars, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df[["time", "open", "high", "low", "close", "volume"]].drop_duplicates("time").sort_values("time").reset_index(drop=True)


# ---------------- public ----------------

def get_1m_bars(days: int = 3) -> tuple[pd.DataFrame | None, str]:
    """(bars, provider). bars None if nothing worked."""
    for name, fn in (("dukascopy_xauusd_ticks", _dukascopy_bars),
                     ("kraken_paxg_1m", _kraken_bars)):
        try:
            bars = fn(days)
        except Exception:
            bars = None
        if bars is not None and len(bars) >= 120:
            return bars, name
    return None, "none"


def microstructure_from_bars(bars: pd.DataFrame) -> dict:
    """Real microstructure from 1m bars: session VWAP, z-score, volume profile."""
    df = bars.copy().reset_index(drop=True)
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].replace(0, np.nan).fillna(vol.median() if (vol := df["volume"]).median() > 0 else 1.0)
    # session = last 24h of bars
    sess = df.tail(1440)
    s_typ = (sess["high"] + sess["low"] + sess["close"]) / 3.0
    s_vol = sess["volume"].replace(0, np.nan).fillna(1.0)
    vwap = float((s_typ * s_vol).sum() / s_vol.sum())
    dev = s_typ - vwap
    sess_std = float(dev.std() + 1e-12)
    vwap_z = float(dev.iloc[-1] / sess_std)
    # volume profile -> POC/VAH/VAL (70% value area)
    lo, hi = float(s_typ.min()), float(s_typ.max())
    bins = np.linspace(lo, hi, 51)
    idx = np.clip(np.digitize(s_typ, bins) - 1, 0, 49)
    prof = np.zeros(50)
    np.add.at(prof, idx, s_vol.values)
    poc_i = int(np.argmax(prof))
    order = np.argsort(-prof)
    cum, target, chosen = 0.0, prof.sum() * 0.70, []
    for i in order:
        cum += prof[i]
        chosen.append(i)
        if cum >= target:
            break
    vah = float(bins[max(chosen) + 1])
    val = float(bins[min(chosen)])
    poc = float((bins[poc_i] + bins[poc_i + 1]) / 2)
    vol_ratio = float(s_vol.tail(60).sum() / (s_vol.mean() * 60 + 1e-12))
    return {"vwap": round(vwap, 2), "vwap_zscore": round(vwap_z, 2),
            "poc": round(poc, 2), "vah": round(vah, 2), "val": round(val, 2),
            "volume_ratio": round(vol_ratio, 2), "n_bars": len(df),
            "sess_std": round(sess_std, 4),
            "last_close": round(float(df["close"].iloc[-1]), 2)}
