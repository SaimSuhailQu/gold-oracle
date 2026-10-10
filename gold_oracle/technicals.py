"""
Technical analysis layer for Gold Oracle (checklist items 2, 4-15, 17, 20, 23).

Computed from two data sources:
- Daily bars (yfinance): EMA 20/50/200, RSI(14), MACD(12,26,9), ADX(14), ATR(14),
  swing support/resistance.
- 1-minute bars (intraday.py): resampled to M5/M15/H1/H4 for multi-timeframe
  trend alignment, market structure (BOS/CHOCH), liquidity sweeps,
  breakout+retest, candlestick patterns, session filter.

Everything degrades gracefully when bars are short. All outputs are plain
numbers/strings — no black boxes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Any

import numpy as np
import pandas as pd


# ---------------- indicators ----------------

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = -d.clip(upper=0).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
    m = ema(close, fast) - ema(close, slow)
    s = ema(m, sig)
    return m, s, m - s


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    up, dn = np.diff(h, prepend=h[0]), -np.diff(l, prepend=l[0])
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum.reduce([h - l, np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))])
    tr[0] = h[0] - l[0]
    atr_s = pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    atr_s = np.where(atr_s == 0, np.nan, atr_s)
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1 / n, adjust=False).mean().to_numpy() / atr_s
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1 / n, adjust=False).mean().to_numpy() / atr_s
    denom = (plus_di + minus_di)
    dx = np.where(denom == 0, 0.0, 100 * np.abs(plus_di - minus_di) / np.where(denom == 0, 1, denom))
    return pd.Series(dx, index=df.index).ewm(alpha=1 / n, adjust=False).mean()


# ---------------- structure ----------------

def swings(df: pd.DataFrame, left: int = 2, right: int = 2, lookback: int = 120):
    """Fractal swing highs/lows over the recent lookback bars."""
    d = df.tail(lookback).reset_index(drop=True)
    highs, lows = [], []
    for i in range(left, len(d) - right):
        if d["high"].iloc[i] == d["high"].iloc[i - left:i + right + 1].max():
            highs.append((i, float(d["high"].iloc[i])))
        if d["low"].iloc[i] == d["low"].iloc[i - left:i + right + 1].min():
            lows.append((i, float(d["low"].iloc[i])))
    return highs, lows


def market_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """BOS/CHOCH from swing breaks on the given timeframe."""
    highs, lows = swings(df)
    if len(highs) < 2 or len(lows) < 2:
        return {"state": "UNKNOWN", "last_event": None}
    close = float(df["close"].iloc[-1])
    prev_close = float(df["close"].iloc[-2])
    last_high = highs[-1][1]
    last_low = lows[-1][1]
    # bias from swing ordering: higher highs + higher lows = up
    hh = highs[-1][1] > highs[-2][1]
    hl = lows[-1][1] > lows[-2][1]
    bias = "UP" if (hh and hl) else ("DOWN" if (not hh and not hl) else "RANGE")
    event = None
    if prev_close <= last_high and close > last_high:
        event = "BOS_UP"
    elif prev_close >= last_low and close < last_low:
        event = "BOS_DOWN"
    return {"state": bias, "last_event": event,
            "last_swing_high": round(last_high, 2), "last_swing_low": round(last_low, 2)}


def liquidity_sweeps(df: pd.DataFrame, atr_v: float) -> Dict[str, Any]:
    """Wick beyond a recent swing extreme that closes back inside = sweep."""
    if atr_v <= 0 or len(df) < 30:
        return {"bullish_sweep": False, "bearish_sweep": False}
    highs, lows = swings(df.tail(60))
    if not highs or not lows:
        return {"bullish_sweep": False, "bearish_sweep": False}
    last = df.iloc[-1]
    bull = (last["low"] < lows[-1][1] - 0.1 * atr_v and last["close"] > lows[-1][1])
    bear = (last["high"] > highs[-1][1] + 0.1 * atr_v and last["close"] < highs[-1][1])
    return {"bullish_sweep": bool(bull), "bearish_sweep": bool(bear)}


def breakout_retest(df: pd.DataFrame) -> Dict[str, Any]:
    """Close beyond a swing extreme, then touch it from the other side and hold."""
    highs, lows = swings(df, lookback=90)
    if len(highs) < 1 or len(lows) < 1 or len(df) < 12:
        return {"bullish": False, "bearish": False}
    d = df.reset_index(drop=True)
    bull = bear = False
    rh, rl = highs[-1][1], lows[-1][1]
    for i in range(len(d) - 8, len(d)):
        if d["close"].iloc[i] > rh and i + 1 < len(d):
            # retest: subsequent low touches rh zone and closes above
            zone = d.iloc[i + 1:]
            touched = (zone["low"] <= rh * 1.001).any()
            held = zone["close"].iloc[-1] > rh
            if touched and held:
                bull = True
        if d["close"].iloc[i] < rl and i + 1 < len(d):
            zone = d.iloc[i + 1:]
            touched = (zone["high"] >= rl * 0.999).any()
            held = zone["close"].iloc[-1] < rl
            if touched and held:
                bear = True
    return {"bullish": bool(bull), "bearish": bool(bear)}


def candle_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """Last-bar engulfing / pin bar / doji."""
    if len(df) < 2:
        return {"pattern": None}
    a, b = df.iloc[-2], df.iloc[-1]
    o1, c1, o2, c2 = a["open"], a["close"], b["open"], b["close"]
    h2, l2 = b["high"], b["low"]
    body = abs(c2 - o2)
    rng = h2 - l2
    pat = None
    if c2 > o2 and c1 < o1 and o2 <= c1 and c2 >= o1 and body > abs(c1 - o1):
        pat = "BULLISH_ENGULFING"
    elif c2 < o2 and c1 > o1 and o2 >= c1 and c2 <= o1 and body > abs(c1 - o1):
        pat = "BEARISH_ENGULFING"
    elif rng > 0 and max(h2 - max(o2, c2), min(o2, c2) - l2) > 2 * body and body < 0.3 * rng:
        pat = "PIN_BAR_BULLISH" if c2 > o2 else "PIN_BAR_BEARISH"
    elif rng > 0 and body < 0.1 * rng:
        pat = "DOJI"
    return {"pattern": pat}


# ---------------- sessions ----------------

def current_session(now: datetime | None = None) -> Dict[str, Any]:
    """London 07-12 UTC, New York 12-17 UTC (overlap 12-16); trade the majors."""
    now = now or datetime.now(timezone.utc)
    h = now.hour + now.minute / 60.0
    if 7 <= h < 12:
        sess = "LONDON"
    elif 12 <= h < 17:
        sess = "NEW_YORK"
    elif 0 <= h < 7:
        sess = "ASIAN"
    else:
        sess = "OFF_HOURS"
    trade_ok = sess in ("LONDON", "NEW_YORK")
    return {"session": sess, "trade_ok": trade_ok, "utc_hour": round(h, 2)}


# ---------------- main ----------------

def resample_ohlc(bars: pd.DataFrame, rule: str) -> pd.DataFrame:
    d = bars.set_index("time").sort_index()
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return d.resample(rule).agg(agg).dropna().reset_index()


def compute_technicals(daily: pd.DataFrame, bars_1m: pd.DataFrame | None) -> Dict[str, Any]:
    """Full technical stack. daily: DataFrame with open/high/low/close(.volume).
    bars_1m: 1-minute bars or None (structure pieces degrade gracefully)."""
    t: Dict[str, Any] = {}
    d = daily.reset_index(drop=True)

    # --- daily indicators ---
    t["ema20"] = round(float(ema(d["close"], 20).iloc[-1]), 2)
    t["ema50"] = round(float(ema(d["close"], 50).iloc[-1]), 2)
    t["ema200"] = round(float(ema(d["close"], 200).iloc[-1]), 2) if len(d) >= 200 else None
    t["rsi14"] = round(float(rsi(d["close"]).iloc[-1]), 1)
    mline, sline, hist = macd(d["close"])
    t["macd_hist"] = round(float(hist.iloc[-1]), 3)
    t["macd_bullish"] = bool(mline.iloc[-1] > sline.iloc[-1])
    t["adx14"] = round(float(adx(d).iloc[-1]), 1)
    t["atr14"] = round(float(atr(d).iloc[-1]), 2)
    t["trend_strong"] = bool(t["adx14"] > 25)

    # daily S/R from swings
    highs, lows = swings(d)
    t["resistance"] = round(highs[-1][1], 2) if highs else None
    t["support"] = round(lows[-1][1], 2) if lows else None

    # --- multi-timeframe trend (needs 1m bars) ---
    mtf: Dict[str, Any] = {}
    struct: Dict[str, Any] = {"state": "UNKNOWN", "last_event": None}
    sweeps = {"bullish_sweep": False, "bearish_sweep": False}
    br = {"bullish": False, "bearish": False}
    pat = {"pattern": None}
    if bars_1m is not None and len(bars_1m) >= 60:
        tfs = {}
        for name, rule, min_bars in (("M5", "5min", 60), ("M15", "15min", 60),
                                     ("H1", "1h", 60), ("H4", "4h", 12)):
            try:
                rb = resample_ohlc(bars_1m, rule)
            except Exception:
                continue
            if len(rb) < min_bars:
                continue
            e20, e50 = ema(rb["close"], 20), ema(rb["close"], 50)
            if len(rb) >= 50:
                tr = 1 if e20.iloc[-1] > e50.iloc[-1] else -1
            else:
                tr = 1 if rb["close"].iloc[-1] > e50.iloc[-1] else -1
            tfs[name] = tr
        mtf = tfs
        # structure on M15, patterns on H1
        try:
            m15 = resample_ohlc(bars_1m, "15min")
            if len(m15) >= 60:
                struct = market_structure(m15)
                a14 = float(atr(m15).iloc[-1])
                sweeps = liquidity_sweeps(m15, a14)
                br = breakout_retest(m15)
            h1 = resample_ohlc(bars_1m, "1h")
            if len(h1) >= 12:
                pat = candle_patterns(h1)
        except Exception:
            pass
    t["mtf_trends"] = mtf
    t["mtf_score"] = round(sum(mtf.values()) / len(mtf), 2) if mtf else 0.0
    t["mtf_aligned_bull"] = all(v > 0 for v in mtf.values()) if mtf else False
    t["mtf_aligned_bear"] = all(v < 0 for v in mtf.values()) if mtf else False
    t["market_structure"] = struct
    t["liquidity_sweep"] = sweeps
    t["breakout_retest"] = br
    t["candle_pattern"] = pat["pattern"]

    # --- sessions ---
    t["session"] = current_session()

    # --- composite technical score [-1, 1] ---
    score = 0.0
    score += 0.30 * t["mtf_score"]
    rsi_v = t["rsi14"]
    score += 0.15 * max(-1.0, min(1.0, (rsi_v - 50) / 30))
    score += 0.15 * (1 if t["macd_bullish"] else -1)
    score += 0.10 * (1 if (t["trend_strong"] and t["mtf_score"] > 0) else (-1 if (t["trend_strong"] and t["mtf_score"] < 0) else 0))
    if struct.get("last_event") == "BOS_UP":
        score += 0.10
    elif struct.get("last_event") == "BOS_DOWN":
        score -= 0.10
    if sweeps["bullish_sweep"] or br["bullish"] or pat["pattern"] in ("BULLISH_ENGULFING", "PIN_BAR_BULLISH"):
        score += 0.10
    if sweeps["bearish_sweep"] or br["bearish"] or pat["pattern"] in ("BEARISH_ENGULFING", "PIN_BAR_BEARISH"):
        score -= 0.10
    t["tech_score"] = round(max(-1.0, min(1.0, score)), 3)
    return t
