#!/usr/bin/env python3
"""Gold Oracle dashboard server (stdlib only).

Serves the GUI at http://localhost:8081 and JSON APIs the page polls:
  /api/state  -> current signal + history + backtest + EA status

Usage:  python3 dashboard/serve.py [--port 8081]
Refresh the model first:  python3 -m gold_oracle.signal
"""
import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(ROOT)

# Mirrors the MQL5 EA's inputs (mql5/GoldOracle_Exness.mq5)
EA = {
    "risk_pct": 1.0,
    "sl_dollars": 30,
    "tp_dollars": 60,
    "max_spread_points": 500,
    "signal_max_age_hrs": 36,
    "symbol": "XAUUSD",
    "note": "EA reads signals/gold_signal.json from MT5/MQL5/Files every 60s",
}

FACTOR_META = {
    "real_rates": ("Real yields", "TIP ETF · 20d return (rises when real yields fall)"),
    "nom_yield": ("Nominal yield", "10Y Treasury · 20d change, inverted"),
    "dollar": ("US dollar", "Dollar index (UUP) · 20d return, inverted"),
    "fed_stance": ("Fed stance", "13-week T-bill · 20d change, inverted"),
    "fear": ("Fear gauge", "VIX · 20d change"),
    "risk_app": ("Risk appetite", "S&P 500 · 20d return, inverted"),
    "btc_comp": ("Bitcoin bid", "BTC · 20d return, inverted"),
    "etf_flow": ("ETF flows", "GLD volume surge × price direction"),
    "momentum": ("Momentum", "Gold · 60d return"),
    "curve": ("Yield curve", "10Y–3M slope level"),
    "duration": ("Duration bid", "TLT (20Y Treasury) · 20d return"),
    "cb_regime": ("Central-bank buying", "WGC quarterly regime · manual input"),
}

# Plain-English translations for the user-friendly layer
FACTOR_PLAIN = {
    "real_rates": "inflation-adjusted interest rates",
    "nom_yield": "bond yields",
    "dollar": "the strength of the US dollar",
    "fed_stance": "the Fed's posture on rates",
    "fear": "fear in the markets",
    "risk_app": "investors' appetite for risk",
    "btc_comp": "Bitcoin (gold's rival for speculative money)",
    "etf_flow": "money flowing into gold ETFs",
    "momentum": "gold's own price momentum",
    "curve": "the shape of the yield curve",
    "duration": "demand for long-term bonds",
    "cb_regime": "central-bank gold buying",
}


def _translate_flaw(flaw: str, dc: dict) -> str | None:
    f = flaw.upper()
    if "INSUFFICIENT PROBABILITY" in f:
        p = dc.get("directional_probability", 0)
        need = dc.get("min_required_probability", 0.80)
        return (f"My estimated odds are only {p:.0%} — I need at least {need:.0%} "
                f"before I risk real money.")
    if "NEWS BLACKOUT" in f:
        return ("A big economic announcement is landing right now — "
                "I never trade into that storm.")
    if "SESSION FILTER" in f:
        return ("It's the market's quiet hours — I only trade when "
                "London or New York are open.")
    if "POOR R:R" in f or "RISK-TO-REWARD" in f or "R:R" in f:
        return "The potential reward isn't big enough compared to the risk."
    if "MACRO" in f and "HEADWIND" in f:
        return "The big economic picture is fighting against this trade."
    return None


def plain_summary(sig: dict) -> dict:
    """One-paragraph, jargon-free explanation of the bot's current mind."""
    dc = sig.get("decision_critic", {}) or {}
    stance = dc.get("final_audited_signal") or sig.get("signal", "FLAT")
    p = float(sig.get("p_up", 0.5) or 0.5)
    p_dir = float(dc.get("directional_probability") or max(p, 1 - p))
    flaws = dc.get("fatal_flaws", []) or []
    crits = dc.get("self_criticisms", []) or []

    translated = [t for t in (_translate_flaw(f, dc) for f in flaws[:2]) if t]

    if stance == "LONG":
        headline = (f"I'm buying gold — I give it a {p:.0%} chance of rising "
                    f"over the next 20 days.")
        change = ("If my estimated odds fall below 65%, I'll step aside. "
                  "A sudden spike in the dollar or bond yields would also make me reconsider.")
    elif stance == "SHORT":
        headline = (f"I'm betting against gold — I give it a {1 - p:.0%} chance of "
                    f"falling over the next 20 days.")
        change = ("If my estimated odds fall below 65%, I'll step aside. "
                  "Strong central-bank buying or falling real yields would make me reconsider.")
    else:
        if any("NEWS BLACKOUT" in f.upper() for f in flaws):
            headline = "I'm sitting out — big economic news is landing and I never trade into that storm."
        elif any("SESSION FILTER" in f.upper() for f in flaws):
            headline = "I'm sitting out — it's the market's quiet hours. I'll look again when London or New York opens."
        elif any("INSUFFICIENT PROBABILITY" in f.upper() for f in flaws):
            headline = (f"I'm sitting out — the odds ({p_dir:.0%}) aren't good enough "
                        f"for me to risk money on.")
        else:
            headline = "I'm sitting out — nothing in the market convinces me right now."
        need_q = dc.get("quarter_probability", 0.65)
        need_f = dc.get("min_required_probability", 0.80)
        change = (f"If my estimated odds reach {need_q:.0%}, I'd take a small test position; "
                  f"{need_f:.0%} for full size.")

    reasons = list(translated)
    # top 2 driving factors, in plain words
    factors = sig.get("factors", {}) or {}
    zscores = sig.get("factor_zscores", {}) or {}
    top = sorted(factors.items(), key=lambda kv: abs(kv[1]), reverse=True)[:2]
    for name, contrib in top:
        if abs(contrib) < 0.03:
            continue
        plain = FACTOR_PLAIN.get(name, name.replace("_", " "))
        z = zscores.get(name, 0)
        level = "unusually high" if z > 1 else ("unusually low" if z < -1 else "running hot" if z > 0.5 else "soft" if z < -0.5 else "about normal")
        direction = "pushing me toward buying gold" if contrib > 0 else "pushing me away from buying gold"
        reasons.append(f"{plain.capitalize()} — {level}, which is {direction}.")

    if crits and len(reasons) < 4:
        n = len(crits)
        reasons.append(f"My inner skeptic has {n} warning{'s' if n != 1 else ''} — I'm cautious even about my own math.")

    return {
        "headline": headline,
        "stance": stance,
        "reasons": reasons[:4],
        "what_would_change_my_mind": change,
    }


def read_json(rel, default):
    try:
        with open(os.path.join(BASE, rel)) as f:
            return json.load(f)
    except Exception:
        return default


def read_history(limit=300):
    out = []
    try:
        with open(os.path.join(BASE, "signals", "history.jsonl")) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        pass
    return out[-limit:]


def ea_status(sig):
    """What the MT5 bot would do RIGHT NOW, mirroring the EA's logic."""
    now = int(time.time())
    age_hrs = (now - sig.get("timestamp_epoch", now)) / 3600.0 if sig else 999
    stale = age_hrs > EA["signal_max_age_hrs"]
    if not sig or stale:
        action, why = "FLAT", "signal stale or missing — EA does nothing by design"
    else:
        action = sig.get("signal", "FLAT")
        why = {"LONG": f"P(up)={sig['p_up']:.2f} ≥ 0.60 — EA opens/keeps LONG",
               "SHORT": f"P(up)={sig['p_up']:.2f} ≤ 0.40 — EA opens/keeps SHORT",
               "FLAT": f"P(up)={sig['p_up']:.2f} in no-trade zone — EA stays flat",
               }.get(action, "")
    return {"action": action, "stale": stale,
            "age_hrs": round(age_hrs, 1), "why": why, "inputs": EA}


class Handler(BaseHTTPRequestHandler):
    def _send(self, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            with open(os.path.join(ROOT, "index.html"), "rb") as f:
                self._send(f.read(), "text/html; charset=utf-8")
        elif path == "/api/state":
            sig = read_json("signals/gold_signal.json", {})
            self._send(json.dumps({
                "signal": sig,
                "plain": plain_summary(sig),
                "history": read_history(),
                "backtest": read_json("data/backtest_summary.json", {}),
                "ea": ea_status(sig),
                "factor_meta": {k: {"title": v[0], "desc": v[1]}
                                for k, v in FACTOR_META.items()},
            }))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8081)
    args = ap.parse_args()
    srv = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Gold Oracle dashboard: http://localhost:{args.port}", flush=True)
    srv.serve_forever()
