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
