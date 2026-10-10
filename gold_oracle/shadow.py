"""
Shadow-logging pipeline for the unproven ML/AI layers.

The ensemble, narrative, microstructure and critic layers have no walk-forward
history. Instead of routing them to capital on faith, every signal run appends
one line to data/shadow_log.jsonl recording what EACH layer recommended:

    {date, p_base, p_ensemble, narrative_z, critic_tier, base_stance,
     ensemble_stance, critic_stance, gold_price}

Nothing here touches execution. After ~3 months, `python -m gold_oracle.shadow
--report` grades mature entries (20-day forward gold move) per layer, giving
a statistical out-of-sample verdict before any capital allocation.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from typing import Any, Dict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHADOW_FILE = os.path.join(BASE, "data", "shadow_log.jsonl")
SIGNAL_FILE = os.path.join(BASE, "signals", "gold_signal.json")
HORIZON_DAYS = 20


def _stance(p: float, long_t: float, short_t: float) -> str:
    return "LONG" if p >= long_t else ("SHORT" if p <= short_t else "FLAT")


def log_daily(sig: Dict[str, Any]) -> Dict[str, Any]:
    """Record one shadow entry from a generated signal dict. No execution."""
    ens = sig.get("ensemble_ml", {}) or {}
    p_base = float(ens.get("p_linear_logistic", sig.get("p_up", 0.5)))
    p_ens = float(sig.get("p_up", 0.5))
    long_t = float(sig.get("long_t", 0.80))
    short_t = float(sig.get("short_t", 0.20))
    rec = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "ts": datetime.now(timezone.utc).isoformat(),
        "p_base": round(p_base, 4),
        "p_ensemble": round(p_ens, 4),
        "narrative_z": (sig.get("ai_sentiment", {}) or {}).get("narrative_zscore", {}).get("z"),
        "critic_tier": (sig.get("decision_critic", {}) or {}).get("size_tier", "FLAT"),
        "base_stance": _stance(p_base, long_t, short_t),
        "ensemble_stance": _stance(p_ens, long_t, short_t),
        "critic_stance": sig.get("signal", "FLAT"),
        "gold_price": sig.get("gold_price"),
        "graded": False,
    }
    os.makedirs(os.path.dirname(SHADOW_FILE), exist_ok=True)
    with open(SHADOW_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def _gold_now() -> float | None:
    try:
        with open(SIGNAL_FILE, encoding="utf-8") as fh:
            return float(json.load(fh).get("gold_price"))
    except Exception:
        return None


def grade() -> int:
    """Grade entries old enough to judge (20d forward move). Returns # graded."""
    if not os.path.exists(SHADOW_FILE):
        return 0
    gold_now = _gold_now()
    if gold_now is None:
        return 0
    now = datetime.now(timezone.utc)
    recs, n = [], 0
    with open(SHADOW_FILE, encoding="utf-8") as fh:
        for line in fh:
            try:
                recs.append(json.loads(line))
            except Exception:
                continue
    for r in recs:
        if r.get("graded"):
            continue
        try:
            age = (now - datetime.fromisoformat(r["ts"])).days
        except Exception:
            continue
        if age < HORIZON_DAYS or r.get("gold_price") is None:
            continue
        up = gold_now > r["gold_price"]
        for layer, key in (("base", "base_stance"), ("ensemble", "ensemble_stance"),
                           ("critic", "critic_stance")):
            st = r[key]
            r[f"{layer}_hit"] = None if st == "FLAT" else bool((st == "LONG") == up)
        r["graded"] = True
        n += 1
    with open(SHADOW_FILE, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    return n


def report() -> Dict[str, Any]:
    """Per-layer out-of-sample hit rates on graded entries."""
    grade()
    layers = ["base", "ensemble", "critic"]
    stats = {l: {"hits": 0, "n": 0} for l in layers}
    total = 0
    if os.path.exists(SHADOW_FILE):
        with open(SHADOW_FILE, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                total += 1
                if not r.get("graded"):
                    continue
                for l in layers:
                    h = r.get(f"{l}_hit")
                    if h is not None:
                        stats[l]["n"] += 1
                        stats[l]["hits"] += int(h)
    out = {"entries": total, "layers": {}}
    for l in layers:
        s = stats[l]
        out["layers"][l] = {"hit_rate": round(s["hits"] / s["n"], 3) if s["n"] else None,
                            "n": s["n"]}
    return out


if __name__ == "__main__":
    import sys
    if "--report" in sys.argv:
        r = report()
        print(f"shadow entries: {r['entries']}")
        for l, s in r["layers"].items():
            hr = f"{s['hit_rate']:.1%}" if s["hit_rate"] is not None else "n/a"
            print(f"  {l:8s} hit rate {hr}  (n={s['n']})")
        print("Need ~60+ graded entries per layer before trusting a layer with capital.")
    else:
        print("usage: python -m gold_oracle.shadow --report")
