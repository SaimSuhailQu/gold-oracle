#!/usr/bin/env python3
"""EA JSON contract tests — mirrors mql5/GoldOracle_Exness.mq5 parsing logic.

The EA itself can't be compiled here (no MetaEditor/MT5), so this validates
the *contract*: that the Python signal files parse the way the EA expects,
including valid, stale, legacy (no size_mult), and malformed inputs.

Usage: python scripts/test_ea_contract.py
"""

import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SIGNAL_MAX_AGE_HRS = 36


# --- mirror of the EA's JsonGet* helpers ---
def jstr(d: str, key: str) -> str:
    m = re.search(r'"' + re.escape(key) + r'"\s*:\s*"([^"]*)"', d)
    return m.group(1) if m else ""


def jnum(d: str, key: str) -> float:
    m = re.search(r'"' + re.escape(key) + r'"\s*:\s*([-+0-9.eE]+)', d)
    try:
        return float(m.group(1)) if m else 0.0
    except ValueError:
        return 0.0


def evaluate(path: str, has_position: bool = False, pos_type: str = "") -> str:
    """Returns the EA's decision string for a signal file."""
    try:
        with open(path, encoding="utf-8") as fh:
            raw = fh.read()
    except OSError:
        return "NO SIGNAL FILE"
    if not (10 < len(raw) <= 65536):
        return "BAD FILE SIZE"
    sig = jstr(raw, "signal")
    p_up = jnum(raw, "p_up")
    ts = jnum(raw, "timestamp_epoch")
    now = time.time()
    if ts <= 0 or now - ts > SIGNAL_MAX_AGE_HRS * 3600:
        return "STALE SIGNAL (do nothing)"
    if sig not in ("LONG", "SHORT"):
        return f"FLAT -> close position (p_up={p_up:.2f})" if has_position else f"FLAT (p_up={p_up:.2f})"
    if has_position and pos_type == sig:
        return f"HOLD {sig}"
    size_mult = jnum(raw, "size_mult")
    size_mult = size_mult if 0.0 < size_mult <= 1.0 else 1.0
    exposure = jnum(raw, "exposure_mult")
    exposure = exposure if 0.0 < exposure <= 2.0 else 1.0
    action = "reverse+open" if has_position else "open"
    return f"{action} {sig} (vol x {size_mult:.2f} x {exposure:.2f})"


def main() -> int:
    tmp = "/tmp/ea_contract"
    os.makedirs(tmp, exist_ok=True)
    now = int(time.time())
    cases = {
        "valid_long": ({"timestamp_epoch": now, "symbol": "XAUUSD", "p_up": 0.85,
                         "signal": "LONG", "size_mult": 1.0, "exposure_mult": 1.5}, False, ""),
        "valid_quarter": ({"timestamp_epoch": now, "symbol": "XAUUSD", "p_up": 0.70,
                            "signal": "LONG", "size_mult": 0.25, "exposure_mult": 1.0}, False, ""),
        "legacy_no_mult": ({"timestamp_epoch": now, "symbol": "XAUUSD", "p_up": 0.15,
                             "signal": "SHORT"}, False, ""),
        "flat_closes": ({"timestamp_epoch": now, "symbol": "XAUUSD", "p_up": 0.5,
                         "signal": "FLAT"}, True, "LONG"),
        "stale": ({"timestamp_epoch": now - 40 * 3600, "symbol": "XAUUSD",
                   "p_up": 0.9, "signal": "LONG"}, False, ""),
        "missing_ts": ({"symbol": "XAUUSD", "p_up": 0.9, "signal": "LONG"}, False, ""),
        "truncated": ('{"timestamp_epoch": %d, "sign' % now, False, ""),
        "empty": ("", False, ""),
        "hold": ({"timestamp_epoch": now, "symbol": "XAUUSD", "p_up": 0.85,
                  "signal": "LONG", "size_mult": 1.0}, True, "LONG"),
    }
    expect = {
        "valid_long": "open LONG (vol x 1.00 x 1.50)",
        "valid_quarter": "open LONG (vol x 0.25 x 1.00)",
        "legacy_no_mult": "open SHORT (vol x 1.00 x 1.00)",
        "flat_closes": "FLAT -> close position (p_up=0.50)",
        "stale": "STALE SIGNAL (do nothing)",
        "missing_ts": "STALE SIGNAL (do nothing)",
        "truncated": "FLAT (p_up=0.00)",
        "empty": "BAD FILE SIZE",
        "hold": "HOLD LONG",
    }
    fails = 0
    for name, (payload, hp, pt) in cases.items():
        p = os.path.join(tmp, name + ".json")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(payload if isinstance(payload, str) else json.dumps(payload))
        got = evaluate(p, hp, pt)
        ok = got == expect[name]
        fails += not ok
        print(f"[{'OK' if ok else 'FAIL'}] {name:16s} -> {got}")
    # live file
    live = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "signals", "gold_signal.json")
    if os.path.exists(live):
        print(f"[INFO] live signal -> {evaluate(live)}")
    else:
        print("[INFO] no live signal file yet")
    print("ALL PASS" if fails == 0 else f"{fails} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
