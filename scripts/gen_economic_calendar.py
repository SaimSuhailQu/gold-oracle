#!/usr/bin/env python3
"""Generate data/economic_calendar.json for the news blackout filter.

- NFP (Non-Farm Payrolls): first Friday of each month, 08:30 ET — computed.
- CPI / FOMC: templates — VERIFY exact dates at forexfactory.com/calendar
  before relying on them; wrong dates are worse than no dates.

Usage: python scripts/gen_economic_calendar.py [--months 12]
"""

import argparse
import calendar
import json
import os
from datetime import datetime, timezone, timedelta

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "data", "economic_calendar.json")


def first_friday(year: int, month: int) -> datetime:
    c = calendar.monthcalendar(year, month)
    day = c[0][calendar.FRIDAY] or c[1][calendar.FRIDAY]
    return datetime(year, month, day)


def et_to_utc(year: int, month: int, day: int, h: int = 8, m: int = 30) -> datetime:
    # DST 2026: Mar 8 – Nov 1 (EDT = UTC-4); else EST = UTC-5. Approx by month.
    off = 4 if 3 <= month <= 11 else 5
    return datetime(year, month, day, h, m, tzinfo=timezone.utc) + timedelta(hours=off)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=12)
    args = ap.parse_args()

    events = []
    now = datetime.now(timezone.utc)
    y, mo = now.year, now.month
    for _ in range(args.months):
        d = first_friday(y, mo)
        # NFP refers to the prior month; skip if already passed
        if d.replace(tzinfo=timezone.utc) > now - timedelta(days=7):
            t = et_to_utc(y, mo, d.day)
            events.append({"name": "NFP Non-Farm Payrolls", "utc": t.isoformat(),
                           "impact": "high", "auto": True})
        mo += 1
        if mo > 12:
            mo, y = 1, y + 1

    events.append({"name": "CPI TEMPLATE — verify date at forexfactory.com",
                   "utc": "VERIFY-ME", "impact": "high", "auto": False})
    events.append({"name": "FOMC TEMPLATE — verify date at forexfactory.com",
                   "utc": "VERIFY-ME", "impact": "high", "auto": False})

    events = [e for e in events if not e["utc"].startswith("VERIFY") or True]
    # drop un-verified templates from the active file (keep them as docs)
    active = [e for e in events if not str(e["utc"]).startswith("VERIFY")]
    active.sort(key=lambda e: e["utc"])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(active, fh, indent=1)
    print(f"wrote {len(active)} events to {OUT}")
    print("CPI/FOMC: add verified dates manually — see script header.")


if __name__ == "__main__":
    main()
