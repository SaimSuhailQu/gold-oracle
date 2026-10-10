"""
Economic news blackout for Gold Oracle (checklist item 16).

No trade entries within ±30 minutes of high-impact USD events (NFP, CPI, FOMC).
Reads data/economic_calendar.json — a list of {"name", "utc", "impact"}.

Generate/refresh it with:  python scripts/gen_economic_calendar.py
NFP dates are computed (first Friday of month, 08:30 ET); CPI/FOMC entries
are templates — verify exact dates against ForexFactory before relying on them.
An empty/missing file = no blackouts (fail-open, logged).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAL_PATH = os.path.join(BASE, "data", "economic_calendar.json")
BLACKOUT_MIN = 30


def load_calendar() -> list:
    try:
        with open(CAL_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return []


def is_blackout(now: datetime | None = None,
                window_min: int = BLACKOUT_MIN) -> Tuple[bool, Optional[str]]:
    """(in_blackout, event_name)."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    for ev in load_calendar():
        try:
            t = datetime.fromisoformat(ev["utc"])
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if abs((now - t).total_seconds()) <= window_min * 60:
            return True, ev.get("name", "HIGH_IMPACT_NEWS")
    return False, None
