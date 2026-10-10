"""
Narrative sentiment calibration for Gold Oracle.

Problem it fixes: raw lexicon sentiment lives in [-1, 1] and pegs at 1.0 on
any vaguely bullish news day, so the ensemble's ±10% nudge fires at maximum
on noise. Instead we track every narrative score in data/narrative_history.jsonl
and convert the current score to a rolling 30-day z-score. Only genuine
anomalies (|z| > 2) move the needle; an ordinary bullish day scores ~+0.5
and barely nudges the ensemble.

Also defines the pluggable sentiment backend: LexiconBackend (current,
keyless) today; FinBERTBackend (HuggingFace `ProsusAI/finbert`, needs
torch+transformers, ~500MB download) or any LLM API can be dropped in via
set_backend() without touching signal.py.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY_FILE = os.path.join(BASE, "data", "narrative_history.jsonl")
WINDOW = 30
MIN_POINTS = 5


class SentimentBackend:
    """Pluggable scorer: headline -> [-1, 1]."""
    name = "base"

    def score(self, text: str) -> float:
        raise NotImplementedError


class LexiconBackend(SentimentBackend):
    """Current keyless lexicon scorer (FinBERT proxy)."""
    name = "lexicon"
    BULL = {"cut", "cuts", "easing", "rally", "soar", "record", "buying",
            "accumulate", "debasement", "deficit", "stimulus", "crisis",
            "tension", "war", "safe haven", "hedge", "inflows", "gold highs"}
    BEAR = {"hike", "hawkish", "higher for longer", "selloff", "plunge",
            "tumble", "outflows", "strong dollar", "yields surge"}

    def score(self, text: str) -> float:
        t = text.lower()
        b = sum(1 for w in self.BULL if w in t)
        r = sum(1 for w in self.BEAR if w in t)
        return (b - r) / (b + r) if (b + r) else 0.0


class FinBERTBackend(SentimentBackend):
    """Finance-specific transformer. Needs: pip install torch transformers.

    Downloads ProsusAI/finbert (~440MB) on first use. If anything is missing
    it raises at construction so callers can fall back to the lexicon.
    """
    name = "finbert"

    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.model.eval()

    def score(self, text: str) -> float:
        import torch
        with torch.no_grad():
            ids = self.tok(text[:512], return_tensors="pt", truncation=True)
            probs = self.model(**ids).logits.softmax(dim=-1)[0]
        # finbert labels: 0=positive, 1=negative, 2=neutral
        return float(probs[0] - probs[1])


_BACKEND: SentimentBackend = LexiconBackend()


def set_backend(b: SentimentBackend) -> None:
    global _BACKEND
    _BACKEND = b


def get_backend() -> SentimentBackend:
    return _BACKEND


def append_history(entry: Dict[str, Any]) -> None:
    entry = {"ts": datetime.now(timezone.utc).isoformat(), **entry}
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def _load_scores(window: int = WINDOW) -> List[float]:
    if not os.path.exists(HISTORY_FILE):
        return []
    scores = []
    with open(HISTORY_FILE, encoding="utf-8") as fh:
        for line in fh:
            try:
                scores.append(float(json.loads(line)["blended"]))
            except Exception:
                continue
    return scores[-window:]


def zscore_current(blended: float, window: int = WINDOW) -> Dict[str, Any]:
    """Rolling z-score of today's narrative vs trailing window.

    Returns z clipped to [-3, 3]; z=None until MIN_POINTS of history exist
    (then the nudge is 0 — no calibration, no opinion).
    """
    hist = _load_scores(window)
    if len(hist) < MIN_POINTS:
        return {"z": None, "mean": None, "std": None, "n": len(hist)}
    mean = sum(hist) / len(hist)
    var = sum((x - mean) ** 2 for x in hist) / len(hist)
    std = var ** 0.5
    z = 0.0 if std < 1e-9 else (blended - mean) / std
    z = max(-3.0, min(3.0, z))
    return {"z": round(z, 3), "mean": round(mean, 4), "std": round(std, 4), "n": len(hist)}
