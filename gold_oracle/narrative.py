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
    """Keyless lexicon scorer (FinBERT proxy) with negation, intensifiers
    and clause boundaries — so "not hawkish" doesn't count as bearish and
    "gold rallies as dollar strengthens" doesn't cancel itself out."""
    name = "lexicon"
    BULL = {"cut", "cuts", "cutting", "easing", "ease", "rally", "rallies",
            "soar", "soars", "record", "records", "buying", "buys",
            "accumulate", "accumulation", "debasement", "deficit", "deficits",
            "stimulus", "crisis", "tension", "tensions", "war",
            "safe haven", "safe-haven", "hedge", "inflows", "inflow",
            "gold highs", "all-time high", "breakout", "surge", "surged",
            "jumps", "climbs", "dovish", "pivot"}
    BEAR = {"hike", "hikes", "hiking", "hawkish", "higher for longer",
            "selloff", "sell-off", "plunge", "plunges", "plunged",
            "tumble", "tumbles", "tumbled", "outflows", "outflow",
            "strong dollar", "dollar strength", "yields surge",
            "yields jump", "sell", "sells", "drops", "falls", "slump"}
    NEGATORS = {"not", "no", "never", "n't", "without", "fails", "failed",
                "unlikely", "doubt", "delays", "delayed", "pauses", "paused",
                "backs away", "rules out", "dismisses"}
    INTENSIFIERS = {"sharply": 1.5, "steeply": 1.5, "surged": 1.4,
                    "plunged": 1.4, "soaring": 1.4, "collapses": 1.4,
                    "very": 1.3, "highly": 1.3, "strongly": 1.3,
                    "slightly": 0.6, "modestly": 0.6, "marginally": 0.5}

    def score(self, text: str) -> float:
        # split into clauses so mixed headlines don't cancel out
        clauses = [c.strip() for c in text.lower().replace(";", ",").split(",")]
        total, weight = 0.0, 0.0
        for clause in clauses:
            words = clause.split()
            c_score, c_w = 0.0, 0.0
            # multi-word negators ("rules out", "backs away") arm negation
            # for the whole clause — headlines are short enough for this
            negated = any(p in clause for p in self.NEGATORS if " " in p)
            mult = 1.0
            for w in words:
                w = w.strip(".,:!?()\"'")
                if w in self.NEGATORS or w.endswith("n't"):
                    negated = True
                    continue
                if w in self.INTENSIFIERS:
                    mult = self.INTENSIFIERS[w]
                    continue
                s = 0
                if w in self.BULL:
                    s = 1
                elif w in self.BEAR:
                    s = -1
                else:
                    # multi-word phrases
                    continue
                if negated:
                    s = -s
                    negated = False  # negator applies to the next sentiment word
                c_score += s * mult
                c_w += mult
                mult = 1.0
            # whole-phrase check for phrases the word loop missed
            for phrase in self.BULL:
                if " " in phrase and phrase in clause:
                    c_score += 1.0
                    c_w += 1.0
            for phrase in self.BEAR:
                if " " in phrase and phrase in clause:
                    c_score -= 1.0
                    c_w += 1.0
            if c_w:
                total += c_score / c_w
                weight += 1.0
        if not weight:
            return 0.0
        return max(-1.0, min(1.0, total / weight))


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
