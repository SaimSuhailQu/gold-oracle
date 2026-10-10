"""
Research-paper intelligence for Gold Oracle (free & keyless).

Pulls gold forecasting papers from the arXiv API (no key), scores each
paper's directional stance from its title+abstract with a transparent
lexicon, and — more usefully — extracts which predictors the literature
keeps finding predictive, mapped onto our 12 factors. That mapping is the
real "learning": if papers keep citing the dollar and real yields as top
gold predictors, our factor weights get literature backing (or a warning
when they don't).

Memory: data/papers_memory.json (dedupe by arXiv ID, recency-weighted
consensus). refresh() does network I/O (daemon calls it);
get_papers_context() is offline and safe to call from signal generation.
"""

from __future__ import annotations

import html
import json
import os
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMORY_FILE = os.path.join(BASE, "data", "papers_memory.json")

QUERIES = [
    'ti:gold AND ti:forecasting',
    'ti:"gold price" AND abs:predict',
    'abs:"gold price forecasting"',
]
MAX_PER_QUERY = 20
RECENCY_HALFLIFE_D = 180.0  # papers stay relevant longer than videos

# predictor phrase -> our factor key
FACTOR_PHRASES: Dict[str, List[str]] = {
    "real_rates": ["real interest rate", "real yield", "tips", "inflation-adjusted yield"],
    "nom_yield": ["treasury yield", "10-year", "bond yield", "interest rate", "federal funds"],
    "dollar": ["dollar", "usd", "dxy", "exchange rate", "currency"],
    "fed_stance": ["federal reserve", "fomc", "monetary policy", "central bank policy"],
    "fear": ["vix", "volatility", "uncertainty", "risk aversion", "market fear", "geopolitical"],
    "risk_app": ["s&p", "stock market", "equity market", "risk appetite", "stock index"],
    "btc_comp": ["bitcoin", "cryptocurrency", "crypto"],
    "etf_flow": ["etf", "gold holdings", "fund flow", "investment demand"],
    "momentum": ["momentum", "trend", "technical indicator", "moving average"],
    "curve": ["yield curve", "term spread", "term structure"],
    "duration": ["duration", "long-term bond", "long bond"],
    "cb_regime": ["central bank", "gold reserve", "official sector", "gold purchase"],
}

BULLISH_FINDINGS = [
    "prices are expected to rise", "forecast an increase", "predict an increase",
    "upward trend", "bullish", "positive outlook", "expected to appreciate",
    "will increase", "upward momentum", "hedge effectiveness",
]
BEARISH_FINDINGS = [
    "prices are expected to fall", "forecast a decrease", "predict a decline",
    "downward trend", "bearish", "negative outlook", "expected to depreciate",
    "will decrease", "downward pressure",
]


class PapersEngine:
    def __init__(self, memory_file: str = MEMORY_FILE):
        self.memory_file = memory_file
        self.headers = {"User-Agent": "gold-oracle-research/1.0"}
        self.memory = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                pass
        return {"papers": {}, "consensus": 0.0, "last_updated": None}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        with open(self.memory_file, "w", encoding="utf-8") as fh:
            json.dump(self.memory, fh, indent=1)

    def _get(self, url: str, timeout: int = 25) -> str:
        req = urllib.request.Request(url, headers=self.headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", "ignore")

    @staticmethod
    def _clean(s: str) -> str:
        return html.unescape(re.sub(r"\s+", " ", s)).strip()

    def _parse(self, raw: str) -> List[Dict[str, Any]]:
        out = []
        for entry in re.findall(r"<entry>(.*?)</entry>", raw, re.DOTALL):
            try:
                aid = re.search(r"<id>(.*?)</id>", entry).group(1).split("/abs/")[-1]
                title = self._clean(re.search(r"<title>(.*?)</title>", entry, re.DOTALL).group(1))
                abstract = self._clean(re.search(r"<summary>(.*?)</summary>", entry, re.DOTALL).group(1))
                pub = re.search(r"<published>(.*?)</published>", entry).group(1)[:10]
                out.append({"arxiv_id": aid, "title": title, "abstract": abstract,
                            "published": pub})
            except Exception:
                continue
        return out

    @staticmethod
    def score_paper(title: str, abstract: str) -> Dict[str, Any]:
        text = f"{title} {abstract}".lower()
        bull = sum(1 for w in BULLISH_FINDINGS if w in text)
        bear = sum(1 for w in BEARISH_FINDINGS if w in text)
        total = bull + bear
        s = (bull - bear) / total if total else 0.0
        factors = sorted({f for f, phrases in FACTOR_PHRASES.items()
                          if any(p in text for p in phrases)})
        return {"sentiment": round(s, 3), "bull_hits": bull, "bear_hits": bear,
                "factors_mentioned": factors}

    def refresh(self) -> Dict[str, Any]:
        """Fetch new papers from arXiv. Network I/O — daemon calls this."""
        now = datetime.now(timezone.utc)
        new = 0
        for q in QUERIES:
            try:
                url = ("http://export.arxiv.org/api/query?search_query="
                       + urllib.parse.quote(q)
                       + f"&sortBy=submittedDate&sortOrder=descending&max_results={MAX_PER_QUERY}")
                for p in self._parse(self._get(url)):
                    aid = p["arxiv_id"]
                    if aid in self.memory["papers"]:
                        continue
                    score = self.score_paper(p["title"], p["abstract"])
                    self.memory["papers"][aid] = {**p, **score,
                                                  "added_at": now.isoformat()}
                    new += 1
            except Exception:
                continue
        self._recompute(now)
        self._save()
        ctx = self.get_papers_context()
        ctx["new_papers"] = new
        return ctx

    def _recompute(self, now: datetime) -> None:
        scored = []
        for p in self.memory["papers"].values():
            try:
                age_d = max(0.0, (now - datetime.fromisoformat(p["published"])).total_seconds() / 86400)
            except Exception:
                age_d = 365.0
            w = 0.5 ** (age_d / RECENCY_HALFLIFE_D)
            scored.append((w, p["sentiment"]))
        if scored:
            tw = sum(w for w, _ in scored)
            self.memory["consensus"] = round(sum(w * s for w, s in scored) / tw, 3) if tw else 0.0
        else:
            self.memory["consensus"] = 0.0
        self.memory["last_updated"] = now.isoformat()

    def get_papers_context(self) -> Dict[str, Any]:
        """Offline read for signal generation."""
        papers = list(self.memory.get("papers", {}).values())
        factor_votes: Dict[str, int] = {}
        for p in papers:
            for f in p.get("factors_mentioned", []):
                factor_votes[f] = factor_votes.get(f, 0) + 1
        top_factors = sorted(factor_votes.items(), key=lambda kv: -kv[1])[:6]
        recent = sorted(papers, key=lambda p: p.get("published", ""), reverse=True)[:5]
        return {"consensus": self.memory.get("consensus", 0.0),
                "n_papers": len(papers),
                "factor_reinforcement": dict(top_factors),
                "recent_titles": [p["title"][:90] for p in recent],
                "last_updated": self.memory.get("last_updated")}
