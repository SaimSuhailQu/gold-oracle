"""
YouTube Channel Intelligence + Learning Memory Engine (free & keyless).

Watches a curated list of gold/macro YouTube channels via their official RSS
feeds (no API key), scores each video's gold stance with a transparent
lexicon, and persists everything to data/youtube_memory.json.

Learning loop: every processed video stores the gold price at watch time.
Once a video is >= 20 days old the engine checks whether gold actually moved
the way the video's stance implied, building a per-channel hit-rate. Channels
that cry wolf get down-weighted automatically in the collective sentiment.

Channel list lives in data/watch_channels.json (handles, human-editable).
Handles are resolved to channel IDs at runtime and cached.
"""

from __future__ import annotations

import html
import json
import os
import re
import statistics
import urllib.request
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMORY_FILE = os.path.join(BASE, "data", "youtube_memory.json")
CHANNELS_FILE = os.path.join(BASE, "data", "watch_channels.json")
SIGNAL_FILE = os.path.join(BASE, "signals", "gold_signal.json")

# Verified working (handle -> resolved live, Oct 2026). Edit watch_channels.json
# to add/remove — handles are resolved to channel IDs on first run.
DEFAULT_CHANNELS = [
    {"handle": "@BloombergTelevision", "label": "Bloomberg Television"},
    {"handle": "@RealVisionFinance", "label": "Real Vision Finance"},
    {"handle": "@PeterSchiff", "label": "Peter Schiff"},
    {"handle": "@MilkRoadMacro", "label": "Milk Road Macro"},
]

HORIZON_DAYS = 20          # same horizon the model predicts
RECENCY_HALFLIFE_D = 7.0   # recent videos count more in collective sentiment

BULLISH_TERMS = [
    "gold rally", "buy gold", "buying gold", "breakout", "all-time high",
    "all time high", "rate cut", "rate cuts", "easing", "central bank buying",
    "dollar collapse", "dollar weakness", "weaker dollar", "inflation hedge",
    "devaluation", "safe haven", "gold to $", "gold $", "new high",
    "bull market", "bullish gold", "target raised", "outperform",
]
BEARISH_TERMS = [
    "gold crash", "sell gold", "selling gold", "top is in", "rate hike",
    "hawkish", "strong dollar", "dollar strength", "yields rising",
    "rising yields", "bearish gold", "reversal", "dump", "bubble",
    "overbought", "correction coming", "gold plunge",
]


class YouTubeMemoryEngine:
    def __init__(self, memory_file: str = MEMORY_FILE,
                 channels_file: str = CHANNELS_FILE):
        self.memory_file = memory_file
        self.channels_file = channels_file
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        self.memory = self._load_memory()
        self._ensure_channels()

    # ---------- persistence ----------
    def _load_memory(self) -> Dict[str, Any]:
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, encoding="utf-8") as fh:
                    m = json.load(fh)
                m.setdefault("channels", [])
                m.setdefault("processed_videos", {})
                return m
            except Exception:
                pass
        return {"channels": [], "processed_videos": {},
                "collective_sentiment": 0.0, "last_updated": None}

    def _save_memory(self) -> None:
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        with open(self.memory_file, "w", encoding="utf-8") as fh:
            json.dump(self.memory, fh, indent=1)

    def _ensure_channels(self) -> None:
        """Seed data/watch_channels.json with defaults if missing."""
        if not os.path.exists(self.channels_file):
            os.makedirs(os.path.dirname(self.channels_file), exist_ok=True)
            with open(self.channels_file, "w", encoding="utf-8") as fh:
                json.dump(DEFAULT_CHANNELS, fh, indent=1)

    def _load_channel_list(self) -> List[Dict[str, str]]:
        try:
            with open(self.channels_file, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return list(DEFAULT_CHANNELS)

    # ---------- fetching ----------
    def _get(self, url: str, timeout: int = 12) -> str:
        req = urllib.request.Request(url, headers=self.headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", "ignore")

    def resolve_handle(self, handle: str) -> Optional[str]:
        """@Handle -> UC... channel ID via the page's canonical link (cached)."""
        cache = self.memory.setdefault("handle_cache", {})
        if handle in cache:
            return cache[handle]
        try:
            raw = self._get(f"https://www.youtube.com/{handle}")
            m = (re.search(r'<link rel="canonical" href="https://www.youtube.com/channel/(UC[^"]+)"', raw)
                 or re.search(r'"externalId":"(UC[^"]+)"', raw))
            if m:
                cache[handle] = m.group(1)
                self._save_memory()
                return m.group(1)
        except Exception:
            pass
        return None

    def _fetch_via_ytdlp(self, channel_id: str, limit: int) -> List[Dict[str, str]]:
        """Optional fallback: yt-dlp if installed (handles some blocks RSS can't)."""
        import shutil
        import subprocess
        if not shutil.which("yt-dlp"):
            return []
        try:
            out = subprocess.run(
                ["yt-dlp", "--no-check-certificate", "--flat-playlist",
                 "--print", "%(id)s\t%(title)s\t%(timestamp)s",
                 "--playlist-end", str(limit),
                 f"https://www.youtube.com/channel/{channel_id}/videos"],
                capture_output=True, text=True, timeout=90)
            videos = []
            for line in out.stdout.splitlines():
                parts = line.split("\t")
                if len(parts) >= 2 and len(parts[0]) == 11:
                    pub = ""
                    if len(parts) >= 3 and parts[2].isdigit():
                        pub = datetime.fromtimestamp(int(parts[2]), tz=timezone.utc).isoformat()
                    videos.append({"video_id": parts[0], "title": parts[1],
                                   "description": "", "published": pub,
                                   "url": f"https://www.youtube.com/watch?v={parts[0]}"})
            return videos
        except Exception:
            return []

    def fetch_channel_videos(self, channel_id: str, limit: int = 8) -> List[Dict[str, str]]:
        """Latest videos: official keyless RSS feed first, yt-dlp fallback."""
        videos = self._fetch_via_rss(channel_id, limit)
        if videos:
            self.memory["fetch_status"] = "rss_ok"
            return videos
        videos = self._fetch_via_ytdlp(channel_id, limit)
        if videos:
            self.memory["fetch_status"] = "ytdlp_ok"
            return videos
        self.memory["fetch_status"] = (
            "blocked_from_this_network (YouTube 404s the feed for datacenter IPs; "
            "works on residential connections; optional: install yt-dlp)")
        return []

    def _fetch_via_rss(self, channel_id: str, limit: int = 8) -> List[Dict[str, str]]:
        """Latest videos via the official keyless RSS feed (title + description)."""
        url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
        videos: List[Dict[str, str]] = []
        try:
            raw = self._get(url)
            for entry in re.findall(r"<entry>(.*?)</entry>", raw, re.DOTALL)[:limit]:
                vid = re.search(r"<yt:videoId>(.*?)</yt:videoId>", entry)
                title = re.search(r"<title>(.*?)</title>", entry)
                pub = re.search(r"<published>(.*?)</published>", entry)
                desc = re.search(r"<media:description>(.*?)</media:description>", entry, re.DOTALL)
                if vid and title:
                    videos.append({
                        "video_id": vid.group(1),
                        "title": html.unescape(title.group(1)).strip(),
                        "description": html.unescape(desc.group(1)).strip()[:600] if desc else "",
                        "published": pub.group(1) if pub else "",
                        "url": f"https://www.youtube.com/watch?v={vid.group(1)}",
                    })
        except Exception:
            pass
        return videos

    def extract_transcript(self, video_id: str) -> str:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            items = YouTubeTranscriptApi.get_transcript(video_id)
            return " ".join(i.get("text", "") for i in items)[:4000]
        except Exception:
            return ""

    # ---------- analysis ----------
    @staticmethod
    def analyze_content(text: str, title: str) -> Dict[str, Any]:
        combined = f"{title} {text}".lower()
        bull = sum(1 for w in BULLISH_TERMS if w in combined)
        bear = sum(1 for w in BEARISH_TERMS if w in combined)
        total = bull + bear
        s = (bull - bear) / total if total else 0.0
        return {"sentiment_score": round(s, 3), "bullish_signals": bull,
                "bearish_signals": bear,
                "summary_tag": "BULLISH" if s > 0.2 else ("BEARISH" if s < -0.2 else "NEUTRAL")}

    @staticmethod
    def _gold_now() -> Optional[float]:
        """Gold price from the latest signal file (no network)."""
        try:
            with open(SIGNAL_FILE, encoding="utf-8") as fh:
                return float(json.load(fh).get("gold_price"))
        except Exception:
            return None

    # ---------- learning loop ----------
    def process_all_channels(self) -> Dict[str, Any]:
        """Fetch new videos, score them, and grade old predictions. Network I/O."""
        now = datetime.now(timezone.utc)
        gold_now = self._gold_now()
        new_videos = 0

        for ch in self._load_channel_list():
            handle = ch.get("handle", "")
            cid = self.resolve_handle(handle)
            if not cid:
                continue
            for v in self.fetch_channel_videos(cid):
                vid = v["video_id"]
                if vid in self.memory["processed_videos"]:
                    continue
                text = v["description"] or self.extract_transcript(vid)
                analysis = self.analyze_content(text, v["title"])
                self.memory["processed_videos"][vid] = {
                    "channel": ch.get("label", handle),
                    "handle": handle,
                    "title": v["title"], "url": v["url"],
                    "published": v.get("published"),
                    "analysis": analysis,
                    "gold_at_watch": gold_now,
                    "processed_at": now.isoformat(),
                    "graded": False, "hit": None,
                }
                new_videos += 1

        graded = self._grade_old_videos(now, gold_now)
        self._recompute(now)
        self._save_memory()
        return {"new_videos_processed": new_videos,
                "newly_graded": graded,
                "total_stored_videos": len(self.memory["processed_videos"]),
                "collective_sentiment": self.memory["collective_sentiment"],
                "channels_tracked": len(self._load_channel_list())}

    def _grade_old_videos(self, now: datetime, gold_now: Optional[float]) -> int:
        """Videos old enough to judge: did gold move with their stance?"""
        if gold_now is None:
            return 0
        graded = 0
        cutoff = now - timedelta(days=HORIZON_DAYS)
        for vid, rec in self.memory["processed_videos"].items():
            if rec.get("graded") or rec.get("gold_at_watch") is None:
                continue
            try:
                watched = datetime.fromisoformat(rec["processed_at"])
            except Exception:
                continue
            if watched > cutoff:
                continue
            s = rec["analysis"]["sentiment_score"]
            if abs(s) < 0.2:  # neutral takes no stance — not graded
                rec["graded"] = True
                continue
            moved_up = gold_now > rec["gold_at_watch"]
            rec["hit"] = bool((s > 0) == moved_up)
            rec["graded"] = True
            graded += 1
        return graded

    def _recompute(self, now: datetime) -> None:
        """Recency-weighted collective sentiment; per-channel accuracy."""
        scored = []
        for rec in self.memory["processed_videos"].values():
            try:
                age_d = max(0.0, (now - datetime.fromisoformat(rec["processed_at"])).total_seconds() / 86400)
            except Exception:
                age_d = 30.0
            w = 0.5 ** (age_d / RECENCY_HALFLIFE_D)
            # down-weight channels with a proven-bad record (min 10 graded)
            acc = self.channel_accuracy(rec.get("handle", ""))
            if acc is not None and acc[1] >= 10 and acc[0] < 0.4:
                w *= 0.5
            scored.append((w, rec["analysis"]["sentiment_score"]))
        if scored:
            tw = sum(w for w, _ in scored)
            self.memory["collective_sentiment"] = round(sum(w * s for w, s in scored) / tw, 3) if tw else 0.0
        else:
            self.memory["collective_sentiment"] = 0.0
        self.memory["last_updated"] = now.isoformat()

    def channel_accuracy(self, handle: str) -> Optional[tuple]:
        """(hit_rate, n_graded) for one channel, or None if nothing graded."""
        hits = [r["hit"] for r in self.memory["processed_videos"].values()
                if r.get("handle") == handle and r.get("graded") and r.get("hit") is not None]
        if not hits:
            return None
        return (sum(hits) / len(hits), len(hits))

    # ---------- read API (no network) ----------
    def get_memory_context(self) -> Dict[str, Any]:
        chans = []
        for ch in self._load_channel_list():
            acc = self.channel_accuracy(ch.get("handle", ""))
            chans.append({"label": ch.get("label"), "handle": ch.get("handle"),
                          "hit_rate": round(acc[0], 3) if acc else None,
                          "graded": acc[1] if acc else 0})
        return {"collective_sentiment": self.memory.get("collective_sentiment", 0.0),
                "total_memory_nodes": len(self.memory.get("processed_videos", {})),
                "channels_tracked": len(self._load_channel_list()),
                "channels": chans,
                "fetch_status": self.memory.get("fetch_status", "not_run_yet"),
                "last_updated": self.memory.get("last_updated")}
