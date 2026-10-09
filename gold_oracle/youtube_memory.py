"""
YouTube Channel & Video Intelligence Memory Engine
Free & Keyless:
1. Monitors specified YouTube channel RSS feeds (No Google API key required)
2. Extracts video metadata and automated/manual transcripts
3. Analyzes video transcripts for macro, gold, and central bank trading insights
4. Persists processed insights into an episodic memory store (`data/youtube_memory.json`)
5. Feeds collective channel memory scores directly into the Gold Oracle AI decision brain
"""

from typing import Dict, List, Any, Optional
import os
import re
import json
import urllib.request
import urllib.parse
from datetime import datetime, timezone


MEMORY_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "youtube_memory.json"
)


class YouTubeMemoryEngine:
    """
    Watches YouTube channels, transcribes/analyzes their commentary,
    and updates long-term trading memory.
    """

    def __init__(self, memory_file_path: str = MEMORY_FILE):
        self.memory_file = memory_file_path
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        self.memory = self._load_memory()

    def _load_memory(self) -> Dict[str, Any]:
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "channels": [],
            "processed_videos": {},
            "collective_sentiment": 0.0,
            "last_updated": None
        }

    def _save_memory(self):
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, indent=2)

    def add_channel(self, channel_id_or_url: str, label: str = ""):
        """
        Adds a channel to watch list. Accepts channel ID (e.g. UC...) or handle.
        """
        if channel_id_or_url not in [c.get("id") for c in self.memory["channels"]]:
            self.memory["channels"].append({
                "id": channel_id_or_url,
                "label": label or channel_id_or_url,
                "added_at": datetime.now(timezone.utc).isoformat()
            })
            self._save_memory()

    def fetch_channel_videos(self, channel_id: str) -> List[Dict[str, str]]:
        """
        Fetches latest videos from a YouTube channel via keyless official XML feed.
        URL format: https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}
        """
        url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
        videos = []
        try:
            req = urllib.request.Request(url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                xml_data = resp.read().decode("utf-8")
                
                # Extract video IDs and titles via regex without external xml dependencies
                entries = re.findall(r"<entry>(.*?)</entry>", xml_data, re.DOTALL)
                for entry in entries[:5]:  # Process top 5 recent videos
                    vid_match = re.search(r"<yt:videoId>(.*?)</yt:videoId>", entry)
                    title_match = re.search(r"<title>(.*?)</title>", entry)
                    published_match = re.search(r"<published>(.*?)</published>", entry)

                    if vid_match and title_match:
                        vid_id = vid_match.group(1)
                        title = title_match.group(1)
                        pub = published_match.group(1) if published_match else ""
                        videos.append({
                            "video_id": vid_id,
                            "title": title,
                            "published": pub,
                            "url": f"https://www.youtube.com/watch?v={vid_id}"
                        })
        except Exception:
            pass
        return videos

    def extract_transcript_or_text(self, video_id: str) -> str:
        """
        Attempts to fetch video transcript using youtube-transcript-api if installed,
        or extracts description/subtitles fallback.
        """
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            full_text = " ".join([item.get("text", "") for item in transcript_list])
            return full_text
        except Exception:
            # Fallback text if transcript library is absent or captions disabled
            return ""

    def analyze_content(self, text: str, title: str) -> Dict[str, Any]:
        """
        Analyzes video content for financial stances, inflation urgency, and price targets.
        """
        combined = f"{title} {text}".lower()

        bullish_terms = [
            "gold rally", "buy gold", "breakout", "ath", "all time high", "rate cuts",
            "central bank buying", "dollar collapse", "inflation hedge", "devaluation",
            "safe haven", "target 3000", "target 2800", "bull market"
        ]
        bearish_terms = [
            "gold crash", "sell gold", "top is in", "rate hike", "hawkish", "strong dollar",
            "yields rising", "bearish", "reversal down", "dump", "bubble burst"
        ]

        bull_hits = sum(1 for w in bullish_terms if w in combined)
        bear_hits = sum(1 for w in bearish_terms if w in combined)

        total = bull_hits + bear_hits
        sentiment = (bull_hits - bear_hits) / total if total > 0 else 0.0

        return {
            "sentiment_score": round(sentiment, 3),
            "bullish_signals": bull_hits,
            "bearish_signals": bear_hits,
            "summary_tag": "BULLISH" if sentiment > 0.2 else ("BEARISH" if sentiment < -0.2 else "NEUTRAL")
        }

    def process_all_channels(self) -> Dict[str, Any]:
        """
        Scans all tracked channels, processes new unvisited videos,
        stores insights, and recalculates collective memory sentiment.
        """
        new_processed = 0
        all_sentiments = []

        for ch in self.memory["channels"]:
            ch_id = ch.get("id")
            videos = self.fetch_channel_videos(ch_id)
            for v in videos:
                vid_id = v["video_id"]
                if vid_id not in self.memory["processed_videos"]:
                    transcript = self.extract_transcript_or_text(vid_id)
                    analysis = self.analyze_content(transcript, v["title"])

                    self.memory["processed_videos"][vid_id] = {
                        "channel_label": ch.get("label"),
                        "title": v["title"],
                        "url": v["url"],
                        "published": v.get("published"),
                        "analysis": analysis,
                        "processed_at": datetime.now(timezone.utc).isoformat()
                    }
                    new_processed += 1

        # Calculate collective memory sentiment across the last 20 processed videos
        recent_videos = list(self.memory["processed_videos"].values())[-20:]
        if recent_videos:
            scores = [v["analysis"]["sentiment_score"] for v in recent_videos if "analysis" in v]
            collective_score = float(np.mean(scores)) if scores else 0.0
        else:
            collective_score = 0.0

        self.memory["collective_sentiment"] = round(collective_score, 3)
        self.memory["last_updated"] = datetime.now(timezone.utc).isoformat()
        self._save_memory()

        return {
            "new_videos_processed": new_processed,
            "total_stored_videos": len(self.memory["processed_videos"]),
            "collective_sentiment": self.memory["collective_sentiment"],
            "tracked_channels_count": len(self.memory["channels"])
        }

    def get_memory_context(self) -> Dict[str, Any]:
        """Returns the active memory profile to inject into the trading brain."""
        return {
            "collective_sentiment": self.memory.get("collective_sentiment", 0.0),
            "total_memory_nodes": len(self.memory.get("processed_videos", {})),
            "channels_tracked": len(self.memory.get("channels", [])),
            "last_updated": self.memory.get("last_updated")
        }
