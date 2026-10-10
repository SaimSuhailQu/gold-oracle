#!/usr/bin/env python3
"""Backfill the YouTube learning memory with the last N days of videos.

Usage:
    python scripts/backfill_youtube.py --days 45

Pulls up to --days of history per tracked channel (RSS feed first, yt-dlp
fallback), scores each video's gold stance, and stores it in
data/youtube_memory.json. Videos older than 20 days are graded immediately
against subsequent gold moves, so the per-channel hit rates start working
right away.

Run this on a network where YouTube is reachable (residential connection);
datacenter IPs often get 404s on the feed endpoint. Needs a gold price
reference for grading — uses the latest signals/gold_signal.json, so run
`python -m gold_oracle.signal` first.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gold_oracle.youtube_memory import YouTubeMemoryEngine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=45,
                    help="how far back to pull (default 45)")
    ap.add_argument("--limit", type=int, default=60,
                    help="max videos per channel (default 60)")
    args = ap.parse_args()

    eng = YouTubeMemoryEngine()
    print(f"Backfilling {args.days} days (up to {args.limit} videos/channel)...")
    res = eng.process_all_channels(limit=args.limit, since_days=args.days)
    print(f"  new videos: {res['new_videos_processed']}")
    print(f"  graded immediately: {res['newly_graded']}")
    print(f"  collective sentiment: {res['collective_sentiment']}")
    ctx = eng.get_memory_context()
    for c in ctx["channels"]:
        print(f"  - {c['label']}: hit_rate={c['hit_rate']} graded={c['graded']}")
    print(f"  feed status: {ctx['fetch_status']}")


if __name__ == "__main__":
    main()
