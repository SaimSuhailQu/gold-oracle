"""
Continuous Autonomous Daemon for Gold Oracle
Continuously runs in the background to:
1. Stream free live macro & financial news from Public APIs
2. Re-compute AI NLP sentiment & Ensemble ML probabilities
3. Re-evaluate Auction Profile, VWAP, and Order Flow Microstructure
4. Run Adversarial Self-Criticism (Devil's Advocate Audit)
5. Enforce strict >= 80% (0.80) win probability before trade clearance
6. Automatically overwrite signals/gold_signal.json on an ongoing loop
"""

import os
import sys
import time
import json
import traceback
from datetime import datetime, timezone

from gold_oracle.signal import generate, SIGNAL_PATH


def run_continuous_daemon(poll_interval_seconds: int = 120):
    print("=" * 70)
    print("  GOLD ORACLE AUTONOMOUS AI DAEMON")
    print(f"  Poll Interval : {poll_interval_seconds}s")
    print(f"  Signal Target : {SIGNAL_PATH}")
    print(f"  Threshold     : Win Probability >= 80% (0.80) & Adversarial Pass")
    print("=" * 70)

    iteration = 0
    while True:
        iteration += 1
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"\n[{now_str}] Iteration #{iteration}: Fetching news & re-evaluating model...")

        try:
            # Check and process tracked YouTube channels (watch + learn)
            from gold_oracle.youtube_memory import YouTubeMemoryEngine
            yt_eng = YouTubeMemoryEngine()
            yt_res = yt_eng.process_all_channels()
            if yt_res.get("new_videos_processed", 0) > 0:
                print(f"  [YouTube Memory] Learned {yt_res['new_videos_processed']} new video(s). Collective Sentiment: {yt_res['collective_sentiment']}")
            if yt_res.get("newly_graded", 0) > 0:
                print(f"  [YouTube Memory] Graded {yt_res['newly_graded']} old video(s) against actual gold moves.")

            # Refresh research-paper intelligence (arXiv gold-forecasting literature)
            from gold_oracle.papers import PapersEngine
            papers_eng = PapersEngine()
            papers_res = papers_eng.refresh()
            if papers_res.get("new_papers", 0) > 0:
                print(f"  [Papers] Learned {papers_res['new_papers']} new paper(s). Consensus: {papers_res['consensus']}")

            # Generate new signal and multi-criteria critique
            sig = generate(use_cache=True)
            
            # Persist to signal path
            os.makedirs(os.path.dirname(SIGNAL_PATH), exist_ok=True)
            with open(SIGNAL_PATH, "w") as fh:
                json.dump(sig, fh, indent=2)

            # Shadow-log every layer's recommendation (no execution impact)
            from gold_oracle.shadow import log_daily, grade
            log_daily(sig)
            graded = grade()
            if graded:
                print(f"  [Shadow] Graded {graded} mature shadow entries.")

            # Display evaluation summary
            p_up = sig.get("p_up", 0.0)
            p_down = sig.get("p_down", 0.0)
            signal = sig.get("signal", "FLAT")
            critic = sig.get("decision_critic", {})
            prob_dir = critic.get("directional_probability", 0.0)
            cleared = critic.get("is_cleared_to_trade", False)

            print(f"  P(Up): {p_up*100:.1f}% | P(Down): {p_down*100:.1f}%")
            print(f"  Best Directional Prob: {prob_dir*100:.1f}% (Required: >= 80%)")
            print(f"  Signal Output: {signal}")
            print(f"  Self-Critic Verdict: {critic.get('audit_verdict_summary', 'N/A')}")
            
            flaws = critic.get("fatal_flaws", [])
            if flaws:
                for flaw in flaws:
                    print(f"    - [FATAL FLAW]: {flaw}")

            criticisms = critic.get("self_criticisms", [])
            if criticisms:
                for c in criticisms:
                    print(f"    - [CRITICISM]: {c}")

        except Exception as e:
            print(f"  [ERROR] Encountered error during evaluation loop: {e}")
            traceback.print_exc()

        print(f"  Sleeping for {poll_interval_seconds} seconds...")
        time.sleep(poll_interval_seconds)


if __name__ == "__main__":
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    run_continuous_daemon(poll_interval_seconds=interval)
