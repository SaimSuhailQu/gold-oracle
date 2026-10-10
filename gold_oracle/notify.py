"""
Cron observability for the Gold Oracle signal pipeline.

Wrap your scheduled signal run so a failure pages you instead of the bot
silently going flat for three days:

    python -m gold_oracle.notify -- python -m gold_oracle.signal

Environment:
    HEALTHCHECKS_URL     e.g. https://hc-ping.com/<your-uuid>
                         (pings /start, success, and /fail automatically)
    DISCORD_WEBHOOK_URL  posts an embed on failure (set NOTIFY_ON_SUCCESS=1
                         to also post successes)

Exit code mirrors the wrapped command. No env set -> just runs the command
(a warning is printed so a missing config is never silent).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone


def _post(url: str, data: bytes | None = None, timeout: int = 15) -> bool:
    try:
        req = urllib.request.Request(url, data=data,
                                     headers={"User-Agent": "gold-oracle-notify/1.0",
                                              "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except Exception as e:
        print(f"[notify] ping failed: {e}", flush=True)
        return False


def _discord(webhook: str, ok: bool, cmd: str, detail: str) -> None:
    color = 0x34D399 if ok else 0xF87171
    title = "Gold Oracle signal OK" if ok else "Gold Oracle signal FAILED"
    payload = {"embeds": [{"title": title, "color": color,
                           "description": f"`{' '.join(cmd)}`\n{detail[:1500]}",
                           "timestamp": datetime.now(timezone.utc).isoformat()}]}
    _post(webhook, json.dumps(payload).encode())


def main(argv: list[str]) -> int:
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    if not argv:
        print(__doc__)
        return 2
    hc = os.environ.get("HEALTHCHECKS_URL", "").strip()
    discord = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if not hc and not discord:
        print("[notify] WARNING: neither HEALTHCHECKS_URL nor DISCORD_WEBHOOK_URL is set — "
              "running unwatched.", flush=True)
    if hc:
        _post(hc + "/start")
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=1800)
        ok, detail = proc.returncode == 0, (proc.stderr or proc.stdout)[-1500:]
    except Exception as e:
        ok, detail = False, f"exception launching command: {e}"
        proc = None
    print(detail[-3000:] if detail else "", flush=True)
    if hc:
        _post(hc + ("/fail" if not ok else ""))
    if discord and (not ok or os.environ.get("NOTIFY_ON_SUCCESS") == "1"):
        _discord(discord, ok, argv, detail)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
