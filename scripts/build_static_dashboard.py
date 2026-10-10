#!/usr/bin/env python3
"""Build a self-contained static snapshot of the dashboard for GitHub Pages.

Usage:
    python -m gold_oracle.signal            # fresh signal first
    python scripts/build_static_dashboard.py

Reads dashboard/index.html, embeds the exact /api/state payload as
window.__STATE__, and writes site/index.html — no backend needed.
A scheduled GitHub Action rebuilds it every 6 hours.
"""

import json
import os
import sys
from datetime import datetime, timezone, timedelta

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "dashboard"))
import serve  # noqa: E402  (reuses the /api/state builders — no drift)


def main() -> None:
    sig = serve.read_json("signals/gold_signal.json", {})
    built = datetime.now(timezone.utc)
    state = {
        "signal": sig,
        "history": serve.read_history(),
        "backtest": serve.read_json("data/backtest_summary.json", {}),
        "ea": serve.ea_status(sig),
        "factor_meta": {k: {"title": v[0], "desc": v[1]}
                        for k, v in serve.FACTOR_META.items()},
        "static": True,
        "built_utc": built.strftime("%Y-%m-%d %H:%M UTC"),
        "next_update_utc": (built + timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M UTC"),
    }
    with open(os.path.join(BASE, "dashboard", "index.html"), encoding="utf-8") as fh:
        html = fh.read()

    payload = json.dumps(state).replace("</script>", "<\\/script>")
    inject = (
        f'<script>window.__STATE__={payload};</script>\n'
        f'<div id="staticbanner" style="background:#1a1a2e;color:#aab;padding:6px '
        f'12px;font-size:.8rem;text-align:center">Static snapshot · built '
        f'{state["built_utc"]} · I rethink every 30 minutes (next ~{state["next_update_utc"]}) · '
        f'<a href="https://github.com/SaimSuhailQu/gold-oracle" style="color:#7aa2ff">source</a></div>\n'
    )
    html = html.replace("<script>", inject + "<script>", 1)
    # serve the embedded state instead of fetching the API
    html = html.replace(
        'const d = await (await fetch("/api/state")).json();',
        'const d = window.__STATE__;',
    )
    # load() is async only because of fetch — keep it working either way
    out_dir = os.path.join(BASE, "site")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "index.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"wrote {out} ({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
