import os
import json
import time
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_SIGNAL = os.path.join(BASE, "signals", "gold_signal.json")
MT5_SIGNAL = r"C:\Users\saim_\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Files\gold_signal.json"

def sync_signal(p_up=None, signal=None):
    if not os.path.exists(SRC_SIGNAL):
        print(f"Source signal not found: {SRC_SIGNAL}")
        return
    with open(SRC_SIGNAL, "r", encoding="utf-8") as f:
        data = json.load(f)

    now = datetime.now(timezone.utc)
    data["timestamp"] = now.isoformat()
    data["timestamp_epoch"] = int(now.timestamp())
    
    if p_up is not None:
        data["p_up"] = round(float(p_up), 4)
        data["p_down"] = round(1.0 - float(p_up), 4)
    if signal is not None:
        data["signal"] = signal

    data["size_tier"] = "FULL"
    data["size_mult"] = 1.0

    os.makedirs(os.path.dirname(MT5_SIGNAL), exist_ok=True)
    with open(MT5_SIGNAL, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    with open(SRC_SIGNAL, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"[{now.strftime('%H:%M:%S')}] Signal synced -> {data.get('signal')} | P_up: {data.get('p_up')} | MT5: {MT5_SIGNAL}")

if __name__ == "__main__":
    sync_signal()
