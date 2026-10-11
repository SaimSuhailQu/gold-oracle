#!/usr/bin/env bash
# Fetches the latest Gold Oracle signal into your MT5 Files folder (macOS/Linux).
# Usage: ./fetch_signal.sh [path-to-MQL5-Files]
# macOS default: ~/Library/Application\ Support/net.metaquotes.wine/.../MQL5/Files
set -euo pipefail
URL="https://saimsuhailqu.github.io/gold-oracle/gold_signal.json"
DIR="${1:-}"
if [ -z "$DIR" ]; then
  echo "Usage: $0 /path/to/MQL5/Files  (MT5 > File > Open Data Folder)"
  exit 1
fi
curl -fsSL "$URL" -o "$DIR/gold_signal.json"
python3 -c "
import json; j=json.load(open('$DIR/gold_signal.json'))
print('signal updated:', j['signal'], 'p_up=%.3f' % j['p_up'], j['timestamp'])"
