# Fetches the latest Gold Oracle signal into your MT5 Files folder.
# Run this on a schedule (Task Scheduler every 5 min) or manually before trading.
# Usage: .\fetch_signal.ps1 -FilesDir "C:\Users\YOU\AppData\Roaming\MetaQuotes\Terminal\<ID>\MQL5\Files"
param(
    [string]$FilesDir = ""
)
$ErrorActionPreference = "Stop"
if (-not $FilesDir) {
    # try to auto-find the first MT5 terminal's Files dir
    $t = Get-ChildItem "$env:APPDATA\MetaQuotes\Terminal" -Directory -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($t) { $FilesDir = Join-Path $t.FullName "MQL5\Files" }
}
if (-not $FilesDir -or -not (Test-Path $FilesDir)) {
    Write-Host "Pass -FilesDir with your MT5 MQL5\Files path. Find it via MT5 > File > Open Data Folder."
    exit 1
}
$url = "https://saimsuhailqu.github.io/gold-oracle/gold_signal.json"
$out = Join-Path $FilesDir "gold_signal.json"
Invoke-WebRequest -Uri $url -OutFile $out -UseBasicParsing
$j = Get-Content $out -Raw | ConvertFrom-Json
Write-Host "signal updated: $($j.signal) p_up=$($j.p_up) ts=$($j.timestamp)"
