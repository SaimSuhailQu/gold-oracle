$editor = "C:\Program Files\MetaTrader 5\metaeditor64.exe"
$src = "C:\Users\saim_\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Experts\GoldOracle_Exness.mq5"
$log = "C:\Users\saim_\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Experts\compile.log"

Write-Host "Compiling $src ..."
$proc = Start-Process -FilePath $editor -ArgumentList "/compile:`"$src`"", "/log:`"$log`"" -PassThru -Wait
Start-Sleep -Seconds 2

if (Test-Path $log) {
    Get-Content $log
} else {
    Write-Host "No compile.log found."
}

$ex5 = "C:\Users\saim_\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Experts\GoldOracle_Exness.ex5"
if (Test-Path $ex5) {
    $item = Get-Item $ex5
    Write-Host "EX5 updated: $($item.LastWriteTime)"
}
