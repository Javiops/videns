param([string]$TaskName='BuildTracker 16.17 Footwear Parallel Audit')
$ErrorActionPreference='Stop'
$footwearRoot=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $footwearRoot
$env:PYTHONUTF8='1'
$footwearExit=1
try {
    & (Join-Path $footwearRoot '.venv\Scripts\python.exe') scripts/run_footwear_timer_parallel.py
    $footwearExit=$LASTEXITCODE
} finally {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
}
exit $footwearExit
