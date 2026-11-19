param([string]$TaskName='BuildTracker 16.18 Reconstructible V2')
$ErrorActionPreference='Stop'
$reconstructibleRoot=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $reconstructibleRoot
$env:PYTHONUTF8='1'
$base=Join-Path $reconstructibleRoot 'data\acceptance\16.18_reconstructible_20260913'
$corpus=Join-Path $reconstructibleRoot 'data\corpora\16.18-reconstructible-v2'
$reconstructibleExit=1
try {
    & (Join-Path $reconstructibleRoot '.venv\Scripts\python.exe') scripts/run_reconstructible_1618.py --base $base --corpus $corpus --max-wall-seconds 21600 *>> (Join-Path $base 'launcher.log')
    $reconstructibleExit=$LASTEXITCODE
} finally {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
}
exit $reconstructibleExit
