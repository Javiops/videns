param([string]$TaskName='BuildTracker 16.18 Reconstructible Sources')
$ErrorActionPreference='Stop'
$reconstructibleRoot=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $reconstructibleRoot
$env:PYTHONUTF8='1'
$reconstructibleExit=1
try {
    & (Join-Path $reconstructibleRoot '.venv\Scripts\python.exe') scripts/run_reconstructible_1618.py *>> (Join-Path $reconstructibleRoot 'data\acceptance\16.18_reconstructible_20260912\launcher.log')
    $reconstructibleExit=$LASTEXITCODE
} finally {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
}
exit $reconstructibleExit
