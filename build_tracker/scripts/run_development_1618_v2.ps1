param([string]$TaskName='BuildTracker 16.18 Development Generation V2')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$env:PYTHONUTF8='1'
$log=Join-Path $root 'data\acceptance\16.18_reconstructible_20260913\admission_launcher.log'
$code=1
try {
    & (Join-Path $root '.venv\Scripts\python.exe') scripts/run_development_1618_v2.py *>> $log
    $code=$LASTEXITCODE
} finally {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
}
exit $code
