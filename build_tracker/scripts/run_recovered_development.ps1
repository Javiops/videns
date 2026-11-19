param([string]$TaskName='BuildTracker 16.17 Development V2')
$ErrorActionPreference='Stop'
$recoveredRoot=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $recoveredRoot
$env:PYTHONUTF8='1'
$recoveredLauncherLog=Join-Path $recoveredRoot 'data\logs\16.17-development-v2-launcher.log'
New-Item -ItemType Directory -Path (Split-Path -Parent $recoveredLauncherLog) -Force | Out-Null
$recoveredExit=1
try {
    & (Join-Path $recoveredRoot '.venv\Scripts\python.exe') scripts/run_recovered_development.py *>> $recoveredLauncherLog
    $recoveredExit=$LASTEXITCODE
} finally {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
}
exit $recoveredExit
