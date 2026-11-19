param([string]$TaskName='BuildTracker Morning Recovery 20260912')
$ErrorActionPreference='Stop'
$morningRoot=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $morningRoot
$env:PYTHONUTF8='1'
$env:PYTHONUNBUFFERED='1'
$morningLog=Join-Path $morningRoot 'data\collection_recovery\20260912\launcher.log'
$morningExit=1
try {
    & (Join-Path $morningRoot '.venv\Scripts\python.exe') scripts/resume_morning_collection.py *>> $morningLog
    $morningExit=$LASTEXITCODE
} finally {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
}
exit $morningExit
