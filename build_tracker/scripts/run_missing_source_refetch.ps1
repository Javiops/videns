param([string]$TaskName='BuildTracker 16.17 Missing Source Refetch')
$ErrorActionPreference='Stop'
$refetchRoot=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $refetchRoot
$env:PYTHONUTF8='1'
$env:PYTHONUNBUFFERED='1'
$refetchLog=Join-Path $refetchRoot 'data\source_refetch\16.17-missing-20260911\launcher.log'
$refetchExit=1
try {
    & (Join-Path $refetchRoot '.venv\Scripts\python.exe') scripts/refetch_missing_sources.py supervise *>> $refetchLog
    $refetchExit=$LASTEXITCODE
} finally {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
}
exit $refetchExit
