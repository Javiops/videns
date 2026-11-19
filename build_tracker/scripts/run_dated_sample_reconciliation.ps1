param([string]$TaskName='BuildTracker Dated Sample Reconciliation 20260920')
$ErrorActionPreference='Stop'
$reconciliationRoot=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $reconciliationRoot
$env:PYTHONUTF8='1'
$env:PYTHONUNBUFFERED='1'
$reconciliationLog=Join-Path $reconciliationRoot 'data\acceptance\16.18_dated_samples_20260920\reconciliation_launcher.log'
$reconciliationExit=1
try {
    & (Join-Path $reconciliationRoot '.venv\Scripts\python.exe') scripts/run_dated_sample_reconciliation.py *>> $reconciliationLog
    $reconciliationExit=$LASTEXITCODE
} finally {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
}
exit $reconciliationExit
