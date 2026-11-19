param([string]$TaskName='BuildTracker Dated Sample Acceptance 20260920')
$ErrorActionPreference = 'Stop'
$sampleRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $sampleRoot
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$sampleBase = Join-Path $sampleRoot 'data\acceptance\16.18_dated_samples_20260920\uncovered'
$sampleCorpus = Join-Path $sampleRoot 'data\corpora\16.18-dated-samples-uncovered-20260920'
$sampleLog = Join-Path $sampleBase 'launcher.log'
$sampleExit = 1
try {
    if (-not (Test-Path -LiteralPath (Join-Path $sampleBase 'source_plan.json'))) {
        throw 'The reviewed source plan must exist before this task starts.'
    }
    & (Join-Path $sampleRoot '.venv\Scripts\python.exe') scripts/run_reconstructible_1618.py --base $sampleBase --corpus $sampleCorpus --max-wall-seconds 7200 *>> $sampleLog
    $sampleExit = $LASTEXITCODE
} finally {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
}
exit $sampleExit
