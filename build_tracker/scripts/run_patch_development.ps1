param([string]$TaskName='BuildTracker 16.17 Development V1')
$ErrorActionPreference='Stop'
$developmentRoot=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $developmentRoot
$env:PYTHONUTF8='1'
$developmentExit=1
try {
    & (Join-Path $developmentRoot '.venv\Scripts\python.exe') scripts/run_patch_development.py --source-plan data/acceptance/16.17_development_20260911/source_plan.json --name 16.17-development-v1
    $developmentExit=$LASTEXITCODE
} finally {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
}
exit $developmentExit
