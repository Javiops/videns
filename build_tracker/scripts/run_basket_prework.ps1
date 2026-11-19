# Phase-3 prework: checkpoint curves + size-mass calibration + enriched rows.
# Self-unregistering one-shot task.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$taskName = 'BuildTracker Basket Prework'
$log = Join-Path $root ('data\logs\basket_prework_' + (Get-Date -Format yyyyMMdd_HHmmss) + '.log')
New-Item -ItemType Directory -Force (Join-Path $root 'data\logs') | Out-Null
Start-Transcript -Path $log
try {
    $env:PREFIX_ML_DIR = Join-Path $root 'data\dataset_generations\16.18-reconstructible-development-v3'
    $review = Join-Path $root 'data\corpora\16.18-reconstructible-v3\static_review\static_review.json'
    Write-Output '=== basket phase-3 prework (full) ==='
    & .venv\Scripts\python.exe scripts\footwear_runtime.py --entry scripts\basket_phase3_prework.py --review $review
    if ($LASTEXITCODE -ne 0) { throw "prework failed (exit $LASTEXITCODE)" }
    Write-Output 'PREWORK COMPLETE'
}
finally {
    try {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    } catch {}
    Stop-Transcript
}
