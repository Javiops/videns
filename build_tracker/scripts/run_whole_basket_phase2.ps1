# Phase 2 of 16.18-whole-basket: bounded loss contrast Bs vs Bf.
# Protocol: docs/experiment_whole_basket_v2_loss_contrast.md.
# Self-unregistering one-shot task (register_task.ps1 contract).
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$taskName = 'BuildTracker Whole Basket Phase2'
$log = Join-Path $root ('data\logs\whole_basket_p2_' + (Get-Date -Format yyyyMMdd_HHmmss) + '.log')
New-Item -ItemType Directory -Force (Join-Path $root 'data\logs') | Out-Null
Start-Transcript -Path $log
try {
    $env:PREFIX_ML_DIR = Join-Path $root 'data\dataset_generations\16.18-reconstructible-development-v3'
    $review = Join-Path $root 'data\corpora\16.18-reconstructible-v3\static_review\static_review.json'
    if (-not (Test-Path -LiteralPath $review)) { throw "missing static review: $review" }
    Write-Output '=== whole-basket PHASE 2: loss contrast (full) ==='
    & .venv\Scripts\python.exe scripts\footwear_runtime.py --entry scripts\train_basket_phase2.py --review $review
    if ($LASTEXITCODE -ne 0) { throw "phase 2 failed (exit $LASTEXITCODE)" }
    Write-Output 'PHASE 2 COMPLETE - report under data/experiments/16.18-whole-basket-v2/'
}
finally {
    try {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    } catch {}
    Stop-Transcript
}
