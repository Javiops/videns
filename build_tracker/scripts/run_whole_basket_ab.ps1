# Experiment 16.18-whole-basket-v1: full A/B contrast on the frozen 2000/500
# chronological TRAIN split. Protocol: docs/experiment_whole_basket_v1.md.
# One-shot scheduled task; unregisters itself when done (register_task.ps1
# contract). 16.18 generations train ONLY through footwear_runtime.py with an
# explicit --review (qualification.json predates the repo move).
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$taskName = 'BuildTracker Whole Basket AB'
$log = Join-Path $root ('data\logs\whole_basket_ab_' + (Get-Date -Format yyyyMMdd_HHmmss) + '.log')
New-Item -ItemType Directory -Force (Join-Path $root 'data\logs') | Out-Null
Start-Transcript -Path $log
try {
    $env:PREFIX_ML_DIR = Join-Path $root 'data\dataset_generations\16.18-reconstructible-development-v3'
    $review = Join-Path $root 'data\corpora\16.18-reconstructible-v3\static_review\static_review.json'
    if (-not (Test-Path -LiteralPath $review)) { throw "missing static review: $review" }
    Write-Output '=== whole-basket A/B (full, sampled-softmax B, reuse arm A) ==='
    & .venv\Scripts\python.exe scripts\footwear_runtime.py --entry scripts\train_basket_ab.py --review $review -- --reuse-arm-a
    if ($LASTEXITCODE -ne 0) { throw "A/B run failed (exit $LASTEXITCODE)" }
    Write-Output 'WHOLE-BASKET AB COMPLETE - report under data/experiments/16.18-whole-basket-v1/'
}
finally {
    try {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    } catch {}
    Stop-Transcript
}
