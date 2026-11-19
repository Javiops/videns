# Frozen 50/50 mixture on VAL + full-circuit latency. Self-unregistering one-shot.
# Protocol: docs/experiment_mixture_val_v1.md
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$taskName = 'BuildTracker Mixture VAL'
$log = Join-Path $root ('data\logs\mixture_val_' + (Get-Date -Format yyyyMMdd_HHmmss) + '.log')
New-Item -ItemType Directory -Force (Join-Path $root 'data\logs') | Out-Null
Start-Transcript -Path $log
try {
    $env:PREFIX_ML_DIR = Join-Path $root 'data\dataset_generations\16.18-reconstructible-development-v3'
    $review = Join-Path $root 'data\corpora\16.18-reconstructible-v3\static_review\static_review.json'
    Write-Output '=== mixture VAL (full) ==='
    & .venv\Scripts\python.exe scripts\footwear_runtime.py --entry scripts\basket_mixture_val.py --review $review
    if ($LASTEXITCODE -ne 0) { throw "mixture VAL failed (exit $LASTEXITCODE)" }
    Write-Output 'MIXTURE VAL COMPLETE'
}
finally {
    try { Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue } catch {}
    Stop-Transcript
}
