# Frozen dev baseline on the VAL temporal window. Self-unregistering one-shot.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$taskName = 'BuildTracker Val Window Check'
$log = Join-Path $root ('data\logs\val_window_check_' + (Get-Date -Format yyyyMMdd_HHmmss) + '.log')
New-Item -ItemType Directory -Force (Join-Path $root 'data\logs') | Out-Null
Start-Transcript -Path $log
try {
    $env:PREFIX_ML_DIR = Join-Path $root 'data\dataset_generations\16.18-reconstructible-development-v3'
    $review = Join-Path $root 'data\corpora\16.18-reconstructible-v3\static_review\static_review.json'
    Write-Output '=== val window check (full) ==='
    & .venv\Scripts\python.exe scripts\footwear_runtime.py --entry scripts\basket_val_window_check.py --review $review
    if ($LASTEXITCODE -ne 0) { throw "val check failed (exit $LASTEXITCODE)" }
    Write-Output 'VAL WINDOW CHECK COMPLETE'
}
finally {
    try {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    } catch {}
    Stop-Transcript
}
