# Residual economic-features contrast: cache build (bench-gated) then the
# matched two-arm trainer. Self-unregistering one-shot task.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$taskName = 'BuildTracker Basket Residual Econ'
$log = Join-Path $root ('data\logs\basket_residual_econ_' + (Get-Date -Format yyyyMMdd_HHmmss) + '.log')
New-Item -ItemType Directory -Force (Join-Path $root 'data\logs') | Out-Null
Start-Transcript -Path $log
try {
    $env:PREFIX_ML_DIR = Join-Path $root 'data\dataset_generations\16.18-reconstructible-development-v3'
    $review = Join-Path $root 'data\corpora\16.18-reconstructible-v3\static_review\static_review.json'
    Write-Output '=== econ cache (full, bench-gated) ==='
    & .venv\Scripts\python.exe scripts\footwear_runtime.py --entry scripts\basket_econ_cache.py --review $review
    if ($LASTEXITCODE -ne 0) { throw "cache failed (exit $LASTEXITCODE)" }
    $reportPath = Join-Path $root 'data\experiments\16.18-basket-residual-econ-v1\cache_report_full.json'
    $cacheReport = Get-Content $reportPath -Raw | ConvertFrom-Json
    if ($cacheReport.aborted) { throw ('cache aborted by bench gate: ' + $cacheReport.aborted) }
    Write-Output '=== residual contrast (control vs econ) ==='
    & .venv\Scripts\python.exe scripts\footwear_runtime.py --entry scripts\basket_residual_econ.py --review $review
    if ($LASTEXITCODE -ne 0) { throw "trainer failed (exit $LASTEXITCODE)" }
    Write-Output 'RESIDUAL ECON COMPLETE'
}
finally {
    try {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    } catch {}
    Stop-Transcript
}
