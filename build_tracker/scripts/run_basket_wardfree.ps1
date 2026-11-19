# Ward-free basket vocabulary experiment. Self-unregistering one-shot.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$taskName = 'BuildTracker Basket Wardfree'
$log = Join-Path $root ('data\logs\basket_wardfree_' + (Get-Date -Format yyyyMMdd_HHmmss) + '.log')
New-Item -ItemType Directory -Force (Join-Path $root 'data\logs') | Out-Null
Start-Transcript -Path $log
try {
    $env:PREFIX_ML_DIR = Join-Path $root 'data\dataset_generations\16.18-reconstructible-development-v3'
    $review = Join-Path $root 'data\corpora\16.18-reconstructible-v3\static_review\static_review.json'
    Write-Output '=== wardfree vocabulary (full) ==='
    & .venv\Scripts\python.exe scripts\footwear_runtime.py --entry scripts\basket_wardfree.py --review $review
    if ($LASTEXITCODE -ne 0) { throw "wardfree failed (exit $LASTEXITCODE)" }
    Write-Output 'WARDFREE COMPLETE'
}
finally {
    try {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    } catch {}
    Stop-Transcript
}
