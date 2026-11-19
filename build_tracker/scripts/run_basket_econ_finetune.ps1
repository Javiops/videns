# Twin economic fine-tune (fixed feature domain). Self-unregistering one-shot.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$taskName = 'BuildTracker Basket Econ Finetune'
$log = Join-Path $root ('data\logs\basket_econ_finetune_' + (Get-Date -Format yyyyMMdd_HHmmss) + '.log')
New-Item -ItemType Directory -Force (Join-Path $root 'data\logs') | Out-Null
Start-Transcript -Path $log
try {
    $env:PREFIX_ML_DIR = Join-Path $root 'data\dataset_generations\16.18-reconstructible-development-v3'
    $review = Join-Path $root 'data\corpora\16.18-reconstructible-v3\static_review\static_review.json'
    Write-Output '=== twin econ fine-tune (full) ==='
    & .venv\Scripts\python.exe scripts\footwear_runtime.py --entry scripts\basket_econ_finetune.py --review $review
    if ($LASTEXITCODE -ne 0) { throw "finetune failed (exit $LASTEXITCODE)" }
    Write-Output 'ECON FINETUNE COMPLETE'
}
finally {
    try {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    } catch {}
    Stop-Transcript
}
