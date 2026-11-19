# Capacity contrast d64 vs d128. Waits for the val window checker so only
# one GPU job runs; pilot + disk gates live inside the python script.
# Self-unregistering one-shot.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$taskName = 'BuildTracker Capacity AB'
$log = Join-Path $root ('data\logs\capacity_ab_' + (Get-Date -Format yyyyMMdd_HHmmss) + '.log')
New-Item -ItemType Directory -Force (Join-Path $root 'data\logs') | Out-Null
Start-Transcript -Path $log
try {
    $valReport = Join-Path $root 'data\experiments\16.18-val-window-check-v1\val_window_report_full.json'
    $deadline = (Get-Date).AddHours(3)
    while (-not (Test-Path $valReport)) {
        $valTask = Get-ScheduledTask -TaskName 'BuildTracker Val Window Check' -ErrorAction SilentlyContinue
        if (-not $valTask) { break }   # checker gone (finished or failed): proceed
        if ((Get-Date) -gt $deadline) { throw 'val checker still running after 3h wait; not starting a second GPU job' }
        Write-Output ('waiting for val checker... ' + (Get-Date -Format HH:mm:ss))
        Start-Sleep -Seconds 60
    }
    Start-Sleep -Seconds 90            # let the checker process fully exit
    $env:PREFIX_ML_DIR = Join-Path $root 'data\dataset_generations\16.18-reconstructible-development-v3'
    $review = Join-Path $root 'data\corpora\16.18-reconstructible-v3\static_review\static_review.json'
    Write-Output '=== capacity contrast d64 vs d128 (full) ==='
    & .venv\Scripts\python.exe scripts\footwear_runtime.py --entry scripts\basket_capacity_ab.py --review $review
    if ($LASTEXITCODE -ne 0) { throw "capacity run failed (exit $LASTEXITCODE)" }
    Write-Output 'CAPACITY AB COMPLETE'
}
finally {
    try {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    } catch {}
    Stop-Transcript
}
