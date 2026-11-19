# History pilot: build the full history caches, then run the twin contrast.
# Self-unregistering one-shot. Protocol: docs/experiment_history_contrast_v1.md
#
# Python stdout/stderr go to PLAIN files via Start-Process -Redirect*, never
# through Start-Transcript (which buffers a native command's output until it
# exits and left a 2h56 run unobservable on 2026-09-22).
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$taskName = 'BuildTracker History Pilot'
$ts = Get-Date -Format yyyyMMdd_HHmmss
New-Item -ItemType Directory -Force (Join-Path $root 'data\logs') | Out-Null
Start-Transcript -Path (Join-Path $root ('data\logs\history_pilot_' + $ts + '.log'))

function Invoke-Entry {
    # Returns ONLY the exit code. Write-Output here would append the message
    # to the function's output stream and a successful run would be reported
    # as failed (observed 2026-09-22). Messages go to the host.
    param([string]$Entry, [string]$Tag)
    $out = Join-Path $root ('data\logs\' + $Tag + '_' + $ts + '.out')
    $err = Join-Path $root ('data\logs\' + $Tag + '_' + $ts + '.err')
    Write-Host ("--> $Entry ; progreso en vivo: $out")
    $p = Start-Process -FilePath (Join-Path $root '.venv\Scripts\python.exe') `
        -ArgumentList @('scripts\footwear_runtime.py', '--entry', $Entry,
                        '--review', $script:review) `
        -WorkingDirectory $root -NoNewWindow -PassThru -Wait `
        -RedirectStandardOutput $out -RedirectStandardError $err
    return [int]$p.ExitCode
}

try {
    $env:PREFIX_ML_DIR = Join-Path $root 'data\dataset_generations\16.18-reconstructible-development-v3'
    $script:review = Join-Path $root 'data\corpora\16.18-reconstructible-v3\static_review\static_review.json'

    Write-Host '=== 1/2 cachas de historial (FIT + probe, completas) ==='
    [int]$code = Invoke-Entry -Entry 'scripts\build_history_cache.py' -Tag 'history_cache'
    if ($code -ne 0) { throw "history cache build failed (exit $code)" }
    Write-Host 'CACHES OK'

    Write-Host '=== 2/2 contraste gemelo de historial (2 brazos x 6 epocas) ==='
    [int]$code = Invoke-Entry -Entry 'scripts\basket_history_contrast.py' -Tag 'history_contrast'
    if ($code -ne 0) { throw "history contrast failed (exit $code)" }
    Write-Host 'HISTORY PILOT COMPLETE'
}
finally {
    try { Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue } catch {}
    Stop-Transcript
}
