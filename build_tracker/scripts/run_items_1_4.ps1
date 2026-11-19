# Owner's "haz 1-4": learned gate (item 4, short) then learning curve (item 1, ~1.5 h).
# Self-unregistering. Python stdout/stderr to plain files (Start-Transcript buffers).
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$taskName = 'BuildTracker Items 1-4'
$ts = Get-Date -Format yyyyMMdd_HHmmss
New-Item -ItemType Directory -Force (Join-Path $root 'data\logs') | Out-Null
Start-Transcript -Path (Join-Path $root ('data\logs\items_1_4_' + $ts + '.log'))
function Invoke-Entry {
    # returns ONLY [int]; messages via Write-Host (Write-Output would pollute the return)
    param([string]$Entry, [string]$Tag)
    $out = Join-Path $root ('data\logs\' + $Tag + '_' + $ts + '.out')
    $err = Join-Path $root ('data\logs\' + $Tag + '_' + $ts + '.err')
    Write-Host ("--> $Entry ; progreso en vivo: $out")
    $p = Start-Process -FilePath (Join-Path $root '.venv\Scripts\python.exe') `
        -ArgumentList @('scripts\footwear_runtime.py', '--entry', $Entry, '--review', $script:review) `
        -WorkingDirectory $root -NoNewWindow -PassThru -Wait `
        -RedirectStandardOutput $out -RedirectStandardError $err
    return [int]$p.ExitCode
}
try {
    $env:PREFIX_ML_DIR = Join-Path $root 'data\dataset_generations\16.18-reconstructible-development-v3'
    $script:review = Join-Path $root 'data\corpora\16.18-reconstructible-v3\static_review\static_review.json'
    Write-Host '=== item 4: learned gate ==='
    [int]$code = Invoke-Entry -Entry 'scripts\basket_learned_gate.py' -Tag 'learned_gate'
    if ($code -ne 0) { throw "learned gate failed (exit $code)" }
    Write-Host '=== item 1: learning curve ==='
    [int]$code = Invoke-Entry -Entry 'scripts\basket_learning_curve.py' -Tag 'learning_curve'
    if ($code -ne 0) { throw "learning curve failed (exit $code)" }
    Write-Host 'ITEMS 1-4 COMPLETE'
}
finally {
    try { Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue } catch {}
    Stop-Transcript
}
