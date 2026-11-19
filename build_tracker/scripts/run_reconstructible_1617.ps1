# 16.17 HISTORY reconstruction/qualification (prior-patch transfer, owner GO 2026-09-27).
# Self-unregistering one-shot; the python supervisor owns budgets, jobs and status.
# Parameters (review 2026-09-27): task name and acceptance/corpus directories so a smoke chain and the
# full chain can coexist; the wall budget (21600 s = 6 h) is the supervisor's, passed through unchanged.
param([string]$TaskName = 'BuildTracker History 1617', [string]$Base = '', [string]$Corpus = '', [int]$MaxWallSeconds = 21600)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$taskName = $TaskName
$pyArgs = @('scripts\run_reconstructible_1617.py', '--max-wall-seconds', "$MaxWallSeconds")
if ($Base) { $pyArgs += @('--base', $Base) }
if ($Corpus) { $pyArgs += @('--corpus', $Corpus) }
$ts = Get-Date -Format yyyyMMdd_HHmmss
New-Item -ItemType Directory -Force (Join-Path $root 'data\logs') | Out-Null
$out = Join-Path $root ('data\logs\history_1617_' + $ts + '.out')
$err = Join-Path $root ('data\logs\history_1617_' + $ts + '.err')
$env:PYTHONUTF8 = '1'
$code = 1
try {
    $p = Start-Process -FilePath (Join-Path $root '.venv\Scripts\python.exe') `
        -ArgumentList $pyArgs `
        -WorkingDirectory $root -NoNewWindow -PassThru -Wait `
        -RedirectStandardOutput $out -RedirectStandardError $err
    $code = [int]$p.ExitCode
}
finally {
    try { Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue } catch {}
}
exit $code
