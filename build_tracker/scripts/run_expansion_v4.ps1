# Design-B expansion chain (owner GO 2026-09-23). Self-unregistering one-shot.
# The python runner supervises every stage itself (per-stage logs, digests,
# disk reserve, wall budget, status.json, tests gate); this wrapper only
# detaches it and keeps its stdout in a plain file written as it flushes.
#
# Resume: if data\experiments\16.18-expansion-v4\resume_from.txt exists, its
# single word is passed as --from-stage (explicit, visible in run_plan.json),
# and the file is removed so a later launch cannot silently skip stages.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$taskName = 'BuildTracker Expansion V4'
$ts = Get-Date -Format yyyyMMdd_HHmmss
New-Item -ItemType Directory -Force (Join-Path $root 'data\logs') | Out-Null
$out = Join-Path $root ('data\logs\expansion_v4_' + $ts + '.out')
$err = Join-Path $root ('data\logs\expansion_v4_' + $ts + '.err')
$env:PYTHONUTF8 = '1'
$argList = @('scripts\run_expansion_v4.py')
$resume = Join-Path $root 'data\experiments\16.18-expansion-v4\resume_from.txt'
if (Test-Path -LiteralPath $resume) {
    $stage = (Get-Content -LiteralPath $resume -Raw).Trim()
    if ($stage) { $argList += @('--from-stage', $stage) }
    Remove-Item -LiteralPath $resume -Force
}
$code = 1
try {
    $p = Start-Process -FilePath (Join-Path $root '.venv\Scripts\python.exe') `
        -ArgumentList $argList `
        -WorkingDirectory $root -NoNewWindow -PassThru -Wait `
        -RedirectStandardOutput $out -RedirectStandardError $err
    $code = [int]$p.ExitCode
}
finally {
    try { Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue } catch {}
}
exit $code
