# Stage-2 chain on the v4 generation (owner GO 2026-09-25). Self-unregistering
# one-shot; the python orchestrator supervises every segment (jobs, budgets,
# pilots, status.json); this wrapper only detaches it and keeps its stdout in
# a plain file as it flushes.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$taskName = 'BuildTracker Stage2 V4'
$ts = Get-Date -Format yyyyMMdd_HHmmss
New-Item -ItemType Directory -Force (Join-Path $root 'data\logs') | Out-Null
$out = Join-Path $root ('data\logs\stage2_v4_' + $ts + '.out')
$err = Join-Path $root ('data\logs\stage2_v4_' + $ts + '.err')
$env:PYTHONUTF8 = '1'
$argList = @('scripts\run_stage2_v4.py')
$resume = Join-Path $root 'data\experiments\16.18-stage2-v4\run\resume.txt'
if (Test-Path -LiteralPath $resume) { $argList += @('--resume'); Remove-Item -LiteralPath $resume -Force }
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
