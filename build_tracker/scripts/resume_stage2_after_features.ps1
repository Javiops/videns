# 2026-09-26: the chain's `features` segment stopped on an inflated bench (GC
# heap + Windows maintenance window) and was re-run standalone with the fix.
# The running chain (wardfree only) will finish on its own. This detached
# watcher relaunches the chain with --resume once BOTH the standalone cache
# build has completed (report without an abort) AND the chain has ended, so
# finetune + mixture + final run with every completed segment reused.
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$log = Join-Path $root ('data\logs\resume_stage2_after_features_' + (Get-Date -Format yyyyMMdd_HHmmss) + '.log')
function Log($m) { Add-Content -LiteralPath $log -Value ((Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + ' ' + $m) }
$report = Join-Path $root 'data\experiments\16.18-stage2-v4\cache_report_full.json'
$status = Join-Path $root 'data\experiments\16.18-stage2-v4\run\status.json'
Log 'waiting for the standalone cache build and the running chain to finish'
while ($true) {
    $cacheDone = (Test-Path $report) -and ((Get-Content $report -Raw) -match '"rows_per_second"') -and ((Get-Content $report -Raw) -notmatch '"aborted"')
    $chainState = if (Test-Path $status) { (Get-Content $status -Raw | ConvertFrom-Json).state } else { 'missing' }
    $taskUp = [bool](Get-ScheduledTask -TaskName 'BuildTracker Stage2 V4' -ErrorAction SilentlyContinue)
    if ($cacheDone -and ($chainState -ne 'running') -and (-not $taskUp)) {
        Set-Content -LiteralPath (Join-Path $root 'data\experiments\16.18-stage2-v4\run\resume.txt') -Value 'resume' -Encoding ascii -NoNewline
        Copy-Item $status (Join-Path $root 'data\experiments\16.18-stage2-v4\run\status_attempt2_wardfree_only.json') -Force
        $out = & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root 'scripts\register_task.ps1') -TaskName 'BuildTracker Stage2 V4' -ScriptName 'scripts\run_stage2_v4.ps1' -StartInMinutes 1 2>&1
        Log ('chain relaunched with --resume (cache done, chain state ' + $chainState + '): ' + ($out -join ' | '))
        break
    }
    if ((Test-Path $report) -and ((Get-Content $report -Raw) -match '"aborted"')) { Log 'cache build aborted again; not relaunching'; break }
    Start-Sleep -Seconds 60
}
