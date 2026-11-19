param([string]$TaskName='BuildTracker Gap Window Pull 20260916')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$env:PYTHONUTF8='1'
$env:PYTHONUNBUFFERED='1'
$log=Join-Path $root 'data\gap_window_pull_20260916-launcher.log'
function Log([string]$text) {
    try { Add-Content -Path $log -Value ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] " + $text) -ErrorAction Stop }
    catch { }
}
$code=1
try {
    # Recovers the collection window missed on Sep 15 (Windows Update reboot at
    # 05:36) and Sep 16 (expired key at 10:11). No --hours: auto_window_hours()
    # anchors on the last run that printed "pipeline finished" (Sep 14 21:05),
    # so the whole gap is inside one window. The dated sample stage then
    # prepares 16.18_20260916 fresh under the current pins; the 20260914
    # checkpoint stays preserved and unused (its pins are stale).
    # Distinct task name: the collector's scheduler guard only fails closed on
    # 'BuildTracker Daily Pull' / 'BuildTracker Causal Recovery'.
    Log "gap window pull starting (auto window since the last completed run; collection + dated sample)"
    $py = Join-Path $root '.venv\Scripts\python.exe'
    $p = Start-Process -FilePath $py -ArgumentList @('scripts/daily_pull.py') `
        -WorkingDirectory $root -Wait -PassThru `
        -RedirectStandardOutput ($log + '.out') `
        -RedirectStandardError ($log + '.err')
    $code = $p.ExitCode
    Log ("finished with exit " + $code)
} catch {
    Log ("launcher failed: " + $_.Exception.Message)
} finally {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
}
exit $code
