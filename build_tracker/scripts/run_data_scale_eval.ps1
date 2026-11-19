param([string]$TaskName='BuildTracker Data Scale Arm 1617 Eval')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$env:PYTHONUTF8='1'
$env:PYTHONUNBUFFERED='1'
$dir=Join-Path $root 'data\experiments'
$log=Join-Path $dir 'data-scale-v1-arm-1617-launcher.log'
# Evidence the launcher process started at all, independent of any file another
# process may hold open (a vanished task with no log line cost a run tonight).
New-Item -ItemType File -Force -Path (Join-Path $dir ('data-scale-v1-started_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.marker')) | Out-Null
# Logging must never terminate the launcher: with ErrorActionPreference Stop a
# transient write failure kills the run and leaves nothing to explain it.
function Log([string]$text) {
    try { Add-Content -Path $log -Value ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] " + $text) -ErrorAction Stop }
    catch { }
}
$code=1
try {
    Log "starting paired evaluation runner"
    # Start-Process owns the redirection: PowerShell 5.1 + ErrorActionPreference
    # Stop turns the first native stderr line under *>> into a terminating error
    # (docs/history.md trap 4; this killed the first diagnostic launch).
    $py = Join-Path $root '.venv\Scripts\python.exe'
    $p = Start-Process -FilePath $py -ArgumentList @('scripts/run_data_scale_eval.py') `
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
