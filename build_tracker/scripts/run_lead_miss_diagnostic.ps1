param([string]$TaskName='BuildTracker Lead Miss Diagnostic V1')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$env:PYTHONUTF8='1'
$env:PYTHONUNBUFFERED='1'
$env:PREFIX_ML_DIR=Join-Path $root 'data\dataset_generations\16.18-reconstructible-development-v2'
$out=Join-Path $root 'data\reports\lead_miss_diagnostic_v1'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$log=Join-Path $out 'run.log'
# Evidence that the launcher process started at all, independent of any file
# another process may hold open: the 01:41 attempt vanished without a log line.
New-Item -ItemType File -Force -Path (Join-Path $out ('started_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.marker')) | Out-Null
# Logging must never terminate the launcher: with ErrorActionPreference Stop a
# transient write failure (a tail holding the file, a locked handle) kills the
# whole run and leaves nothing behind to explain it.
function Log([string]$text) {
    try { Add-Content -Path $log -Value ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] " + $text) -ErrorAction Stop }
    catch { }
}
$code=1
try {
    Log "launcher started; waiting for the data-scale eval task to finish"
    $waited=0
    while ($waited -lt 43200) {
        $busy = Get-ScheduledTask -TaskName 'BuildTracker Data Scale Arm 1617 Eval' -ErrorAction SilentlyContinue
        if ($null -eq $busy) { break }
        Start-Sleep -Seconds 60
        $waited += 60
    }
    Log ("lead-miss diagnostic starting (TRAIN only; waited " + $waited + "s)")
    # Start-Process owns the redirection: PowerShell 5.1 + ErrorActionPreference
    # Stop turns the first native stderr line under *>> into a terminating error
    # (torch warnings killed the 01:32 attempt; docs/history.md trap 4).
    $py = Join-Path $root '.venv\Scripts\python.exe'
    $p = Start-Process -FilePath $py -ArgumentList @(
        'scripts/footwear_runtime.py','--entry','scripts/mine_lead_misses_v1.py','--',
        '--dataset', (Join-Path $root 'data\dataset_generations\16.18-reconstructible-development-v2'),
        '--artifact', (Join-Path $root 'data\experiments\basket-composition-v1\candidate.pt'),
        '--out', $out
    ) -WorkingDirectory $root -Wait -PassThru `
        -RedirectStandardOutput (Join-Path $out 'diagnostic.out.log') `
        -RedirectStandardError (Join-Path $out 'diagnostic.err.log')
    $code = $p.ExitCode
    Log ("finished with exit " + $code)
} catch {
    Log ("launcher failed: " + $_.Exception.Message)
} finally {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
}
exit $code
