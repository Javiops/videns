param([string]$TaskName='BuildTracker Gold Posterior 20260917')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$env:PYTHONUTF8='1'
$env:PYTHONUNBUFFERED='1'
$log=Join-Path $root 'data\gold_posterior_20260917-launcher.log'
function Log([string]$text) {
    try { Add-Content -Path $log -Value ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] " + $text) -ErrorAction Stop }
    catch { }
}

$code=1
try {
    # docs/experiment_gold_posterior_training_v1.md, frozen before this ran.
    # tests -> trunk -> graft -> slack sweep -> miss profile, and it STOPS there:
    # the paired validation comparison is not run by this chain.
    #
    # The runner owns the environment. It builds and validates its own frozen run
    # plan, so nothing is set here beyond the launcher's own plumbing.
    Log "gold posterior training starting (tests -> trunk -> graft -> sweep -> miss profile)"
    $py = Join-Path $root '.venv\Scripts\python.exe'
    $p = Start-Process -FilePath $py -ArgumentList @('scripts/run_gold_posterior_training_v1.py') `
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
