param([string]$TaskName='BuildTracker Phase D Resume V3')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$env:PYTHONUTF8='1'
$env:PYTHONUNBUFFERED='1'
$log=Join-Path $root 'data\experiments\16.18-reconstructible-development-v3-resume-launcher.log'
New-Item -ItemType File -Force -Path (Join-Path $root ('data\experiments\phase_d_resume_started_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.marker')) | Out-Null
function Log([string]$text) {
    try { Add-Content -Path $log -Value ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] " + $text) -ErrorAction Stop }
    catch { }
}
$code=1
try {
    # The runner picks its own entry point: full reconstruction when the
    # corpus is absent, audit-only when just the audit was lost. Do not claim
    # one of them here — the stage it actually enters is in status.json.
    Log "resuming Phase D (entry point chosen by the runner) -> ... -> verify -> admission -> coverage -> export -> tests -> trunk -> graft -> policy"
    $py = Join-Path $root '.venv\Scripts\python.exe'
    $p = Start-Process -FilePath $py -ArgumentList @('scripts/resume_development_1618_v3.py') `
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
