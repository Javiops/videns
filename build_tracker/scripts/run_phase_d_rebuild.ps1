param([string]$TaskName='BuildTracker Phase D Rebuild V3')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$env:PYTHONUTF8='1'
$env:PYTHONUNBUFFERED='1'
$log=Join-Path $root 'data\experiments\16.18-reconstructible-development-v3-launcher.log'
New-Item -ItemType File -Force -Path (Join-Path $root ('data\experiments\phase_d_started_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.marker')) | Out-Null
function Log([string]$text) {
    try { Add-Content -Path $log -Value ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] " + $text) -ErrorAction Stop }
    catch { }
}
$code=1
try {
    Log "launcher started; waiting for the window sample to finish"
    $waited=0
    while ($waited -lt 18000) {
        $busy = Get-ScheduledTask -TaskName 'BuildTracker Window Sample 20260914' -ErrorAction SilentlyContinue
        if ($null -eq $busy) { break }
        Start-Sleep -Seconds 60
        $waited += 60
    }
    Log ("starting Phase D chain (waited " + $waited + "s): prepare -> recon -> admission -> export -> tests -> trunk -> graft -> policy")
    $py = Join-Path $root '.venv\Scripts\python.exe'
    $p = Start-Process -FilePath $py -ArgumentList @('scripts/run_development_1618_v3.py') `
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
