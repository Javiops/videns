param([string]$TaskName='BuildTracker Oracle Budget Ceiling V1')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$env:PYTHONUTF8='1'
$env:PYTHONUNBUFFERED='1'
$env:PREFIX_ML_DIR=Join-Path $root 'data\dataset_generations\16.18-reconstructible-development-v3'
$out=Join-Path $root 'data\reports\oracle_budget_ceiling_v1'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$log=Join-Path $out 'run.log'
function Log([string]$text) {
    try { Add-Content -Path $log -Value ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] " + $text) -ErrorAction Stop }
    catch { }
}
$code=1
try {
    Log "oracle budget ceiling starting (TRAIN only, label-derived diagnostic)"
    $py = Join-Path $root '.venv\Scripts\python.exe'
    $p = Start-Process -FilePath $py -ArgumentList @(
        'scripts/footwear_runtime.py','--entry','scripts/oracle_budget_ceiling_v1.py','--',
        '--dataset', (Join-Path $root 'data\dataset_generations\16.18-reconstructible-development-v3'),
        '--artifact', (Join-Path $root 'data\experiments\16.18-reconstructible-development-v3\candidate.pt'),
        '--out', $out
    ) -WorkingDirectory $root -Wait -PassThru `
        -RedirectStandardOutput (Join-Path $out 'oracle.out.log') `
        -RedirectStandardError (Join-Path $out 'oracle.err.log')
    $code = $p.ExitCode
    Log ("finished with exit " + $code)
} catch {
    Log ("launcher failed: " + $_.Exception.Message)
} finally {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
}
exit $code
