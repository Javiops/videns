param([string]$TaskName='BuildTracker Gold Harness Phase A')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$env:PYTHONUTF8='1'
$env:PYTHONUNBUFFERED='1'
$out=Join-Path $root 'data\reports\gold_prequential_v3_phase_a'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$log=Join-Path $out 'run.log'
New-Item -ItemType File -Force -Path (Join-Path $out ('started_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.marker')) | Out-Null
function Log([string]$text) {
    try { Add-Content -Path $log -Value ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] " + $text) -ErrorAction Stop }
    catch { }
}
$code=1
try {
    Log "prequential gold harness starting (TRAIN only, full pass)"
    $py = Join-Path $root '.venv\Scripts\python.exe'
    $p = Start-Process -FilePath $py -ArgumentList @(
        'scripts/prequential_gold_harness.py',
        '--dataset', (Join-Path $root 'data\dataset_generations\16.18-reconstructible-development-v2'),
        '--out', $out
    ) -WorkingDirectory $root -Wait -PassThru `
        -RedirectStandardOutput (Join-Path $out 'harness.out.log') `
        -RedirectStandardError (Join-Path $out 'harness.err.log')
    $code = $p.ExitCode
    Log ("finished with exit " + $code)
} catch {
    Log ("launcher failed: " + $_.Exception.Message)
} finally {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
}
exit $code
