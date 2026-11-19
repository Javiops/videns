param([string]$TaskName='BuildTracker Phase C Sample Effect')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$env:PYTHONUTF8='1'
$env:PYTHONUNBUFFERED='1'
$out=Join-Path $root 'data\reports\opening_sale_sample_effect_v1'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$log=Join-Path $out 'run.log'
New-Item -ItemType File -Force -Path (Join-Path $out ('started_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.marker')) | Out-Null
function Log([string]$text) {
    try { Add-Content -Path $log -Value ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] " + $text) -ErrorAction Stop }
    catch { }
}
$code=1
try {
    Log "phase C sample effect starting (old=git HEAD vs new, 300 TRAIN games)"
    $py = Join-Path $root '.venv\Scripts\python.exe'
    $p = Start-Process -FilePath $py -ArgumentList @(
        'scripts/sample_reconstruction_effect_v1.py',
        '--dataset', (Join-Path $root 'data\dataset_generations\16.18-reconstructible-development-v2'),
        '--out', $out, '--games', '300'
    ) -WorkingDirectory $root -Wait -PassThru `
        -RedirectStandardOutput (Join-Path $out 'sample.out.log') `
        -RedirectStandardError (Join-Path $out 'sample.err.log')
    $code = $p.ExitCode
    Log ("sample finished with exit " + $code)
    if ($code -eq 0) {
        Log "re-running gold harness for post-change visit overcredit"
        $out2=Join-Path $root 'data\reports\gold_prequential_v3_phase_a_after'
        New-Item -ItemType Directory -Force -Path $out2 | Out-Null
        $p2 = Start-Process -FilePath $py -ArgumentList @(
            'scripts/prequential_gold_harness.py',
            '--dataset', (Join-Path $root 'data\dataset_generations\16.18-reconstructible-development-v2'),
            '--out', $out2
        ) -WorkingDirectory $root -Wait -PassThru `
            -RedirectStandardOutput (Join-Path $out2 'harness.out.log') `
            -RedirectStandardError (Join-Path $out2 'harness.err.log')
        $code = $p2.ExitCode
        Log ("harness rerun finished with exit " + $code)
    }
} catch {
    Log ("launcher failed: " + $_.Exception.Message)
} finally {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
}
exit $code
