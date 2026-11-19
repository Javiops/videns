param([string]$TaskName='BuildTracker Posterior Day 20260917')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$env:PYTHONUTF8='1'
$env:PYTHONUNBUFFERED='1'
$log=Join-Path $root 'data\posterior_day_20260917-launcher.log'
function Log([string]$text) {
    try { Add-Content -Path $log -Value ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] " + $text) -ErrorAction Stop }
    catch { }
}

$code=1
try {
    # Phase 2b of docs/experiment_mask_slack_v1.md (addendum frozen before this
    # ran) plus the miss profile. TRAIN-only, no validation or test read, no
    # ledger spend, no retraining, nothing promotable. The candidate is the
    # Phase D artifact, unchanged; only the decode-time budget varies.
    $name='16.18-reconstructible-development-v3'
    $env:PREFIX_ML_DIR   = Join-Path $root ("data\dataset_generations\" + $name)
    $env:PREFIX_RUN_DIR  = Join-Path $root ("data\experiments\" + $name)
    $env:PREFIX_RUNTIME_REVIEW = Join-Path $root 'data\acceptance\16.18_reconstructible_20260914\inputs\static_review.json'
    $env:PREFIX_DMODEL='128'; $env:PREFIX_LAYERS='4'; $env:PREFIX_FF='256'; $env:PREFIX_HEADS='8'
    $env:PREFIX_COSINE='1'; $env:PREFIX_EXTRAS='1'; $env:PREFIX_HISTORY='0'
    $env:PREFIX_GOLDEST='1'; $env:PREFIX_GOLDX='1'; $env:PREFIX_RUNES='1'
    $env:PREFIX_POSW='8'; $env:PREFIX_POSW_SCHED='8:24'; $env:PREFIX_POSW_CUSTOM='0'
    $env:PREFIX_TBLEND='0'; $env:PREFIX_SPLICE='0'
    $py = Join-Path $root '.venv\Scripts\python.exe'
    $posterior = Join-Path $root 'data\reports\gold_posterior_v1\posterior.json'

    # Stage 1 — miss profile. Shorter, and it answers a standing question, so it
    # runs first: if the day is interrupted, this is the part already banked.
    Log "stage 1/2: miss profile (const:0 vs quant:0.65, 500 games)"
    $p = Start-Process -FilePath $py -ArgumentList @(
            'scripts/miss_profile_v1.py',
            '--arms', 'const:0,quant:0.65', '--games', '500', '--seed', '16',
            '--out', (Join-Path $root 'data\reports\miss_profile_v1')) `
        -WorkingDirectory $root -Wait -PassThru `
        -RedirectStandardOutput ($log + '.miss.out') `
        -RedirectStandardError ($log + '.miss.err')
    Log ("stage 1/2 exited " + $p.ExitCode)
    if ($p.ExitCode -ne 0) { $code = $p.ExitCode; exit $code }

    # Stage 2 — quantile arms of the fitted posterior, with rate:2 carried
    # forward so phases 2 and 2b share one paired axis.
    $arms='const:0,quant:0.50,quant:0.55,quant:0.60,quant:0.65,quant:0.70,quant:0.75,quant:0.80,quant:0.85,quant:0.90,rate:2'
    Log "stage 2/2: quantile sweep (11 arms, 500 games, seed 16, bootstrap 400)"
    $p = Start-Process -FilePath $py -ArgumentList @(
            'scripts/mask_slack_sweep_v1.py',
            '--arms', $arms, '--posterior', $posterior,
            '--games', '500', '--seed', '16', '--bootstrap', '400',
            '--out', (Join-Path $root 'data\reports\mask_slack_sweep_v1b')) `
        -WorkingDirectory $root -Wait -PassThru `
        -RedirectStandardOutput ($log + '.sweep.out') `
        -RedirectStandardError ($log + '.sweep.err')
    $code = $p.ExitCode
    Log ("stage 2/2 exited " + $code)
    Log ("finished with exit " + $code)
} catch {
    Log ("launcher failed: " + $_.Exception.Message)
} finally {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
}
exit $code
