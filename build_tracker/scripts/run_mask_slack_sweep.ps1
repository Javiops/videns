param(
  [string]$TaskName='BuildTracker Mask Slack Sweep 20260916',
  [string]$WaitFor='BuildTracker Gap Window Pull 20260916',
  [int]$WaitMinutes=240
)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$env:PYTHONUTF8='1'
$env:PYTHONUNBUFFERED='1'
$log=Join-Path $root 'data\mask_slack_sweep_20260916-launcher.log'
function Log([string]$text) {
    try { Add-Content -Path $log -Value ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] " + $text) -ErrorAction Stop }
    catch { }
}

$code=1
try {
    # Phases 1 and 2 of docs/experiment_mask_slack_v1.md (frozen before this ran).
    # TRAIN-only selection: no validation or test read, no ledger spend, no
    # retraining, nothing promotable. The gold estimator and GOLD_DRIFT are
    # untouched; only the decode-time budget varies between arms.
    #
    # Waits for the gap window pull so the two do not compete for the machine.
    # The pull is Riot-API bound and this is CPU bound, but the pull's dated
    # sample stage is the first clean end-to-end test of its fixed guard and it
    # gets the machine to itself.
    if ($WaitFor) {
        $deadline=(Get-Date).AddMinutes($WaitMinutes)
        $waited=$false
        while ((Get-Date) -lt $deadline) {
            $t = Get-ScheduledTask -TaskName $WaitFor -ErrorAction SilentlyContinue
            if (-not $t -or $t.State -ne 'Running') { break }
            if (-not $waited) { Log ("waiting for '" + $WaitFor + "' to finish"); $waited=$true }
            Start-Sleep -Seconds 60
        }
        $t = Get-ScheduledTask -TaskName $WaitFor -ErrorAction SilentlyContinue
        if ($t -and $t.State -eq 'Running') {
            Log ("'" + $WaitFor + "' still running after " + $WaitMinutes + " min; not starting")
            $code=2
            exit $code
        }
    }

    $name='16.18-reconstructible-development-v3'
    $env:PREFIX_ML_DIR   = Join-Path $root ("data\dataset_generations\" + $name)
    $env:PREFIX_RUN_DIR  = Join-Path $root ("data\experiments\" + $name)
    $env:PREFIX_RUNTIME_REVIEW = Join-Path $root 'data\acceptance\16.18_reconstructible_20260914\inputs\static_review.json'
    # Same model geometry the Phase D candidate was trained and scored under.
    $env:PREFIX_DMODEL='128'; $env:PREFIX_LAYERS='4'; $env:PREFIX_FF='256'; $env:PREFIX_HEADS='8'
    $env:PREFIX_COSINE='1'; $env:PREFIX_EXTRAS='1'; $env:PREFIX_HISTORY='0'
    $env:PREFIX_GOLDEST='1'; $env:PREFIX_GOLDX='1'; $env:PREFIX_RUNES='1'
    $env:PREFIX_POSW='8'; $env:PREFIX_POSW_SCHED='8:24'; $env:PREFIX_POSW_CUSTOM='0'
    $env:PREFIX_TBLEND='0'; $env:PREFIX_SPLICE='0'

    $arms='const:0,const:50,const:75,const:100,const:150,const:200,const:300,const:500,rate:1,rate:2,rate:4,rate:6'
    $out=Join-Path $root 'data\reports\mask_slack_sweep_v1'
    Log ("mask slack sweep starting: 12 arms, 500 train games, seed 16, bootstrap 400")
    $py = Join-Path $root '.venv\Scripts\python.exe'
    $p = Start-Process -FilePath $py -ArgumentList @(
            'scripts/mask_slack_sweep_v1.py',
            '--arms', $arms, '--games', '500', '--seed', '16',
            '--bootstrap', '400', '--out', $out) `
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
