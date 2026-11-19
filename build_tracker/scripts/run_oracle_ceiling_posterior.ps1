param(
  [string]$TaskName='BuildTracker Oracle Ceiling Posterior 20260917',
  [string]$WaitFor='BuildTracker Gold Posterior 20260917',
  [int]$WaitMinutes=120
)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$env:PYTHONUTF8='1'
$env:PYTHONUNBUFFERED='1'
$log=Join-Path $root 'data\oracle_ceiling_posterior_20260917-launcher.log'
function Log([string]$text) {
    try { Add-Content -Path $log -Value ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] " + $text) -ErrorAction Stop }
    catch { }
}

$code=1
try {
    # The oracle-budget ceiling, re-run on the uncertainty-trained candidate with
    # the SAME method that produced 0.6006 on the point candidate on Sep 15:
    # same script, full TRAIN purchases, control at budget_slack=0, oracle wallet
    # = max(gold_est, exact recursive cost of the recorded basket).
    #
    # LABEL-DERIVED CEILING. TRAIN only. Never a feature, never an evaluation
    # reference, never promotable. It answers one question: with a perfect
    # wallet, how much does each model already know?
    if ($WaitFor) {
        $deadline=(Get-Date).AddMinutes($WaitMinutes)
        while ((Get-Date) -lt $deadline) {
            $t = Get-ScheduledTask -TaskName $WaitFor -ErrorAction SilentlyContinue
            if (-not $t -or $t.State -ne 'Running') { break }
            Start-Sleep -Seconds 60
        }
        $t = Get-ScheduledTask -TaskName $WaitFor -ErrorAction SilentlyContinue
        if ($t -and $t.State -eq 'Running') {
            Log ("'" + $WaitFor + "' still running after " + $WaitMinutes + " min; not starting")
            $code=2; exit $code
        }
    }

    $name='16.18-reconstructible-development-v3'
    $env:PREFIX_ML_DIR  = Join-Path $root ("data\dataset_generations\" + $name)
    $env:PREFIX_RUN_DIR = Join-Path $root 'data\experiments\16.18-gold-posterior-v1'
    $env:PREFIX_RUNTIME_REVIEW = Join-Path $root 'data\acceptance\16.18_reconstructible_20260914\inputs\static_review.json'
    $env:PREFIX_DMODEL='128'; $env:PREFIX_LAYERS='4'; $env:PREFIX_FF='256'; $env:PREFIX_HEADS='8'
    $env:PREFIX_COSINE='1'; $env:PREFIX_EXTRAS='1'; $env:PREFIX_HISTORY='0'
    $env:PREFIX_GOLDEST='1'; $env:PREFIX_GOLDX='1'; $env:PREFIX_RUNES='1'
    $env:PREFIX_POSW='8'; $env:PREFIX_POSW_SCHED='8:24'; $env:PREFIX_POSW_CUSTOM='0'
    $env:PREFIX_TBLEND='0'; $env:PREFIX_SPLICE='0'

    $py = Join-Path $root '.venv\Scripts\python.exe'
    Log "oracle ceiling starting on the posterior candidate (full TRAIN, same method as 2026-09-15)"
    $p = Start-Process -FilePath $py -ArgumentList @(
            'scripts/footwear_runtime.py', '--entry', 'scripts/oracle_budget_ceiling_v1.py', '--',
            '--dataset', (Join-Path $root ("data\dataset_generations\" + $name)),
            '--artifact', (Join-Path $root 'data\experiments\16.18-gold-posterior-v1\candidate.pt'),
            '--out', (Join-Path $root 'data\reports\oracle_budget_ceiling_posterior_v1')) `
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
