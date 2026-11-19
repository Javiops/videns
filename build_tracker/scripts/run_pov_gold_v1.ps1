# Experiment 16.18-pov-gold-v1: trunk -> graft -> paired conditional eval.
# Plan: data/experiments/16.18-pov-gold-v1/run_plan.json (validated by
# scripts/make_pov_run_plan.py). Protocol: docs/experiment_pov_gold_v1.md.
# One-shot scheduled task; unregisters itself at the end (register_task.ps1
# contract) so nothing lingers in the scheduler.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$taskName = 'BuildTracker POV Gold V1'
$run = Join-Path $root 'data\experiments\16.18-pov-gold-v1'
$log = Join-Path $root ('data\logs\pov_gold_v1_' + (Get-Date -Format yyyyMMdd_HHmmss) + '.log')
New-Item -ItemType Directory -Force (Join-Path $root 'data\logs') | Out-Null
Start-Transcript -Path $log

function Set-FixedEnv {
    $env:PREFIX_COSINE = '1';  $env:PREFIX_DMODEL = '128'
    $env:PREFIX_EXTRAS = '1';  $env:PREFIX_FF = '256'
    $env:PREFIX_GOLDEST = '1'; $env:PREFIX_GOLDX = '1'
    $env:PREFIX_HEADS = '8';   $env:PREFIX_HISTORY = '0'
    $env:PREFIX_LAYERS = '4';  $env:PREFIX_POSW = '8'
    $env:PREFIX_POSW_CUSTOM = '0'; $env:PREFIX_POSW_SCHED = '8:24'
    $env:PREFIX_RUNES = '1';   $env:PREFIX_SPLICE = '0'
    $env:PREFIX_TBLEND = '0'
    $env:PREFIX_POV_GOLD = '1'
    $env:PREFIX_ML_DIR = Join-Path $root 'data\dataset_generations\16.18-reconstructible-development-v3'
    $env:PREFIX_RUN_DIR = $run
    $env:PREFIX_RUN_PLAN = Join-Path $run 'run_plan.json'
}

# The 16.18 generation trains through footwear_runtime.py, which activates the
# reconstructible-16.18-v2 runtime (patched RECONSTRUCTION_VERSION, pinned
# dragon, footwear economics) BEFORE importing the trainer — calling
# train_prefix.py directly refuses the corpus ("requires decision-start-v3"),
# which is how the first launch of this experiment died in 7 seconds.
# --review is explicit because qualification.json still records the corpus at
# the pre-move repository root.
$review = Join-Path $root 'data\corpora\16.18-reconstructible-v3\static_review\static_review.json'
if (-not (Test-Path -LiteralPath $review)) { throw "missing static review: $review" }

try {
    Set-FixedEnv
    $env:PREFIX_STAGE = 'trunk'
    $env:PREFIX_EPOCHS = '16'; $env:PREFIX_FREEZE = '0'
    $env:PREFIX_INIT_FROM = ''; $env:PREFIX_OUT = 'trunk.pt'
    $env:PREFIX_SAVEW = '0'; $env:PREFIX_TARGETW = '0'
    Write-Output '=== stage: trunk ==='
    & .venv\Scripts\python.exe scripts\footwear_runtime.py --entry scripts\train_prefix.py --review $review
    if ($LASTEXITCODE -ne 0) { throw "trunk failed (exit $LASTEXITCODE)" }

    $env:PREFIX_STAGE = 'graft'
    $env:PREFIX_EPOCHS = '6'; $env:PREFIX_FREEZE = '1'
    $env:PREFIX_INIT_FROM = 'trunk.pt'; $env:PREFIX_OUT = 'candidate.pt'
    $env:PREFIX_SAVEW = '0.5'; $env:PREFIX_TARGETW = '0.25'
    Write-Output '=== stage: graft ==='
    & .venv\Scripts\python.exe scripts\footwear_runtime.py --entry scripts\train_prefix.py --review $review
    if ($LASTEXITCODE -ne 0) { throw "graft failed (exit $LASTEXITCODE)" }

    Write-Output '=== stage: paired conditional evaluation (full val) ==='
    & .venv\Scripts\python.exe scripts\footwear_runtime.py --entry scripts\eval_policy.py --review $review `
        -- --artifact candidate.pt --compare-conditional --split val --games 0 --seed 16 --bootstrap 400
    if ($LASTEXITCODE -ne 0) { throw "evaluation failed (exit $LASTEXITCODE)" }

    Write-Output 'EXPERIMENT COMPLETE — report under data/experiments/16.18-pov-gold-v1/'
}
finally {
    try {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    } catch {}
    Stop-Transcript
}
