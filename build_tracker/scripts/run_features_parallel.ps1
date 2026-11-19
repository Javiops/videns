# Parallel economic-feature cache on the v4 generation (owner, 2026-09-26:
# "no voy a utilizar el pc en todo el dia... usa mas recursos"). Four shard
# workers (two 100k-row shards each) + one cohort worker, all CPU-only, then
# the merge, which verifies every shard and writes the manifest + report the
# chain's --resume expects. Bit-identity with the sequential build was
# asserted on the smoke before this was launched.
param([int]$Workers = 4, [int]$Shards = 8)
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$ts = Get-Date -Format yyyyMMdd_HHmmss
$logdir = Join-Path $root 'data\logs'
New-Item -ItemType Directory -Force $logdir | Out-Null
$log = Join-Path $logdir ("features_parallel_" + $ts + ".log")
function Log($m) { Add-Content -LiteralPath $log -Value ((Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + ' ' + $m) }
$env:PYTHONUTF8 = '1'; $env:PYTHONUNBUFFERED = '1'; $env:CUDA_VISIBLE_DEVICES = ''
$env:PREFIX_ML_DIR = Join-Path $root 'data\dataset_generations\16.18-reconstructible-development-v4'
$review = Join-Path $root 'data\acceptance\16.18_reconstructible_20260923\inputs\static_review.json'
$py = Join-Path $root '.venv\Scripts\python.exe'
$base = @('scripts\footwear_runtime.py', '--entry', 'scripts\basket_econ_cache_newgen.py', '--review', $review, '--', '--phase', 'features')
$procs = @()
$t0 = Get-Date
for ($w = 0; $w -lt $Workers; $w++) {
    $mine = @(); for ($k = $w; $k -lt $Shards; $k += $Workers) { $mine += $k }
    $args = $base + @('--shards', ($mine -join ','))
    $p = Start-Process -FilePath $py -ArgumentList $args -WorkingDirectory $root -NoNewWindow -PassThru `
        -RedirectStandardOutput (Join-Path $logdir ("features_worker" + $w + "_" + $ts + ".out")) `
        -RedirectStandardError (Join-Path $logdir ("features_worker" + $w + "_" + $ts + ".err"))
    $procs += $p; Log ("worker $w pid " + $p.Id + " shards " + ($mine -join ','))
}
$pc = Start-Process -FilePath $py -ArgumentList ($base + @('--cohort-only')) -WorkingDirectory $root -NoNewWindow -PassThru `
    -RedirectStandardOutput (Join-Path $logdir ("features_cohort_" + $ts + ".out")) `
    -RedirectStandardError (Join-Path $logdir ("features_cohort_" + $ts + ".err"))
$procs += $pc; Log ("cohort worker pid " + $pc.Id)
$procs | ForEach-Object { $_.WaitForExit() }
$codes = $procs | ForEach-Object { $_.ExitCode }
$wall = ((Get-Date) - $t0).TotalSeconds
Log ("workers finished in " + [int]$wall + " s; exit codes " + ($codes -join ','))
if (($codes | Where-Object { $_ -ne 0 }).Count -gt 0) { Log 'a worker failed; no merge'; exit 1 }
$rps = [math]::Round((757975 + 146597) / $wall, 2)
$pm = Start-Process -FilePath $py -ArgumentList ($base + @('--merge', '--bench-ms', '16.9', '--rows-per-second', "$rps")) -WorkingDirectory $root -NoNewWindow -PassThru -Wait `
    -RedirectStandardOutput (Join-Path $logdir ("features_merge_" + $ts + ".out")) `
    -RedirectStandardError (Join-Path $logdir ("features_merge_" + $ts + ".err"))
Log ("merge exit " + $pm.ExitCode + " (aggregate " + $rps + " rows/s)")
exit $pm.ExitCode
