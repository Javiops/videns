# 2026-09-26 16:05: the parallel cache build lost worker 3 (shards 3,7) and the
# cohort worker (both died before printing anything) and the launcher's
# exit-code check misread $null as failure (PowerShell: Start-Process
# -PassThru only populates ExitCode if .Handle was touched first). This runs
# the three missing pieces with the fix, then the merge; the resume watcher
# relaunches the chain once the merge report exists.
param([string]$Shards = "3,7", [switch]$NoCohort)
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$ts = Get-Date -Format yyyyMMdd_HHmmss
$logdir = Join-Path $root 'data\logs'
$log = Join-Path $logdir ("features_remaining_" + $ts + ".log")
function Log($m) { Add-Content -LiteralPath $log -Value ((Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + ' ' + $m) }
$env:PYTHONUTF8 = '1'; $env:PYTHONUNBUFFERED = '1'; $env:CUDA_VISIBLE_DEVICES = ''
$env:PREFIX_ML_DIR = Join-Path $root 'data\dataset_generations\16.18-reconstructible-development-v4'
$review = Join-Path $root 'data\acceptance\16.18_reconstructible_20260923\inputs\static_review.json'
$py = Join-Path $root '.venv\Scripts\python.exe'
$base = @('scripts\footwear_runtime.py', '--entry', 'scripts\basket_econ_cache_newgen.py', '--review', $review, '--', '--phase', 'features')
function Launch($tag, $extra) {
    $p = Start-Process -FilePath $py -ArgumentList ($base + $extra) -WorkingDirectory $root -NoNewWindow -PassThru `
        -RedirectStandardOutput (Join-Path $logdir ("features_" + $tag + "_" + $ts + ".out")) `
        -RedirectStandardError (Join-Path $logdir ("features_" + $tag + "_" + $ts + ".err"))
    $null = $p.Handle          # keep the handle so ExitCode is populated after exit
    Log ("$tag pid " + $p.Id + " args " + ($extra -join ' '))
    return $p
}
$t0 = Get-Date
$procs = @()
foreach ($k in $Shards.Split(',')) { if ($k) { $procs += Launch ("shard" + $k) @('--shards', $k) } }
if (-not $NoCohort) { $procs += Launch 'cohort' @('--cohort-only') }
$procs | ForEach-Object { $_.WaitForExit() }
$codes = $procs | ForEach-Object { $_.ExitCode }
Log ("pieces finished in " + [int]((Get-Date) - $t0).TotalSeconds + " s; exit codes " + ($codes -join ','))
if (($codes | Where-Object { $_ -ne 0 }).Count -gt 0) { Log 'a piece failed; no merge'; exit 1 }
$pm = Launch 'merge' @('--merge', '--bench-ms', '16.9', '--rows-per-second', '215')
$pm.WaitForExit()
Log ("merge exit " + $pm.ExitCode)
exit $pm.ExitCode
