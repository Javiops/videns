# Prior-patch transfer SMOKE chain (2026-09-27): export the qualified 300-game 16.17 smoke corpus as a history
# corpus (validator at export), build the common-vocabulary caches for both patches (validator again), run the
# trainer smoke (both arms, U=40, 1 epoch, 1,500 cohort rows, R1/R2 digest-bound) and the steady-state pilots
# (declared-budget projection). Self-unregistering one-shot (register_task.ps1); stops at the first non-zero exit.
# Nothing here touches app/, the v4 generation, M2 or the full history chain.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$taskName = 'BuildTracker Transfer Smoke'
$ts = Get-Date -Format yyyyMMdd_HHmmss
New-Item -ItemType Directory -Force (Join-Path $root 'data\logs') | Out-Null
$py = Join-Path $root '.venv\Scripts\python.exe'
$base = Join-Path $root 'data\acceptance\16.17_history_smoke_20260927'
$corpus = Join-Path $root 'data\corpora\16.17-history-smoke'
$history = Join-Path $root 'data\history\16.17-history-smoke'
$out = Join-Path $root 'data\experiments\16.18-prior-patch-transfer-smoke'
$gen = Join-Path $root 'data\dataset_generations\16.18-reconstructible-development-v4'
$review = Join-Path $root 'data\acceptance\16.18_reconstructible_20260923\inputs\static_review.json'
$env:PYTHONUTF8 = '1'; $env:PYTHONUNBUFFERED = '1'
$env:HTTP_PROXY = 'http://127.0.0.1:9'; $env:HTTPS_PROXY = 'http://127.0.0.1:9'; $env:ALL_PROXY = 'http://127.0.0.1:9'; $env:NO_PROXY = ''
$steps = @(
  @{ name = 'export';    v4 = $false; args = @('scripts\export_history_1617.py', '--base', $base, '--corpus', $corpus, '--out', $history, '--allow-limited') },
  @{ name = 'cache1617'; v4 = $false; args = @('scripts\prepare_transfer_caches.py', '--patch', '16.17', '--smoke', '--history', $history, '--out', $out) },
  @{ name = 'cache1618'; v4 = $true;  args = @('scripts\footwear_runtime.py', '--entry', 'scripts\prepare_transfer_caches.py', '--review', $review, '--', '--patch', '16.18', '--smoke', '--history', $history, '--out', $out) },
  @{ name = 'smoke';     v4 = $true;  args = @('scripts\footwear_runtime.py', '--entry', 'scripts\basket_transfer_ab.py', '--review', $review, '--', '--smoke', '--history', $history, '--out', $out) },
  @{ name = 'pilot';     v4 = $true;  args = @('scripts\footwear_runtime.py', '--entry', 'scripts\basket_transfer_ab.py', '--review', $review, '--', '--pilot-only', '--smoke', '--history', $history, '--out', $out) }
)
$code = 0
try {
    foreach ($s in $steps) {
        if ($s.v4) { $env:PREFIX_ML_DIR = $gen } else { Remove-Item Env:PREFIX_ML_DIR -ErrorAction SilentlyContinue }
        $o = Join-Path $root ('data\logs\transfer_smoke_' + $s.name + '_' + $ts + '.out')
        $e = Join-Path $root ('data\logs\transfer_smoke_' + $s.name + '_' + $ts + '.err')
        $p = Start-Process -FilePath $py -ArgumentList $s.args -WorkingDirectory $root -NoNewWindow -PassThru -Wait `
            -RedirectStandardOutput $o -RedirectStandardError $e
        $null = $p.Handle
        $code = [int]$p.ExitCode
        Add-Content -Path (Join-Path $root ('data\logs\transfer_smoke_' + $ts + '.log')) -Value ((Get-Date -Format s) + ' ' + $s.name + ' exit ' + $code)
        if ($code -ne 0) { break }
    }
}
finally {
    try { Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue } catch {}
}
exit $code
