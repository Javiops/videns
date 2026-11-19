# Prior-patch transfer FULL preparation (owner GO 2026-09-27 12:5x): waits for the complete 16.17 history
# reconstruction (snapshot r2), then export (validator at export) -> common-vocabulary caches for both patches
# (validator again) -> steady-state pilots (--pilot-only, declared-budget projection). The two long trainings
# are NOT started here (they wait for the measured budget and the owner's GO). Self-unregistering one-shot
# (register_task.ps1); stops at the first non-zero exit; every step waits for >= 2.5 GB of available RAM.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$taskName = 'BuildTracker Transfer Prep'
$ts = Get-Date -Format yyyyMMdd_HHmmss
New-Item -ItemType Directory -Force (Join-Path $root 'data\logs') | Out-Null
$py = Join-Path $root '.venv\Scripts\python.exe'
$base = Join-Path $root 'data\acceptance\16.17_history_20260927'
$corpus = Join-Path $root 'data\corpora\16.17-history-v1'
$history = Join-Path $root 'data\history\16.17-history-v1'
$out = Join-Path $root 'data\experiments\16.18-prior-patch-transfer-v1'
$gen = Join-Path $root 'data\dataset_generations\16.18-reconstructible-development-v4'
$review = Join-Path $root 'data\acceptance\16.18_reconstructible_20260923\inputs\static_review.json'
$log = Join-Path $root ('data\logs\transfer_prep_' + $ts + '.log')
$env:PYTHONUTF8 = '1'; $env:PYTHONUNBUFFERED = '1'
$env:HTTP_PROXY = 'http://127.0.0.1:9'; $env:HTTPS_PROXY = 'http://127.0.0.1:9'; $env:ALL_PROXY = 'http://127.0.0.1:9'; $env:NO_PROXY = ''
function Log($msg) { Add-Content -Path $log -Value ((Get-Date -Format s) + ' ' + $msg) }
function AvailableGB { [double]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory) / 1MB }
$code = 1
try {
    # ---- 1. the reconstruction must have COMPLETED (build + audit + qualification); a failed run stops here ----
    $status = Join-Path $base 'run\status.json'
    $completion = Join-Path $base 'run\completion.json'
    Log 'waiting for the 16.17 history reconstruction to complete'
    $deadline = (Get-Date).AddHours(8)            # the supervisor's own wall budget is 6 h
    $lastWrite = Get-Date
    while ($true) {
        if (Test-Path -LiteralPath $completion) { break }
        if (Test-Path -LiteralPath $status) {
            try {
                $s = Get-Content -LiteralPath $status -Raw -ErrorAction Stop | ConvertFrom-Json
                if ($s.state -eq 'failed') { Log ('reconstruction FAILED: ' + $s.reason); exit 2 }
                $lastWrite = (Get-Item -LiteralPath $status).LastWriteTime   # rewritten every 10 s while alive
            } catch { Log ('transient read of status.json failed: ' + $_.Exception.Message) }
        }
        if ((Get-Date) -gt $deadline) { Log 'reconstruction did not complete within 8 h; stopping'; exit 4 }
        if (((Get-Date) - $lastWrite).TotalMinutes -gt 15) { Log 'status.json stale for 15 min without completion: supervisor dead; stopping'; exit 5 }
        Start-Sleep -Seconds 60
    }
    Log 'reconstruction complete'
    $steps = @(
      @{ name = 'export';    v4 = $false; args = @('scripts\export_history_1617.py', '--base', $base, '--corpus', $corpus, '--out', $history) },
      @{ name = 'cache1617'; v4 = $false; args = @('scripts\prepare_transfer_caches.py', '--patch', '16.17', '--history', $history, '--out', $out) },
      @{ name = 'cache1618'; v4 = $true;  args = @('scripts\footwear_runtime.py', '--entry', 'scripts\prepare_transfer_caches.py', '--review', $review, '--', '--patch', '16.18', '--history', $history, '--out', $out) },
      @{ name = 'pilot';     v4 = $true;  args = @('scripts\footwear_runtime.py', '--entry', 'scripts\basket_transfer_ab.py', '--review', $review, '--', '--pilot-only', '--history', $history, '--out', $out) }
    )
    foreach ($s in $steps) {
        $waited = 0
        while ((AvailableGB) -lt 2.5) {          # 2.5 GB: the desktop keeps 1-3 GB free all day; the steps stream rows and hold <= 1.6 GB of tensors
            if ($waited -eq 0) { Log ('waiting for RAM before ' + $s.name + ' (available ' + ('{0:N2}' -f (AvailableGB)) + ' GB)') }
            Start-Sleep -Seconds 60; $waited += 60
            if ($waited -ge 10800) { Log ('RAM never reached 2.5 GB in 3 h before ' + $s.name + '; stopping'); exit 3 }
        }
        if ($s.v4) { $env:PREFIX_ML_DIR = $gen } else { Remove-Item Env:PREFIX_ML_DIR -ErrorAction SilentlyContinue }
        $o = Join-Path $root ('data\logs\transfer_prep_' + $s.name + '_' + $ts + '.out')
        $e = Join-Path $root ('data\logs\transfer_prep_' + $s.name + '_' + $ts + '.err')
        Log ('start ' + $s.name)
        $p = Start-Process -FilePath $py -ArgumentList $s.args -WorkingDirectory $root -NoNewWindow -PassThru -Wait `
            -RedirectStandardOutput $o -RedirectStandardError $e
        $null = $p.Handle
        $code = [int]$p.ExitCode
        Log ($s.name + ' exit ' + $code)
        if ($code -ne 0) { break }
    }
}
finally {
    try { Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue } catch {}
}
exit $code
