# Prior-patch transfer: the two long arms (T then H), full mode. NOT registered by default: it waits for the
# measured pilot report (transfer prep chain), the declared bounds and the owner's GO. When registered
# (register_task.ps1 -TaskName 'BuildTracker Transfer Train' -ScriptName scripts\run_transfer_train.ps1) it runs
# basket_transfer_ab.py once for both arms: references bound, R1 reproduction gate, pilots vs bounds, T, H, scoring,
# paired comparisons and the decision block. Self-unregistering one-shot; the trainer's own exit codes are kept
# (2 = integrity gate failed, 3 = pilot above a declared bound).
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$taskName = 'BuildTracker Transfer Train'
$ts = Get-Date -Format yyyyMMdd_HHmmss
New-Item -ItemType Directory -Force (Join-Path $root 'data\logs') | Out-Null
$py = Join-Path $root '.venv\Scripts\python.exe'
$history = Join-Path $root 'data\history\16.17-history-v1'
$out = Join-Path $root 'data\experiments\16.18-prior-patch-transfer-v1'
$gen = Join-Path $root 'data\dataset_generations\16.18-reconstructible-development-v4'
$review = Join-Path $root 'data\acceptance\16.18_reconstructible_20260923\inputs\static_review.json'
$log = Join-Path $root ('data\logs\transfer_train_' + $ts + '.log')
$env:PYTHONUTF8 = '1'; $env:PYTHONUNBUFFERED = '1'; $env:PREFIX_ML_DIR = $gen
$env:HTTP_PROXY = 'http://127.0.0.1:9'; $env:HTTPS_PROXY = 'http://127.0.0.1:9'; $env:ALL_PROXY = 'http://127.0.0.1:9'; $env:NO_PROXY = ''
function Log($msg) { Add-Content -Path $log -Value ((Get-Date -Format s) + ' ' + $msg) }
function AvailableGB { [double]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory) / 1MB }
$code = 1
try {
    if (-not (Test-Path -LiteralPath (Join-Path $out 'pilot_report_full.json'))) { Log 'no full pilot report: the budget was not measured; refusing to train'; exit 6 }
    if (-not (Test-Path -LiteralPath (Join-Path $history 'history_manifest.json'))) { Log 'no full history export; refusing to train'; exit 6 }
    $waited = 0
    while ((AvailableGB) -lt 2.5) {          # 2.5 GB: the desktop keeps 1-3 GB free; the trainer's host side holds the raw cohort rows only
        if ($waited -eq 0) { Log ('waiting for RAM (available ' + ('{0:N2}' -f (AvailableGB)) + ' GB)') }
        Start-Sleep -Seconds 60; $waited += 60
        if ($waited -ge 10800) { Log 'RAM never reached 2.5 GB in 3 h; stopping'; exit 3 }
    }
    $o = Join-Path $root ('data\logs\transfer_train_' + $ts + '.out')
    $e = Join-Path $root ('data\logs\transfer_train_' + $ts + '.err')
    Log 'start basket_transfer_ab.py (both arms, full)'
    $p = Start-Process -FilePath $py -ArgumentList @('scripts\footwear_runtime.py', '--entry', 'scripts\basket_transfer_ab.py', '--review', $review, '--', '--history', $history, '--out', $out) `
        -WorkingDirectory $root -NoNewWindow -PassThru -Wait -RedirectStandardOutput $o -RedirectStandardError $e
    $null = $p.Handle
    $code = [int]$p.ExitCode
    Log ('trainer exit ' + $code)
}
finally {
    try { Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue } catch {}
}
exit $code
