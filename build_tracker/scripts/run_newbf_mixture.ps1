# Frozen 50/50 of the new Bf (v4) with the old mixture package, on the v4
# cohort. Protocol: docs/experiment_newbf_mixture_v1.md. Self-unregistering
# one-shot; stdout goes to a plain file as the child flushes (Start-Transcript
# buffers native stdout until exit -- docs/history.md).
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$taskName = 'BuildTracker NewBf Mixture'
$ts = Get-Date -Format yyyyMMdd_HHmmss
New-Item -ItemType Directory -Force (Join-Path $root 'data\logs') | Out-Null
$out = Join-Path $root ('data\logs\newbf_mixture_' + $ts + '.out')
$err = Join-Path $root ('data\logs\newbf_mixture_' + $ts + '.err')
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:PREFIX_ML_DIR = Join-Path $root 'data\dataset_generations\16.18-reconstructible-development-v4'
$review = Join-Path $root 'data\acceptance\16.18_reconstructible_20260923\inputs\static_review.json'
$code = 1
try {
    $p = Start-Process -FilePath (Join-Path $root '.venv\Scripts\python.exe') `
        -ArgumentList @('scripts\footwear_runtime.py', '--entry', 'scripts\basket_newbf_mixture.py', '--review', $review) `
        -WorkingDirectory $root -NoNewWindow -PassThru -Wait `
        -RedirectStandardOutput $out -RedirectStandardError $err
    $code = [int]$p.ExitCode
}
finally {
    try { Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue } catch {}
}
exit $code
