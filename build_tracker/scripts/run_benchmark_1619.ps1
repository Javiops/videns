# Patch 16.19 benchmark chain (docs/benchmark_1619_v1.md; owner's job 2, 2026-09-29). Waits for the complete
# 16.19 reconstruction, then: admission with the declared forward boundary -> coverage -> publication ->
# frozen R1 scoring (through the 16.18 runtime) -> frozen M2 scoring (package) -> paired benchmark report.
# No training. Self-unregistering one-shot (register_task.ps1); stops at the first non-zero exit; each step
# waits for >= 2.5 GB of available RAM.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$taskName = 'BuildTracker Benchmark 1619'
$ts = Get-Date -Format yyyyMMdd_HHmmss
New-Item -ItemType Directory -Force (Join-Path $root 'data\logs') | Out-Null
$py = Join-Path $root '.venv\Scripts\python.exe'
$base = Join-Path $root 'data\acceptance\16.19_reconstructible_20260929'
$corpus = Join-Path $root 'data\corpora\16.19-reconstructible-v1'
$prepared = Join-Path $root 'data\acceptance\16.19_development_v1_admission'
$coverage = Join-Path $root 'data\acceptance\16.19_development_v1_admission_coverage'
$genName = '16.19-development-v1'
$generation = Join-Path $root ('data\dataset_generations\' + $genName)
$out = Join-Path $root 'data\experiments\16.19-benchmark-v1'
$gen18 = Join-Path $root 'data\dataset_generations\16.18-reconstructible-development-v4'
$review18 = Join-Path $root 'data\acceptance\16.18_reconstructible_20260923\inputs\static_review.json'
$boundary = '2026-09-27T00:00:00+00:00'
$log = Join-Path $root ('data\logs\benchmark_1619_' + $ts + '.log')
$env:PYTHONUTF8 = '1'; $env:PYTHONUNBUFFERED = '1'
$env:HTTP_PROXY = 'http://127.0.0.1:9'; $env:HTTPS_PROXY = 'http://127.0.0.1:9'; $env:ALL_PROXY = 'http://127.0.0.1:9'; $env:NO_PROXY = ''
function Log($msg) { Add-Content -Path $log -Value ((Get-Date -Format s) + ' ' + $msg) }
function AvailableGB { [double]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory) / 1MB }
$code = 1
try {
    $status = Join-Path $base 'run\status.json'
    $completion = Join-Path $base 'run\completion.json'
    Log 'waiting for the 16.19 reconstruction to complete'
    $deadline = (Get-Date).AddHours(8); $lastWrite = Get-Date
    while ($true) {
        if (Test-Path -LiteralPath $completion) { break }
        if (Test-Path -LiteralPath $status) {
            try {
                $s = Get-Content -LiteralPath $status -Raw -ErrorAction Stop | ConvertFrom-Json
                if ($s.state -eq 'failed') { Log ('reconstruction FAILED: ' + $s.reason); exit 2 }
                $lastWrite = (Get-Item -LiteralPath $status).LastWriteTime
            } catch { Log ('transient read of status.json failed: ' + $_.Exception.Message) }
        }
        if ((Get-Date) -gt $deadline) { Log 'reconstruction did not complete within 8 h; stopping'; exit 4 }
        if (((Get-Date) - $lastWrite).TotalMinutes -gt 15) { Log 'status.json stale for 15 min; supervisor dead; stopping'; exit 5 }
        Start-Sleep -Seconds 60
    }
    Log 'reconstruction complete'
    $steps = @(
      @{ name = 'admission'; v4 = $false; args = @('scripts\prepare_development_1619.py', '--source-root', $root, '--base', $base, '--corpus', $corpus, '--out', $prepared, '--benchmark-id', $genName, '--eval-version', '1', '--forward-boundary-utc', $boundary, '--boundary-document', 'docs\benchmark_1619_v1.md') },
      @{ name = 'coverage';  v4 = $false; args = @('scripts\benchmark_1619_entry.py', '--corpus', $corpus, '--entry', 'report_corpus_coverage', '--', '--qualification', (Join-Path $prepared 'qualification.json'), '--out', $coverage, '--benchmark-id', $genName, '--eval-version', '1', '--purpose', 'patch_development_1619', '--split-assignment', (Join-Path $prepared 'preserved_split_assignment.json'), '--split-metadata', (Join-Path $prepared 'split_metadata.json')) },
      @{ name = 'publish';   v4 = $false; args = @('scripts\benchmark_1619_entry.py', '--corpus', $corpus, '--entry', 'publish_patch_dataset', '--', '--coverage', $coverage, '--name', $genName) },
      @{ name = 'score_R1';  v4 = $true;  args = @('scripts\footwear_runtime.py', '--entry', 'scripts\score_frozen_on_generation.py', '--review', $review18, '--', '--model', 'R1', '--generation', $generation, '--out', $out) },
      @{ name = 'score_M2';  v4 = $false; args = @('scripts\score_frozen_on_generation.py', '--model', 'M2', '--generation', $generation, '--out', $out) },
      @{ name = 'pair';      v4 = $true;  args = @('scripts\footwear_runtime.py', '--entry', 'scripts\pair_frozen_on_generation.py', '--review', $review18, '--', '--generation', $generation, '--out', $out) }
    )
    foreach ($s in $steps) {
        $waited = 0
        while ((AvailableGB) -lt 2.5) {
            if ($waited -eq 0) { Log ('waiting for RAM before ' + $s.name + ' (available ' + ('{0:N2}' -f (AvailableGB)) + ' GB)') }
            Start-Sleep -Seconds 60; $waited += 60
            if ($waited -ge 10800) { Log ('RAM never reached 2.5 GB in 3 h before ' + $s.name + '; stopping'); exit 3 }
        }
        if ($s.v4) { $env:PREFIX_ML_DIR = $gen18 } else { Remove-Item Env:PREFIX_ML_DIR -ErrorAction SilentlyContinue }
        $o = Join-Path $root ('data\logs\benchmark_1619_' + $s.name + '_' + $ts + '.out')
        $e = Join-Path $root ('data\logs\benchmark_1619_' + $s.name + '_' + $ts + '.err')
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
