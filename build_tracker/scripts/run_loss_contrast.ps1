# After the mixture VAL frees the GPU: (1) A-vs-M latency benchmark (short,
# non-fatal), then (2) the ward-weight loss contrast. Self-unregistering.
#
# Python stdout/stderr go to PLAIN files via Start-Process -Redirect*, NOT
# through Start-Transcript: the transcript buffers a native command's output
# until it exits, which left the mixture VAL run blind for two hours. These
# files are written as Python flushes, so progress is readable live.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$taskName = 'BuildTracker Loss Contrast'
$ts = Get-Date -Format yyyyMMdd_HHmmss
New-Item -ItemType Directory -Force (Join-Path $root 'data\logs') | Out-Null
$transcript = Join-Path $root ('data\logs\loss_contrast_' + $ts + '.log')
Start-Transcript -Path $transcript

function Invoke-Entry {
    # MUST return ONLY the exit code.
    #
    # BUG FIXED 2026-09-22: this used Write-Output for the progress message.
    # In PowerShell that appends the string to the FUNCTION'S OUTPUT STREAM,
    # so the caller received Object[] = (message, code). `$code -ne 0` then
    # filters the array, returns the non-zero message, and the `if` fires:
    # a run that exited 0 was announced as failed. It happened to the latency
    # benchmark that night ("AVISO: latency bench fallo (exit --> ... 0)")
    # while the benchmark had in fact completed and written its report.
    # Messages go to the host; only [int] is returned.
    param([string]$Entry, [string]$Tag)
    $out = Join-Path $root ('data\logs\' + $Tag + '_' + $ts + '.out')
    $err = Join-Path $root ('data\logs\' + $Tag + '_' + $ts + '.err')
    Write-Host ("--> $Entry ; progreso en vivo: $out")
    $p = Start-Process -FilePath (Join-Path $root '.venv\Scripts\python.exe') `
        -ArgumentList @('scripts\footwear_runtime.py', '--entry', $Entry,
                        '--review', $script:review) `
        -WorkingDirectory $root -NoNewWindow -PassThru -Wait `
        -RedirectStandardOutput $out -RedirectStandardError $err
    return [int]$p.ExitCode
}

try {
    # one training process at a time on this machine
    while ($true) {
        $other = Get-ScheduledTask -TaskName 'BuildTracker Mixture VAL' -ErrorAction SilentlyContinue
        if (-not $other -or $other.State -ne 'Running') { break }
        Write-Host ('esperando a BuildTracker Mixture VAL... ' + (Get-Date -Format HH:mm:ss))
        Start-Sleep -Seconds 60
    }
    $env:PREFIX_ML_DIR = Join-Path $root 'data\dataset_generations\16.18-reconstructible-development-v3'
    $script:review = Join-Path $root 'data\corpora\16.18-reconstructible-v3\static_review\static_review.json'

    Write-Host '=== A-vs-M latency benchmark (fixed FIT sample) ==='
    [int]$code = Invoke-Entry -Entry 'scripts\basket_latency_bench.py' -Tag 'latency_bench'
    if ($code -ne 0) { Write-Host "AVISO: latency bench fallo (exit $code); sigo con el contraste" }

    Write-Host '=== ward-weight loss contrast (full, 2 brazos x 8 epocas) ==='
    [int]$code = Invoke-Entry -Entry 'scripts\basket_loss_contrast.py' -Tag 'loss_contrast'
    if ($code -ne 0) { throw "loss contrast failed (exit $code)" }
    Write-Host 'LOSS CONTRAST COMPLETE'
}
finally {
    try { Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue } catch {}
    Stop-Transcript
}
