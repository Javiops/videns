$ErrorActionPreference='Stop'
$taskName='Videns Rich Ablation'
$root=Split-Path -Parent $PSScriptRoot
$runDir=Join-Path $root 'data\runs\rich-ablation-20260929-v1'
if (Test-Path -LiteralPath $runDir) { throw 'Existing output; refusing overwrite' }
New-Item -ItemType Directory -Path $runDir -Force | Out-Null
try {
    $busy=@(Get-ScheduledTask | Where-Object { ($_.TaskName -like 'BuildTracker*' -or $_.TaskName -eq 'Videns Rich State Experiment') -and $_.State -eq 'Running' })
    if ($busy.Count -gt 0) { throw 'Another experiment is running' }
    if (@(Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue).Count -gt 0) { throw 'Game is running' }
    [System.Diagnostics.Process]::GetCurrentProcess().PriorityClass='BelowNormal'
    $env:OMP_NUM_THREADS='1'
    $env:OPENBLAS_NUM_THREADS='1'
    $env:MKL_NUM_THREADS='1'
    & (Join-Path $root '.venv\Scripts\python.exe') -u (Join-Path $root 'scripts\ablate_rich.py') --rich-run (Join-Path $root 'data\runs\rich-20260929-v1\experiment') --output (Join-Path $runDir 'experiment') --deadline-seconds 1100 1> (Join-Path $runDir 'worker.log') 2> (Join-Path $runDir 'worker.err')
    if ($LASTEXITCODE -ne 0) { throw ('Worker failed; exit='+$LASTEXITCODE) }
} finally {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
}
