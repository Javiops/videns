param(
    [string]$Version='boots-support-20260930-v1',
    [ValidateSet('boots_support.py','undo_pairs_support.py')][string]$WorkerScript='boots_support.py',
    [ValidateSet('Videns Boots Support','Videns Undo Pairs')][string]$TaskName='Videns Boots Support'
)
$ErrorActionPreference='Stop'
if ($Version -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Unsafe experiment version' }
$root=Split-Path -Parent $PSScriptRoot
$runDir=Join-Path $root ('data\runs\'+$Version)
if (Test-Path -LiteralPath $runDir) { throw 'Existing output; refusing overwrite' }
New-Item -ItemType Directory -Path $runDir -Force | Out-Null
$worker=$null
$deadline=(Get-Date).AddMinutes(25)
function Set-State([string]$status,[string]$reason) {
    @{status=$status;reason=$reason;utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runDir 'launcher_status.json') -Encoding UTF8
}
function Stop-OwnWorker {
    if ($null -ne $worker -and -not $worker.HasExited) { & taskkill.exe /PID $worker.Id /T /F | Out-Null }
}
try {
    $busy=@(Get-ScheduledTask | Where-Object { ($_.TaskName -like 'BuildTracker*' -or ($_.TaskName -like 'Videns*' -and $_.TaskName -ne $taskName)) -and $_.State -eq 'Running' })
    if ($busy.Count -gt 0) { throw 'Another job is running' }
    if (@(Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue).Count -gt 0) { throw 'Game is running' }
    $env:OMP_NUM_THREADS='1'
    $env:OPENBLAS_NUM_THREADS='1'
    $env:MKL_NUM_THREADS='1'
    $source=Join-Path $root 'data\runs\w1-pilot-20260929-v3\cache'
    $script=Join-Path (Join-Path $root 'scripts') $WorkerScript
    $static=Join-Path $root 'artifacts\boots-static\20260930-v1\boots_static.json'
    $arguments=@('-u',('"'+$script+'"'),'--cache',('"'+(Join-Path $source 'states.npz')+'"'),'--source-manifest',('"'+(Join-Path $source 'source_manifest.json')+'"'),'--static-export',('"'+$static+'"'),'--output',('"'+(Join-Path $runDir 'experiment')+'"'),'--deadline-seconds','1400')
    Set-State 'running' ('outcome-blind shop support census: '+$WorkerScript)
    $worker=Start-Process -FilePath (Join-Path $root '.venv\Scripts\python.exe') -ArgumentList $arguments -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runDir 'worker.log') -RedirectStandardError (Join-Path $runDir 'worker.err')
    $workerHandle=$worker.Handle
    $worker.PriorityClass='BelowNormal'
    while (-not $worker.HasExited) {
        if ((Get-Date) -gt $deadline) { Stop-OwnWorker; throw 'Deadline reached; own worker stopped' }
        if (@(Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue).Count -gt 0) { Stop-OwnWorker; throw 'Game started; own worker stopped' }
        $busy=@(Get-ScheduledTask | Where-Object { $_.TaskName -like 'BuildTracker*' -and $_.State -eq 'Running' })
        if ($busy.Count -gt 0) { Stop-OwnWorker; throw 'Sibling job started; own worker stopped' }
        $os=Get-CimInstance Win32_OperatingSystem
        if ($os.FreePhysicalMemory/1MB -lt 0.6) { Stop-OwnWorker; throw 'Memory pressure; own worker stopped' }
        Start-Sleep -Seconds 15
        $worker.Refresh()
    }
    $worker.WaitForExit()
    if ($worker.ExitCode -ne 0) { throw ('Worker failed; exit='+$worker.ExitCode) }
    if (-not (Test-Path -LiteralPath (Join-Path $runDir 'experiment\report.json'))) { throw 'No completed report' }
    Set-State 'completed' 'support measured; no outcome comparison made'
} catch {
    Set-State 'failed' $_.Exception.Message
    throw
} finally {
    Stop-OwnWorker
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
}
