param([string]$Version='rich-20260929-v1')
$ErrorActionPreference='Stop'
$taskName='Videns Rich State Experiment'
$root=Split-Path -Parent $PSScriptRoot
$runDir=Join-Path $root ('data\runs\' + $Version)
if (Test-Path -LiteralPath $runDir) { throw 'Existing output; refusing overwrite' }
New-Item -ItemType Directory -Path $runDir -Force | Out-Null
$worker=$null
$deadline=(Get-Date).AddMinutes(45)
function Set-State([string]$status,[string]$reason) {
    @{status=$status;reason=$reason;utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runDir 'launcher_status.json') -Encoding UTF8
}
function Stop-OwnWorker {
    if ($null -ne $worker -and -not $worker.HasExited) { & taskkill.exe /PID $worker.Id /T /F | Out-Null }
}
try {
    while ($true) {
        if ((Get-Date) -gt $deadline) { throw '45 minute deadline reached' }
        $busy=@(Get-ScheduledTask | Where-Object { ($_.TaskName -like 'BuildTracker*' -or $_.TaskName -eq 'Videns Winprob W1 Pilot' -or $_.TaskName -eq 'Videns Setback Experiment') -and $_.State -eq 'Running' })
        $game=@(Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue)
        $os=Get-CimInstance Win32_OperatingSystem
        if ($busy.Count -eq 0 -and $game.Count -eq 0 -and $os.FreePhysicalMemory/1MB -ge 2.5) { break }
        Set-State 'waiting_for_resources' ('jobs='+$busy.Count+';freeGiB='+[math]::Round($os.FreePhysicalMemory/1MB,2))
        Start-Sleep -Seconds 30
    }
    $env:OMP_NUM_THREADS='1'
    $env:OPENBLAS_NUM_THREADS='1'
    $env:MKL_NUM_THREADS='1'
    $source=Join-Path $root 'data\runs\w1-pilot-20260929-v3'
    $setback=Join-Path $root 'data\runs\setback-20260929-v1\experiment'
    $script=Join-Path $root 'scripts\rich_experiment.py'
    $arguments=@('-u',('"'+$script+'"'),'--cache',('"'+(Join-Path $source 'cache\states.npz')+'"'),'--source-manifest',('"'+(Join-Path $source 'cache\source_manifest.json')+'"'),'--w1-artifacts',('"'+(Join-Path $source 'experiment')+'"'),'--setback-artifacts',('"'+$setback+'"'),'--output',('"'+(Join-Path $runDir 'experiment')+'"'))
    Set-State 'running' 'matched games; fixed model recipes; geometry and gold allocation'
    $worker=Start-Process -FilePath (Join-Path $root '.venv\Scripts\python.exe') -ArgumentList $arguments -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runDir 'worker.log') -RedirectStandardError (Join-Path $runDir 'worker.err')
    $workerHandle=$worker.Handle
    $worker.PriorityClass='BelowNormal'
    while (-not $worker.HasExited) {
        if ((Get-Date) -gt $deadline) { Stop-OwnWorker; throw 'Deadline reached; own worker stopped' }
        if (@(Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue).Count -gt 0) { Stop-OwnWorker; throw 'Game started; own worker stopped' }
        $os=Get-CimInstance Win32_OperatingSystem
        if ($os.FreePhysicalMemory/1MB -lt 0.6) { Stop-OwnWorker; throw 'Memory pressure; own worker stopped' }
        $busy=@(Get-ScheduledTask | Where-Object { $_.TaskName -like 'BuildTracker*' -and $_.State -eq 'Running' })
        if ($busy.Count -gt 0) { Stop-OwnWorker; throw 'Sibling job started; own worker stopped' }
        Start-Sleep -Seconds 15
        $worker.Refresh()
    }
    $worker.WaitForExit()
    if ($worker.ExitCode -ne 0) { throw ('Worker failed; exit='+$worker.ExitCode) }
    if (-not (Test-Path -LiteralPath (Join-Path $runDir 'experiment\report.json'))) { throw 'No completed report' }
    Set-State 'completed' 'rich state comparison ready'
} catch {
    Set-State 'failed' $_.Exception.Message
    throw
} finally {
    Stop-OwnWorker
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
}
