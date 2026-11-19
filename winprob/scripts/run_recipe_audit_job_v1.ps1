param([Parameter(Mandatory=$true)][string]$Version)
$ErrorActionPreference='Stop'
if($Version -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Unsafe version' }
$root=Split-Path -Parent $PSScriptRoot
$source=Join-Path $root 'data\runs\recipe-transition-fit-20261001-v1\experiment'
$taskName='Videns Recipe Audit v1'
$run=Join-Path $root ('data\runs\'+$Version)
if(Test-Path -LiteralPath $run) { throw 'Existing immutable audit run' }
if(-not(Test-Path -LiteralPath (Join-Path $source 'report.json'))) { throw 'Complete fit report required' }
New-Item -ItemType Directory -Path $run | Out-Null
$worker=$null;$resourceDeadline=(Get-Date).AddMinutes(15)
function State([string]$state,[string]$reason) {
    @{state=$state;reason=$reason;utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $run 'launcher_status.json') -Encoding UTF8
}
function Busy {
    $jobs=@(Get-ScheduledTask | Where-Object { ($_.TaskName -like 'BuildTracker*' -or ($_.TaskName -like 'Videns*' -and $_.TaskName -ne $taskName)) -and $_.State -eq 'Running' })
    $capture=@(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match '(capture_live|live_capture|run_capture|reconstruct|extract_frames\.py)' })
    return ($jobs.Count -gt 0 -or $capture.Count -gt 0)
}
function Stop-OwnWorker {
    if($null -ne $worker -and -not $worker.HasExited) { & taskkill.exe /PID $worker.Id /T /F | Out-Null }
}
try {
    while($true) {
        if((Get-Date) -gt $resourceDeadline) { throw 'Resource wait deadline reached' }
        $free=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
        if(-not(Busy) -and -not @(Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue).Count -and $free -ge 3.25) { break }
        State 'waiting_for_resources' ('freeGiB='+[math]::Round($free,2));Start-Sleep -Seconds 15
    }
    $env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1'
    $output=Join-Path $run 'audit';$script=Join-Path $root 'scripts\audit_recipe_transition_v1.py'
    $workerArgs=@('-u',('"'+$script+'"'),'--directory',('"'+$source+'"'),'--output',('"'+$output+'"'))
    $deadline=(Get-Date).AddMinutes(31)
    State 'running' ('source='+$source)
    $worker=Start-Process -FilePath (Join-Path $root '.venv\Scripts\python.exe') -ArgumentList $workerArgs -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $run 'worker.log') -RedirectStandardError (Join-Path $run 'worker.err')
    $workerHandle=$worker.Handle;$worker.PriorityClass='BelowNormal'
    @{worker_pid=$worker.Id;utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $run 'process.json') -Encoding UTF8
    while(-not $worker.HasExited) {
        if((Get-Date) -gt $deadline) { Stop-OwnWorker;throw 'Audit worker deadline reached' }
        if(@(Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue).Count -or (Busy)) { Stop-OwnWorker;throw 'Game or sibling workload started' }
        if((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB -lt .75) { Stop-OwnWorker;throw 'Host RAM floor reached' }
        Start-Sleep -Seconds 5;$worker.Refresh()
    }
    $worker.WaitForExit()
    if($worker.ExitCode -ne 0) { throw ('Audit worker exited '+$worker.ExitCode) }
    $report=Join-Path $output 'report.json'
    if(-not(Test-Path -LiteralPath $report)) { throw 'Audit report missing' }
    $result=Get-Content -Raw -LiteralPath $report | ConvertFrom-Json
    if($result.state -ne 'complete' -or -not $result.passed -or $result.correction_queries_replayed -ne 8700 -or $result.test_payload_reads -or $result.confirmation_payload_reads) { throw 'Audit completion/boundary failed' }
    State 'completed' ('report='+$report)
} catch { State 'failed' $_.Exception.Message;throw }
finally { Stop-OwnWorker;Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue }
