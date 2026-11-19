param(
    [Parameter(Mandatory=$true)][ValidateSet('prepare','preflight','train')][string]$Mode,
    [Parameter(Mandatory=$true)][string]$Version,
    [string]$HistoryVersion='history-cache-20260930-v1'
)
$ErrorActionPreference='Stop'
if($Version -notmatch '^[A-Za-z0-9_.-]+$' -or $HistoryVersion -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Unsafe version' }
$root=Split-Path -Parent $PSScriptRoot
$taskName='Videns History '+$Mode
$run=Join-Path $root ('data\runs\'+$Version)
if(Test-Path -LiteralPath $run) { throw 'Existing immutable history job directory' }
New-Item -ItemType Directory -Path $run | Out-Null
$worker=$null
$minimumFreeGiB=if($Mode -eq 'prepare') {1.5} else {3.25}
$deadline=(Get-Date).AddMinutes($(if($Mode -eq 'prepare') {25} elseif($Mode -eq 'preflight') {15} else {60}))
$marks=@()
function State([string]$state,[string]$reason) {
    @{state=$state;reason=$reason;mode=$Mode;utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $run 'launcher_status.json') -Encoding UTF8
}
function Stop-OwnWorker {
    if($null -ne $worker -and -not $worker.HasExited) { & taskkill.exe /PID $worker.Id /T /F | Out-Null }
}
try {
    while($true) {
        if((Get-Date) -gt $deadline) { throw 'History launch deadline reached while waiting' }
        $busy=@(Get-ScheduledTask | Where-Object { ($_.TaskName -like 'BuildTracker*' -or ($_.TaskName -like 'Videns*' -and $_.TaskName -ne $taskName)) -and $_.State -eq 'Running' })
        $game=@(Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue)
        $os=Get-CimInstance Win32_OperatingSystem
        $gpuReady=$true
        if($Mode -ne 'prepare') {
            $gpuFree=@(& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
            $gpuReady=$LASTEXITCODE -eq 0 -and $gpuFree.Count -gt 0 -and [int]$gpuFree[0] -ge 2048
        }
        if($busy.Count -eq 0 -and $game.Count -eq 0 -and $os.FreePhysicalMemory/1MB -ge $minimumFreeGiB -and $gpuReady) { break }
        State 'waiting_for_resources' ('jobs='+$busy.Count+';game='+$game.Count+';freeGiB='+[math]::Round($os.FreePhysicalMemory/1MB,2)+';gpuReady='+$gpuReady)
        Start-Sleep -Seconds 20
    }
    $env:OMP_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
    if($Mode -eq 'prepare') {
        $output=Join-Path $run 'cache'
        $script=Join-Path $root 'scripts\prepare_history_probe.py'
        $workerArguments=@('-u',('"'+$script+'"'),'--base-manifest',('"'+(Join-Path $root 'data\runs\iql-cache-20260930-v1\cache\manifest.json')+'"'),'--protocol',('"'+(Join-Path $root 'docs\history_probe_protocol_v1.md')+'"'),'--output',('"'+$output+'"'),'--deadline-seconds','900')
        $completion=Join-Path $output 'manifest.json'
    } else {
        $output=Join-Path $run 'experiment'
        $script=Join-Path $root 'scripts\train_history_probe_v2.py'
        $history=Join-Path $root ('data\runs\'+$HistoryVersion+'\cache\manifest.json')
        $workerArguments=@('-u',('"'+$script+'"'),'--history-manifest',('"'+$history+'"'),'--output',('"'+$output+'"'),'--device','cuda:0','--deadline-seconds','2700')
        if($Mode -eq 'train') { $workerArguments+='--execute' }
        $completion=Join-Path $output $(if($Mode -eq 'train') {'report.json'} else {'preflight.json'})
    }
    State 'running' ($Mode+'; output='+$output)
    $worker=Start-Process -FilePath (Join-Path $root '.venv\Scripts\python.exe') -ArgumentList $workerArguments -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $run 'worker.log') -RedirectStandardError (Join-Path $run 'worker.err')
    $workerHandle=$worker.Handle
    $worker.PriorityClass='BelowNormal'
    while(-not $worker.HasExited) {
        if((Get-Date) -gt $deadline) { Stop-OwnWorker; throw 'History job wall deadline reached' }
        if(@(Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue).Count) { Stop-OwnWorker; throw 'Game started; own history worker stopped' }
        $busy=@(Get-ScheduledTask | Where-Object { $_.TaskName -like 'BuildTracker*' -and $_.State -eq 'Running' })
        if($busy.Count) { Stop-OwnWorker; throw 'Sibling workload started; own history worker stopped' }
        $os=Get-CimInstance Win32_OperatingSystem
        $ownIds=@($worker.Id)+@(Get-CimInstance Win32_Process -Filter ('ParentProcessId = '+$worker.Id) | Select-Object -ExpandProperty ProcessId)
        $ownWorkers=@(Get-Process -Id $ownIds -ErrorAction SilentlyContinue)
        $marks+=@{utc=(Get-Date).ToUniversalTime().ToString('o');available_host_bytes=[long]$os.FreePhysicalMemory*1024;own_worker_working_bytes=($ownWorkers | Measure-Object -Property WorkingSet64 -Sum).Sum;own_worker_private_bytes=($ownWorkers | Measure-Object -Property PrivateMemorySize64 -Sum).Sum}
        @{start_gate_GiB=$minimumFreeGiB;stop_gate_GiB=.75;marks=$marks} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $run 'resource_marks.json') -Encoding UTF8
        if($os.FreePhysicalMemory/1MB -lt .75) { Stop-OwnWorker; throw 'Host memory pressure; own history worker stopped' }
        Start-Sleep -Seconds 5
        $worker.Refresh()
    }
    $worker.WaitForExit()
    if($worker.ExitCode -ne 0) { throw ('History worker exited '+$worker.ExitCode+'; see worker.err') }
    if(-not(Test-Path -LiteralPath $completion)) { throw 'No completed history artifact' }
    State 'completed' ('artifact='+$completion)
} catch { State 'failed' $_.Exception.Message; throw }
finally { Stop-OwnWorker; Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue }
