param(
    [Parameter(Mandatory=$true)][ValidateSet('train','audit')][string]$Mode,
    [Parameter(Mandatory=$true)][string]$Version,
    [string]$ExperimentVersion='wp-players-fit-20260930-v1'
)
$ErrorActionPreference='Stop'
if($Version -notmatch '^[A-Za-z0-9_.-]+$' -or $ExperimentVersion -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Unsafe version' }
$root=Split-Path -Parent $PSScriptRoot
$taskName='Videns Player WP '+$Mode
$run=Join-Path $root ('data\runs\'+$Version)
if(Test-Path -LiteralPath $run) { throw 'Existing immutable player-WP run directory' }
New-Item -ItemType Directory -Path $run | Out-Null
$worker=$null
$deadline=(Get-Date).AddMinutes($(if($Mode -eq 'train') {120} else {60}))
$marks=@()
function State([string]$state,[string]$reason) {
    @{state=$state;reason=$reason;mode=$Mode;utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $run 'launcher_status.json') -Encoding UTF8
}
function Stop-OwnWorker {
    if($null -ne $worker -and -not $worker.HasExited) { & taskkill.exe /PID $worker.Id /T /F | Out-Null }
}
try {
    while($true) {
        if((Get-Date) -gt $deadline) { throw 'Player-WP launch deadline reached while waiting' }
        $busy=@(Get-ScheduledTask | Where-Object { ($_.TaskName -like 'BuildTracker*' -or ($_.TaskName -like 'Videns*' -and $_.TaskName -ne $taskName)) -and $_.State -eq 'Running' })
        $game=@(Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue)
        $os=Get-CimInstance Win32_OperatingSystem
        $gpuStats=@(& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
        $gpuReady=$LASTEXITCODE -eq 0 -and $gpuStats.Count -gt 0 -and [int]$gpuStats[0] -ge 4096
        if($busy.Count -eq 0 -and $game.Count -eq 0 -and $os.FreePhysicalMemory/1MB -ge 2.5 -and $gpuReady) { break }
        State 'waiting_for_resources' ('jobs='+$busy.Count+';game='+$game.Count+';freeGiB='+[math]::Round($os.FreePhysicalMemory/1MB,2)+';gpuReady='+$gpuReady)
        Start-Sleep -Seconds 20
    }
    $env:OMP_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
    if($Mode -eq 'train') {
        $output=Join-Path $run 'experiment'
        $script=Join-Path $root 'scripts\train_wp_players_v1.py'
        $cache=Join-Path $root 'data\runs\wp-players-cache-20260930-v1\cache\manifest.json'
        $workerArguments=@('-u',('"'+$script+'"'),'--cache-manifest',('"'+$cache+'"'),'--output',('"'+$output+'"'),'--device','cuda:0')
        $completion=Join-Path $output 'report.json'
    } else {
        $output=Join-Path $run 'audit'
        $script=Join-Path $root 'scripts\audit_wp_players_result_v2.py'
        $experiment=Join-Path $root ('data\runs\'+$ExperimentVersion+'\experiment')
        $workerArguments=@('-u',('"'+$script+'"'),'--experiment',('"'+$experiment+'"'),'--output',('"'+$output+'"'),'--device','cuda:0')
        $completion=Join-Path $output 'report.json'
    }
    if(-not (Test-Path -LiteralPath $script)) { throw 'Required stage script does not exist' }
    State 'running' ($Mode+'; output='+$output)
    $worker=Start-Process -FilePath (Join-Path $root '.venv\Scripts\python.exe') -ArgumentList $workerArguments -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $run 'worker.log') -RedirectStandardError (Join-Path $run 'worker.err')
    $workerHandle=$worker.Handle;$worker.PriorityClass='BelowNormal'
    while(-not $worker.HasExited) {
        if((Get-Date) -gt $deadline) { Stop-OwnWorker;throw 'Player-WP outer deadline reached' }
        if(@(Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue).Count) { Stop-OwnWorker;throw 'Game started; own player-WP worker stopped' }
        $busy=@(Get-ScheduledTask | Where-Object { $_.TaskName -like 'BuildTracker*' -and $_.State -eq 'Running' })
        if($busy.Count) { Stop-OwnWorker;throw 'Sibling workload started; own player-WP worker stopped' }
        $os=Get-CimInstance Win32_OperatingSystem
        $ownIds=@($worker.Id)+@(Get-CimInstance Win32_Process -Filter ('ParentProcessId = '+$worker.Id) | Select-Object -ExpandProperty ProcessId)
        $ownWorkers=@(Get-Process -Id $ownIds -ErrorAction SilentlyContinue)
        $marks+=@{utc=(Get-Date).ToUniversalTime().ToString('o');available_host_bytes=[long]$os.FreePhysicalMemory*1024;own_worker_working_bytes=($ownWorkers | Measure-Object -Property WorkingSet64 -Sum).Sum;own_worker_private_bytes=($ownWorkers | Measure-Object -Property PrivateMemorySize64 -Sum).Sum}
        @{start_gate_GiB=2.5;stop_gate_GiB=.75;marks=$marks} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $run 'resource_marks.json') -Encoding UTF8
        if($os.FreePhysicalMemory/1MB -lt .75) { Stop-OwnWorker;throw 'Memory pressure; own player-WP worker stopped' }
        Start-Sleep -Seconds 10;$worker.Refresh()
    }
    $worker.WaitForExit()
    if($worker.ExitCode -ne 0) { throw ('Player-WP worker failed: '+$worker.ExitCode+'; see worker.err') }
    if(-not (Test-Path -LiteralPath $completion)) { throw 'No completed stage artifact' }
    State 'completed' ('artifact='+$completion)
} catch {
    State 'failed' $_.Exception.Message
    throw
} finally {
    Stop-OwnWorker
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
}
