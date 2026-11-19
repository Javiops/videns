param(
    [Parameter(Mandatory=$true)][ValidateSet('preflight','train')][string]$Mode,
    [Parameter(Mandatory=$true)][string]$Version
)
$ErrorActionPreference='Stop'
if($Version -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Unsafe version' }
$root=Split-Path -Parent $PSScriptRoot
$source=Join-Path (Split-Path -Parent $root) 'build_tracker'
$taskName='Videns IQL '+$Mode
$run=Join-Path $root ('data\runs\'+$Version)
if(Test-Path -LiteralPath $run) { throw 'Existing immutable IQL job directory' }
New-Item -ItemType Directory -Path $run | Out-Null
$worker=$null
$deadline=(Get-Date).AddMinutes($(if($Mode -eq 'train') {95} else {10}))
function State([string]$state,[string]$reason) {
    @{state=$state;reason=$reason;mode=$Mode;utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $run 'launcher_status.json') -Encoding UTF8
}
function Stop-OwnWorker {
    if($null -ne $worker -and -not $worker.HasExited) { & taskkill.exe /PID $worker.Id /T /F | Out-Null }
}
try {
    while($true) {
        if((Get-Date) -gt $deadline) { throw 'IQL launch deadline reached while waiting' }
        $busy=@(Get-ScheduledTask | Where-Object { ($_.TaskName -like 'BuildTracker*' -or ($_.TaskName -like 'Videns*' -and $_.TaskName -ne $taskName)) -and $_.State -eq 'Running' })
        $game=@(Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue)
        $os=Get-CimInstance Win32_OperatingSystem
        if($busy.Count -eq 0 -and $game.Count -eq 0 -and $os.FreePhysicalMemory/1MB -ge 2.5) { break }
        State 'waiting_for_resources' ('jobs='+$busy.Count+';game='+$game.Count+';freeGiB='+[math]::Round($os.FreePhysicalMemory/1MB,2))
        Start-Sleep -Seconds 20
    }
    $env:OMP_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
    $output=Join-Path $run 'experiment'
    $workerArguments=@('-u',('"'+(Join-Path $root 'scripts\train_iql.py')+'"'),'--dataset',('"'+(Join-Path $root 'data\runs\iql-cache-20260930-v1\cache')+'"'),'--catalogue',('"'+(Join-Path $source 'data\exports\winprob-action-catalogue\full-action-catalogue-20260930-v1\catalogue.json')+'"'),'--reference',('"'+(Join-Path $source 'data\exports\winprob-m2-pairs\m2-preference-20260930-v1')+'"'),'--static-export',('"'+(Join-Path $root 'artifacts\boots-static\20260930-v1\boots_static.json')+'"'),'--output',('"'+$output+'"'),'--device','cuda:0','--deadline-seconds','5400')
    if($Mode -eq 'train') { $workerArguments+='--execute' }
    State 'running' ('CUDA '+$Mode+'; output='+$output)
    $worker=Start-Process -FilePath (Join-Path $root '.venv\Scripts\python.exe') -ArgumentList $workerArguments -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $run 'worker.log') -RedirectStandardError (Join-Path $run 'worker.err')
    $workerHandle=$worker.Handle
    $worker.PriorityClass='BelowNormal'
    while(-not $worker.HasExited) {
        if((Get-Date) -gt $deadline) { Stop-OwnWorker; throw 'IQL job wall deadline reached' }
        if(@(Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue).Count) { Stop-OwnWorker; throw 'Game started; own IQL worker stopped' }
        $busy=@(Get-ScheduledTask | Where-Object { $_.TaskName -like 'BuildTracker*' -and $_.State -eq 'Running' })
        if($busy.Count) { Stop-OwnWorker; throw 'Sibling workload started; own IQL worker stopped' }
        if((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB -lt .6) { Stop-OwnWorker; throw 'Host memory pressure; own IQL worker stopped' }
        Start-Sleep -Seconds 5; $worker.Refresh()
    }
    $worker.WaitForExit()
    if($worker.ExitCode -ne 0) { throw ('IQL worker exited '+$worker.ExitCode+'; see worker.err') }
    $status=Get-Content -LiteralPath (Join-Path $output 'status.json') -Raw | ConvertFrom-Json
    $expected=if($Mode -eq 'train') {'completed'} else {'preflight_completed_no_real_training'}
    if($status.stage -ne $expected) { throw 'IQL completion marker differs from requested mode' }
    State 'completed' ('experiment='+$output)
} catch { State 'failed' $_.Exception.Message; throw }
finally { Stop-OwnWorker; Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue }
