param(
    [Parameter(Mandatory=$true)][string]$Version,
    [string]$ProducerVersion='future-goal-probe-20260930-v1'
)
$ErrorActionPreference='Stop'
if($Version -notmatch '^[A-Za-z0-9_.-]+$' -or $ProducerVersion -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Unsafe version' }
$root=Split-Path -Parent $PSScriptRoot
$taskName='Videns Future Goal audit'
$run=Join-Path $root ('data\runs\'+$Version)
$producer=Join-Path $root ('data\runs\'+$ProducerVersion)
$experiment=Join-Path $producer 'experiment'
if(Test-Path -LiteralPath $run) { throw 'Existing immutable audit run directory' }
New-Item -ItemType Directory -Path $run | Out-Null
$deadline=(Get-Date).AddMinutes(100)
$worker=$null
$marks=@()
function State([string]$state,[string]$reason) {
    @{state=$state;reason=$reason;producer=$ProducerVersion;utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $run 'launcher_status.json') -Encoding UTF8
}
function Stop-OwnWorker {
    if($null -ne $worker -and -not $worker.HasExited) { & taskkill.exe /PID $worker.Id /T /F | Out-Null }
}
try {
    while($true) {
        if((Get-Date) -gt $deadline) { throw 'Audit dependency/resource waiting deadline reached' }
        $producerLive=@(Get-CimInstance Win32_Process | Where-Object {
            $_.Name -eq 'python.exe' -and $_.CommandLine -match 'train_future_goal.py' -and $_.CommandLine -match [regex]::Escape($experiment)
        })
        $trainingTask=Get-ScheduledTask -TaskName 'Videns Future Goal train' -ErrorAction SilentlyContinue
        $producerLauncher=Get-Content -LiteralPath (Join-Path $producer 'launcher_status.json') -Raw | ConvertFrom-Json
        if($producerLive.Count -gt 0 -or ($null -ne $trainingTask -and $trainingTask.State -eq 'Running')) {
            State 'waiting_for_producer' ('livePython='+$producerLive.Count+'; producerState='+$producerLauncher.state)
            Start-Sleep -Seconds 20
            continue
        }
        if($producerLauncher.state -ne 'completed') { throw ('Producer is terminal without a completed experiment: '+$producerLauncher.state+'; '+$producerLauncher.reason) }
        if(-not(Test-Path -LiteralPath (Join-Path $experiment 'report.json'))) { throw 'Completed producer has no final report' }
        $finalStatus=Get-Content -LiteralPath (Join-Path $experiment 'status.json') -Raw | ConvertFrom-Json
        if($finalStatus.stage -ne 'complete') { throw 'Producer has not published terminal completion' }
        $busy=@(Get-ScheduledTask | Where-Object { ($_.TaskName -like 'BuildTracker*' -or ($_.TaskName -like 'Videns*' -and $_.TaskName -ne $taskName)) -and $_.State -eq 'Running' })
        $game=@(Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue)
        $os=Get-CimInstance Win32_OperatingSystem
        if($busy.Count -eq 0 -and $game.Count -eq 0 -and $os.FreePhysicalMemory/1MB -ge 3.25) { break }
        State 'waiting_for_resources' ('jobs='+$busy.Count+';game='+$game.Count+';freeGiB='+[math]::Round($os.FreePhysicalMemory/1MB,2))
        Start-Sleep -Seconds 20
    }
    $env:OMP_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:MKL_NUM_THREADS='1'
    $output=Join-Path $run 'audit.json'
    $script=Join-Path $root 'data\reports\future_goal_result_audit_v1.py'
    $auditDeadline=(Get-Date).AddMinutes(6)
    State 'running' ('terminal producer verified; output='+$output)
    $worker=Start-Process -FilePath (Join-Path $root '.venv\Scripts\python.exe') -ArgumentList @('-u',('"'+$script+'"'),'--run',('"'+$experiment+'"'),'--output',('"'+$output+'"'),'--completed-run-confirmed') -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $run 'worker.log') -RedirectStandardError (Join-Path $run 'worker.err')
    $workerHandle=$worker.Handle
    $worker.PriorityClass='BelowNormal'
    while(-not $worker.HasExited) {
        if((Get-Date) -gt $auditDeadline -or (Get-Date) -gt $deadline) { Stop-OwnWorker;throw 'Audit worker wall deadline reached' }
        if(@(Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue).Count) { Stop-OwnWorker;throw 'Game started; own audit stopped' }
        $busy=@(Get-ScheduledTask | Where-Object { $_.TaskName -like 'BuildTracker*' -and $_.State -eq 'Running' })
        if($busy.Count) { Stop-OwnWorker;throw 'Sibling benchmark started; own audit stopped' }
        $os=Get-CimInstance Win32_OperatingSystem
        $ownIds=@($worker.Id)+@(Get-CimInstance Win32_Process -Filter ('ParentProcessId = '+$worker.Id) | Select-Object -ExpandProperty ProcessId)
        $ownWorkers=@(Get-Process -Id $ownIds -ErrorAction SilentlyContinue)
        $marks+=@{utc=(Get-Date).ToUniversalTime().ToString('o');available_host_bytes=[long]$os.FreePhysicalMemory*1024;own_working_bytes=($ownWorkers|Measure-Object -Property WorkingSet64 -Sum).Sum;own_private_bytes=($ownWorkers|Measure-Object -Property PrivateMemorySize64 -Sum).Sum}
        @{start_gate_GiB=3.25;stop_gate_GiB=.75;marks=$marks} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $run 'resource_marks.json') -Encoding UTF8
        if($os.FreePhysicalMemory/1MB -lt .75) { Stop-OwnWorker;throw 'Host memory pressure; own audit stopped' }
        Start-Sleep -Seconds 3
        $worker.Refresh()
    }
    $worker.WaitForExit()
    if($worker.ExitCode -ne 0) { throw ('Audit exited '+$worker.ExitCode+'; see worker.err') }
    if(-not(Test-Path -LiteralPath $output)) { throw 'Audit did not publish a final artifact' }
    $result=Get-Content -LiteralPath $output -Raw | ConvertFrom-Json
    if($result.state -ne 'passed') { throw 'Final audit did not pass' }
    State 'completed' ('artifact='+$output)
} catch { State 'failed' $_.Exception.Message;throw }
finally { Stop-OwnWorker;Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue }
