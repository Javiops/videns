param([Parameter(Mandatory=$true)][string]$Version)
$ErrorActionPreference='Stop'
if($Version -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Unsafe version' }
$root=Split-Path -Parent $PSScriptRoot
$taskName='Videns Purchase Memory v1'
$run=Join-Path $root ('data\runs\'+$Version)
if(Test-Path -LiteralPath $run) { throw 'Existing immutable run directory' }
New-Item -ItemType Directory -Path $run | Out-Null
$worker=$null
$deadline=(Get-Date).AddMinutes(30)
function State([string]$state,[string]$reason) {
    @{state=$state;reason=$reason;utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $run 'launcher_status.json') -Encoding UTF8
}
function Busy {
    $jobs=@(Get-ScheduledTask | Where-Object { ($_.TaskName -like 'BuildTracker*' -or ($_.TaskName -like 'Videns*' -and $_.TaskName -ne $taskName)) -and $_.State -eq 'Running' })
    $processes=@(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match '(capture_live|live_capture|run_capture|reconstruct|extract_frames\.py)' })
    return ($jobs.Count -gt 0 -or $processes.Count -gt 0)
}
function Stop-OwnWorker {
    if($null -ne $worker -and -not $worker.HasExited) { & taskkill.exe /PID $worker.Id /T /F | Out-Null }
}
try {
    while($true) {
        if((Get-Date) -gt $deadline) { throw 'Resource waiting deadline reached' }
        $free=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
        $game=@(Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue).Count
        $busy=Busy
        $gpuFree=@(& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
        $gpuOK=$LASTEXITCODE -eq 0 -and $gpuFree.Count -gt 0 -and [int]$gpuFree[0] -ge 1024
        if(-not $busy -and -not $game -and $free -ge 2.5 -and $gpuOK) { break }
        State 'waiting_for_resources' ('freeGiB='+[math]::Round($free,2)+';game='+$game+';busy='+$busy+';gpuOK='+$gpuOK)
        Start-Sleep -Seconds 15
    }
    $env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1'
    $output=Join-Path $run 'experiment'
    $script=Join-Path $root 'scripts\probe_purchase_memory_v1.py'
    $workerArgs=@('-u',('"'+$script+'"'),'--output',('"'+$output+'"'),'--deadline-seconds','1200')
    State 'running' ('output='+$output)
    $worker=Start-Process -FilePath (Join-Path $root '.venv\Scripts\python.exe') -ArgumentList $workerArgs -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $run 'worker.log') -RedirectStandardError (Join-Path $run 'worker.err')
    $workerHandle=$worker.Handle
    $worker.PriorityClass='BelowNormal'
    @{worker_pid=$worker.Id;created_utc=(Get-Date).ToUniversalTime().ToString('o');version=$Version;task=$taskName} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $run 'process.json') -Encoding UTF8
    while(-not $worker.HasExited) {
        if((Get-Date) -gt $deadline) { Stop-OwnWorker; throw 'Job wall deadline reached' }
        if(@(Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue).Count -or (Busy)) { Stop-OwnWorker; throw 'Game or source workload started; own worker stopped' }
        if((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB -lt .75) { Stop-OwnWorker; throw 'Live host memory floor reached' }
        Start-Sleep -Seconds 5
        $worker.Refresh()
    }
    $worker.WaitForExit()
    if($worker.ExitCode -ne 0) { throw ('Worker exited '+$worker.ExitCode+'; see worker.err') }
    $report=Join-Path $output 'report.json'
    if(-not(Test-Path -LiteralPath $report)) { throw 'No completed report' }
    if((Get-Content -Raw -LiteralPath $report | ConvertFrom-Json).state -ne 'complete') { throw 'Report not complete' }
    State 'completed' ('artifact='+$report)
} catch { State 'failed' $_.Exception.Message; throw }
finally { Stop-OwnWorker; Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue }
