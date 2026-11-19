param([Parameter(Mandatory=$true)][string]$Version)
$ErrorActionPreference='Stop'
if($Version -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Unsafe version' }
$root=Split-Path -Parent $PSScriptRoot
$sourceRoot=Join-Path (Split-Path -Parent $root) 'build_tracker'
$taskName='Videns Route Prior Release v1'
$run=Join-Path $root ('data\runs\'+$Version)
$output=Join-Path $sourceRoot ('data\exports\winprob-route-prior\'+$Version)
if((Test-Path -LiteralPath $run) -or (Test-Path -LiteralPath $output)) { throw 'Existing immutable recipe native M2 export' }
New-Item -ItemType Directory -Path $run | Out-Null
$worker=$null;$deadline=(Get-Date).AddMinutes(210);$resourceDeadline=(Get-Date).AddMinutes(30)
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
        if((Get-Date) -gt $resourceDeadline) { throw 'Resource waiting deadline reached' }
        $free=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
        $gpuFree=@(& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
        $gpuOK=$LASTEXITCODE -eq 0 -and $gpuFree.Count -gt 0 -and [int]$gpuFree[0] -ge 3072
        if(-not (Busy) -and -not @(Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue).Count -and $free -ge 2.5 -and $gpuOK) { break }
        State 'waiting_for_resources' ('freeGiB='+[math]::Round($free,2));Start-Sleep -Seconds 15
    }
    $env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1'
    $script=Join-Path $sourceRoot 'scripts\export_route_prior_v1.py'
    $workerArgs=@('-u',('"'+$script+'"'),'--output',('"'+$output+'"'))
    State 'running' ('output='+$output)
    $worker=Start-Process -FilePath (Join-Path $sourceRoot '.venv\Scripts\python.exe') -ArgumentList $workerArgs -WorkingDirectory $sourceRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $run 'worker.log') -RedirectStandardError (Join-Path $run 'worker.err')
    $workerHandle=$worker.Handle;$worker.PriorityClass='BelowNormal'
    @{worker_pid=$worker.Id;utc=(Get-Date).ToUniversalTime().ToString('o');output=$output} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $run 'process.json') -Encoding UTF8
    while(-not $worker.HasExited) {
        if((Get-Date) -gt $deadline) { Stop-OwnWorker;throw 'Job deadline reached' }
        if(@(Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue).Count -or (Busy)) { Stop-OwnWorker;throw 'Game/source workload started' }
        if((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB -lt .75) { Stop-OwnWorker;throw 'Host memory floor reached' }
        Start-Sleep -Seconds 5;$worker.Refresh()
    }
    $worker.WaitForExit()
    if($worker.ExitCode -ne 0) { throw ('Recipe native M2 worker exited '+$worker.ExitCode+'; see worker.err') }
    $manifest=Join-Path $output 'manifest.json'
    if(-not(Test-Path -LiteralPath $manifest)) { throw 'No completed export manifest' }
    $result=Get-Content -Raw -LiteralPath $manifest | ConvertFrom-Json
    if($result.state -ne 'complete' -or $result.states -ne 56288 -or $result.complete_probability_rows_replayed_exactly -ne 56288 -or $result.saved_rows_verified_exactly -ne 56288 -or $result.target_file_opened -or $result.training_performed -or $result.accuracy_computed) { throw 'Native export completion/boundary failed' }
    State 'completed' ('manifest='+$manifest)
} catch { State 'failed' $_.Exception.Message;throw }
finally { Stop-OwnWorker;Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue }

