param([Parameter(Mandatory=$true)][string]$Version,[Parameter(Mandatory=$true)][string]$TrainingVersion)
$ErrorActionPreference='Stop'
if($Version -notmatch '^[A-Za-z0-9_.-]+$' -or $TrainingVersion -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Unsafe version' }
$root=Split-Path -Parent $PSScriptRoot
$taskName='Videns Causal Student Predict v2'
$run=Join-Path $root ('data\runs\'+$Version)
if(Test-Path -LiteralPath $run) { throw 'Existing immutable fit run' }
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
        $gpuFree=@(& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
        $gpuOK=$LASTEXITCODE -eq 0 -and $gpuFree.Count -gt 0 -and [int]$gpuFree[0] -ge 3072
        if(-not(Busy) -and -not @(Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue).Count -and $free -ge 2.5 -and $gpuOK) { break }
        State 'waiting_for_resources' ('freeGiB='+[math]::Round($free,2));Start-Sleep -Seconds 15
    }
    $env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1'
    $output=Join-Path $run 'experiment';$script=Join-Path $root 'scripts\predict_causal_student_v2.py'
    $trainingDirectory=Join-Path $root ('data\runs\'+$TrainingVersion+'\experiment')
    $workerArgs=@('-u',('"'+$script+'"'),'--training-directory',('"'+$trainingDirectory+'"'),'--output',('"'+$output+'"'))
    $deadline=(Get-Date).AddMinutes(61)
    State 'running' 'teacher-free prediction from unchanged final models'
    $worker=Start-Process -FilePath (Join-Path $root '.venv\Scripts\python.exe') -ArgumentList $workerArgs -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $run 'worker.log') -RedirectStandardError (Join-Path $run 'worker.err')
    $workerHandle=$worker.Handle;$worker.PriorityClass='BelowNormal'
    @{worker_pid=$worker.Id;utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $run 'process.json') -Encoding UTF8
    while(-not $worker.HasExited) {
        if((Get-Date) -gt $deadline) { Stop-OwnWorker;throw 'Fit worker deadline reached' }
        if(@(Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue).Count -or (Busy)) { Stop-OwnWorker;throw 'Game or sibling workload started' }
        if((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB -lt .75) { Stop-OwnWorker;throw 'Host RAM floor reached' }
        Start-Sleep -Seconds 5;$worker.Refresh()
    }
    $worker.WaitForExit()
    if($worker.ExitCode -ne 0) { throw ('Fit worker exited '+$worker.ExitCode) }
    $report=Join-Path $output 'report.json'
    if(-not(Test-Path -LiteralPath $report)) { throw 'Complete fit report missing' }
    $result=Get-Content -Raw -LiteralPath $report | ConvertFrom-Json
    if($result.state -ne 'complete' -or $result.updates -ne 26460 -or $result.query_games -ne 2900 -or -not $result.training_performed -or $result.student_inference_uses_teacher_or_labels -or $result.accuracy_computed -or $result.test_payload_reads -or $result.confirmation_payload_reads) { throw 'Fit completion/boundary failed' }
    State 'completed' ('report='+$report)
} catch { State 'failed' $_.Exception.Message;throw }
finally { Stop-OwnWorker;Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue }









