param([Parameter(Mandatory=$true)][string]$Version,[Parameter(Mandatory=$true)][string]$ForecastVersion,[Parameter(Mandatory=$true)][string]$NativeVersion)
$ErrorActionPreference='Stop'
if($Version -notmatch '^[A-Za-z0-9_.-]+$' -or $ForecastVersion -notmatch '^[A-Za-z0-9_.-]+$' -or $NativeVersion -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Unsafe version' }
$root=Split-Path -Parent $PSScriptRoot
$sourceRoot=Join-Path (Split-Path -Parent $root) 'build_tracker'
$taskName='Videns Bayes Proxy Evaluation v1'
$run=Join-Path $root ('data\runs\'+$Version)
$sourceRun=Join-Path $sourceRoot ('data\exports\winprob-bayes-proxy\'+$NativeVersion)
$fit=Join-Path $root ('data\runs\'+$ForecastVersion+'\experiment')
$distManifest=Join-Path $fit 'distribution\manifest.json'
if(Test-Path -LiteralPath $run) { throw 'Existing immutable evaluation' }
if(-not(Test-Path -LiteralPath $distManifest)) { throw 'Completed full-catalogue fit required first' }
if((Get-Content -LiteralPath $distManifest -Raw | ConvertFrom-Json).state -ne 'complete') { throw 'Incomplete fit distribution' }
New-Item -ItemType Directory -Path $run | Out-Null
$deadline=(Get-Date).AddMinutes(245);$worker=$null
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
function Wait-Resources([bool]$needsGpu) {
    $resourceDeadline=(Get-Date).AddMinutes(15)
    while($true) {
        if((Get-Date) -gt $deadline -or (Get-Date) -gt $resourceDeadline) { throw 'Resource deadline reached' }
        $free=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
        $gpuOK=$true
        if($needsGpu) {
            $gpuFree=@(& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
            $gpuOK=$LASTEXITCODE -eq 0 -and $gpuFree.Count -gt 0 -and [int]$gpuFree[0] -ge 3072
        }
        if(-not(Busy) -and -not @(Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue).Count -and $free -ge 2.5 -and $gpuOK) { return }
        State 'waiting_for_resources' ('freeGiB='+[math]::Round($free,2));Start-Sleep -Seconds 15
    }
}
function Stage([string]$name,[string]$python,[string]$scriptPath,[string[]]$stageArgs,[string]$cwd,[string]$completion,[int]$minutes,[bool]$needsGpu) {
    Wait-Resources $needsGpu;State 'running' $name
    $workerArguments=@('-u',('"'+$scriptPath+'"'))+$stageArgs
    $script:worker=Start-Process -FilePath $python -ArgumentList $workerArguments -WorkingDirectory $cwd -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $run ($name+'.log')) -RedirectStandardError (Join-Path $run ($name+'.err'))
    $workerHandle=$worker.Handle;$worker.PriorityClass='BelowNormal';$stageDeadline=(Get-Date).AddMinutes($minutes)
    @{worker_pid=$worker.Id;stage=$name;utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $run 'process.json') -Encoding UTF8
    while(-not $worker.HasExited) {
        if((Get-Date) -gt $deadline -or (Get-Date) -gt $stageDeadline) { Stop-OwnWorker;throw 'Stage deadline reached' }
        if(@(Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue).Count -or (Busy)) { Stop-OwnWorker;throw 'Game or source workload started' }
        if((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB -lt .75) { Stop-OwnWorker;throw 'Host RAM floor reached' }
        Start-Sleep -Seconds 5;$worker.Refresh()
    }
    $worker.WaitForExit()
    if($worker.ExitCode -ne 0) { throw ($name+' worker exited '+$worker.ExitCode) }
    if(-not(Test-Path -LiteralPath $completion)) { throw ('Missing completion: '+$name) }
    if((Get-Content -Raw -LiteralPath $completion | ConvertFrom-Json).state -ne 'complete') { throw ('Incomplete artifact: '+$name) }
}
try {
    $env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1'
    $ownPython=Join-Path $root '.venv\Scripts\python.exe';$sourcePython=Join-Path $sourceRoot '.venv\Scripts\python.exe'
    $decodedManifest=Join-Path $sourceRun 'decoded\manifest.json';$nativeManifest=Join-Path $sourceRun 'native17\manifest.json'
    foreach($path in @($decodedManifest,$nativeManifest)) { if(-not(Test-Path -LiteralPath $path) -or (Get-Content -LiteralPath $path -Raw | ConvertFrom-Json).state -ne 'complete') { throw 'Complete native manifests required' } }
    $forecastAudit=Join-Path $run 'forecast_audit';$forecastAuditReport=Join-Path $forecastAudit 'report.json'
    Stage 'forecast_audit' $ownPython (Join-Path $root 'scripts\audit_bayes_proxy_forecasts_v1.py') @('--directory',('"'+$fit+'"'),'--output',('"'+$forecastAudit+'"')) $root $forecastAuditReport 121 $true
    $verified=Get-Content -LiteralPath $forecastAuditReport -Raw | ConvertFrom-Json
    if(-not $verified.passed -or $verified.anchored_distributions_replayed_exactly -ne 8700) { throw 'Forecast replay incomplete' }
    $evaluation=Join-Path $run 'evaluation';$report=Join-Path $evaluation 'report.json'
    Stage 'evaluation' $ownPython (Join-Path $root 'scripts\evaluate_bayes_proxy_v1.py') @('--directory',('"'+$fit+'"'),'--decoded-manifest',('"'+$decodedManifest+'"'),'--native-manifest',('"'+$nativeManifest+'"'),'--output',('"'+$evaluation+'"')) $root $report 31 $false
    $audit=Join-Path $run 'audit';$auditReport=Join-Path $audit 'report.json'
    Stage 'audit' $ownPython (Join-Path $root 'scripts\audit_bayes_proxy_metrics_v1.py') @('--report',('"'+$report+'"'),'--forecast-audit',('"'+$forecastAuditReport+'"'),'--output',('"'+$audit+'"')) $root $auditReport 31 $false
    $checked=Get-Content -LiteralPath $auditReport -Raw | ConvertFrom-Json
    if(-not $checked.passed -or $checked.test_payload_reads -or $checked.confirmation_payload_reads) { throw 'Final metric audit failed' }
    State 'completed' ('report='+$report+'; audit='+$auditReport)
} catch { State 'failed' $_.Exception.Message;throw }
finally { Stop-OwnWorker;Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue }
