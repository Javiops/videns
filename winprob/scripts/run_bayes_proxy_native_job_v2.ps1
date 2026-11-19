param([Parameter(Mandatory=$true)][string]$Version,[Parameter(Mandatory=$true)][string]$ForecastVersion)
$ErrorActionPreference='Stop'
if($Version -notmatch '^[A-Za-z0-9_.-]+$' -or $ForecastVersion -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Unsafe version' }
$root=Split-Path -Parent $PSScriptRoot
$sourceRoot=Join-Path (Split-Path -Parent $root) 'build_tracker'
$taskName='Videns Bayes Proxy Native v2'
$run=Join-Path $root ('data\runs\'+$Version)
$sourceRun=Join-Path $sourceRoot ('data\exports\winprob-bayes-proxy\'+$Version)
$fit=Join-Path $root ('data\runs\'+$ForecastVersion+'\experiment')
$distManifest=Join-Path $fit 'distribution\manifest.json'
if((Test-Path -LiteralPath $run) -or (Test-Path -LiteralPath $sourceRun)) { throw 'Existing immutable evaluation' }
if(-not(Test-Path -LiteralPath $distManifest)) { throw 'Completed full-catalogue fit required first' }
if((Get-Content -LiteralPath $distManifest -Raw | ConvertFrom-Json).state -ne 'complete') { throw 'Incomplete fit distribution' }
New-Item -ItemType Directory -Path $run | Out-Null
$deadline=(Get-Date).AddMinutes(95);$worker=$null
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
    $decoded=Join-Path $sourceRun 'decoded';$decodedManifest=Join-Path $decoded 'manifest.json'
    Stage 'decode' $sourcePython (Join-Path $sourceRoot 'scripts\decode_bayes_proxy_v2.py') @('--mode','decode','--distribution-manifest',('"'+$distManifest+'"'),'--output',('"'+$decoded+'"')) $sourceRoot $decodedManifest 31 $true
    $native17=Join-Path $sourceRun 'native17';$nativeManifest=Join-Path $native17 'manifest.json'
    Stage 'native17' $sourcePython (Join-Path $sourceRoot 'scripts\decode_bayes_proxy_v2.py') @('--mode','audit17','--distribution-manifest',('"'+$distManifest+'"'),'--decoded-manifest',('"'+$decodedManifest+'"'),'--output',('"'+$native17+'"')) $sourceRoot $nativeManifest 31 $false
    State 'completed' ('decoded='+$decodedManifest+'; native17='+$nativeManifest)
} catch { State 'failed' $_.Exception.Message;throw }
finally { Stop-OwnWorker;Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue }

