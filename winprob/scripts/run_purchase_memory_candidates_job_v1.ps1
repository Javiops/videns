param([Parameter(Mandatory=$true)][string]$Version)
$ErrorActionPreference='Stop'
if($Version -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Unsafe version' }
$root=Split-Path -Parent $PSScriptRoot
$sourceRoot=Join-Path (Split-Path -Parent $root) 'build_tracker'
$taskName='Videns Purchase Memory Candidates v1'
$run=Join-Path $root ('data\runs\'+$Version)
$sourceRun=Join-Path $sourceRoot ('data\exports\winprob-memory\'+$Version)
if((Test-Path -LiteralPath $run) -or (Test-Path -LiteralPath $sourceRun)) { throw 'Existing immutable run' }
New-Item -ItemType Directory -Path $run | Out-Null
$deadline=(Get-Date).AddMinutes(30)
$worker=$null
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
function Wait-Resources {
    while($true) {
        if((Get-Date) -gt $deadline) { throw 'Resource deadline reached' }
        $free=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
        if(-not (Busy) -and -not @(Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue).Count -and $free -ge 2.5) { return }
        State 'waiting_for_resources' ('freeGiB='+[math]::Round($free,2))
        Start-Sleep -Seconds 15
    }
}
function Stage([string]$name,[string]$python,[string]$script,[string[]]$stageArgs,[string]$cwd,[string]$completion) {
    Wait-Resources
    State 'running' $name
    $arguments=@('-u',('"'+$script+'"'))+$stageArgs
    $script:worker=Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $cwd -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $run ($name+'.log')) -RedirectStandardError (Join-Path $run ($name+'.err'))
    $workerHandle=$worker.Handle;$worker.PriorityClass='BelowNormal';$stageDeadline=(Get-Date).AddMinutes(11)
    @{worker_pid=$worker.Id;stage=$name;utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $run 'process.json') -Encoding UTF8
    while(-not $worker.HasExited) {
        if((Get-Date) -gt $deadline -or (Get-Date) -gt $stageDeadline) { Stop-OwnWorker; throw 'Worker deadline reached' }
        if(@(Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue).Count -or (Busy)) { Stop-OwnWorker; throw 'Game/source workload started; own worker stopped' }
        if((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB -lt .75) { Stop-OwnWorker; throw 'Host memory floor reached' }
        Start-Sleep -Seconds 5;$worker.Refresh()
    }
    $worker.WaitForExit()
    if($worker.ExitCode -ne 0) { throw ($name+' exited '+$worker.ExitCode+'; inspect its error log') }
    if(-not (Test-Path -LiteralPath $completion)) { throw ('Missing completion for '+$name) }
    if((Get-Content -Raw -LiteralPath $completion | ConvertFrom-Json).state -ne 'complete') { throw ('Incomplete artifact for '+$name) }
}
try {
    $env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1'
    $ownPython=Join-Path $root '.venv\Scripts\python.exe';$sourcePython=Join-Path $sourceRoot '.venv\Scripts\python.exe'
    $distribution=Join-Path $run 'distribution';$distManifest=Join-Path $distribution 'manifest.json'
    Stage 'distribution' $ownPython (Join-Path $root 'scripts\export_purchase_memory_distribution_v1.py') @('--output',('"'+$distribution+'"')) $root $distManifest
    $decoded=Join-Path $sourceRun 'decoded';$decodedManifest=Join-Path $decoded 'manifest.json'
    Stage 'decode' $sourcePython (Join-Path $sourceRoot 'scripts\decode_purchase_memory_v1.py') @('--mode','decode','--distribution-manifest',('"'+$distManifest+'"'),'--output',('"'+$decoded+'"')) $sourceRoot $decodedManifest
    $audit=Join-Path $sourceRun 'native19';$auditManifest=Join-Path $audit 'manifest.json'
    Stage 'native19' $sourcePython (Join-Path $sourceRoot 'scripts\decode_purchase_memory_v1.py') @('--mode','audit19','--distribution-manifest',('"'+$distManifest+'"'),'--decoded-manifest',('"'+$decodedManifest+'"'),'--output',('"'+$audit+'"')) $sourceRoot $auditManifest
    $evaluation=Join-Path $run 'evaluation';$report=Join-Path $evaluation 'report.json'
    Stage 'evaluation' $ownPython (Join-Path $root 'scripts\evaluate_purchase_memory_candidates_v1.py') @('--distribution-manifest',('"'+$distManifest+'"'),'--decoded-manifest',('"'+$decodedManifest+'"'),'--native-manifest',('"'+$auditManifest+'"'),'--output',('"'+$evaluation+'"')) $root $report
    State 'completed' ('report='+$report)
} catch { State 'failed' $_.Exception.Message; throw }
finally { Stop-OwnWorker; Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue }
