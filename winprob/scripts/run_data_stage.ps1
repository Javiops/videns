param(
    [Parameter(Mandatory=$true)][ValidateSet('revision-contexts','trajectory-smoke','undo-expanded','revision-align-full','revision-score-full','ipo-fit','iql-trajectories','iql-cache','ordinary-contexts','ordinary-actor','ordinary-reference','ordinary-native19','ordinary-evaluate','ordinary-feasibility-18','ordinary-feasibility-19','goal-census','commitment-census','wp-distill','wp-distill-audit','current-schema','current-combat-cache','current-combat-fit','wp-scale-bind','wp-scale-cache','wp-scale-cache-audit','wp-scale-fit','wp-scale-audit','wp-players-cache','wp-players-cache-audit')][string]$Stage,
    [Parameter(Mandatory=$true)][string]$Version
)
$ErrorActionPreference='Stop'
if ($Version -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Unsafe version' }
$root=Split-Path -Parent $PSScriptRoot
$taskName='Videns Data '+$Stage
$runDir=Join-Path $root ('data\runs\'+$Version)
if (Test-Path -LiteralPath $runDir) { throw 'Existing run; refusing overwrite' }
New-Item -ItemType Directory -Path $runDir -Force | Out-Null
$worker=$null
$minimumFreeGiB=if($Stage -in @('goal-census','commitment-census','current-schema','wp-scale-bind')) { 1.5 } elseif($Stage -in @('ordinary-feasibility-18','ordinary-feasibility-19')) { 1.25 } elseif($Stage -in @('ordinary-reference','ordinary-native19','ordinary-evaluate','wp-distill','wp-distill-audit','current-combat-cache','current-combat-fit')) { 2.0 } else { 2.5 }
$liveFloorGiB=if($Stage -in @('commitment-census','wp-distill','wp-distill-audit','current-schema','current-combat-cache','current-combat-fit','wp-scale-bind','wp-scale-cache','wp-scale-cache-audit','wp-scale-fit','wp-scale-audit','wp-players-cache','wp-players-cache-audit')) { .75 } else { .6 }
$memoryMarks=@()
$deadline=(Get-Date).AddMinutes($(if($Stage -in @('wp-scale-cache','wp-scale-cache-audit','wp-scale-fit','wp-scale-audit','wp-players-cache','wp-players-cache-audit')) {55} elseif($Stage -in @('iql-cache','ordinary-reference')) { 45 } elseif($Stage -in @('iql-trajectories','current-combat-cache')) { 40 } else { 25 }))
function Set-State([string]$state,[string]$reason) {
    @{state=$state;reason=$reason;stage=$Stage;utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runDir 'launcher_status.json') -Encoding UTF8
}
function Stop-OwnWorker {
    if ($null -ne $worker -and -not $worker.HasExited) { & taskkill.exe /PID $worker.Id /T /F | Out-Null }
}
try {
    while ($true) {
        if ((Get-Date) -gt $deadline) { throw 'Stage deadline reached while waiting' }
        $busy=@(Get-ScheduledTask | Where-Object { ($_.TaskName -like 'BuildTracker*' -or ($_.TaskName -like 'Videns*' -and $_.TaskName -ne $taskName)) -and $_.State -eq 'Running' })
        $game=@(Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue)
        $os=Get-CimInstance Win32_OperatingSystem
        if ($busy.Count -eq 0 -and $game.Count -eq 0 -and $os.FreePhysicalMemory/1MB -ge $minimumFreeGiB) { break }
        Set-State 'waiting_for_resources' ('jobs='+$busy.Count+';freeGiB='+[math]::Round($os.FreePhysicalMemory/1MB,2))
        Start-Sleep -Seconds 30
    }
    $env:OMP_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
    if ($Stage -eq 'revision-contexts') {
        $working=$root
        $python=Join-Path $root '.venv\Scripts\python.exe'
        $script=Join-Path $root 'scripts\prepare_revision_contexts.py'
        $output=Join-Path $runDir 'cache'
        $arguments=@('-u',('"'+$script+'"'),'--admission',('"'+(Join-Path $root 'artifacts\admission\w1-admission-20260929-v2\admission_manifest.json')+'"'),'--base-cache',('"'+(Join-Path $root 'data\runs\w1-pilot-20260929-v3\cache')+'"'),'--output',('"'+$output+'"'))
        $completion=Join-Path $output 'cache_metadata.json'
    } elseif ($Stage -eq 'undo-expanded') {
        $working=$root
        $python=Join-Path $root '.venv\Scripts\python.exe'
        $script=Join-Path $root 'scripts\undo_pairs_expanded.py'
        $cache=Join-Path $root 'data\runs\revision-contexts-20260930-v1\cache'
        $output=Join-Path $runDir 'experiment'
        $arguments=@('-u',('"'+$script+'"'),'--cache',('"'+(Join-Path $cache 'states.npz')+'"'),'--source-manifest',('"'+(Join-Path $cache 'source_manifest.json')+'"'),'--static-export',('"'+(Join-Path $root 'artifacts\boots-static\20260930-v1\boots_static.json')+'"'),'--output',('"'+$output+'"'),'--deadline-seconds','1200')
        $completion=Join-Path $output 'report.json'
    } elseif ($Stage -eq 'revision-align-full') {
        $working=$root
        $python='powershell.exe'
        $output=$runDir
        $script=Join-Path $root 'scripts\align_revision_patches.ps1'
        $arguments=@('-NoProfile','-ExecutionPolicy','Bypass','-WindowStyle','Hidden','-File',('"'+$script+'"'),'-RunDirectory',('"'+$runDir+'"'),'-Version',$Version)
        $completion=Join-Path $runDir 'alignment_summary.json'
    } elseif ($Stage -eq 'revision-score-full') {
        $working=Join-Path (Split-Path -Parent $root) 'build_tracker'
        $python=Join-Path $working '.venv\Scripts\python.exe'
        $script=Join-Path $working 'scripts\score_winprob_m2_pairs_v2.py'
        $output=Join-Path $working ('data\exports\winprob-m2-pairs\'+$Version)
        $arguments=@('-u',('"'+$script+'"'),'--full','--output',('"'+$output+'"'))
        foreach ($patch in @('16.17','16.18','16.19')) {
            $context=Join-Path $working ('data\exports\winprob-revision-contexts\'+$patch+'-revision-alignment-20260930-v1')
            $arguments+=@('--contexts',('"'+$context+'"'))
        }
        $completion=Join-Path $output 'manifest.json'
    } elseif ($Stage -eq 'ipo-fit') {
        $working=$root
        $python=Join-Path $root '.venv\Scripts\python.exe'
        $script=Join-Path $root 'scripts\fit_ipo_residual.py'
        $reference=Join-Path (Split-Path -Parent $root) 'build_tracker\data\exports\winprob-m2-pairs\m2-preference-20260930-v1'
        $output=Join-Path $runDir 'experiment'
        $arguments=@('-u',('"'+$script+'"'),'--reference',('"'+$reference+'"'),'--admission',('"'+(Join-Path $root 'artifacts\admission\w1-admission-20260929-v2\admission_manifest.json')+'"'),'--static-export',('"'+(Join-Path $root 'artifacts\boots-static\20260930-v1\boots_static.json')+'"'),'--output',('"'+$output+'"'),'--evaluate-development','--deadline-seconds','1200')
        $completion=Join-Path $output 'report.json'
    } elseif ($Stage -eq 'iql-trajectories') {
        $working=$root
        $python='powershell.exe'
        $output=$runDir
        $script=Join-Path $root 'scripts\export_iql_patches.ps1'
        $arguments=@('-NoProfile','-ExecutionPolicy','Bypass','-WindowStyle','Hidden','-File',('"'+$script+'"'),'-RunDirectory',('"'+$runDir+'"'),'-Version',$Version)
        $completion=Join-Path $runDir 'trajectory_summary.json'
    } elseif ($Stage -eq 'iql-cache') {
        $working=$root
        $python=Join-Path $root '.venv\Scripts\python.exe'
        $script=Join-Path $root 'scripts\prepare_iql_dataset.py'
        $sourceRoot=Join-Path (Split-Path -Parent $root) 'build_tracker'
        $output=Join-Path $runDir 'cache'
        $arguments=@('-u',('"'+$script+'"'),'--admission',('"'+(Join-Path $root 'artifacts\admission\w1-admission-20260929-v2\admission_manifest.json')+'"'),'--catalogue',('"'+(Join-Path $sourceRoot 'data\exports\winprob-action-catalogue\full-action-catalogue-20260930-v1\catalogue.json')+'"'),'--static-export',('"'+(Join-Path $root 'artifacts\boots-static\20260930-v1\boots_static.json')+'"'),'--output',('"'+$output+'"'),'--deadline-seconds','2400')
        foreach ($patch in @('16.17','16.18')) {
            $source=Join-Path $sourceRoot ('data\exports\winprob-fit-trajectories\'+$patch+'-iql-trajectories-20260930-v1')
            $arguments+=@('--source',('"'+$source+'"'))
        }
        $completion=Join-Path $output 'manifest.json'
    } elseif ($Stage -eq 'ordinary-contexts') {
        $working=Join-Path (Split-Path -Parent $root) 'build_tracker'
        $python=Join-Path $working '.venv\Scripts\python.exe'
        $script=Join-Path $working 'scripts\export_bc20_ordinary_contexts_v1.py'
        $output=Join-Path $working ('data\exports\winprob-ordinary\'+$Version)
        $selection=Join-Path $working 'data\exports\winprob-ordinary\ordinary-selection-20260930-v1\selection.json'
        $arguments=@('-u',('"'+$script+'"'),'contexts','--selection',('"'+$selection+'"'),'--full','--limit','0','--output',('"'+$output+'"'))
        $completion=Join-Path $output 'manifest.json'
    } elseif ($Stage -eq 'ordinary-actor') {
        $working=$root
        $python=Join-Path $root '.venv\Scripts\python.exe'
        $script=Join-Path $root 'scripts\export_ordinary_bc20_v2.py'
        $source=Join-Path (Split-Path -Parent $root) 'build_tracker\data\exports\winprob-ordinary\ordinary-contexts-20260930-v1\manifest.json'
        $output=Join-Path $runDir 'export'
        $arguments=@('-u',('"'+$script+'"'),'--source-manifest',('"'+$source+'"'),'--output',('"'+$output+'"'),'--deadline-seconds','600')
        $completion=Join-Path $output 'manifest.json'
    } elseif ($Stage -in @('ordinary-reference','ordinary-native19')) {
        $working=Join-Path (Split-Path -Parent $root) 'build_tracker'
        $python=Join-Path $working '.venv\Scripts\python.exe'
        $source=Join-Path $working 'data\exports\winprob-ordinary\ordinary-contexts-20260930-v1\manifest.json'
        $output=Join-Path $working ('data\exports\winprob-ordinary\'+$Version)
        if ($Stage -eq 'ordinary-reference') {
            $script=Join-Path $working 'scripts\export_bc20_ordinary_reference_v2.py'
            $actor=Join-Path $root 'data\runs\ordinary-actor-20260930-v2\export\manifest.json'
            $arguments=@('-u',('"'+$script+'"'),'--source-manifest',('"'+$source+'"'),'--actor-manifest',('"'+$actor+'"'),'--output',('"'+$output+'"'),'--batch-size','16','--deadline-seconds','2400')
        } else {
            $script=Join-Path $working 'scripts\audit_bc20_ordinary_native19_v2.py'
            $reference=Join-Path $working 'data\exports\winprob-ordinary\ordinary-reference-20260930-v3\manifest.json'
            $arguments=@('-u',('"'+$script+'"'),'--source-manifest',('"'+$source+'"'),'--reference-manifest',('"'+$reference+'"'),'--output',('"'+$output+'"'),'--deadline-seconds','600')
        }
        $completion=Join-Path $output 'manifest.json'
    } elseif ($Stage -eq 'ordinary-evaluate') {
        $working=$root
        $python=Join-Path $root '.venv\Scripts\python.exe'
        $script=Join-Path $root 'scripts\evaluate_ordinary_bc20.py'
        $sourceRoot=Join-Path (Split-Path -Parent $root) 'build_tracker\data\exports\winprob-ordinary'
        $source=Join-Path $sourceRoot 'ordinary-contexts-20260930-v1\manifest.json'
        $reference=Join-Path $sourceRoot 'ordinary-reference-20260930-v3\manifest.json'
        $audit=Join-Path $sourceRoot 'ordinary-native19-20260930-v3\manifest.json'
        $actor=Join-Path $root 'data\runs\ordinary-actor-20260930-v2\export\manifest.json'
        $output=Join-Path $runDir 'experiment'
        $arguments=@('-u',('"'+$script+'"'),'--source-manifest',('"'+$source+'"'),'--actor-manifest',('"'+$actor+'"'),'--reference-manifest',('"'+$reference+'"'),'--audit-manifest',('"'+$audit+'"'),'--output',('"'+$output+'"'))
        $completion=Join-Path $output 'report.json'
    } elseif ($Stage -in @('ordinary-feasibility-18','ordinary-feasibility-19')) {
        $working=Join-Path (Split-Path -Parent $root) 'build_tracker'
        $python=Join-Path $working '.venv\Scripts\python.exe'
        $script=Join-Path $working 'scripts\diagnose_bc20_ordinary_feasibility_v1.py'
        $sourceRoot=Join-Path $working 'data\exports\winprob-ordinary'
        $source=Join-Path $sourceRoot 'ordinary-contexts-20260930-v1\manifest.json'
        $targets=Join-Path $sourceRoot 'ordinary-contexts-20260930-v1\targets.jsonl'
        $reference=Join-Path $sourceRoot 'ordinary-reference-20260930-v3\manifest.json'
        $audit=Join-Path $sourceRoot 'ordinary-native19-20260930-v3\manifest.json'
        $actor=Join-Path $root 'data\runs\ordinary-actor-20260930-v2\export\manifest.json'
        $evaluation=Join-Path $root 'data\runs\ordinary-evaluation-20260930-v1\experiment\report.json'
        $patch=if($Stage -eq 'ordinary-feasibility-18') { '16.18' } else { '16.19' }
        $output=Join-Path $sourceRoot $Version
        $arguments=@('-u',('"'+$script+'"'),'--evaluation-report',('"'+$evaluation+'"'),'--source-manifest',('"'+$source+'"'),'--targets',('"'+$targets+'"'),'--actor-manifest',('"'+$actor+'"'),'--reference-manifest',('"'+$reference+'"'),'--native19-manifest',('"'+$audit+'"'),'--patch',$patch,'--output',('"'+$output+'"'),'--read-observed-targets','--deadline-seconds','120')
        $completion=Join-Path $output 'manifest.json'
    } elseif ($Stage -in @('wp-scale-cache-audit','wp-scale-audit')) {
        $working=$root
        $python=Join-Path $root '.venv\Scripts\python.exe'
        $file=if($Stage -eq 'wp-scale-cache-audit') {'audit_wp_scale_cache_v1'} else {'audit_wp_scale_result_v2'}
        $script=Join-Path $root ('scripts\'+$file+'.py')
        $output=Join-Path $root ('data\reports\'+$(if($Stage -eq 'wp-scale-cache-audit') {'wp_scale_cache_audit_v1.json'} else {'wp_scale_result_audit_v2.json'}))
        $arguments=@('-u',('"'+$script+'"'))
        $completion=$output
    } elseif ($Stage -eq 'wp-players-cache-audit') {
        $working=$root
        $python=Join-Path $root '.venv\Scripts\python.exe'
        $script=Join-Path $root 'scripts\audit_wp_players_cache_v1.py'
        $output=Join-Path $root 'data\reports\wp_players_cache_audit_v1.json'
        $arguments=@('-u',('"'+$script+'"'))
        $completion=$output
    } elseif ($Stage -eq 'wp-players-cache') {
        $working=$root
        $python=Join-Path $root '.venv\Scripts\python.exe'
        $script=Join-Path $root 'scripts\prepare_wp_players_v1.py'
        $output=Join-Path $runDir 'cache'
        $arguments=@('-u',('"'+$script+'"'),'--output',('"'+$output+'"'))
        $completion=Join-Path $output 'manifest.json'
    } elseif ($Stage -eq 'wp-scale-fit') {
        $working=$root
        $python=Join-Path $root '.venv\Scripts\python.exe'
        $script=Join-Path $root 'scripts\train_wp_scale_v2.py'
        $output=Join-Path $runDir 'experiment'
        $cache=Join-Path $root 'data\runs\wp-scale-cache-20260930-v1\cache\manifest.json'
        $arguments=@('-u',('"'+$script+'"'),'--cache-manifest',('"'+$cache+'"'),'--output',('"'+$output+'"'))
        $completion=Join-Path $output 'report.json'
    } elseif ($Stage -eq 'wp-scale-cache') {
        $working=$root
        $python=Join-Path $root '.venv\Scripts\python.exe'
        $script=Join-Path $root 'scripts\prepare_wp_scale_cache_v1.py'
        $output=Join-Path $runDir 'cache'
        $arguments=@('-u',('"'+$script+'"'),'--output',('"'+$output+'"'))
        $completion=Join-Path $output 'manifest.json'
    } elseif ($Stage -eq 'wp-scale-bind') {
        $working=$root
        $python=Join-Path $root '.venv\Scripts\python.exe'
        $script=Join-Path $root 'scripts\prepare_wp_scale_scope_v1.py'
        $output=Join-Path $runDir 'bindings'
        $scope=Join-Path $root 'artifacts\wp-scale-scope\20260930-v1\scope.json'
        $arguments=@('-u',('"'+$script+'"'),'bind','--scope',('"'+$scope+'"'),'--output',('"'+$output+'"'))
        $completion=Join-Path $output 'sources.json'
    } elseif ($Stage -eq 'current-combat-fit') {
        $working=$root
        $python=Join-Path $root '.venv\Scripts\python.exe'
        $script=Join-Path $root 'scripts\train_current_combat_v1.py'
        $output=Join-Path $runDir 'experiment'
        $cache=Join-Path $root 'data\runs\current-combat-cache-20260930-v1\cache\manifest.json'
        $arguments=@('-u',('"'+$script+'"'),'--cache-manifest',('"'+$cache+'"'),'--output',('"'+$output+'"'))
        $completion=Join-Path $output 'report.json'
    } elseif ($Stage -eq 'current-combat-cache') {
        $working=$root
        $python=Join-Path $root '.venv\Scripts\python.exe'
        $script=Join-Path $root 'scripts\prepare_current_combat_v1.py'
        $output=Join-Path $runDir 'cache'
        $arguments=@('-u',('"'+$script+'"'),'--output',('"'+$output+'"'))
        $completion=Join-Path $output 'manifest.json'
    } elseif ($Stage -eq 'current-schema') {
        $working=$root
        $python=Join-Path $root '.venv\Scripts\python.exe'
        $script=Join-Path $root 'scripts\inspect_current_state_schema_v1.py'
        $output=Join-Path $runDir 'inspection'
        $arguments=@('-u',('"'+$script+'"'),'--output',('"'+$output+'"'))
        $completion=Join-Path $output 'report.json'
    } elseif ($Stage -eq 'wp-distill-audit') {
        $working=$root
        $python=Join-Path $root '.venv\Scripts\python.exe'
        $script=Join-Path $root 'data\reports\wp_distillation_audit_v1.py'
        $output=Join-Path $root 'data\reports\wp_distillation_audit_v1.json'
        $arguments=@('-u',('"'+$script+'"'))
        $completion=$output
    } elseif ($Stage -eq 'wp-distill') {
        $working=$root
        $python=Join-Path $root '.venv\Scripts\python.exe'
        $script=Join-Path $root 'scripts\run_wp_distillation_v1.py'
        $output=Join-Path $runDir 'experiment'
        $arguments=@('-u',('"'+$script+'"'),'--output',('"'+$output+'"'))
        $completion=Join-Path $output 'report.json'
    } elseif ($Stage -eq 'commitment-census') {
        $working=$root
        $python=Join-Path $root '.venv\Scripts\python.exe'
        $script=Join-Path $root 'scripts\commitment_census_v1.py'
        $output=Join-Path $runDir 'census'
        $arguments=@('-u',('"'+$script+'"'),'--output',('"'+$output+'"'))
        $completion=Join-Path $output 'report.json'
    } elseif ($Stage -eq 'goal-census') {
        $working=$root
        $python=Join-Path $root '.venv\Scripts\python.exe'
        $script=Join-Path $root 'scripts\prepare_goal_census_v2.py'
        $definition=Join-Path $root 'artifacts\goal-definition\20260930-v2\definition.json'
        $output=Join-Path $runDir 'census'
        $arguments=@('-u',('"'+$script+'"'),'census','--definition',('"'+$definition+'"'),'--output',('"'+$output+'"'),'--deadline-seconds','300')
        $completion=Join-Path $output 'report.json'
    } else {
        $working=Join-Path (Split-Path -Parent $root) 'build_tracker'
        $python=Join-Path $working '.venv\Scripts\python.exe'
        $script=Join-Path $working 'scripts\export_winprob_trajectories_v1.py'
        $output=Join-Path $working ('data\exports\winprob-trajectories\'+$Version)
        $arguments=@('-u',('"'+$script+'"'),'--limit','10','--output',('"'+$output+'"'))
        $completion=Join-Path $output 'manifest.json'
    }
    Set-State 'running' ('source-owned bounded stage; output='+$output)
    $worker=Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $working -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runDir 'worker.log') -RedirectStandardError (Join-Path $runDir 'worker.err')
    $workerHandle=$worker.Handle
    $worker.PriorityClass='BelowNormal'
    while (-not $worker.HasExited) {
        if ((Get-Date) -gt $deadline) { Stop-OwnWorker; throw 'Stage deadline reached' }
        if (@(Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue).Count -gt 0) { Stop-OwnWorker; throw 'Game started; own worker stopped' }
        $busy=@(Get-ScheduledTask | Where-Object { $_.TaskName -like 'BuildTracker*' -and $_.State -eq 'Running' })
        if ($busy.Count -gt 0) { Stop-OwnWorker; throw 'Sibling workload started; own worker stopped' }
        $os=Get-CimInstance Win32_OperatingSystem
        $ownWorkerIds=@($worker.Id)+@(Get-CimInstance Win32_Process -Filter ('ParentProcessId = '+$worker.Id) | Select-Object -ExpandProperty ProcessId)
        $ownWorkers=@(Get-Process -Id $ownWorkerIds -ErrorAction SilentlyContinue)
        $memoryMarks+=@{utc=(Get-Date).ToUniversalTime().ToString('o');available_host_bytes=[long]$os.FreePhysicalMemory*1024;own_worker_working_bytes=($ownWorkers | Measure-Object -Property WorkingSet64 -Sum).Sum;own_worker_private_bytes=($ownWorkers | Measure-Object -Property PrivateMemorySize64 -Sum).Sum}
        @{start_gate_GiB=$minimumFreeGiB;stop_gate_GiB=$liveFloorGiB;marks=$memoryMarks} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $runDir 'resource_marks.json') -Encoding UTF8
        if ($os.FreePhysicalMemory/1MB -lt $liveFloorGiB) { Stop-OwnWorker; throw 'Memory pressure; own worker stopped' }
        Start-Sleep -Seconds $(if($Stage -in @('ordinary-feasibility-18','ordinary-feasibility-19')) { 1 } else { 10 })
        $worker.Refresh()
    }
    $worker.WaitForExit()
    if ($worker.ExitCode -ne 0) { throw ('Worker failed: '+$worker.ExitCode+'; see worker.err') }
    if (-not (Test-Path -LiteralPath $completion)) { throw 'No completed artifact' }
    Set-State 'completed' ('artifact='+$completion)
} catch {
    Set-State 'failed' $_.Exception.Message
    throw
} finally {
    Stop-OwnWorker
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
}
