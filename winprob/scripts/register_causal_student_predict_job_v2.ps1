param([Parameter(Mandatory=$true)][string]$Version,[Parameter(Mandatory=$true)][string]$TrainingVersion)
$ErrorActionPreference='Stop'
if($Version -notmatch '^[A-Za-z0-9_.-]+$' -or $TrainingVersion -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Unsafe version' }
$root=Split-Path -Parent $PSScriptRoot
$taskName='Videns Causal Student Predict v2'
if(Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) { throw 'Existing task; inspect rather than replace' }
if(@(Get-ScheduledTask | Where-Object { ($_.TaskName -like 'BuildTracker*' -or $_.TaskName -like 'Videns*') -and $_.State -eq 'Running' }).Count) { throw 'Another protected task is active; register only after it finishes' }
if(Test-Path -LiteralPath (Join-Path $root ('data\runs\'+$Version))) { throw 'Existing immutable run' }
$trainingDirectory=Join-Path $root ('data\runs\'+$TrainingVersion+'\experiment')
$statusPath=Join-Path (Split-Path -Parent $trainingDirectory) 'launcher_status.json'
if(-not(Test-Path -LiteralPath $statusPath) -or (Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json).state -ne 'failed') { throw 'Original training must be terminal before metadata-load recovery' }
if(-not(Test-Path -LiteralPath (Join-Path $trainingDirectory 'models_lock.json'))) { throw 'All six final models must be locked first' }
$script=Join-Path $root 'scripts\run_causal_student_predict_job_v2.ps1'
$taskArguments='-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "'+$script+'" -Version '+$Version+' -TrainingVersion '+$TrainingVersion
$action=New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $taskArguments -WorkingDirectory $root
$settings=New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 85)
$settings.DisallowStartIfOnBatteries=$false;$settings.StopIfGoingOnBatteries=$false
$principal=New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $taskName -Action $action -Settings $settings -Principal $principal | Out-Null
if((Get-ScheduledTask -TaskName $taskName).Actions[0].Arguments -ne $taskArguments) { throw 'Registered action mismatch' }
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State | ConvertTo-Json -Compress









