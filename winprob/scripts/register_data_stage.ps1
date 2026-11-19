param(
    [Parameter(Mandatory=$true)][ValidateSet('revision-contexts','trajectory-smoke','undo-expanded','revision-align-full','revision-score-full','ipo-fit','iql-trajectories','iql-cache','ordinary-contexts','ordinary-actor','ordinary-reference','ordinary-native19','ordinary-evaluate','ordinary-feasibility-18','ordinary-feasibility-19','goal-census','commitment-census','wp-distill','wp-distill-audit','current-schema','current-combat-cache','current-combat-fit','wp-scale-bind','wp-scale-cache','wp-scale-cache-audit','wp-scale-fit','wp-scale-audit','wp-players-cache','wp-players-cache-audit')][string]$Stage,
    [Parameter(Mandatory=$true)][string]$Version
)
$ErrorActionPreference='Stop'
if ($Version -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Unsafe version' }
$root=Split-Path -Parent $PSScriptRoot
$taskName='Videns Data '+$Stage
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) { throw 'Existing task; inspect rather than replace' }
$script=Join-Path $root 'scripts\run_data_stage.ps1'
$arguments='-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "'+$script+'" -Stage '+$Stage+' -Version '+$Version
$action=New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arguments -WorkingDirectory $root
$trigger=New-ScheduledTaskTrigger -Once -At ((Get-Date).AddMinutes(1))
$settings=New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes $(if($Stage -in @('wp-scale-cache','wp-scale-cache-audit','wp-scale-fit','wp-scale-audit','wp-players-cache','wp-players-cache-audit')) {60} elseif($Stage -in @('iql-cache','ordinary-reference')) {50} elseif($Stage -in @('iql-trajectories','current-combat-cache')) {45} else {30}))
$settings.DisallowStartIfOnBatteries=$false; $settings.StopIfGoingOnBatteries=$false
$principal=New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal | Out-Null
if ((Get-ScheduledTask -TaskName $taskName).Actions[0].Arguments -ne $arguments) { throw 'Registered action mismatch' }
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State | ConvertTo-Json -Compress
