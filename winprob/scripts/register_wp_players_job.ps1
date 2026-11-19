param(
    [Parameter(Mandatory=$true)][ValidateSet('train','audit')][string]$Mode,
    [Parameter(Mandatory=$true)][string]$Version,
    [string]$ExperimentVersion='wp-players-fit-20260930-v1'
)
$ErrorActionPreference='Stop'
if($Version -notmatch '^[A-Za-z0-9_.-]+$' -or $ExperimentVersion -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Unsafe version' }
$root=Split-Path -Parent $PSScriptRoot
$taskName='Videns Player WP '+$Mode
if(Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) { throw 'Existing player-WP task; inspect rather than replace' }
$script=Join-Path $root 'scripts\run_wp_players_job.ps1'
$taskArguments='-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "'+$script+'" -Mode '+$Mode+' -Version '+$Version+' -ExperimentVersion '+$ExperimentVersion
$action=New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $taskArguments -WorkingDirectory $root
$trigger=New-ScheduledTaskTrigger -Once -At ((Get-Date).AddMinutes(1))
$settings=New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes $(if($Mode -eq 'train') {125} else {65}))
$settings.DisallowStartIfOnBatteries=$false;$settings.StopIfGoingOnBatteries=$false
$principal=New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal | Out-Null
if((Get-ScheduledTask -TaskName $taskName).Actions[0].Arguments -ne $taskArguments) { throw 'Registered player-WP action mismatch' }
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State | ConvertTo-Json -Compress
