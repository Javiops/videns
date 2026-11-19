param([Parameter(Mandatory=$true)][ValidateSet('preflight','train')][string]$Mode,
      [Parameter(Mandatory=$true)][string]$Version)
$ErrorActionPreference='Stop'
if($Version -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Unsafe version' }
$root=Split-Path -Parent $PSScriptRoot
$taskName='Videns IQL '+$Mode
if(Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) { throw 'Existing job; inspect rather than replace' }
$script=Join-Path $root 'scripts\run_iql_job.ps1'
$taskArguments='-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "'+$script+'" -Mode '+$Mode+' -Version '+$Version
$action=New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $taskArguments -WorkingDirectory $root
$trigger=New-ScheduledTaskTrigger -Once -At ((Get-Date).AddMinutes(1))
$settings=New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes $(if($Mode -eq 'train') {100} else {15}))
$settings.DisallowStartIfOnBatteries=$false; $settings.StopIfGoingOnBatteries=$false
$principal=New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal | Out-Null
if((Get-ScheduledTask -TaskName $taskName).Actions[0].Arguments -ne $taskArguments) { throw 'Registered IQL action mismatch' }
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State | ConvertTo-Json -Compress
