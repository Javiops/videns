param([Parameter(Mandatory=$true)][string]$Version)
$ErrorActionPreference='Stop'
if($Version -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Unsafe version' }
$root=Split-Path -Parent $PSScriptRoot
$taskName='Videns Scoreboard Export v1'
if(Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) { throw 'Existing task; inspect rather than replace' }
if(Test-Path -LiteralPath (Join-Path $root ('data\runs\'+$Version))) { throw 'Existing immutable run' }
$script=Join-Path $root 'scripts\run_scoreboard_export_job_v1.ps1'
$taskArguments='-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "'+$script+'" -Version '+$Version
$action=New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $taskArguments -WorkingDirectory $root
$settings=New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 35)
$settings.DisallowStartIfOnBatteries=$false;$settings.StopIfGoingOnBatteries=$false
$principal=New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $taskName -Action $action -Settings $settings -Principal $principal | Out-Null
if((Get-ScheduledTask -TaskName $taskName).Actions[0].Arguments -ne $taskArguments) { throw 'Registered action mismatch' }
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State | ConvertTo-Json -Compress
