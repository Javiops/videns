param([Parameter(Mandatory=$true)][string]$Version)
$ErrorActionPreference='Stop'
if($Version -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Unsafe version' }
$root=Split-Path -Parent $PSScriptRoot
$taskName='Videns Recipe Fit v3'
if(Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) { throw 'Existing task; inspect rather than replace' }
if(@(Get-ScheduledTask | Where-Object { ($_.TaskName -like 'BuildTracker*' -or $_.TaskName -like 'Videns*') -and $_.State -eq 'Running' }).Count) { throw 'Another protected task is active; register only after it finishes' }
if(Test-Path -LiteralPath (Join-Path $root ('data\runs\'+$Version))) { throw 'Existing immutable run' }
$source=Join-Path (Split-Path -Parent $root) 'build_tracker\data\exports\winprob-recipe-transition\recipe-m2-release-20261001-v2\manifest.json'
if(-not(Test-Path -LiteralPath $source)) { throw 'Native M2 release not complete' }
$script=Join-Path $root 'scripts\run_recipe_fit_job_v3.ps1'
$taskArguments='-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "'+$script+'" -Version '+$Version
$action=New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $taskArguments -WorkingDirectory $root
$settings=New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 65)
$settings.DisallowStartIfOnBatteries=$false;$settings.StopIfGoingOnBatteries=$false
$principal=New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $taskName -Action $action -Settings $settings -Principal $principal | Out-Null
if((Get-ScheduledTask -TaskName $taskName).Actions[0].Arguments -ne $taskArguments) { throw 'Registered action mismatch' }
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State | ConvertTo-Json -Compress
