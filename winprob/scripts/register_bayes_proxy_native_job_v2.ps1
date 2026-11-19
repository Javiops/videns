param([Parameter(Mandatory=$true)][string]$Version,[Parameter(Mandatory=$true)][string]$ForecastVersion)
$ErrorActionPreference='Stop'
if($Version -notmatch '^[A-Za-z0-9_.-]+$' -or $ForecastVersion -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Unsafe version' }
$root=Split-Path -Parent $PSScriptRoot
$taskName='Videns Bayes Proxy Native v2'
if(Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) { throw 'Existing task; inspect rather than replace' }
if(@(Get-ScheduledTask | Where-Object { ($_.TaskName -like 'BuildTracker*' -or $_.TaskName -like 'Videns*') -and $_.State -eq 'Running' }).Count) { throw 'Another protected task is active' }
if(Test-Path -LiteralPath (Join-Path $root ('data\runs\'+$Version))) { throw 'Existing immutable run' }
$manifest=Join-Path $root ('data\runs\'+$ForecastVersion+'\experiment\distribution\manifest.json')
if(-not(Test-Path -LiteralPath $manifest) -or (Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json).state -ne 'complete') { throw 'Complete fitted distribution required first' }
$script=Join-Path $root 'scripts\run_bayes_proxy_native_job_v2.ps1'
$taskArguments='-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "'+$script+'" -Version '+$Version+' -ForecastVersion '+$ForecastVersion
$action=New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $taskArguments -WorkingDirectory $root
$settings=New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 100)
$settings.DisallowStartIfOnBatteries=$false;$settings.StopIfGoingOnBatteries=$false
$principal=New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $taskName -Action $action -Settings $settings -Principal $principal | Out-Null
if((Get-ScheduledTask -TaskName $taskName).Actions[0].Arguments -ne $taskArguments) { throw 'Registered action mismatch' }
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State | ConvertTo-Json -Compress

