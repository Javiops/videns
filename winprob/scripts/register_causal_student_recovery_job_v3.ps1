param([Parameter(Mandatory=$true)][string]$Version)
$ErrorActionPreference='Stop'
if($Version -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Unsafe version' }
$root=Split-Path -Parent $PSScriptRoot
$taskName='Videns Causal Student Recovery v3'
if(Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) { throw 'Existing task; inspect rather than replace' }
if(@(Get-ScheduledTask | Where-Object { ($_.TaskName -like 'BuildTracker*' -or $_.TaskName -like 'Videns*') -and $_.State -eq 'Running' }).Count) { throw 'Another protected task is active; register only after it finishes' }
if(Test-Path -LiteralPath (Join-Path $root ('data\runs\'+$Version))) { throw 'Existing immutable run' }
$originalProcess=@(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match 'train_causal_student_v1.py' })
if($originalProcess.Count) { throw 'Original training command still exists' }
$original=Join-Path $root 'data\runs\causal-student-train-20261004-v1\experiment'
if(@(Get-ChildItem -LiteralPath $original -Filter '*.pt').Count -ne 5 -or (Test-Path -LiteralPath (Join-Path $original 'models_lock.json'))) { throw 'Expected interrupted five-model family required' }
$script=Join-Path $root 'scripts\run_causal_student_recovery_job_v3.ps1'
$taskArguments='-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "'+$script+'" -Version '+$Version
$action=New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $taskArguments -WorkingDirectory $root
$settings=New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 395)
$settings.DisallowStartIfOnBatteries=$false;$settings.StopIfGoingOnBatteries=$false
$principal=New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $taskName -Action $action -Settings $settings -Principal $principal | Out-Null
if((Get-ScheduledTask -TaskName $taskName).Actions[0].Arguments -ne $taskArguments) { throw 'Registered action mismatch' }
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State | ConvertTo-Json -Compress









