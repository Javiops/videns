param([Parameter(Mandatory=$true)][string]$Version,[Parameter(Mandatory=$true)][string]$PredictionVersion)
$ErrorActionPreference='Stop'
if($Version -notmatch '^[A-Za-z0-9_.-]+$' -or $PredictionVersion -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Unsafe version' }
$root=Split-Path -Parent $PSScriptRoot
$sourceRoot=Join-Path (Split-Path -Parent $root) 'build_tracker'
$taskName='Videns Causal Student Assessment v1'
if(Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) { throw 'Existing task; inspect rather than replace' }
if(@(Get-ScheduledTask | Where-Object { ($_.TaskName -like 'BuildTracker*' -or $_.TaskName -like 'Videns*') -and $_.State -eq 'Running' }).Count) { throw 'Another protected task is active' }
if((Test-Path -LiteralPath (Join-Path $root ('data\runs\'+$Version))) -or (Test-Path -LiteralPath (Join-Path $sourceRoot ('data\exports\winprob-causal-student\'+$Version)))) { throw 'Existing immutable run' }
$fit=Join-Path $root ('data\runs\'+$PredictionVersion+'\experiment')
foreach($leaf in @('report.json','distribution\manifest.json')) {
    $manifest=Join-Path $fit $leaf
    if(-not(Test-Path -LiteralPath $manifest) -or (Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json).state -ne 'complete') { throw 'Complete student prediction required first' }
}
$script=Join-Path $root 'scripts\run_causal_student_assessment_job_v1.ps1'
$taskArguments='-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "'+$script+'" -Version '+$Version+' -PredictionVersion '+$PredictionVersion
$action=New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $taskArguments -WorkingDirectory $root
$settings=New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 340)
$settings.DisallowStartIfOnBatteries=$false;$settings.StopIfGoingOnBatteries=$false
$principal=New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $taskName -Action $action -Settings $settings -Principal $principal | Out-Null
if((Get-ScheduledTask -TaskName $taskName).Actions[0].Arguments -ne $taskArguments) { throw 'Registered action mismatch' }
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State | ConvertTo-Json -Compress
