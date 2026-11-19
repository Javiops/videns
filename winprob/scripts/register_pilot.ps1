param(
    [Parameter(Mandatory=$true)][string]$Version,
    [string]$ManifestVersion='w1-admission-20260929-v2'
)
$ErrorActionPreference='Stop'
if ($Version -notmatch '^[A-Za-z0-9_.-]+$' -or $ManifestVersion -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Unsafe version' }
$taskName='Videns Winprob W1 Pilot'
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) { throw 'Existing pilot task; never replace' }
$root = Split-Path -Parent $PSScriptRoot
$manifest = Join-Path $root ('artifacts\admission\' + $ManifestVersion + '\admission_manifest.json')
if (-not (Test-Path -LiteralPath $manifest)) { throw 'Missing admission manifest' }
$script = Join-Path $root 'scripts\run_pilot.ps1'
$arguments = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $script + '" -Version ' + $Version + ' -ManifestVersion ' + $ManifestVersion
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arguments -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Once -At ((Get-Date).AddMinutes(1))
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 2)
$settings.DisallowStartIfOnBatteries = $false
$settings.StopIfGoingOnBatteries = $false
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal | Out-Null
$registered = Get-ScheduledTask -TaskName $taskName
if ($registered.Actions[0].Arguments -ne $arguments) { throw 'Registered action mismatch' }
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State
