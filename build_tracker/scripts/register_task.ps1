# Registers a one-shot BuildTracker task for a launcher under scripts/.
# The launcher is responsible for unregistering the task when it finishes, so
# nothing lingers in the scheduler (docs/history.md records the near miss caused
# by a task that outlived its run).
#
# -ScriptName is the launcher path relative to the repository root. It is
# verified to exist AND the registered action is read back and compared, because
# a helper that silently registers the wrong script costs a whole run.
param(
  [Parameter(Mandatory=$true)][string]$TaskName,
  [Parameter(Mandatory=$true)][string]$ScriptName,
  [int]$StartInMinutes=2
)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$script=Join-Path $root $ScriptName
if (-not (Test-Path -LiteralPath $script)) { throw "missing launcher: $script" }

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}
$action = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument ('-NoProfile -ExecutionPolicy Bypass -File "' + $script + '"') `
    -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Once -At ((Get-Date).AddMinutes($StartInMinutes))
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit ([TimeSpan]::Zero)
$settings.DisallowStartIfOnBatteries = $false
$settings.StopIfGoingOnBatteries = $false
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal | Out-Null

$t = Get-ScheduledTask -TaskName $TaskName
$registered = $t.Actions[0].Arguments
if ($registered -notlike ('*' + [IO.Path]::GetFileName($script) + '*')) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    throw "registered action does not point at $ScriptName : $registered"
}
Write-Output ("registered: " + $t.TaskName + " state=" + $t.State)
Write-Output ("action:     " + $registered)
Write-Output ("starts at:  " + $t.Triggers[0].StartBoundary)
