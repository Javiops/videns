# Registers the one-shot task that runs phases 1 and 2 of
# docs/experiment_mask_slack_v1.md. The launcher waits for the gap window pull
# to finish before it starts, and unregisters this task when it is done.
param(
  [string]$TaskName='BuildTracker Mask Slack Sweep 20260916',
  [int]$StartInMinutes=2
)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$script=Join-Path $root 'scripts\run_mask_slack_sweep.ps1'
if (-not (Test-Path -LiteralPath $script)) { throw "missing $script" }

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
Write-Output ("registered: " + $t.TaskName + " state=" + $t.State)
Write-Output ("starts at: " + $t.Triggers[0].StartBoundary)
