# Registers and starts a one-shot task running run_reconstructible_1619.ps1 with explicit parameters
# (task name, acceptance directory, corpus directory), so a smoke chain and the full 16.17 history chain
# can coexist. The launcher unregisters the task when it finishes (register_task.ps1 pattern: the
# registered action is read back and compared before anything starts).
param(
  [Parameter(Mandatory=$true)][string]$TaskName,
  [Parameter(Mandatory=$true)][string]$Base,
  [Parameter(Mandatory=$true)][string]$Corpus,
  [int]$MaxWallSeconds=21600,
  [int]$StartInMinutes=1
)
$ErrorActionPreference='Stop'
$Base = $Base.TrimEnd('\', '/'); $Corpus = $Corpus.TrimEnd('\', '/')    # a trailing separator would escape the closing quote of -File arguments
$root=Split-Path -Parent $PSScriptRoot
$script=Join-Path $root 'scripts\run_reconstructible_1619.ps1'
if (-not (Test-Path -LiteralPath $script)) { throw "missing launcher: $script" }
if (-not (Test-Path -LiteralPath (Join-Path $Base 'source_plan.json'))) { throw "no frozen source plan under $Base" }
if (Test-Path -LiteralPath $Corpus) { throw "corpus directory already exists: $Corpus" }
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    throw "task already registered: $TaskName"
}
$argument = ('-NoProfile -ExecutionPolicy Bypass -File "' + $script + '" -TaskName "' + $TaskName + '" -Base "' + $Base +
             '" -Corpus "' + $Corpus + '" -MaxWallSeconds ' + $MaxWallSeconds)
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $argument -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Once -At ((Get-Date).AddMinutes($StartInMinutes))
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero)
$settings.DisallowStartIfOnBatteries = $false
$settings.StopIfGoingOnBatteries = $false
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal | Out-Null
$t = Get-ScheduledTask -TaskName $TaskName
$registered = $t.Actions[0].Arguments
if ($registered -ne $argument) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    throw "registered action differs from the requested one: $registered"
}
Write-Output ("registered: " + $t.TaskName + " state=" + $t.State)
Write-Output ("action:     " + $registered)
Write-Output ("starts at:  " + $t.Triggers[0].StartBoundary)
