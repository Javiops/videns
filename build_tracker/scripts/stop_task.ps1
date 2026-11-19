# Stops a BuildTracker one-shot task and unregisters it, then reports what is
# left running. Used when a launcher was registered against the wrong script.
# Only touches the named task and the processes that task started.
param([Parameter(Mandatory=$true)][string]$TaskName)
$ErrorActionPreference='Continue'

$t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if (-not $t) {
    Write-Output ("not registered: " + $TaskName)
} else {
    if ($t.State -eq 'Running') {
        Stop-ScheduledTask -TaskName $TaskName
        Start-Sleep -Seconds 3
    }
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Output ("stopped and unregistered: " + $TaskName)
}

# Stop-ScheduledTask does not always reap grandchildren; clear any python left
# running our sweep or profile scripts, and nothing else.
$ours = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'mask_slack_sweep_v1|miss_profile_v1|gold_error_shape_v1|budget_shortfall_shape_v1|run_gold_posterior_training_v1|train_prefix' }
foreach ($p in $ours) {
    Write-Output ("stopping pid " + $p.ProcessId)
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
}

Start-Sleep -Seconds 1
Write-Output "--- remaining BuildTracker tasks ---"
Get-ScheduledTask -TaskName 'BuildTracker*' | Select-Object TaskName,State | Format-Table -AutoSize | Out-String | Write-Output
$left = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Measure-Object
Write-Output ("python processes still running: " + $left.Count)
