param(
    [string]$TaskName = 'Videns Winprob W1 Pilot',
    [string]$Version = 'w1-pilot-20260929-v1',
    [string]$ManifestVersion = 'w1-admission-20260929-v2'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$runDir = Join-Path $root ('data\runs\' + $Version)
$manifest = Join-Path $root ('artifacts\admission\' + $ManifestVersion + '\admission_manifest.json')
if (Test-Path -LiteralPath $runDir) { throw 'Pilot output already exists; never overwrite a run' }
New-Item -ItemType Directory -Path $runDir -Force | Out-Null
$log = Join-Path $runDir 'launcher.log'
$deadline = (Get-Date).AddMinutes(90)
$worker = $null
function Stop-OwnWorker {
    if ($null -ne $worker -and -not $worker.HasExited) {
        & taskkill.exe /PID $worker.Id /T /F | Out-Null
    }
}
function Write-State([string]$state, [string]$reason) {
    $entry = @{ status=$state; reason=$reason; utc=(Get-Date).ToUniversalTime().ToString('o') }
    $entry | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runDir 'launcher_status.json') -Encoding UTF8
    Add-Content -LiteralPath $log -Value ($entry.utc + ' ' + $state + ' ' + $reason)
}
try {
    while ($true) {
        if ((Get-Date) -gt $deadline) { throw '90 minute launch/run deadline reached' }
        $busy = @(Get-ScheduledTask | Where-Object { $_.TaskName -like 'BuildTracker*' -and $_.State -eq 'Running' })
        $game = @(Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue)
        $os = Get-CimInstance Win32_OperatingSystem
        $availableGiB = $os.FreePhysicalMemory / 1MB
        if ($busy.Count -eq 0 -and $game.Count -eq 0 -and $availableGiB -ge 2.5) { break }
        Write-State 'waiting_for_resources' ('runningSiblingJobs=' + $busy.Count + '; games=' + $game.Count + '; freeGiB=' + [math]::Round($availableGiB,2))
        Start-Sleep -Seconds 30
    }
    $env:OMP_NUM_THREADS='1'
    $env:OPENBLAS_NUM_THREADS='1'
    $env:MKL_NUM_THREADS='1'
    Write-State 'running' 'single CPU thread; BelowNormal; no GPU'
    $python = Join-Path $root '.venv\Scripts\python.exe'
    $script = Join-Path $root 'scripts\run_pilot.py'
    $arguments = @('-u', ('"' + $script + '"'), '--manifest', ('"' + $manifest + '"'), '--output', ('"' + $runDir + '"'), '--max-games', '6000')
    $worker = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runDir 'worker.log') -RedirectStandardError (Join-Path $runDir 'worker.err')
    $workerHandle = $worker.Handle
    $worker.PriorityClass = 'BelowNormal'
    while (-not $worker.HasExited) {
        if ((Get-Date) -gt $deadline) { Stop-OwnWorker; throw '90 minute deadline; stopped own pilot worker tree' }
        if (@(Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue).Count -gt 0) { Stop-OwnWorker; throw 'Game started; stopped own pilot worker tree' }
        $newBusy = @(Get-ScheduledTask | Where-Object { $_.TaskName -like 'BuildTracker*' -and $_.State -eq 'Running' })
        if ($newBusy.Count -gt 0) { Stop-OwnWorker; throw 'Sibling job started; stopped own pilot worker tree' }
        $os = Get-CimInstance Win32_OperatingSystem
        if ($os.FreePhysicalMemory / 1MB -lt 0.6) { Stop-OwnWorker; throw 'Memory pressure; stopped own pilot worker tree' }
        Start-Sleep -Seconds 15
        $worker.Refresh()
    }
    $worker.WaitForExit()
    if ($worker.ExitCode -ne 0) { throw ('Worker failed; see worker.err; exit=' + $worker.ExitCode) }
    if (-not (Test-Path -LiteralPath (Join-Path $runDir 'experiment\report.json'))) { throw 'Worker did not publish a report' }
    Write-State 'completed' 'development pilot only; practical benefit still requires validation'
} catch {
    Write-State 'failed' $_.Exception.Message
    throw
} finally {
    Stop-OwnWorker
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
}
