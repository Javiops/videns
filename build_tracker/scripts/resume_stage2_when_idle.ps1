# Owner's instruction (2026-09-25 23:45): the stage-2 chain must not run while
# they play; resume it as soon as they stop. This detached watcher polls for
# the game process ("League of Legends.exe") and, once it has been absent for
# $IdleMinutes, arms run\resume.txt and registers the chain task (which then
# pauses itself if a game starts again -- run_stage2_v4.py). One-shot: exits
# after launching.
param([int]$IdleMinutes = 3)
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$log = Join-Path $root ('data\logs\resume_stage2_watch_' + (Get-Date -Format yyyyMMdd_HHmmss) + '.log')
function Log($m) { $line = (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + ' ' + $m; Add-Content -LiteralPath $log -Value $line }
Log "watching for the game to stop (idle threshold $IdleMinutes min)"
$idleSince = $null
while ($true) {
    $game = Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue
    $chain = Get-ScheduledTask -TaskName 'BuildTracker Stage2 V4' -ErrorAction SilentlyContinue
    if ($chain) { Log 'chain task already registered; exiting'; break }
    if ($game) { if ($idleSince) { Log 'game running again; idle timer reset' }; $idleSince = $null }
    elseif (-not $idleSince) { $idleSince = Get-Date; Log 'game not running; idle timer started' }
    elseif (((Get-Date) - $idleSince).TotalMinutes -ge $IdleMinutes) {
        Set-Content -LiteralPath (Join-Path $root 'data\experiments\16.18-stage2-v4\run\resume.txt') -Value 'resume' -Encoding ascii -NoNewline
        $out = & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root 'scripts\register_task.ps1') -TaskName 'BuildTracker Stage2 V4' -ScriptName 'scripts\run_stage2_v4.ps1' -StartInMinutes 1 2>&1
        Log ('chain relaunched with --resume: ' + ($out -join ' | '))
        break
    }
    Start-Sleep -Seconds 30
}
