# Patch 16.19 ladder pull, KR + EUW (owner: "quieres pullear games en paralelo... done you can pull", 2026-09-28).
# Runs the runbook command (app.ingest --ladder --region both) for the reviewed patch 16.19 as a detached, low-priority
# task beside the transfer training (network + CPU + SQLite only; no GPU). The Riot dev key expires every 24 h; if
# Riot rejects the key the pull is retried every 10 minutes for up to 8 h so a key pasted later still gets used.
# Self-unregistering one-shot (register_task.ps1). The collector lease (data/riot_collector.lock) keeps the 04:30
# daily pull from racing this sequence.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$taskName = 'BuildTracker Ladder Pull 1619'
$ts = Get-Date -Format yyyyMMdd_HHmmss
New-Item -ItemType Directory -Force (Join-Path $root 'data\logs') | Out-Null
$py = Join-Path $root '.venv\Scripts\python.exe'
$log = Join-Path $root ('data\logs\ladder_pull_1619_' + $ts + '.log')
$env:PYTHONUTF8 = '1'; $env:PYTHONUNBUFFERED = '1'
function Log($msg) { Add-Content -Path $log -Value ((Get-Date -Format s) + ' ' + $msg) }
$code = 1
try {
    $deadline = (Get-Date).AddHours(8)
    $attempt = 0
    while ($true) {
        $attempt += 1
        $o = Join-Path $root ('data\logs\ladder_pull_1619_' + $ts + '_' + $attempt + '.out')
        $e = Join-Path $root ('data\logs\ladder_pull_1619_' + $ts + '_' + $attempt + '.err')
        Log ('start attempt ' + $attempt + ': app.ingest --ladder --region both --patch 16.19')
        $p = Start-Process -FilePath $py -ArgumentList @('-m', 'app.ingest', '--ladder', '--region', 'both', '--patch', '16.19') `
            -WorkingDirectory $root -NoNewWindow -PassThru -Wait -RedirectStandardOutput $o -RedirectStandardError $e
        $null = $p.Handle
        $code = [int]$p.ExitCode
        Log ('attempt ' + $attempt + ' exit ' + $code)
        $rejected = $false
        foreach ($f in @($o, $e)) { if ((Test-Path -LiteralPath $f) -and (Select-String -Path $f -Pattern 'rejected the API key|401|403' -Quiet)) { $rejected = $true } }
        if ($code -eq 0 -or -not $rejected) { break }
        if ((Get-Date) -gt $deadline) { Log 'API key still rejected after 8 h; giving up'; break }
        Log 'API key rejected; retrying in 10 min (paste a fresh RIOT_API_KEY into build_tracker\.env)'
        Start-Sleep -Seconds 600
    }
}
finally {
    try { Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue } catch {}
}
exit $code
