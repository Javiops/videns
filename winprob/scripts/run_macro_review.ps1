$ErrorActionPreference='Stop'
$taskName='MacroCoach Winprob Review'
$researchRoot=Split-Path -Parent $PSScriptRoot
$macroRoot=Join-Path (Split-Path -Parent $researchRoot) 'macro_coach'
$logs=Join-Path $researchRoot 'data\logs'
New-Item -ItemType Directory -Path $logs -Force | Out-Null
try {
    $server=Start-Process -FilePath (Join-Path $macroRoot '.venv\Scripts\python.exe') -ArgumentList @('-m','uvicorn','scripts.review_app:app','--host','127.0.0.1','--port','8011') -WorkingDirectory $macroRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logs 'macro_wp_review.log') -RedirectStandardError (Join-Path $logs 'macro_wp_review.err')
    $serverHandle=$server.Handle
    $server.WaitForExit()
} finally {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
}
