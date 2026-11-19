param([Parameter(Mandatory=$true)][string]$Version)
$ErrorActionPreference='Stop'
if($Version -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Unsafe version' }
$root=Split-Path -Parent $PSScriptRoot
$run=Join-Path $root ('data\runs\'+$Version)
if(Test-Path -LiteralPath $run) { throw 'Existing runtime setup record' }
New-Item -ItemType Directory -Path $run | Out-Null
$child=$null
function State([string]$state,[string]$reason) {
    @{state=$state;reason=$reason;utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $run 'status.json') -Encoding UTF8
}
try {
    $uv=(Get-Command uv.exe).Source
    $python=Join-Path $root '.venv\Scripts\python.exe'
    $env:UV_CONCURRENT_DOWNLOADS='1'; $env:UV_CONCURRENT_INSTALLS='1'; $env:UV_CONCURRENT_BUILDS='1'
    State 'installing' 'Own winprob environment; pinned official PyTorch CUDA130 wheel; no source-runtime reuse'
    $args=@('pip','install','--python',('"'+$python+'"'),'--default-index','https://download.pytorch.org/whl/cu130','--only-binary',':all:','torch==2.13.0+cu130')
    $child=Start-Process -FilePath $uv -ArgumentList $args -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $run 'install.log') -RedirectStandardError (Join-Path $run 'install.err')
    $childHandle=$child.Handle
    $deadline=(Get-Date).AddMinutes(25)
    while(-not $child.HasExited) {
        if((Get-Date) -gt $deadline) { & taskkill.exe /PID $child.Id /T /F | Out-Null; throw 'Runtime installation deadline reached' }
        Start-Sleep -Seconds 5
        $child.Refresh()
    }
    $child.WaitForExit()
    if($child.ExitCode -ne 0) { throw ('Package installation failed: '+$child.ExitCode) }
    & $uv pip freeze --python $python | Set-Content -LiteralPath (Join-Path $run 'requirements.txt') -Encoding UTF8
    if($LASTEXITCODE -ne 0) { throw 'Could not record installed runtime' }
    State 'complete' 'Packages installed and version list recorded; CUDA execution is checked separately'
} catch { State 'failed' $_.Exception.Message; throw }
finally {
    if($null -ne $child -and -not $child.HasExited) { & taskkill.exe /PID $child.Id /T /F | Out-Null }
    Unregister-ScheduledTask -TaskName 'Winprob Runtime Setup' -Confirm:$false -ErrorAction SilentlyContinue
}
