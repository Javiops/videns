param([Parameter(Mandatory=$true)][string]$RunDirectory,[Parameter(Mandatory=$true)][string]$Version)
$ErrorActionPreference='Stop'
if($Version -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Unsafe version' }
$root=Split-Path -Parent $PSScriptRoot
$source=Join-Path (Split-Path -Parent $root) 'build_tracker'
$exports=@()
foreach($patch in @('16.17','16.18')) {
    $output=Join-Path $source ('data\exports\winprob-fit-trajectories\'+$patch+'-'+$Version)
    if(Test-Path -LiteralPath $output) { throw ('Existing immutable export: '+$output) }
    $args=@('-u',('"'+(Join-Path $source 'scripts\export_winprob_fit_trajectories_v2.py')+'"'),'--patch',$patch,'--full','--limit','0','--output',('"'+$output+'"'))
    $child=Start-Process -FilePath (Join-Path $source '.venv\Scripts\python.exe') -ArgumentList $args -WorkingDirectory $source -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $RunDirectory ($patch+'.log')) -RedirectStandardError (Join-Path $RunDirectory ($patch+'.err'))
    $childHandle=$child.Handle
    $child.WaitForExit()
    if($child.ExitCode -ne 0) { throw ('FIT trajectory export failed for '+$patch+': '+$child.ExitCode) }
    $path=Join-Path $output 'manifest.json'
    $result=Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
    if($result.state -ne 'complete' -or $result.schema -ne 'winprob-fit-trajectories-v2') { throw 'Incomplete FIT trajectory artifact' }
    $exports+=@{patch=$patch;directory=$output;manifest_sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLower();eligible_games=$result.eligible_games;counts=$result.counts}
}
@{state='complete';exports=$exports;utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $RunDirectory 'trajectory_summary.json') -Encoding UTF8
