param(
    [Parameter(Mandatory=$true)][string]$RunDirectory,
    [Parameter(Mandatory=$true)][string]$Version
)
$ErrorActionPreference='Stop'
if ($Version -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Unsafe version' }
$root=Split-Path -Parent $PSScriptRoot
$source=Join-Path (Split-Path -Parent $root) 'build_tracker'
$census=Join-Path $root 'data\runs\undo-pairs-20260930-v2\experiment'
$report=Get-Content -LiteralPath (Join-Path $census 'report.json') -Raw | ConvertFrom-Json
if ($report.schema -ne 'undo-pairs-support-v2' -or $report.outcome_targets_extracted -ne $false) { throw 'Wrong census contract' }
$exports=@()
foreach ($patch in @('16.17','16.18','16.19')) {
    $output=Join-Path $source ('data\exports\winprob-revision-contexts\'+$patch+'-'+$Version)
    if (Test-Path -LiteralPath $output) { throw ('Existing immutable export: '+$output) }
    $arguments=@('-u',('"'+(Join-Path $source 'scripts\export_winprob_revision_contexts_v3.py')+'"'),'--census',('"'+$census+'"'),'--patch',$patch,'--full','--limit','0','--output',('"'+$output+'"'))
    if ($patch -eq '16.17') { $arguments+=@('--role','fit') }
    elseif ($patch -eq '16.18') { $arguments+=@('--role','fit','--role','calibration') }
    else { $arguments+=@('--role','development') }
    $child=Start-Process -FilePath (Join-Path $source '.venv\Scripts\python.exe') -ArgumentList $arguments -WorkingDirectory $source -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $RunDirectory ($patch+'.log')) -RedirectStandardError (Join-Path $RunDirectory ($patch+'.err'))
    $childHandle=$child.Handle
    $child.WaitForExit()
    if ($child.ExitCode -ne 0) { throw ('Native patch '+$patch+' failed: '+$child.ExitCode) }
    $path=Join-Path $output 'report.json'
    $result=Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
    if ($result.state -ne 'complete') { throw ('Patch export incomplete: '+$patch) }
    $exports+=@{patch=$patch;directory=$output;report_sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLower();pairs_examined=$result.pairs_examined;status_counts=$result.status_counts}
}
@{state='complete';version=$Version;exports=$exports;utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $RunDirectory 'alignment_summary.json') -Encoding UTF8
