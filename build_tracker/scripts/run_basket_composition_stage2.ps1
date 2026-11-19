param([string]$TaskName='BuildTracker Basket Composition Stage2')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$env:PYTHONUTF8='1'
$log=Join-Path $root 'data\experiments\basket-composition-v1-stage2-launcher.log'
$code=1
try {
    & (Join-Path $root '.venv\Scripts\python.exe') scripts/run_basket_composition_stage2.py *>> $log
    $code=$LASTEXITCODE
} finally {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
}
exit $code
