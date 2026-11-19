# Runs T3 (blind) then the turn simulation sequentially, after v1b has finished.
$root = 'C:\Users\Javier\videns.gg\draft'
$py = "$root\.venv\Scripts\python.exe"
while (-not (Select-String -Path "$root\data\logs\run_v1b.out" -Pattern 'k50:C_noexp - frozen:C_noexp' -Quiet)) {
    if (Select-String -Path "$root\data\logs\run_v1b.err" -Pattern 'Traceback' -Quiet) { exit 1 }
    Start-Sleep -Seconds 5
}
foreach ($s in 'blind_v1', 'turns_v1') {
    $p = Start-Process -FilePath $py -ArgumentList "scripts\$s.py" -WorkingDirectory $root `
        -RedirectStandardOutput "$root\data\logs\$s.out" -RedirectStandardError "$root\data\logs\$s.err" `
        -WindowStyle Hidden -PassThru
    $p.PriorityClass = 'BelowNormal'
    $p.WaitForExit()
}
