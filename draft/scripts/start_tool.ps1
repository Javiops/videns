# Starts the local draft tool (http://127.0.0.1:8030) detached.
$root = Split-Path -Parent $PSScriptRoot
$p = Start-Process -FilePath "$root\.venv\Scripts\python.exe" -ArgumentList "scripts\tool_server.py" -WorkingDirectory $root `
    -RedirectStandardOutput "$root\data\logs\tool.out" -RedirectStandardError "$root\data\logs\tool.err" -WindowStyle Hidden -PassThru
"draft tool pid $($p.Id) -> http://127.0.0.1:8030"
