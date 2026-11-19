@echo off
rem Para el overlay del todo: las ventanas y el servidor local (puerto 8020).
rem Cerrar las ventanas con la X deja el servidor en marcha a propósito, para que /faces siga funcionando.
cd /d "%~dp0"
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { ($_.Name -eq 'python.exe' -or $_.Name -eq 'pythonw.exe') -and $_.CommandLine -like ('*' + (Get-Location).Path + '\.venv*') -and ($_.CommandLine -like '*app.overlay*' -or $_.CommandLine -like '*uvicorn app.main:app*') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; Write-Output ('parado ' + $_.ProcessId + ' ' + $_.Name) }"
