@echo off
rem Ventana del overlay (sin consola). League en modo Sin bordes o Ventana.
cd /d "%~dp0"
start "" ".venv\Scripts\pythonw.exe" -m app.overlay
