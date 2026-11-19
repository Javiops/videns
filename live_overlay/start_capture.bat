@echo off
rem Graba la proxima partida para la landing (seleccion de campeones y partida, con capturas). F9 marca un momento.
cd /d %~dp0
.venv\Scripts\python.exe scripts\capture.py
pause
