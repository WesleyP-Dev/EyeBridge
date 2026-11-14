@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (
  echo Execute instalar_windows.ps1 primeiro.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" main.py %*
if errorlevel 1 pause
