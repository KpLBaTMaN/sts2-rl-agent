@echo off
cd /d "%~dp0..\.."
if "%~1"=="" (
  ".venv\Scripts\python.exe" -u -m sts2_env.watch.player --config local\config\watch.json
) else (
  ".venv\Scripts\python.exe" -u -m sts2_env.watch.player --config "%~1"
)
pause
