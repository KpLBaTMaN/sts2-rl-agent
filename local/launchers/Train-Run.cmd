@echo off
cd /d "%~dp0..\.."
if "%~1"=="" (
  ".venv\Scripts\python.exe" -m sts2_env.training.runner --config local\config\training.json
) else (
  ".venv\Scripts\python.exe" -m sts2_env.training.runner --config "%~1"
)
pause
