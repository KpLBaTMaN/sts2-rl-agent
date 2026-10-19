@echo off
cd /d "%~dp0..\.."
echo Open http://127.0.0.1:6006 to view training logs.
".venv\Scripts\python.exe" -m tensorboard.main --logdir output\experiments --host 127.0.0.1 --port 6006
pause
