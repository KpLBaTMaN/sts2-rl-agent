@echo off
cd /d "%~dp0..\.."
".venv\Scripts\python.exe" scripts\train_combat.py --total-timesteps 500000 --n-envs 4 --device cpu --output-dir output\combat_ppo
pause
