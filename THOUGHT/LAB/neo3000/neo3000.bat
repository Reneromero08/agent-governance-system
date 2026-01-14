@echo off
title NEO3000 KERNEL // BOOT SEQUENCE
echo [SYSTEM] INITIALIZING NEO3000 CORE...

:: Navigate to NEO3000 directory
cd /d "%~dp0"

:: Check for virtual environment
if exist "..\..\..\..\.venv\Scripts\activate.bat" (
    echo [SYSTEM] ACTIVATING NEURAL VENV...
    call ..\..\..\..\.venv\Scripts\activate.bat
) else (
    echo [WARN] .venv NOT DETECTED. ATTEMPTING GLOBAL PYTHON...
)

:: Start model server in background (Port 8421)
echo [SYSTEM] STARTING MODEL SERVER (Port 8421)...
start "NEO3000 Model Server" /MIN python model_server.py

:: Wait for model server to load
echo [SYSTEM] WAITING FOR MODEL SERVER TO LOAD...
timeout /t 5 /nobreak >nul

:: Start feral server (Port 8420)
echo [SYSTEM] STARTING FERAL DASHBOARD (Port 8420)...
echo [SYSTEM] LAUNCHING OPERATOR DECK...
start http://localhost:8420

:: Run feral server (this will show the TUI)
python feral_server.py

pause
