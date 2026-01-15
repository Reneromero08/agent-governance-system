@echo off
title FERAL DASHBOARD // BOOT SEQUENCE
echo [SYSTEM] INITIALIZING FERAL DASHBOARD...

:: Navigate to dashboard directory (inside FERAL_RESIDENT)
cd /d "%~dp0"

:: Check for virtual environment (5 levels up: dashboard -> FERAL_RESIDENT -> LAB -> THOUGHT -> repo)
if exist "..\..\..\..\..\venv\Scripts\activate.bat" (
    echo [SYSTEM] ACTIVATING VENV...
    call ..\..\..\..\..\venv\Scripts\activate.bat
) else if exist "..\..\..\..\..\.venv\Scripts\activate.bat" (
    echo [SYSTEM] ACTIVATING .VENV...
    call ..\..\..\..\..\.venv\Scripts\activate.bat
) else (
    echo [WARN] VENV NOT DETECTED. ATTEMPTING GLOBAL PYTHON...
)

:: Start model server in background (Port 8421)
echo [SYSTEM] STARTING MODEL SERVER (Port 8421)...
start "Feral Model Server" /MIN python model_server.py

:: Wait for model server to load
echo [SYSTEM] WAITING FOR MODEL SERVER TO LOAD...
timeout /t 5 /nobreak >nul

:: Start dashboard server (Port 8420)
echo [SYSTEM] STARTING FERAL DASHBOARD (Port 8420)...
echo [SYSTEM] LAUNCHING OPERATOR DECK...
start http://localhost:8420

:: Run dashboard server (this will show the TUI)
python server.py

pause
