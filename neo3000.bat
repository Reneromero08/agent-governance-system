@echo off
title NEO3000 KERNEL // BOOT SEQUENCE
echo [SYSTEM] INITIALIZING NEO3000 CORE...

:: Check for virtual environment
if exist ".venv\Scripts\activate.bat" (
    echo [SYSTEM] ACTIVATING NEURAL VENV...
    call .venv\Scripts\activate.bat
) else (
    echo [WARN] .venv NOT DETECTED. ATTEMPTING GLOBAL PYTHON...
)

:: Start the browser in a separate process
echo [SYSTEM] LAUNCHING OPERATOR DECK...
start http://localhost:8000

:: Start the server
python TOOLS\neo3000\server.py

pause
