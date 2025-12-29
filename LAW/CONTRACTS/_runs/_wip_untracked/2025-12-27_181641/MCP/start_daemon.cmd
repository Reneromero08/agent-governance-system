@echo off
REM Start AGS MCP Server as a daemon (persistent background process)
REM This keeps the server running independently of your terminal

SET REPO_ROOT=%~dp0..
SET LOG_DIR=%REPO_ROOT%\CONTRACTS\_runs\mcp_logs
SET PID_FILE=%LOG_DIR%\server.pid

REM Create log directory
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM Check if already running
if exist "%PID_FILE%" (
    set /p PID=<"%PID_FILE%"
    tasklist /FI "PID eq !PID!" 2>NUL | find /I "python.exe" >NUL
    if !ERRORLEVEL! EQU 0 (
        echo [RUNNING] AGS MCP Server is already running (PID: !PID!)
        echo.
        echo To stop: taskkill /PID !PID! /F
        exit /b 0
    ) else (
        echo [INFO] PID file exists but process not found, cleaning up...
        del "%PID_FILE%"
    )
)

echo Starting AGS MCP Server as daemon...
echo.

REM Start in a new window that stays open
START "AGS MCP Server" /MIN python "%REPO_ROOT%\MCP\server.py"

REM Wait for process to start
timeout /t 2 /nobreak >NUL

REM Find the Python process and save PID
for /f "tokens=2" %%i in ('tasklist /FI "WINDOWTITLE eq AGS MCP Server" /FO LIST ^| find "PID:"') do (
    echo %%i > "%PID_FILE%"
    echo [STARTED] AGS MCP Server (PID: %%i)
    echo [LOG] Output: %LOG_DIR%\server.log
    echo.
    echo Server is running in a minimized window.
    echo To stop: taskkill /PID %%i /F
)

echo.
pause
