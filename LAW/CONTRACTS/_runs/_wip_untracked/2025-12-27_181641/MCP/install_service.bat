@echo off
REM Install AGS MCP Server as a Windows Service using NSSM
REM Download NSSM from https://nssm.cc/download

SET REPO_ROOT=%~dp0..
SET PYTHON_PATH=python
SET MCP_SERVER=%REPO_ROOT%\MCP\server.py

echo Installing AGS MCP Server as Windows Service...
echo.

REM Check if NSSM is installed
where nssm >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: NSSM not found in PATH
    echo.
    echo Please install NSSM:
    echo   1. Download from https://nssm.cc/download
    echo   2. Extract nssm.exe to a folder
    echo   3. Add that folder to your PATH
    echo.
    pause
    exit /b 1
)

REM Install the service
nssm install AGS-MCP-Server "%PYTHON_PATH%" "%MCP_SERVER%"
nssm set AGS-MCP-Server AppDirectory "%REPO_ROOT%"
nssm set AGS-MCP-Server DisplayName "AGS MCP Server"
nssm set AGS-MCP-Server Description "Model Context Protocol server for Agent Governance System - provides persistent coordination for multiple AI agents and IDE extensions"
nssm set AGS-MCP-Server Start SERVICE_AUTO_START
nssm set AGS-MCP-Server AppStdout "%REPO_ROOT%\CONTRACTS\_runs\mcp_logs\service_stdout.log"
nssm set AGS-MCP-Server AppStderr "%REPO_ROOT%\CONTRACTS\_runs\mcp_logs\service_stderr.log"
nssm set AGS-MCP-Server AppRotateFiles 1
nssm set AGS-MCP-Server AppRotateOnline 1
nssm set AGS-MCP-Server AppRotateBytes 10485760

echo.
echo Service installed successfully!
echo.
echo To start the service:
echo   net start AGS-MCP-Server
echo.
echo To stop the service:
echo   net stop AGS-MCP-Server
echo.
echo To uninstall the service:
echo   nssm remove AGS-MCP-Server confirm
echo.
pause
