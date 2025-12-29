@echo off
REM Simple MCP Server Launcher
REM Starts the server in the current terminal window

cd /d "%~dp0.."
echo Starting AGS MCP Server...
echo Press Ctrl+C to stop
echo.

python MCP\server.py
