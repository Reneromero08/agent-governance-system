@echo off
setlocal
cd /d "%~dp0"

REM One-click packer. Output goes to MEMORY/LLM-PACKER-1.0/_packs/
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0pack.ps1" %*

echo.
echo Done.
pause
