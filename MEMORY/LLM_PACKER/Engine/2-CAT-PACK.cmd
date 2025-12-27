@echo off
setlocal
cd /d "%~dp0"

REM One-click packer for CATALYTIC-DPT only.
REM Output goes to MEMORY/LLM_PACKER/_packs/

if "%~1"=="" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0pack.ps1" -Scope catalytic-dpt -Profile full -SplitLite -Combined -Zip
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0pack.ps1" -Scope catalytic-dpt %*
)

echo.
echo Done.
pause
