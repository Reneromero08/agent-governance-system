@echo off
setlocal
cd /d "%~dp0"

REM One-click packer for CATALYTIC-DPT only.
REM Output goes to MEMORY/LLM_PACKER/_packs/
if "%~1"=="" (
  REM Default: generate a single FULL catalytic-dpt pack folder (includes combined outputs + zip archive).
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0pack.ps1" -Scope catalytic-dpt -Mode full -Profile full -Combined -Zip -SplitLite
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0pack.ps1" -Scope catalytic-dpt %*
)

echo.
echo Done.
pause
