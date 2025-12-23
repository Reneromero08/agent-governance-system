@echo off
setlocal
cd /d "%~dp0"

REM One-click packer. Output goes to MEMORY/LLM_PACKER/_packs/
if "%~1"=="" (
  REM Default: generate a single FULL pack folder with SPLIT + SPLIT_LITE (no huge combined outputs).
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0pack.ps1" -Profile full -SplitLite -NoCombined -NoZip
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0pack.ps1" %*
)

echo.
echo Done.
pause
