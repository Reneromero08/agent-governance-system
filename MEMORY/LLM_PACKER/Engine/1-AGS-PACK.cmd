@echo off
setlocal
cd /d "%~dp0"

REM One-click packer. Output goes to MEMORY/LLM_PACKER/_packs/
if "%~1"=="" (
  REM Default: generate a FULL pack folder with FULL_COMBINED + SPLIT + SPLIT_LITE and archive zips.
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0pack.ps1" -Profile full -SplitLite -Combined -Zip
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0pack.ps1" %*
)

echo.
echo Done.
pause
