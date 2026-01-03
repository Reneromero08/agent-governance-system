@echo off
setlocal
cd /d "%~dp0"

REM One-click packer. Output goes to MEMORY/LLM_PACKER/_packs/
if "%~1"=="" (
  REM Default: generate a pack folder with FULL/ + SPLIT/ + LITE/ and an archive zip.
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0pack.ps1" -Profile full -SplitLite -Combined -Zip
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0pack.ps1" %*
)

echo.
echo Done.
pause
