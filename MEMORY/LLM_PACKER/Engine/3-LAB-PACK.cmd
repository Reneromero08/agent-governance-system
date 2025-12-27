@echo off
setlocal
cd /d "%~dp0"

REM One-click packer for CATALYTIC-DPT/LAB only.
REM Output goes to MEMORY/LLM_PACKER/_packs/ and zips go to MEMORY/LLM_PACKER/_packs/_system/archive/

if "%~1"=="" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0pack.ps1" -Scope lab -Profile full -SplitLite -Combined -Zip
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0pack.ps1" -Scope lab %*
)

echo.
echo Done.
pause
