@echo off
setlocal
cd /d "%~dp0"

REM One-click packer for THOUGHT/LAB only.
REM Output goes to MEMORY/LLM_PACKER/_packs/ and external zips go to MEMORY/LLM_PACKER/_packs/_archive/<pack_name>.zip

if "%~1"=="" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0pack.ps1" -Scope lab -Profile full -SplitLite -Combined -Zip
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0pack.ps1" -Scope lab %*
)

echo.
echo Done.
pause
