@echo off
setlocal
cd /d "%~dp0"

REM One-click packer for CATALYTIC-DPT only (bypasses pack.ps1 to avoid scope/arg drift).
REM Output goes to MEMORY/LLM_PACKER/_packs/

set "STAMP="
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd_HH-mm-ss"') do set "STAMP=%%i"
if "%STAMP%"=="" set "STAMP=manual"

if "%~1"=="" (
  python "%~dp0packer.py" --scope catalytic-dpt --mode full --profile full --combined --zip --out-dir "MEMORY/LLM_PACKER/_packs/catalytic-dpt-pack-%STAMP%" --stamp "catalytic-dpt-pack-%STAMP%"
) else (
  python "%~dp0packer.py" %* --scope catalytic-dpt
)

echo.
echo Done.
pause
