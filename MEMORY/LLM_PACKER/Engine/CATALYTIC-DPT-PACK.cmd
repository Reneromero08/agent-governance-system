@echo off
setlocal
cd /d "%~dp0"

REM One-click packer for CATALYTIC-DPT only.
REM Output goes to MEMORY/LLM_PACKER/_packs/
if "%~1"=="" (
  REM Default: build a MAIN pack + LAB sub-pack, then zip the whole bundle.
  call "%~dp0CAT-DPT1.cmd"
) else (
  echo This launcher does not accept arguments. Use CAT-DPT1.cmd or the Python packers directly.
  exit /b 1
)

echo.
echo Done.
pause
