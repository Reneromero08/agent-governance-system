@echo off
setlocal
cd /d "%~dp0"

REM Packer 2 -- Standalone CAT_CAS codebase cleaner.
REM Copies THOUGHT/LAB/CAT_CAS, strips to .py .md .rs .toml only,
REM removes empty directories, zips. Output goes to MEMORY/LLM_PACKER/_packs/

cd /d "%~dp0..\..\.."

echo.
echo ============================================
echo   Packer 2 - CAT_CAS Codebase Cleaner
echo ============================================
echo.

if "%~1"=="" (
  .\.venv\Scripts\python.exe -m MEMORY.LLM_PACKER.Engine.packer_2
) else (
  .\.venv\Scripts\python.exe -m MEMORY.LLM_PACKER.Engine.packer_2 %*
)

echo.
echo Done.
pause
