@echo off
setlocal
cd /d "%~dp0"

REM One-click packer for CATALYTIC-DPT only (bypasses pack.ps1 to avoid scope/arg drift).
REM Output goes to MEMORY/LLM_PACKER/_packs/

set "STAMP="
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd_HH-mm-ss"') do set "STAMP=%%i"
if "%STAMP%"=="" set "STAMP=manual"

if not "%~1"=="" (
  echo This launcher does not accept arguments.
  echo Run: python "%~dp0packer_cat_dpt_main.py" --help
  echo Run: python "%~dp0packer_cat_dpt_lab.py" --help
  exit /b 1
)

set "BASENAME=catalytic-dpt-pack-%STAMP%"
set "OUT_DIR=MEMORY/LLM_PACKER/_packs/%BASENAME%"

python "%~dp0packer_cat_dpt_main.py" --mode full --profile full --split-lite --combined --out-dir "%OUT_DIR%" --stamp "%BASENAME%"
if errorlevel 1 exit /b 1

python "%~dp0packer_cat_dpt_lab.py" --mode full --profile full --split-lite --combined --out-dir "%OUT_DIR%\\LAB" --stamp "%BASENAME%-LAB"
if errorlevel 1 exit /b 1

REM Zip the entire bundle (MAIN + LAB) after both are built.
for %%I in ("%~dp0..\\..\\_packs") do set "PACKS_DIR=%%~fI"
for %%I in ("%PACKS_DIR%\\%BASENAME%") do set "OUT_DIR_ABS=%%~fI"
for %%I in ("%PACKS_DIR%\\archive") do set "ARCHIVE_DIR=%%~fI"

powershell -NoProfile -ExecutionPolicy Bypass -Command "New-Item -Force -ItemType Directory \"%ARCHIVE_DIR%\" | Out-Null; $zip = Join-Path \"%ARCHIVE_DIR%\" (\"%BASENAME%\" + '.zip'); if (Test-Path $zip) { Remove-Item -Force $zip }; Compress-Archive -Force -Path (Join-Path \"%OUT_DIR_ABS%\" '*') -DestinationPath $zip"
if errorlevel 1 exit /b 1

echo.
echo Done.
pause
