@echo off
REM Qwen CLI - Quick access to local AI assistant
REM Usage: qwen.bat "your question"
REM        qwen.bat "explain this" file.py
REM        qwen.bat (interactive mode)

setlocal enabledelayedexpansion

REM Get the directory where this batch file is located
set "SKILL_DIR=%~dp0"
set "PYTHON=python"

REM Check if Python is available
%PYTHON% --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found in PATH
    echo Please install Python or add it to your PATH
    exit /b 1
)

REM Check if ollama package is installed
%PYTHON% -c "import ollama" >nul 2>&1
if errorlevel 1 (
    echo Error: ollama package not installed
    echo Installing now...
    pip install ollama
    if errorlevel 1 (
        echo Failed to install ollama package
        exit /b 1
    )
)

REM Build command
set "CMD=%PYTHON% "%SKILL_DIR%qwen_cli.py""

REM No arguments - interactive mode
if "%~1"=="" (
    %CMD% --interactive
    exit /b %ERRORLEVEL%
)

REM One argument - simple question
if "%~2"=="" (
    %CMD% %1
    exit /b %ERRORLEVEL%
)

REM Two arguments - question + file
%CMD% %1 --file %2
exit /b %ERRORLEVEL%
