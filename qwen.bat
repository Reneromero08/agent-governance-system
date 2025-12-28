@echo off
REM Global Qwen CLI launcher
REM Can be run from anywhere in your project

REM Get project root (where this file is)
set "PROJECT_ROOT=%~dp0"
set "SKILL_DIR=%PROJECT_ROOT%SKILLS\qwen-cli\"

REM Check if skill exists
if not exist "%SKILL_DIR%qwen.bat" (
    echo Error: Qwen CLI skill not found at %SKILL_DIR%
    exit /b 1
)

REM Forward all arguments to the skill
call "%SKILL_DIR%qwen.bat" %*
exit /b %ERRORLEVEL%
