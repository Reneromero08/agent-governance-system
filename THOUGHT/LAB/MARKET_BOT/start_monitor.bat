@echo off
echo ============================================================
echo PSYCHOHISTORY OPTIONS MONITOR
echo ============================================================
echo.
echo This will run in background and check SPY every hour.
echo When danger detected, ALERT.txt will be created.
echo.
echo Press Ctrl+C to stop.
echo.

cd /d "%~dp0"
python options_signal_bot.py --symbols SPY --interval 60
