@echo off
echo ============================================================
echo PSYCHOHISTORY OPTIONS MONITOR
echo ============================================================
echo.
echo This will run in background and check SPY every hour.
echo When danger detected:
echo   - ALERT.txt will be created
echo   - Desktop notification (if enabled)
echo   - Sound alert (if enabled)
echo   - Telegram/Discord (if configured)
echo.
echo To set up notifications: python notifier.py --setup
echo.
echo Press Ctrl+C to stop.
echo.

cd /d "%~dp0"
python options_signal_bot.py --symbols SPY --interval 60
