@echo off
title Tamako Photos - Service Restart

echo ==========================================
echo      Restarting Tamako Photos Services
echo ==========================================

:: Change to script directory
cd /d "%~dp0"

echo Step 1: Stopping all services...
call scripts\stop-services.bat

echo.
echo Step 2: Waiting for cleanup...
timeout /t 3 /nobreak >/dev/null

echo Step 3: Starting services...
call scripts\start-services.bat
if %ERRORLEVEL% neq 0 (
    echo Error restarting services. Press any key to exit.
    pause >/dev/null
    exit /b 1
)

echo.
echo ==========================================
echo    Services Restarted Successfully!
echo ==========================================

echo Press any key to continue...
pause >/dev/null