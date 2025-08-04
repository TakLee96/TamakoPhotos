@echo off
title Tamako Photos - Application Launcher

echo ==========================================
echo         Tamako Photos Launcher
echo ==========================================

:: Change to script directory
cd /d "%~dp0"

:: Clean up any existing services first
echo Cleaning up existing services...
call scripts\stop-services.bat >/dev/null 2>&1

:: Start services
echo Starting background services...
call scripts\start-services.bat
if %ERRORLEVEL% neq 0 (
    echo Error starting services. Press any key to exit.
    pause >/dev/null
    exit /b 1
)

echo.
echo ==========================================
echo      Starting Tamako Photos App
echo ==========================================

:: Start the Electron app
npm run start:app

echo.
echo ==========================================
echo    Tamako Photos App Has Closed
echo ==========================================

:: Ask if user wants to stop services
set /p STOP_SERVICES="Stop background services? (y/n): "
if /i "%STOP_SERVICES%"=="y" (
    call scripts\stop-services.bat
)

echo Press any key to exit...
pause >/dev/null