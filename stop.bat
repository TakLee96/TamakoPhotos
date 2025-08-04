@echo off
title Tamako Photos - Service Stopper

echo ==========================================
echo      Stopping Tamako Photos Services
echo ==========================================

:: Change to script directory
cd /d "%~dp0"

:: Stop all services
call scripts\stop-services.bat

echo.
echo ==========================================
echo     All Services Stopped Successfully
echo ==========================================

echo Press any key to exit...
pause >/dev/null