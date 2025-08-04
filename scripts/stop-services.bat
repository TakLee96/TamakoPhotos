@echo off
echo Stopping Tamako Photos Services...

:: Stop Python face detection services
echo Stopping face detection service...
taskkill /F /IM python.exe 2>/dev/null
if %ERRORLEVEL% equ 0 (
    echo Success: Face detection service stopped
) else (
    echo Info: No face detection service was running
)

:: Stop any Node.js processes (Electron app)
echo Stopping Electron app...
taskkill /F /IM electron.exe 2>/dev/null
if %ERRORLEVEL% equ 0 (
    echo Success: Electron app stopped
) else (
    echo Info: No Electron app was running
)

:: Clean up orphaned Node.js processes
taskkill /F /IM node.exe 2>/dev/null >/dev/null

echo All services stopped successfully!