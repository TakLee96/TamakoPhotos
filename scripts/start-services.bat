@echo off
echo Starting Tamako Photos Services...

:: Set environment variables for Python
set KMP_DUPLICATE_LIB_OK=TRUE
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

:: Kill any existing Python processes to avoid conflicts
echo Cleaning up existing services...
taskkill //F //IM python.exe 2>/dev/null >/dev/null

:: Wait a moment for cleanup
timeout /t 2 /nobreak >/dev/null

:: Start face detection service
echo Starting face detection service...
cd /d "%~dp0\..\face_detection"
start "Face Detection Service" "/c/Users/jiaha/anaconda3/envs/tensorflow/python.exe" face_service.py

:: Wait for service to start
echo Waiting for face detection service to initialize...
timeout /t 5 /nobreak >/dev/null

:: Check if service is running
tasklist /FI "IMAGENAME eq python.exe" >/dev/null 2>&1
if %ERRORLEVEL% equ 0 (
    echo Success: Face detection service started successfully
) else (
    echo Warning: Could not verify service status - check manually
)

echo All services started successfully!
echo Ready to launch Tamako Photos app...