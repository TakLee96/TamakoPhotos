@echo off
echo Starting Face Detection Service...
cd /d "%~dp0"

REM Activate conda environment
echo Activating conda environment 'tensorflow'...
call conda activate tensorflow

REM Start the service
echo Starting FastAPI service on http://127.0.0.1:8000
python face_service.py

pause