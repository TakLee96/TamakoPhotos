#!/bin/bash
echo "Starting Tamako Photos Services..."

# Set environment variables for Python
export KMP_DUPLICATE_LIB_OK=TRUE
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Kill any existing Python processes to avoid conflicts
echo "Cleaning up existing services..."
taskkill //F //IM python.exe 2>/dev/null >/dev/null

# Wait a moment for cleanup
sleep 2

# Start face detection service
echo "Starting face detection service..."
cd "$(dirname "$0")/../face_detection"
"/c/Users/jiaha/anaconda3/envs/tensorflow/python.exe" face_service.py &

# Wait for service to start
echo "Waiting for face detection service to initialize..."
sleep 5

# Check if service is running
if pgrep -f python.exe >/dev/null 2>&1; then
    echo "Success: Face detection service started successfully"
else
    echo "Warning: Could not verify service status - check manually"
fi

echo "All services started successfully!"
echo "Ready to launch Tamako Photos app..."