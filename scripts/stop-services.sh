#!/bin/bash
echo "Stopping Tamako Photos Services..."

# Stop Python face detection services
echo "Stopping face detection service..."
if taskkill /F /IM python.exe 2>/dev/null; then
    echo "Success: Face detection service stopped"
else
    echo "Info: No face detection service was running"
fi

# Stop any Node.js processes (Electron app)
echo "Stopping Electron app..."
if taskkill /F /IM electron.exe 2>/dev/null; then
    echo "Success: Electron app stopped"
else
    echo "Info: No Electron app was running"
fi

# Clean up orphaned Node.js processes
taskkill /F /IM node.exe 2>/dev/null >/dev/null

echo "All services stopped successfully!"