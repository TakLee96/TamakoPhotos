#!/bin/bash
echo "=========================================="
echo "         Tamako Photos Launcher"
echo "=========================================="

# Clean up any existing services first
echo "Cleaning up existing services..."
bash ./scripts/stop-services.sh >/dev/null 2>&1

# Start services
echo "Starting background services..."
bash ./scripts/start-services.sh
if [ $? -ne 0 ]; then
    echo "Error starting services. Press any key to exit."
    read -n 1
    exit 1
fi

echo ""
echo "=========================================="
echo "      Starting Tamako Photos App"
echo "=========================================="

# Start the Electron app
npm run start:app

echo ""
echo "=========================================="
echo "    Tamako Photos App Has Closed"
echo "=========================================="

# Ask if user wants to stop services
read -p "Stop background services? (y/n): " STOP_SERVICES
if [[ "$STOP_SERVICES" =~ ^[Yy]$ ]]; then
    bash ./scripts/stop-services.sh
fi

echo "Press any key to exit..."
read -n 1