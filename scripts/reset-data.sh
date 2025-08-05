#!/bin/bash

# Tamako Photos - Data Reset Script
# This script cleans all application data for fresh development testing

set -e

echo "🔄 Tamako Photos - Data Reset Script"
echo "This will permanently delete ALL photos, faces, and database content!"
echo ""

# Confirmation prompt
read -p "Are you sure you want to reset all data? (type 'yes' to confirm): " confirmation
if [[ "$confirmation" != "yes" ]]; then
    echo "❌ Reset cancelled."
    exit 0
fi

echo ""
echo "🧹 Starting data cleanup..."

# Stop all services first
echo "⏹️  Stopping services..."
bash "$(dirname "$0")/stop-services.sh" > /dev/null 2>&1 || true

# Wait for services to fully stop
sleep 3

# Function to safely remove files/directories
safe_remove() {
    if [[ -e "$1" ]]; then
        rm -rf "$1"
        echo "✅ Removed: $1"
    else
        echo "ℹ️  Not found: $1"
    fi
}

# Remove SQLite database
safe_remove "photos.db"
safe_remove "photos.db-journal"
safe_remove "photos.db-shm"
safe_remove "photos.db-wal"

# Remove photo directories
safe_remove "photos/"
safe_remove "thumbnails/"

# Remove face detection data
safe_remove "face_detection/faces/"
safe_remove "face_detection/face_embeddings.index"
safe_remove "face_detection/face_metadata.json"
safe_remove "face_detection/face_metadata.json.migrated_backup"

# Remove any Python cache files
safe_remove "face_detection/__pycache__/"
safe_remove "**/*.pyc"

# Remove any logs or temporary files
safe_remove "*.log"
safe_remove "face_detection/*.log"

echo ""
echo "🎉 Data reset complete!"
echo ""
echo "Next steps:"
echo "1. Run 'npm start' to restart the application"
echo "2. Upload photos to test with fresh data"
echo "3. All face detection will start from scratch"
echo ""