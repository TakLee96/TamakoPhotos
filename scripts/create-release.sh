#!/bin/bash

# Tamako Photos Release Script
# Usage: ./scripts/create-release.sh [version]

VERSION=${1:-"1.0.0"}
RELEASE_NAME="Tamako Photos v${VERSION} - AI-Powered Photo Management"
EXECUTABLE="dist/Tamako Photos ${VERSION}.exe"

echo "🚀 Creating release for Tamako Photos v${VERSION}"

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "❌ Error: Executable not found at: $EXECUTABLE"
    echo "Please run 'npm run build' first"
    exit 1
fi

# Check if git tag exists
if ! git tag -l | grep -q "v${VERSION}"; then
    echo "📝 Creating git tag v${VERSION}"
    git tag -a "v${VERSION}" -m "Release v${VERSION}: ${RELEASE_NAME}"
    git push origin "v${VERSION}"
else
    echo "✅ Git tag v${VERSION} already exists"
fi

echo "📦 Release assets ready:"
echo "   - Executable: $EXECUTABLE ($(du -h "$EXECUTABLE" | cut -f1))"
echo "   - Release notes: RELEASE_NOTES.md"

echo ""
echo "🎯 Next steps:"
echo "1. Go to: https://github.com/TakLee96/TamakoPhotos/releases"
echo "2. Click 'Create a new release'"
echo "3. Select tag: v${VERSION}"
echo "4. Title: ${RELEASE_NAME}"
echo "5. Upload: $EXECUTABLE"
echo "6. Description: Copy from RELEASE_NOTES.md"
echo "7. Click 'Publish release'"

echo ""
echo "✨ Release v${VERSION} is ready for distribution!"
