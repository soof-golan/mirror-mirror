#!/bin/bash

# Script to sync Mirror Mirror code to Jetson Orin NX
# Usage: ./sync_to_jetson.sh [jetson-host] [jetson-user]

set -e

# Default values (customize these)
JETSON_HOST="${1:-your-jetson-ip}"
JETSON_USER="${2:-your-jetson-user}"
REMOTE_PATH="~/dev/mirror-mirror"

echo "🪞 Mirror Mirror - Sync to Jetson"
echo "================================="
echo "Host: $JETSON_USER@$JETSON_HOST"
echo "Remote path: $REMOTE_PATH"
echo ""

# Check if we can connect
echo "Testing connection..."
if ! ssh -o ConnectTimeout=5 "$JETSON_USER@$JETSON_HOST" "echo 'Connected successfully'"; then
    echo "❌ Failed to connect to Jetson"
    echo "Please check:"
    echo "  - Jetson is powered on and connected"
    echo "  - SSH is enabled on Jetson"
    echo "  - Correct hostname/IP and username"
    echo "  - SSH keys are set up"
    exit 1
fi

echo "✅ Connection successful"
echo ""

# Create remote directory if it doesn't exist
echo "Creating remote directory..."
ssh "$JETSON_USER@$JETSON_HOST" "mkdir -p $REMOTE_PATH"

# Sync files (excluding unnecessary directories)
echo "Syncing files..."
rsync -avz --progress \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '.pytest_cache' \
    --exclude 'wheels' \
    --exclude '.venv' \
    --exclude 'node_modules' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    ./ "$JETSON_USER@$JETSON_HOST:$REMOTE_PATH/"

echo ""
echo "✅ Sync complete!"
echo ""
echo "Next steps:"
echo "1. SSH to Jetson: ssh $JETSON_USER@$JETSON_HOST"
echo "2. Change directory: cd $REMOTE_PATH"
echo "3. Run setup: ./setup_jetson.sh"
echo "4. Test camera: python test_camera.py"
echo "5. Run diagnostics: python debug_jetson.py diagnose"
echo "6. Start testing: python test_system.py test-simple"
echo ""
echo "For detailed testing guide, see: JETSON_TESTING.md" 