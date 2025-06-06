#!/bin/bash

# Mirror Mirror Test Runner for Jetson Orin NX
# This script helps run tests on the actual hardware

set -e

echo "🪞 Mirror Mirror - Jetson Test Runner"
echo "======================================"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "Checking prerequisites..."

if ! command_exists docker; then
    echo "❌ Docker not found. Please install Docker."
    exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
    echo "❌ Docker Compose not found. Please install Docker Compose plugin."
    exit 1
fi

if ! command_exists uv; then
    echo "❌ uv not found. Please install uv (pip install uv)."
    exit 1
fi

echo "✅ All prerequisites found"

# Sync dependencies
echo "Syncing dependencies..."
uv sync

# Make test script executable
chmod +x test_system.py

echo ""
echo "Available test commands:"
echo "========================"
echo ""
echo "1. Simple test (fake diffusion - no GPU required):"
echo "   python test_system.py test-simple"
echo ""
echo "2. Full test (real diffusion - requires GPU):"
echo "   python test_system.py test-full"
echo ""
echo "3. Manual pipeline control:"
echo "   python test_system.py start-redis"
echo "   python test_system.py start-pipeline --mode=fake"
echo "   python test_system.py status"
echo "   python test_system.py cleanup"
echo ""
echo "4. Test individual components:"
echo "   python test_system.py start-redis"
echo "   uv run python -m mirror_mirror.camera_server"
echo ""

# Run simple test by default
echo "Running simple test..."
echo "Press Ctrl+C to stop"
echo ""

python test_system.py test-simple 