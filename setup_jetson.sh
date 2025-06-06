#!/bin/bash

# Mirror Mirror - Jetson Orin NX Setup Script
# This script sets up the environment for running Mirror Mirror on Jetson

set -e

echo "🪞 Mirror Mirror - Jetson Orin NX Setup"
echo "======================================="

# Check if we're running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "⚠️  Warning: This doesn't appear to be a Jetson device"
    echo "   Continuing anyway..."
else
    echo "✅ Detected Jetson device"
    cat /etc/nv_tegra_release
fi

echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check system requirements
echo "Checking system requirements..."

# Check Python version
if command_exists python3.10; then
    python_version=$(python3.10 --version)
    echo "✅ Python: $python_version"
else
    echo "❌ Python 3.10 not found"
    echo "   Install with: sudo apt install python3.10 python3.10-venv python3.10-dev"
    exit 1
fi

# Check uv
if ! command_exists uv; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.bashrc
    if ! command_exists uv; then
        echo "❌ Failed to install uv"
        exit 1
    fi
fi
echo "✅ uv: $(uv --version)"

# Check Docker
if ! command_exists docker; then
    echo "📦 Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    echo "⚠️  You may need to log out and back in for Docker permissions"
else
    echo "✅ Docker: $(docker --version)"
fi

# Check Docker Compose
if ! command_exists docker-compose; then
    echo "📦 Installing Docker Compose..."
    sudo apt-get update
    sudo apt-get install -y docker-compose
fi
echo "✅ Docker Compose: $(docker-compose --version)"

# Check CUDA
echo ""
echo "Checking CUDA environment..."
if command_exists nvidia-smi; then
    echo "✅ NVIDIA SMI available"
    nvidia-smi --query-gpu=name,memory.total,temperature.gpu --format=csv,noheader
else
    echo "❌ nvidia-smi not found"
    echo "   Make sure CUDA drivers are installed"
fi

# Check camera devices
echo ""
echo "Checking camera devices..."
if ls /dev/video* >/dev/null 2>&1; then
    echo "✅ Camera devices found:"
    for dev in /dev/video*; do
        echo "   $dev"
    done
else
    echo "❌ No camera devices found"
    echo "   Make sure a camera is connected"
fi

# Install system dependencies
echo ""
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3-opencv \
    libopencv-dev \
    python3-dev \
    portaudio19-dev \
    libasound2-dev \
    libv4l-dev \
    v4l-utils

# Set up Python environment
echo ""
echo "Setting up Python environment..."
uv sync

# Test basic imports
echo ""
echo "Testing basic imports..."
uv run python -c "
import cv2
print('✅ OpenCV:', cv2.__version__)

import torch
print('✅ PyTorch:', torch.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✅ CUDA device:', torch.cuda.get_device_name(0))
    print('✅ CUDA memory:', torch.cuda.get_device_properties(0).total_memory // 1024**3, 'GB')

import redis
print('✅ Redis client available')

import faststream
print('✅ FastStream available')

import diffusers
print('✅ Diffusers available')
"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Test camera: python test_camera.py"
echo "2. Run simple test: python test_system.py test-simple"
echo "3. Run full test: python test_system.py test-full" 