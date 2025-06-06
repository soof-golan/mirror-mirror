# Jetson Orin NX Testing Guide

This guide walks you through testing the Mirror Mirror system on Jetson Orin NX hardware.

## 🚀 Quick Start

### 1. Get the Code on Jetson

```bash
# SSH to your Jetson
ssh your-jetson-user@your-jetson-ip

# Clone or sync the repository
git clone https://github.com/soof-golan/mirror-mirror.git
cd mirror-mirror

# Or if you already have it:
git pull origin main
```

### 2. Setup Environment

```bash
# Run the automated setup
./setup_jetson.sh
```

This script will:
- ✅ Check system requirements
- 📦 Install Docker and Docker Compose
- 🐍 Install/update uv
- 🔧 Install system dependencies
- 🧪 Test basic imports

### 3. Quick Validation

```bash
# Test camera first
python test_camera.py

# Run diagnostics
python debug_jetson.py diagnose

# Quick component tests
python debug_jetson.py test-components
```

## 📋 Step-by-Step Testing

### Phase 1: Hardware Validation

**1.1 Check System Resources**
```bash
# Monitor system in real-time
python debug_jetson.py monitor
```

**1.2 Test Camera**
```bash
# Test default camera (usually /dev/video0)
python test_camera.py 0

# Test other cameras if needed
python test_camera.py 1
```

**1.3 Verify CUDA**
```bash
# Check NVIDIA setup
nvidia-smi

# Test PyTorch CUDA
uv run python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
print('Device name:', torch.cuda.get_device_name(0))
print('Memory:', torch.cuda.get_device_properties(0).total_memory // 1024**3, 'GB')
"
```

### Phase 2: Individual Components

**2.1 Test Redis**
```bash
# Start Redis
python test_system.py start-redis

# Test connection
python debug_jetson.py test-components

# Check Redis status
docker compose ps redis
```

**2.2 Test Camera Server**
```bash
# In one terminal
uv run python -m mirror_mirror.camera_server

# In another terminal - check Redis for messages
docker exec -it mirror-mirror-redis redis-cli
> MONITOR
# You should see frame messages
```

**2.3 Test Diffusion (Fake Mode)**
```bash
# Start Redis first
python test_system.py start-redis

# Test fake diffusion
PYTHONPATH=src MODE=fake uv run python -m mirror_mirror.diffusion_server

# In another terminal, publish a prompt
python test_system.py publish-prompt "a beautiful sunset"
```

### Phase 3: Pipeline Testing

**3.1 Simple Pipeline (Fake Diffusion)**
```bash
# This should work without GPU
python test_system.py test-simple
```

**3.2 Monitor During Testing**
```bash
# In another terminal, monitor resources
python debug_jetson.py monitor
```

**3.3 Full Pipeline (Real Diffusion)**
```bash
# Only if CUDA is working and you have enough GPU memory
python test_system.py test-full
```

## 🐛 Troubleshooting

### Camera Issues

**No camera devices found:**
```bash
# List video devices
ls -la /dev/video*

# Check camera permissions
sudo usermod -a -G video $USER
# Log out and back in

# Test with v4l2
v4l2-ctl --list-devices
```

**Camera fails to open:**
```bash
# Kill processes using camera
sudo fuser -k /dev/video0

# Test basic camera access
uv run python -c "
import cv2
cap = cv2.VideoCapture(0)
print('Camera opened:', cap.isOpened())
if cap.isOpened():
    ret, frame = cap.read()
    print('Frame captured:', ret, frame.shape if ret else None)
    cap.release()
"
```

### CUDA/GPU Issues

**CUDA not available:**
```bash
# Check NVIDIA drivers
nvidia-smi

# Check JetPack version
sudo apt show nvidia-jetpack

# Reinstall PyTorch for Jetson if needed
pip3 install --upgrade torch torchvision
```

**Out of GPU memory:**
```bash
# Monitor GPU memory
watch -n 1 nvidia-smi

# Try with smaller batch size or fake mode
MODE=fake python test_system.py start-pipeline
```

### Redis Issues

**Connection refused:**
```bash
# Check Docker daemon
sudo systemctl status docker

# Start Docker if needed
sudo systemctl start docker

# Check Redis container
docker compose ps
docker compose logs redis
```

**Permission denied:**
```bash
# Add user to docker group
sudo usermod -a -G docker $USER
# Log out and back in
```

### Performance Issues

**Low FPS:**
```bash
# Check system load
python debug_jetson.py monitor

# Reduce camera resolution
export FRAME_WIDTH=320
export FRAME_HEIGHT=240

# Use fake diffusion mode
MODE=fake python test_system.py start-pipeline
```

**High CPU/Memory usage:**
```bash
# Monitor processes
htop

# Kill background processes if needed
sudo pkill -f "python.*mirror_mirror"

# Restart with lower settings
python test_system.py cleanup
```

## 📊 Performance Targets

### Expected Performance (Fake Mode)
- **FPS**: 24-30 fps
- **Latency**: <50ms camera to display
- **Memory**: ~2GB RAM, minimal GPU

### Expected Performance (SDXS Mode)
- **FPS**: 10-15 fps (first time slower due to model loading)
- **Latency**: 200-500ms camera to display
- **Memory**: ~4GB RAM, ~3-4GB GPU

### System Resources
- **CPU**: Should stay under 80%
- **Temperature**: Should stay under 80°C
- **Memory**: Leave ~2GB free

## 🔧 Advanced Testing

### Model Loading Test
```bash
# Test model loading time
time uv run python -c "
from diffusers import StableDiffusionPipeline
import torch
pipe = StableDiffusionPipeline.from_pretrained(
    'IDKiro/sdxs-512-dreamshaper',
    torch_dtype=torch.float16
)
pipe.to('cuda')
print('Model loaded successfully')
"
```

### Memory Stress Test
```bash
# Run multiple components and monitor
python debug_jetson.py monitor &
python test_system.py start-pipeline --mode=sdxs --debug
```

### Network Performance Test
```bash
# Test Redis message throughput
python test_system.py start-redis

# Publish many frames quickly
for i in {1..100}; do
  python test_system.py publish-prompt "test message $i"
done
```

## 📝 Test Results Template

Document your test results:

```
# Mirror Mirror Jetson Test Results

**Hardware:**
- Device: Jetson Orin NX
- JetPack Version: 
- RAM: GB
- Storage: GB free

**Test Results:**

## Phase 1: Hardware Validation
- [ ] Camera test passed (camera ID: )
- [ ] CUDA available: Y/N
- [ ] GPU memory: GB
- [ ] System temperature: °C

## Phase 2: Component Tests
- [ ] Redis connection: Y/N
- [ ] Camera server: Y/N
- [ ] Diffusion server (fake): Y/N
- [ ] Display: Y/N

## Phase 3: Pipeline Tests
- [ ] Simple pipeline (fake): Y/N, FPS: 
- [ ] Full pipeline (SDXS): Y/N, FPS: 
- [ ] End-to-end latency: ms

## Issues Encountered:
- 
- 

## Performance Notes:
- Peak GPU memory usage: MB
- Average CPU usage: %
- Temperature under load: °C
```

## 🆘 Getting Help

If you encounter issues:

1. **Run diagnostics**: `python debug_jetson.py diagnose`
2. **Check logs**: Look for error messages in component output
3. **Monitor resources**: `python debug_jetson.py monitor`
4. **Clean restart**: `python test_system.py cleanup`

**Common Solutions:**
- Restart Docker: `sudo systemctl restart docker`
- Clear GPU memory: Reboot the Jetson
- Check permissions: Ensure user is in `docker` and `video` groups
- Update dependencies: `uv sync` 