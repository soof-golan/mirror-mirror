# Mirror Mirror

A real-time camera-based image diffusion system for Jetson Orin NX.

## Architecture

The system follows a modular pipeline architecture using FastStream and Redis for message passing:

```
Camera → Latent Encoder → Diffusion → Latent Decoder → Display
                            ↑
                     Text Prompts
```

### Components

1. **Camera Server** (`camera_server.py`) - Captures webcam frames
2. **Latent Encoder** (`latent_encoder.py`) - Converts frames to latent space using VAE
3. **Diffusion Server** (`diffusion_server.py`) - Applies diffusion effects using SDXS
4. **Latent Decoder** (`latent_decoder.py`) - Converts latents back to images
5. **Display** (`display.py`) - Shows processed frames with performance metrics
6. **Audio Server** (`audio_server.py`) - Voice activity detection and audio capture
7. **Prompt Publisher** (`prompt_publisher.py`) - Publishes text prompts

### Message Flow

- **Redis Channels**:
  - `frames:camera:{id}` - Raw camera frames
  - `latents:camera` - Encoded camera latents
  - `latents:diffused` - Processed latents from diffusion
  - `images:processed` - Final processed images
  - `prompts:*` - Text prompts for diffusion
  - `audio:chunks` - Audio data from microphone

## Quick Start

### Prerequisites

- Jetson Orin NX with JetPack 6
- Docker and Docker Compose
- Python 3.10.7 with uv

### Installation

```bash
git clone https://github.com/soof-golan/mirror-mirror.git
cd mirror-mirror
```

### Running Tests

**Simple test (no GPU required):**
```bash
./run_tests.sh
```

**Or manually:**
```bash
# Install dependencies
uv sync

# Run simple test with fake diffusion
python test_system.py test-simple
```

**Full test with real diffusion:**
```bash
python test_system.py test-full
```

### Manual Control

```bash
# Start Redis
python test_system.py start-redis

# Start the complete pipeline
python test_system.py start-pipeline --mode=sdxs --debug

# Check status
python test_system.py status

# Publish a test prompt
python test_system.py publish-prompt "a beautiful sunset landscape"

# Cleanup
python test_system.py cleanup
```

### Individual Components

```bash
# Start Redis first
python test_system.py start-redis

# Run individual components
uv run python -m mirror_mirror.camera_server
uv run python -m mirror_mirror.latent_encoder
uv run python -m mirror_mirror.diffusion_server
uv run python -m mirror_mirror.latent_decoder
uv run python -m mirror_mirror.display
```

## Configuration

Each component can be configured via environment variables:

```bash
# Camera settings
export CAMERA_ID=0
export FPS=24
export FRAME_WIDTH=640
export FRAME_HEIGHT=480

# Diffusion settings
export MODE=sdxs  # or "fake"
export MODEL_REPO=IDKiro/sdxs-512-dreamshaper
export DEVICE=cuda
export TORCH_DTYPE=float16

# Redis settings
export REDIS_URL=redis://localhost:6379
```

## Performance

The system is optimized for real-time performance on Jetson Orin NX:

- **Target**: 15-30 FPS end-to-end
- **Latency**: <100ms camera to display
- **Memory**: ~4GB GPU memory usage
- **Models**: SDXS (1-step diffusion) for speed

## Controls

When running the display:
- Press `q` to quit
- Press `f` to toggle FPS/performance display

## Development

To add new components:

1. Create a new module in `src/mirror_mirror/`
2. Use the shared models from `models.py`
3. Subscribe/publish to appropriate Redis channels
4. Add to the test system pipeline

## Troubleshooting

**Camera not found:**
```bash
# List available cameras
ls /dev/video*

# Test camera directly
uv run python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
```

**GPU memory issues:**
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Use fake mode for testing without GPU
python test_system.py start-pipeline --mode=fake
```

**Redis connection issues:**
```bash
# Check Redis status
docker-compose ps redis

# View Redis logs
docker-compose logs redis
```

## Future Enhancements

- [ ] Speech-to-text integration
- [ ] ControlNet support for pose/depth guidance
- [ ] Multiple diffusion models
- [ ] Web UI for remote control
- [ ] Audio-reactive diffusion
- [ ] Gesture recognition

 
We're targeting Jetson Orin NX.
I have a few protoype quality files that implement core idea of Mirror Mirror.
Help me continue implementation.

run tests against the actuall hardware.

# Architecture:

Local Redis Broker: (add a docker compose to run it on the device)
Using FastStream for piping messages between components
Each component in its own small python script, sharing api shapes with a common imported module
The Diffusion module accepts both embedded latents and prompts
The Prompts should come from a VAD -> STT -> Rephrading (English to image genration descriptions) -> Tokenzation -> Embedding 
Images should come from Camera (cv2) -> preprocessing (crop resize to_tensor) -> ControlNet -> Encoding To Latent
Diffusion subscribes to both Image and Prompt streams (memoizes the prompt) -> Diffusion -> Decodeing -> Denormalization -> Postprocessing (resize, crop, to_tensor) 
Display module subscribes to the processed images and displays to the screen

