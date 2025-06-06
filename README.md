# Mirror Mirror

A camera-based system with real-time image diffusion effects.

## Components

- **Camera Server**: Captures webcam frames and publishes them to Redis
- **Diffusion Server**: Subscribes to frames, applies effects based on prompts, and publishes processed frames
- **Prompt Publisher**: Sends text prompts to modify diffusion effects

## Requirements

- Python 3.10+
- Redis
- OpenCV
- FastStream

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mirror-mirror.git
cd mirror-mirror

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the System

1. Start Redis server:
   ```bash
   redis-server
   ```

2. Start the diffusion server:
   ```bash
   python -m src.mirror_mirror.diffusion_server
   ```

3. Start the camera server:
   ```bash
   python -m src.mirror_mirror.camera_server
   ```

4. Optionally, use the prompt publisher to send text prompts:
   ```bash
   python -m src.mirror_mirror.prompt_publisher
   ```

## Configuration

All components use environment variables for configuration or default values:

- `REDIS_URL`: Redis connection URL (default: "redis://localhost:6379")
- `CAMERA_ID`: Camera device ID (default: 0)
- `FPS`: Frame rate for camera capture (default: 24)
- `FEED_NAME`: Feed identifier (default: "main")

# Development

To develop on the Jetson Orin NX device, use the following commands to sync the latest code from the repository:

```bash
ssh -XC soof@soof-jetson.tail6f38f.ts.net
cd ~/dev/soof-golan/mirror-mirror/
git pull
uv sync
```
