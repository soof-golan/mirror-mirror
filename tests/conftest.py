import pytest
import numpy as np
import cv2
import time
import asyncio
import torch
from unittest.mock import AsyncMock, MagicMock, Mock
from typing import AsyncGenerator
import logging

from faststream.redis import RedisBroker

from mirror_mirror.models import (
    FrameMessage, 
    LatentsMessage, 
    ProcessedFrameMessage,
    PromptMessage,
    AudioMessage,
    EmbeddingMessage,
    CarrierMessage,
    encode_bytes,
    serialize_array
)

logger = logging.getLogger(__name__)


@pytest.fixture
def sample_frame() -> np.ndarray:
    """Create a sample RGB frame for testing"""
    # Create a simple 640x480 RGB test pattern
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some test patterns
    frame[:, :, 0] = 255  # Red channel
    frame[100:200, 100:200, :] = [0, 255, 0]  # Green square
    frame[300:400, 300:400, :] = [0, 0, 255]  # Blue square
    
    return frame


@pytest.fixture
def sample_frame_jpeg(sample_frame: np.ndarray) -> bytes:
    """Create a JPEG-encoded frame for testing"""
    success, encoded = cv2.imencode('.jpg', sample_frame)
    assert success, "Failed to encode test frame as JPEG"
    return encoded.tobytes()


@pytest.fixture
def sample_latents() -> np.ndarray:
    """Create sample latent array for testing"""
    # Typical SDXL latent shape: (1, 4, 64, 64)
    return np.random.randn(1, 4, 64, 64).astype(np.float32)


@pytest.fixture
def sample_audio() -> tuple[np.ndarray, int]:
    """Create sample audio data for testing"""
    sample_rate = 16000
    duration = 1.0  # 1 second
    samples = int(sample_rate * duration)
    
    # Generate a simple sine wave
    t = np.linspace(0, duration, samples, False)
    frequency = 440  # A4 note
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    return audio, sample_rate


@pytest.fixture
def frame_message(sample_frame_jpeg: bytes) -> FrameMessage:
    """Create a sample FrameMessage"""
    return FrameMessage(
        frame=encode_bytes(sample_frame_jpeg),
        timestamp=time.time(),
        camera_id=0
    )


@pytest.fixture
def latents_message(sample_latents: np.ndarray) -> LatentsMessage:
    """Create a sample LatentsMessage"""
    data, shape, dtype = serialize_array(sample_latents)
    return LatentsMessage(
        latents=data,
        shape=shape,
        dtype=dtype,
        timestamp=time.time(),
        source="camera"
    )


@pytest.fixture
def prompt_message() -> PromptMessage:
    """Create a sample PromptMessage"""
    return PromptMessage(
        prompt="a beautiful landscape with mountains and trees",
        timestamp=time.time()
    )


@pytest.fixture
def processed_frame_message(sample_frame_jpeg: bytes) -> ProcessedFrameMessage:
    """Create a sample ProcessedFrameMessage"""
    return ProcessedFrameMessage(
        frame=encode_bytes(sample_frame_jpeg),
        timestamp=time.time(),
        processing_time=0.123
    )


@pytest.fixture
def mock_camera():
    """Create a mock camera for testing"""
    mock = MagicMock()
    mock.isOpened.return_value = True
    mock.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_WIDTH: 640.0,
        cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
        cv2.CAP_PROP_FPS: 24.0
    }.get(prop, 0.0)
    
    # Return success and a test frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock.read.return_value = (True, test_frame)
    
    return mock


@pytest.fixture
def mock_redis_broker():
    """Create a mock Redis broker for testing"""
    broker = AsyncMock(spec=RedisBroker)
    broker.publish = AsyncMock()
    broker.__aenter__ = AsyncMock(return_value=broker)
    broker.__aexit__ = AsyncMock(return_value=None)
    return broker


@pytest.fixture
async def redis_broker():
    """Create a real Redis broker for integration tests (requires Redis)"""
    broker = RedisBroker("redis://localhost:6379")
    try:
        await broker.connect()
        yield broker
    finally:
        await broker.close()


@pytest.fixture
def mock_vae():
    """Create a mock VAE for testing"""
    mock = MagicMock()
    
    # Mock output for AutoencoderKL
    mock_dist = MagicMock()
    mock_dist.sample.return_value = torch.randn(1, 4, 64, 64)
    
    mock_output = MagicMock()
    mock_output.latent_dist = mock_dist
    
    mock.encode.return_value = mock_output
    mock.config.scaling_factor = 0.18215
    mock.to.return_value = mock
    mock.eval.return_value = mock
    
    return mock


@pytest.fixture
def mock_diffusion_pipeline():
    """Create a mock diffusion pipeline for testing"""
    mock = MagicMock()
    
    # Mock pipeline call
    mock_result = MagicMock()
    mock_result.images = [torch.randn(1, 4, 64, 64)]
    mock.__call__.return_value = mock_result
    
    # Mock encode_prompt
    mock.encode_prompt.return_value = (
        torch.randn(1, 77, 768),  # prompt_embeds
        torch.randn(1, 77, 768)   # negative_prompt_embeds
    )
    
    mock.to.return_value = mock
    mock.set_progress_bar_config = MagicMock()
    
    return mock


@pytest.fixture(autouse=True)
def setup_logging():
    """Setup logging for tests"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)8s] %(name)s: %(message)s'
    )


# Async test utilities
@pytest.fixture
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Performance measurement fixtures
@pytest.fixture
def performance_monitor():
    """Monitor performance of test operations"""
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.measurements = {}
        
        def start(self, operation: str):
            self.start_time = time.time()
            logger.info(f"Starting performance measurement for: {operation}")
        
        def end(self, operation: str):
            if self.start_time is None:
                raise ValueError("Must call start() before end()")
            
            duration = time.time() - self.start_time
            self.measurements[operation] = duration
            logger.info(f"Completed {operation} in {duration:.3f}s")
            self.start_time = None
            return duration
    
    return PerformanceMonitor()


# GPU availability fixture
@pytest.fixture
def gpu_available():
    """Check if GPU is available for testing"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False 