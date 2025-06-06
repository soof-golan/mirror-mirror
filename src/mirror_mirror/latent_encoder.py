import logging
import time
import torch
from contextlib import AsyncExitStack
from faststream import FastStream, Depends
from faststream.redis import RedisBroker, PubSub
from pydantic_settings import BaseSettings
from diffusers import StableDiffusionPipeline, AutoencoderKL, AutoencoderTiny

from mirror_mirror.common import log_errors
from mirror_mirror.decode import decode_frame
from mirror_mirror.models import (
    CarrierMessage, 
    FrameMessage, 
    LatentsMessage, 
    serialize_array,
    decode_bytes
)

logger = logging.getLogger(__name__)


class Config(BaseSettings):
    redis_url: str = "redis://localhost:6379"
    model_repo: str = "IDKiro/sdxs-512-dreamshaper"
    torch_dtype: str = "float16"
    device: str = "cuda"
    target_size: int = 512


config = Config()
broker = RedisBroker(url=config.redis_url)
app = FastStream(broker=broker)


class LatentEncoder:
    def __init__(self):
        self.vae = None
        self.device = config.device
        self.dtype = getattr(torch, config.torch_dtype)
        self.frames_processed = 0
        self.total_processing_time = 0.0
        
    def initialize(self):
        """Initialize the VAE model"""
        if self.vae is None:
            logger.info(f"Loading VAE from {config.model_repo}")
            start_time = time.time()
            
            pipe = StableDiffusionPipeline.from_pretrained(
                config.model_repo, 
                torch_dtype=self.dtype
            )
            self.vae = pipe.vae
            self.vae.to(self.device)
            self.vae.eval()
            
            load_time = time.time() - start_time
            logger.info(f"VAE loaded successfully in {load_time:.2f}s - device: {self.device}, dtype: {self.dtype}")
            
            # Log model info
            total_params = sum(p.numel() for p in self.vae.parameters())
            logger.info(f"VAE model: {total_params:,} parameters")
    
    @torch.inference_mode()
    def encode_frame(self, frame_bytes: bytes) -> tuple[bytes, tuple[int, ...], str]:
        """Encode a frame to latents"""
        self.initialize()
        encoding_start = time.time()
        
        # Decode frame from bytes
        frame = decode_frame(frame_bytes)
        logger.debug(f"Decoded frame: {frame.shape} {frame.dtype}")
        
        # Preprocess image
        # Convert to PIL format and resize
        from PIL import Image
        import numpy as np
        
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
            
        pil_image = Image.fromarray(frame)
        
        # Resize to target size
        pil_image = pil_image.resize((config.target_size, config.target_size), Image.LANCZOS)
        
        # Convert to tensor and normalize
        image_array = np.array(pil_image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(self.device, dtype=self.dtype)
        
        # Normalize to [-1, 1] range
        image_tensor = 2.0 * image_tensor - 1.0
        
        # Encode to latents
        output = self.vae.encode(image_tensor)
        
        # Handle different VAE output formats
        if hasattr(output, 'latent_dist'):
            # AutoencoderKL format
            latents = output.latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        else:
            # AutoencoderTiny format - returns latents directly
            latents = output.latents
            # AutoencoderTiny doesn't need scaling
        
        # Convert to numpy and serialize
        latents_np = latents.cpu().float().numpy()
        
        # Track statistics
        encoding_time = time.time() - encoding_start
        self.frames_processed += 1
        self.total_processing_time += encoding_time
        
        if self.frames_processed % 50 == 0:
            avg_time = self.total_processing_time / self.frames_processed
            logger.info(f"Encoded {self.frames_processed} frames, avg time: {avg_time:.3f}s")
        
        logger.debug(f"Encoded frame to latents {latents_np.shape} in {encoding_time:.3f}s")
        return serialize_array(latents_np)


def get_encoder() -> LatentEncoder:
    return LatentEncoder()


@broker.subscriber(channel=PubSub("frames:camera:*", pattern=True))
@broker.publisher(channel="latents:camera")
@log_errors
async def encode_camera_frames(
    carrier: CarrierMessage,
    encoder: LatentEncoder = Depends(get_encoder),
) -> CarrierMessage | None:
    """Process camera frames and convert to latents"""
    
    if not isinstance(carrier.content, FrameMessage):
        logger.debug(f"Ignoring non-frame message: {type(carrier.content)}")
        return None
    
    frame_msg = carrier.content
    start_time = time.time()
    
    logger.debug(f"Processing frame from camera {frame_msg.camera_id} at {frame_msg.timestamp}")
    
    try:
        # Decode and validate frame data
        frame_bytes = decode_bytes(frame_msg.frame)
        logger.debug(f"Frame data: {len(frame_bytes)} bytes")
        
        # Encode frame to latents
        latents_data, shape, dtype = encoder.encode_frame(frame_bytes)
        
        processing_time = time.time() - start_time
        
        latents_msg = LatentsMessage(
            latents=latents_data,
            shape=shape,
            dtype=dtype,
            timestamp=frame_msg.timestamp,
            source="camera"
        )
        
        logger.debug(
            f"Encoded frame to latents: {shape} in {processing_time:.3f}s"
        )
        
        return CarrierMessage(content=latents_msg)
        
    except Exception as e:
        logger.error(f"Failed to encode frame: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting latent encoder...")
    asyncio.run(app.run()) 