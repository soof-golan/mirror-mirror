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
    serialize_array
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
        
    def initialize(self):
        """Initialize the VAE model"""
        if self.vae is None:
            logger.info(f"Loading VAE from {config.model_repo}")
            pipe = StableDiffusionPipeline.from_pretrained(
                config.model_repo, 
                torch_dtype=self.dtype
            )
            self.vae = pipe.vae
            self.vae.to(self.device)
            self.vae.eval()
            logger.info("VAE loaded successfully")
    
    @torch.inference_mode()
    def encode_frame(self, frame_bytes: bytes) -> tuple[bytes, tuple[int, ...], str]:
        """Encode a frame to latents"""
        self.initialize()
        
        # Decode frame from bytes
        frame = decode_frame(frame_bytes)
        
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
        latents = self.vae.encode(image_tensor).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        
        # Convert to numpy and serialize
        latents_np = latents.cpu().float().numpy()
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
        return None
    
    frame_msg = carrier.content
    start_time = time.time()
    
    try:
        # Encode frame to latents
        latents_data, shape, dtype = encoder.encode_frame(frame_msg.frame)
        
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
        logger.error(f"Failed to encode frame: {e}")
        return None


if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting latent encoder...")
    app.run() 