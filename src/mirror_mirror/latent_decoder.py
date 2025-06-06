import logging
import time
import torch
from faststream import FastStream, Depends
from faststream.redis import RedisBroker, PubSub
from pydantic_settings import BaseSettings
from diffusers import StableDiffusionPipeline, AutoencoderKL, AutoencoderTiny

from mirror_mirror.common import log_errors
from mirror_mirror.decode import encode_frame
from mirror_mirror.models import (
    CarrierMessage, 
    LatentsMessage, 
    ProcessedFrameMessage,
    deserialize_array
)

logger = logging.getLogger(__name__)


class Config(BaseSettings):
    redis_url: str = "redis://localhost:6379"
    model_repo: str = "IDKiro/sdxs-512-dreamshaper"
    torch_dtype: str = "float16"
    device: str = "cuda"


config = Config()
broker = RedisBroker(url=config.redis_url)
app = FastStream(broker=broker)


class LatentDecoder:
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
    def decode_latents(self, latents_bytes: bytes, shape: tuple[int, ...], dtype: str) -> bytes:
        """Decode latents to image bytes"""
        self.initialize()
        
        # Deserialize latents
        latents_array = deserialize_array(latents_bytes, shape, dtype)
        latents_tensor = torch.from_numpy(latents_array).to(self.device, dtype=self.dtype)
        
        # Scale latents back
        latents_tensor = latents_tensor / self.vae.config.scaling_factor
        
        # Decode to image
        with torch.no_grad():
            image = self.vae.decode(latents_tensor).sample
        
        # Denormalize from [-1, 1] to [0, 1]
        image = (image / 2 + 0.5).clamp(0, 1)
        
        # Convert to numpy
        image_np = image.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        
        # Convert to uint8
        image_uint8 = (image_np * 255).astype('uint8')
        
        # Encode as JPEG bytes
        return encode_frame(image_uint8)


def get_decoder() -> LatentDecoder:
    return LatentDecoder()


@broker.subscriber(channel=PubSub("latents:diffused"))
@broker.publisher(channel="images:processed")
@log_errors
async def decode_diffused_latents(
    carrier: CarrierMessage,
    decoder: LatentDecoder = Depends(get_decoder),
) -> CarrierMessage | None:
    """Decode diffused latents to processed images"""
    
    if not isinstance(carrier.content, LatentsMessage):
        return None
    
    latents_msg = carrier.content
    if latents_msg.source != "diffusion":
        return None
    
    start_time = time.time()
    
    try:
        # Decode latents to image
        image_bytes = decoder.decode_latents(
            latents_msg.latents,
            latents_msg.shape,
            latents_msg.dtype
        )
        
        processing_time = time.time() - start_time
        
        processed_msg = ProcessedFrameMessage(
            frame=image_bytes,
            timestamp=latents_msg.timestamp,
            processing_time=processing_time
        )
        
        logger.debug(f"Decoded latents to image in {processing_time:.3f}s")
        
        return CarrierMessage(content=processed_msg)
        
    except Exception as e:
        logger.error(f"Failed to decode latents: {e}")
        return None


if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting latent decoder...")
    asyncio.run(app.run()) 