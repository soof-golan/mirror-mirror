import logging
import time
import torch
from abc import ABC, abstractmethod
from functools import cache
from typing import Literal

import cv2
import numpy as np
import numpy.typing as npt
import faststream
from faststream import FastStream, Depends
from faststream.redis import RedisBroker, PubSub
from pydantic_settings import BaseSettings
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor2_0

from mirror_mirror.common import log_errors, assert_unreachable
from mirror_mirror.decode import decode_frame
from mirror_mirror.models import (
    CarrierMessage, 
    LatentsMessage, 
    EmbeddingMessage, 
    PromptMessage,
    ProcessedFrameMessage,
    deserialize_array,
    serialize_array
)

logger = logging.getLogger(__name__)


class Config(BaseSettings):
    redis_url: str = "redis://localhost:6379"
    mode: Literal["sdxs", "fake"] = "sdxs"
    model_repo: str = "IDKiro/sdxs-512-dreamshaper"
    torch_dtype: str = "float16"
    device: str = "cuda"
    num_inference_steps: int = 1
    guidance_scale: float = 0.0


config = Config()
broker = RedisBroker(url=config.redis_url)
app = FastStream(broker=broker)


class Diffuser(ABC):
    def __init__(self):
        self.current_prompt = "a beautiful landscape"
        self.current_embedding = None
        
    @abstractmethod
    def diffuse_latents(self, latents: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def update_prompt(self, prompt: str):
        pass
    
    @abstractmethod
    def update_embedding(self, embedding: torch.Tensor, text: str):
        pass


class FakeDiffuser(Diffuser):
    def diffuse_latents(self, latents: torch.Tensor) -> torch.Tensor:
        # Just return the input latents unchanged
        return latents
    
    def update_prompt(self, prompt: str):
        self.current_prompt = prompt
        logger.info(f"Updated prompt: {prompt}")
    
    def update_embedding(self, embedding: torch.Tensor, text: str):
        self.current_embedding = embedding
        logger.info(f"Updated embedding for: {text}")


class SDXSDiffuser(Diffuser):
    def __init__(self):
        super().__init__()
        self.pipe = None
        self.device = config.device
        self.dtype = getattr(torch, config.torch_dtype)
        
    def initialize(self):
        """Initialize the diffusion pipeline"""
        if self.pipe is None:
            logger.info(f"Loading SDXS pipeline from {config.model_repo}")
            self.pipe = StableDiffusionPipeline.from_pretrained(
                config.model_repo, 
                torch_dtype=self.dtype
            )
            self.pipe.set_progress_bar_config(disable=True)
            self.pipe.unet.set_attn_processor(AttnProcessor2_0())
            self.pipe.to(self.device)
            self.pipe.vae.eval()
            self.pipe.unet.eval()
            
            # Pre-encode default prompt
            self.update_prompt(self.current_prompt)
            logger.info("SDXS pipeline loaded successfully")
    
    @torch.inference_mode()
    def diffuse_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Apply diffusion to input latents"""
        self.initialize()
        
        latents = latents.to(self.device, dtype=self.dtype)
        
        # Use the current prompt embeddings
        if self.current_embedding is None:
            self.update_prompt(self.current_prompt)
        
        prompt_embeds, negative_prompt_embeds = self.current_embedding
        
        # Apply diffusion
        result = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            latents=latents,
            output_type="latent",
        ).images
        
        return result[0]
    
    def update_prompt(self, prompt: str):
        """Update the current prompt and encode it"""
        self.initialize()
        self.current_prompt = prompt
        
        # Encode prompt
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            [prompt],
            num_images_per_prompt=1,
            device=self.device,
            do_classifier_free_guidance=False,
        )
        
        self.current_embedding = (prompt_embeds, negative_prompt_embeds)
        logger.info(f"Updated prompt: {prompt}")
    
    def update_embedding(self, embedding: torch.Tensor, text: str):
        """Update with pre-computed embeddings"""
        # For now, just update the prompt text
        # In a full implementation, we would use the provided embeddings
        self.update_prompt(text)


@cache
def get_diffuser() -> Diffuser:
    if config.mode == "sdxs":
        return SDXSDiffuser()
    elif config.mode == "fake":
        return FakeDiffuser()
    else:
        return assert_unreachable()


@broker.subscriber(channel=PubSub("latents:camera"))
@broker.publisher(channel="latents:diffused")
@log_errors
async def diffuse_latents(
    carrier: CarrierMessage,
    diffuser: Diffuser = Depends(get_diffuser),
) -> CarrierMessage | None:
    """Apply diffusion to camera latents"""
    
    if not isinstance(carrier.content, LatentsMessage):
        return None
    
    latents_msg = carrier.content
    if latents_msg.source != "camera":
        return None
    
    start_time = time.time()
    
    try:
        # Deserialize latents
        latents_array = deserialize_array(
            latents_msg.latents, 
            latents_msg.shape, 
            latents_msg.dtype
        )
        latents_tensor = torch.from_numpy(latents_array)
        
        # Apply diffusion
        diffused_latents = diffuser.diffuse_latents(latents_tensor)
        
        # Serialize result
        diffused_array = diffused_latents.cpu().float().numpy()
        latents_data, shape, dtype = serialize_array(diffused_array)
        
        processing_time = time.time() - start_time
        
        result_msg = LatentsMessage(
            latents=latents_data,
            shape=shape,
            dtype=dtype,
            timestamp=latents_msg.timestamp,
            source="diffusion"
        )
        
        logger.debug(f"Diffused latents in {processing_time:.3f}s")
        
        return CarrierMessage(content=result_msg)
        
    except Exception as e:
        logger.error(f"Failed to diffuse latents: {e}")
        return None


@broker.subscriber(channel=PubSub("prompts:*", pattern=True))
@log_errors
async def update_prompts(
    carrier: CarrierMessage,
    diffuser: Diffuser = Depends(get_diffuser),
):
    """Update diffuser with new prompts"""
    
    if isinstance(carrier.content, PromptMessage):
        diffuser.update_prompt(carrier.content.prompt)
    elif isinstance(carrier.content, EmbeddingMessage):
        embedding_array = deserialize_array(
            carrier.content.embedding,
            carrier.content.shape,
            carrier.content.dtype
        )
        embedding_tensor = torch.from_numpy(embedding_array)
        diffuser.update_embedding(embedding_tensor, carrier.content.text)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Starting diffusion server in {config.mode} mode...")
    app.run()
