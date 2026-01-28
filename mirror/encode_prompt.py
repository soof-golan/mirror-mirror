import logging
from typing import Any

import torch
from diffusers import (
    StableDiffusionControlNetPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
    ZImageImg2ImgPipeline,
)

from mirror.utils import log_timing

logger = logging.getLogger(__name__)


@log_timing
@torch.inference_mode
def encode_prompt(pipe, prompt: str) -> dict[str, Any]:
    logger.info(f"Updated prompt: {prompt}")
    if isinstance(
        pipe,
        StableDiffusionXLImg2ImgPipeline
        | StableDiffusionXLPipeline
        | StableDiffusionXLControlNetPipeline,
    ):
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt,
            negative_prompt="low quality, blurry, ugly, poor details, deformed, disfigured, bad anatomy, error, cropped",
            device=pipe.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
        }
    elif isinstance(
        pipe,
        StableDiffusionPipeline
        | StableDiffusionControlNetPipeline
        | StableDiffusionImg2ImgPipeline,
    ):
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt,
            negative_prompt="low quality, blurry, ugly, poor details, deformed, disfigured, bad anatomy, error, cropped",
            device=pipe.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
        }
    elif isinstance(pipe, ZImageImg2ImgPipeline):
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt,
            negative_prompt="low quality, blurry, ugly, poor details, deformed, disfigured, bad anatomy, error, cropped, watermark, text, error",
            device=pipe.device,
            do_classifier_free_guidance=False,
        )
        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
        }
    else:
        raise ValueError(f"Unsupported pipeline type: {type(pipe)}")
