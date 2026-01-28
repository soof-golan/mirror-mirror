from typing import Literal, overload

import torch
from diffusers import AutoencoderTiny, AutoPipelineForImage2Image, StableDiffusionImg2ImgPipeline

from mirror.fused_decode_upscale import FusedDecodeUpscale
from mirror.upscale import load_upscaler2, load_upscaler4

type UpscaleOptions = Literal[None, 2, 4]


@overload
def get_pipeline(
    base: Literal["stabilityai/sd-turbo"],
    upscale: UpscaleOptions,
) -> tuple[StableDiffusionImg2ImgPipeline, FusedDecodeUpscale]: ...


def get_pipeline(base: str, upscale: UpscaleOptions, compile_decoder: bool = False):
    vae_name = get_tiny_vae(base)
    vae = AutoencoderTiny.from_pretrained(vae_name, torch_dtype=torch.bfloat16)
    vae.to(memory_format=torch.channels_last)
    vae.eval()
    pipe = AutoPipelineForImage2Image.from_pretrained(
        base,
        variant="fp16",
        vae=vae,
        torch_dtype=torch.bfloat16,
    )
    pipe.unet.eval()
    pipe.text_encoder.eval()
    pipe.set_progress_bar_config(disable=True)
    pipe.to("cuda", torch.bfloat16)
    upscaler = get_upscaler(upscale)
    fused_decoder = FusedDecodeUpscale(
        decoder=vae.decoder.layers,
        upscaler=upscaler,
        scaling_factor=vae.config.scaling_factor,
    )
    fused_decoder.to("cuda", torch.bfloat16)
    fused_decoder.eval()
    fused_decoder = torch.compile(
        fused_decoder, mode="reduce-overhead", disable=not compile_decoder
    )

    return pipe, fused_decoder


def get_tiny_vae(base: str):
    if "sdxl" in base:
        return "madebyollin/taesdxl"
    else:
        return "madebyollin/taesd"


def get_upscaler(upscale: UpscaleOptions):
    if upscale == 2:
        return load_upscaler2()
    elif upscale == 4:
        return load_upscaler4()
    else:
        return torch.nn.Identity()
