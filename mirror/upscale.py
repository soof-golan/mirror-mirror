"""Real-ESRGAN upscale worker."""

import logging
import sys

import torch
import torchvision.transforms.functional as F

logger = logging.getLogger(__name__)

# Patch for basicsr compatibility with torchvision >= 0.18
if not hasattr(F, "rgb_to_grayscale"):
    from torchvision.transforms.functional import rgb_to_grayscale

    F.rgb_to_grayscale = rgb_to_grayscale

sys.modules["torchvision.transforms.functional_tensor"] = F

from basicsr.archs.rrdbnet_arch import RRDBNet  # noqa: E402,I001
from realesrgan import RealESRGANer  # noqa: E402,I001
from basicsr.archs.srvgg_arch import SRVGGNetCompact  # noqa: E402,I001


__all__ = ["load_upscaler2", "load_upscaler4"]


def load_upscaler2(dtype=torch.bfloat16, device="cuda") -> RRDBNet:
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    upscaler_init = RealESRGANer(
        scale=2,
        model_path=url,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        device=torch.device(device),
    )
    model = upscaler_init.model.to(device).to(dtype)
    return model


def load_upscaler4(dtype=torch.bfloat16, device="cuda") -> SRVGGNetCompact:
    model = SRVGGNetCompact(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu"
    )
    url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth"
    upscaler_init = RealESRGANer(
        scale=4,
        model_path=url,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        device=torch.device(device),
    )
    model = upscaler_init.model.to(device).to(dtype)
    return model
