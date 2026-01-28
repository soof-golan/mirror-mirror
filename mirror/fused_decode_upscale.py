from typing import Literal

import numpy as np
import torch

from mirror.utils import log_timing


class FusedDecodeUpscale(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module, upscaler: torch.nn.Module, scaling_factor: float):
        super().__init__()
        self.decoder = decoder
        self.upscaler = upscaler
        self.inverse_scaling_factor = 1.0 / scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.bfloat16)
        x = x.mul(self.inverse_scaling_factor)
        decoded = self.decoder(x)
        upscaled = self.upscaler(decoded)
        return upscaled


@log_timing
def torch_to_np(x: torch.Tensor) -> np.ndarray[tuple[int, int, Literal[3]], np.dtype[np.uint8]]:
    x = torch.clamp(x, 0.0, 1.0)
    if x.dim() == 4:
        x = x.squeeze(0)
    x = x.mul(255).to("cpu", torch.uint8)
    x = x.permute(1, 2, 0).numpy()
    image = x
    return image
