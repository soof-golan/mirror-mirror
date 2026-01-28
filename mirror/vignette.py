import logging
import os
import struct
from contextlib import ExitStack
from typing import NoReturn

import numpy as np
import zmq
from tqdm import tqdm
from typer import Typer

from mirror.config import Config
from mirror.generated.messages import Audio
from mirror.zmq_utils import create_socket

app = Typer()
logger = logging.getLogger(__name__)


class AudioIntensityTracker:
    def __init__(
        self,
        alpha_ewm: float = 0.67,
        alpha_std: float = 0.01,
        alpha_intensity: float = 0.3,
    ):
        self.alpha_ewm = alpha_ewm
        self.alpha_std = alpha_std
        self.alpha_intensity = alpha_intensity
        self.baseline = 0.0
        self.std = 0.01
        self.smoothed_intensity = 0.0
        self.frame_count = 0

    def update(self, audio_chunk: bytes) -> float:
        audio = np.frombuffer(audio_chunk, dtype=np.float32)
        rms = np.sqrt(np.mean(audio**2))

        self.baseline = self.alpha_ewm * rms + (1 - self.alpha_ewm) * self.baseline

        deviation = max(0.0, rms - self.baseline)

        squared_dev = deviation**2
        self.std = np.sqrt(
            self.alpha_std * squared_dev + (1 - self.alpha_std) * self.std**2
        )

        if self.std < 1e-6:
            return 0.0

        normalized = (deviation - 0.5 * self.std) / self.std
        raw_intensity = np.clip(normalized, 0.0, 1.0)

        self.smoothed_intensity = (
            self.alpha_intensity * raw_intensity
            + (1 - self.alpha_intensity) * self.smoothed_intensity
        )

        self.frame_count += 1
        if self.frame_count % 100 == 0:
            logger.debug(
                f"RMS={rms:.4f} baseline={self.baseline:.4f} "
                f"dev={deviation:.4f} std={self.std:.4f} "
                f"raw={raw_intensity:.3f} smoothed={self.smoothed_intensity:.3f}"
            )

        return self.smoothed_intensity


def pack_vignette(intensity: float, hue: float) -> bytes:
    return struct.pack("ff", intensity, hue)


def unpack_vignette(data: bytes) -> tuple[float, float]:
    return struct.unpack("ff", data)


@app.command()
def main(
    hue: float = 200.0,
    alpha_ewm: float = 0.67,
    alpha_std: float = 0.01,
    alpha_intensity: float = 0.3,
    log_level: str = "INFO",
) -> NoReturn:
    logging.basicConfig(level=log_level, format="%(levelname)s\t%(name)s\t%(message)s")
    config = Config()

    os.makedirs(config.ipc_dir, exist_ok=True)

    logger.info("Vignette process starting...")
    logger.info("Subscribing to audio at %s", config.zmq_audio_addr)
    logger.info("Publishing vignette to %s", config.ipc_ui_vignette_addr)
    logger.info(
        "Baseline EWM alpha=%.2f, Std EWM alpha=%.3f, Intensity EWM alpha=%.2f",
        alpha_ewm,
        alpha_std,
        alpha_intensity,
    )

    tracker = AudioIntensityTracker(
        alpha_ewm=alpha_ewm, alpha_std=alpha_std, alpha_intensity=alpha_intensity
    )

    with ExitStack() as stack:
        audio_sub = create_socket(
            zmq.SUB, config.zmq_audio_addr, "connect", conflate=False, exit_stack=stack
        )
        vignette_pub = create_socket(zmq.PUB, config.ipc_ui_vignette_addr, "bind", exit_stack=stack)

        logger.info("Vignette process ready")

        for _ in tqdm(iter(int, 1), unit="frame", disable=log_level != "INFO"):
            raw_audio = audio_sub.recv()
            audio_msg = Audio.parse(raw_audio)
            intensity = tracker.update(audio_msg.data)
            vignette_pub.send(pack_vignette(intensity, hue))


if __name__ == "__main__":
    app()
